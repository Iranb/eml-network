from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eml_mnist.diagnostics import collect_eml_diagnostics
from eml_mnist.experiment_utils import ExperimentLogger, count_parameters
from eml_mnist.head_ablation import build_head, pairwise_prototype_loss
from eml_mnist.metrics import (
    brier_score,
    classification_accuracy,
    expected_calibration_error,
    negative_log_likelihood,
    pearson_corr,
)
from eml_mnist.training import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one classification head on frozen CNN features")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--head", choices=[
        "linear",
        "mlp",
        "cosine_prototype",
        "eml_no_ambiguity",
        "eml_raw_ambiguity",
        "eml_centered_ambiguity",
        "merc_linear",
        "merc_energy",
        "merc_linear_small",
        "merc_energy_small",
    ], required=True)
    parser.add_argument("--runs-root", default="reports/head_ablation/runs")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dataset", default="synthetic_shape")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--pairwise-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-margin", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-evals", type=int, default=1)
    return parser


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _load_tensor(path: Path, device: torch.device | None = None) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    if device is not None:
        value = value.to(device)
    return value


def _optional_tensor(path: Path, device: torch.device | None = None) -> torch.Tensor | None:
    if not path.exists():
        return None
    return _load_tensor(path, device=device)


def _load_split(features_dir: Path, split: str, device: torch.device) -> Dict[str, torch.Tensor | None]:
    return {
        "features": _load_tensor(features_dir / f"features_{split}.pt", device=device).float(),
        "labels": _load_tensor(features_dir / f"labels_{split}.pt", device=device).long(),
        "noise_level": _optional_tensor(features_dir / f"noise_level_{split}.pt", device=device),
        "occlusion_level": _optional_tensor(features_dir / f"occlusion_level_{split}.pt", device=device),
        "resistance_target": _optional_tensor(features_dir / f"resistance_target_{split}.pt", device=device),
        "evidence_target": _optional_tensor(features_dir / f"evidence_target_{split}.pt", device=device),
    }


def _mean_value(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu().item())
    if value is None:
        return float("nan")
    return float(value)


def _batch_indices(num_items: int, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, num_items, (batch_size,), device=device)


def _loss_for_outputs(
    head: torch.nn.Module,
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    pairwise_weight: float,
    pairwise_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ce = F.cross_entropy(outputs["logits"], labels)
    pairwise = pairwise_prototype_loss(head, margin=pairwise_margin)
    return ce + pairwise_weight * pairwise, pairwise


@torch.no_grad()
def evaluate_head(
    head: torch.nn.Module,
    split: Dict[str, torch.Tensor | None],
    batch_size: int = 256,
    warmup_eta: float = 1.0,
) -> Dict[str, Any]:
    features = split["features"]
    labels = split["labels"]
    if not torch.is_tensor(features) or not torch.is_tensor(labels):
        raise ValueError("split must include features and labels")
    head.eval()
    logits_parts: list[torch.Tensor] = []
    metric_values: Dict[str, list[torch.Tensor]] = {}
    for start in range(0, features.size(0), batch_size):
        end = min(features.size(0), start + batch_size)
        resistance_target = split.get("resistance_target")
        if torch.is_tensor(resistance_target):
            out = head(
                features[start:end],
                labels=labels[start:end],
                warmup_eta=warmup_eta,
                resistance_target=resistance_target[start:end],
            )
        else:
            out = head(features[start:end], labels=labels[start:end], warmup_eta=warmup_eta)
        logits_parts.append(out["logits"].detach())
        for key in (
            "positive_logit",
            "hard_negative_logit",
            "margin",
            "positive_drive",
            "hard_negative_drive",
            "positive_resistance",
            "hard_negative_resistance",
        ):
            value = out.get(key)
            if torch.is_tensor(value):
                metric_values.setdefault(key, []).append(value.detach())
        for key in ("sample_uncertainty", "ambiguity", "class_resistance"):
            value = out.get(key)
            if torch.is_tensor(value):
                metric_values.setdefault(key, []).append(value.detach().reshape(value.size(0), -1).mean(dim=-1))
        for key in ("update_gate", "precision"):
            value = out.get(key)
            if torch.is_tensor(value):
                metric_values.setdefault(key, []).append(value.detach().reshape(value.size(0), -1).mean(dim=-1))
        for key in ("support_factors", "conflict_factors"):
            value = out.get(key)
            if torch.is_tensor(value):
                metric_values.setdefault(key, []).append(value.detach().reshape(value.size(0), -1))
    logits = torch.cat(logits_parts, dim=0)
    predictions = logits.argmax(dim=-1)
    metrics: Dict[str, Any] = {
        "loss": negative_log_likelihood(logits, labels),
        "accuracy": classification_accuracy(logits, labels),
        "nll": negative_log_likelihood(logits, labels),
        "ece": expected_calibration_error(logits, labels),
        "brier": brier_score(logits, labels),
        "logits": logits.detach().cpu(),
        "labels": labels.detach().cpu(),
        "predictions": predictions.detach().cpu(),
        "correct": predictions.eq(labels).detach().cpu(),
    }
    for key, chunks in metric_values.items():
        joined = torch.cat(chunks, dim=0)
        metrics[key] = joined.detach().cpu()
        metrics[f"{key}_mean"] = _mean_value(joined)
    positive_resistance = metrics.get("positive_resistance")
    if torch.is_tensor(positive_resistance):
        noise = split.get("noise_level")
        occlusion = split.get("occlusion_level")
        if torch.is_tensor(noise):
            metrics["resistance_noise_corr"] = pearson_corr(positive_resistance.to(noise.device), noise)
        if torch.is_tensor(occlusion):
            metrics["resistance_occlusion_corr"] = pearson_corr(positive_resistance.to(occlusion.device), occlusion)
    evidence_target = split.get("evidence_target")
    support_factors = metrics.get("support_factors")
    if torch.is_tensor(evidence_target) and torch.is_tensor(support_factors):
        metrics["support_evidence_corr"] = pearson_corr(
            support_factors.float().mean(dim=-1).to(evidence_target.device),
            evidence_target.float(),
        )
    conflict_factors = metrics.get("conflict_factors")
    resistance_target = split.get("resistance_target")
    if torch.is_tensor(resistance_target) and torch.is_tensor(conflict_factors):
        metrics["conflict_resistance_corr"] = pearson_corr(
            conflict_factors.float().mean(dim=-1).to(resistance_target.device),
            resistance_target.float(),
        )
    return metrics


def _summary_from_eval(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        f"{prefix}_loss": metrics["loss"],
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_nll": metrics["nll"],
        f"{prefix}_ece": metrics["ece"],
        f"{prefix}_brier": metrics["brier"],
        f"{prefix}_positive_logit_mean": metrics.get("positive_logit_mean", float("nan")),
        f"{prefix}_hard_negative_logit_mean": metrics.get("hard_negative_logit_mean", float("nan")),
        f"{prefix}_margin_mean": metrics.get("margin_mean", float("nan")),
        f"{prefix}_positive_drive_mean": metrics.get("positive_drive_mean", float("nan")),
        f"{prefix}_positive_resistance_mean": metrics.get("positive_resistance_mean", float("nan")),
        f"{prefix}_hard_negative_drive_mean": metrics.get("hard_negative_drive_mean", float("nan")),
        f"{prefix}_hard_negative_resistance_mean": metrics.get("hard_negative_resistance_mean", float("nan")),
        f"{prefix}_sample_uncertainty_mean": metrics.get("sample_uncertainty_mean", float("nan")),
        f"{prefix}_ambiguity_mean": metrics.get("ambiguity_mean", float("nan")),
        f"{prefix}_class_resistance_mean": metrics.get("class_resistance_mean", float("nan")),
        f"{prefix}_update_gate_mean": metrics.get("update_gate_mean", float("nan")),
        f"{prefix}_precision_mean": metrics.get("precision_mean", float("nan")),
        f"{prefix}_resistance_noise_corr": metrics.get("resistance_noise_corr", float("nan")),
        f"{prefix}_resistance_occlusion_corr": metrics.get("resistance_occlusion_corr", float("nan")),
        f"{prefix}_support_evidence_corr": metrics.get("support_evidence_corr", float("nan")),
        f"{prefix}_conflict_resistance_corr": metrics.get("conflict_resistance_corr", float("nan")),
    }


def train_head(args: argparse.Namespace | SimpleNamespace) -> Dict[str, Any]:
    set_seed(args.seed)
    device = _device(args.device)
    features_dir = Path(args.features_dir)
    metadata = json.loads((features_dir / "metadata.json").read_text(encoding="utf-8"))
    train = _load_split(features_dir, "train", device)
    val = _load_split(features_dir, "val", device)
    test = _load_split(features_dir, "test", device)
    features = train["features"]
    labels = train["labels"]
    if not torch.is_tensor(features) or not torch.is_tensor(labels):
        raise ValueError("cached training features are missing")

    input_dim = int(features.size(-1))
    num_classes = int(metadata["num_classes"])
    head = build_head(
        args.head,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        temperature=args.temperature,
    ).to(device)
    run_id = args.run_id or f"frozen_{args.dataset}_{args.head}_seed{args.seed}"
    logger = ExperimentLogger(
        run_id=run_id,
        config={
            "mode": "head_ablation",
            "experiment_type": "frozen_features",
            "task_name": "image_classification",
            "model_name": args.head,
            "dataset_name": args.dataset,
            "seed": args.seed,
            "device": str(device),
            "features_dir": str(features_dir),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "pairwise_weight": args.pairwise_weight,
            "pairwise_margin": args.pairwise_margin,
        },
        root=args.runs_root,
    )
    model_info = logger.set_model_info(head, extra={"feature_dim": input_dim, "num_classes": num_classes})
    optimizer = AdamW(head.parameters(), lr=args.lr)
    best_val_acc = 0.0
    best_step = 0
    eval_count = 0
    stale_evals = 0
    early_stop_triggered = False
    start_time = time.time()
    last_metrics: Dict[str, Any] = {}

    for step in range(1, args.steps + 1):
        head.train()
        index = _batch_indices(features.size(0), args.batch_size, device)
        batch_features = features[index]
        batch_labels = labels[index]
        warmup_eta = min(1.0, step / max(1, args.steps // 2))
        batch_resistance_target = train.get("resistance_target")
        if torch.is_tensor(batch_resistance_target):
            batch_resistance_target = batch_resistance_target[index]
        outputs = head(
            batch_features,
            labels=batch_labels,
            warmup_eta=warmup_eta,
            resistance_target=batch_resistance_target,
        )
        loss, pairwise = _loss_for_outputs(head, outputs, batch_labels, args.pairwise_weight, args.pairwise_margin)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        accuracy = classification_accuracy(outputs["logits"].detach(), batch_labels)
        diagnostics = collect_eml_diagnostics(outputs)
        metrics = {
            "step": step,
            "train_loss": float(loss.detach().cpu().item()),
            "train_accuracy": accuracy,
            "pairwise_loss": float(pairwise.detach().cpu().item()),
            "learning_rate": args.lr,
            "wall_clock_time_sec": time.time() - start_time,
        }
        if step % max(1, args.eval_interval) == 0 or step == args.steps:
            val_metrics = evaluate_head(head, val, warmup_eta=warmup_eta)
            metrics.update(_summary_from_eval("val", val_metrics))
            eval_count += 1
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = float(val_metrics["accuracy"])
                best_step = step
                stale_evals = 0
            else:
                stale_evals += 1
        logger.log_step(metrics, diagnostics)
        last_metrics = metrics
        if (
            args.early_stop_patience > 0
            and eval_count >= args.early_stop_min_evals
            and stale_evals >= args.early_stop_patience
        ):
            early_stop_triggered = True
            break

    val_metrics = evaluate_head(head, val)
    test_metrics = evaluate_head(head, test)
    predictions_path = logger.run_dir / "eval_predictions.pt"
    torch.save(
        {
            "test_logits": test_metrics["logits"],
            "test_labels": test_metrics["labels"],
            "test_predictions": test_metrics["predictions"],
            "test_correct": test_metrics["correct"],
            "val_logits": val_metrics["logits"],
            "val_labels": val_metrics["labels"],
            "val_predictions": val_metrics["predictions"],
            "val_correct": val_metrics["correct"],
        },
        predictions_path,
    )
    torch.save(head.state_dict(), logger.run_dir / "head.pt")
    logger.add_artifact("eval_predictions", str(predictions_path))
    logger.add_artifact("head_state", str(logger.run_dir / "head.pt"))
    total_time = time.time() - start_time
    summary = {
        **_summary_from_eval("val", val_metrics),
        **_summary_from_eval("test", test_metrics),
        "best_metric": best_val_acc,
        "best_val_accuracy": best_val_acc,
        "best_step": best_step,
        "final_metric": test_metrics["accuracy"],
        "steps_run": step,
        "early_stop_triggered": early_stop_triggered,
        "final_train_loss": last_metrics.get("train_loss", float("nan")),
        "final_train_accuracy": last_metrics.get("train_accuracy", float("nan")),
        "total_train_time_sec": total_time,
    }
    logger.finalize(summary=summary, model_info=model_info)
    return {"run_dir": str(logger.run_dir), "summary": summary}


def main() -> None:
    result = train_head(build_parser().parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
