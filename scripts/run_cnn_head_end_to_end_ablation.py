from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eml_mnist.diagnostics import collect_eml_diagnostics
from eml_mnist.experiment_utils import ExperimentLogger
from eml_mnist.head_ablation import build_head, has_prototypes, pairwise_prototype_loss
from eml_mnist.merc import MERCResidualBlock
from eml_mnist.metrics import (
    brier_score,
    classification_accuracy,
    expected_calibration_error,
    negative_log_likelihood,
    pearson_corr,
)
from eml_mnist.model import ConvBackbone, EMLResidualBankBlock
from eml_mnist.training import set_seed

from extract_cnn_features import _build_datasets


MODEL_SPECS = [
    ("linear", "linear", False),
    ("mlp", "mlp", False),
    ("cosine_prototype", "cosine_prototype", False),
    ("eml_no_ambiguity", "eml_no_ambiguity", False),
    ("eml_centered_ambiguity", "eml_centered_ambiguity", False),
    ("eml_bank_centered_ambiguity", "eml_centered_ambiguity", True),
]

MERC_MODEL_SPECS = [
    ("merc_linear", "merc_linear", False, False),
    ("merc_energy", "merc_energy", False, False),
    ("merc_block_linear", "merc_linear", False, True),
    ("merc_block_energy", "merc_energy", False, True),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run end-to-end CNN plus head ablations")
    parser.add_argument("--dataset", choices=["synthetic_shape", "synthetic_evidence", "cifar10"], default="synthetic_shape")
    parser.add_argument("--mode", choices=["smoke", "medium", "full"], default="smoke")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", default="~/dataset")
    parser.add_argument("--runs-root", default="reports/head_ablation/runs")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--pairwise-weight", type=float, default=0.05)
    parser.add_argument("--pairwise-margin", type=float, default=0.0)
    parser.add_argument("--include-merc", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-evals", type=int, default=1)
    return parser


def _defaults(args: argparse.Namespace) -> tuple[int, int, int, int]:
    if args.mode == "smoke":
        return 256, 128, 128, args.steps or 30
    if args.mode == "medium":
        return 2048, 512, 512, args.steps or 300
    return 8192, 2048, 2048, args.steps or 1000


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _loader(dataset: torch.utils.data.Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )


def _cycle(loader: DataLoader) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


def _batch_to_image_label(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["image"].to(device), batch["label"].to(device).long()
    image, label = batch
    return image.to(device), label.to(device).long()


def _meta(batch: Any, key: str, device: torch.device) -> torch.Tensor | None:
    if isinstance(batch, dict) and key in batch and torch.is_tensor(batch[key]):
        return batch[key].to(device).float()
    return None


class CNNHeadModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        feature_dim: int,
        num_classes: int,
        head_name: str,
        use_bank: bool,
        use_merc_block: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = ConvBackbone(feature_dim=feature_dim, input_channels=input_channels)
        self.bank = (
            EMLResidualBankBlock(
                input_dim=feature_dim,
                hidden_dim=max(96, feature_dim * 2),
                bank_dim=feature_dim,
                dropout=0.0,
                gate_bias=-1.0,
            )
            if use_bank
            else None
        )
        self.merc_block = (
            MERCResidualBlock(
                input_dim=feature_dim,
                hidden_dim=max(96, feature_dim * 2),
                output_dim=feature_dim,
                num_support_factors=4,
                num_conflict_factors=4,
                init_gamma=0.3,
                old_confidence_init=4.0,
                update_threshold=1.0,
            )
            if use_merc_block
            else None
        )
        self.head = build_head(
            head_name,
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=max(96, feature_dim * 2),
            temperature=0.25,
        )

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        warmup_eta: float = 1.0,
    ) -> Dict[str, Any]:
        features = self.backbone(images)
        block_stats = None
        if self.bank is not None:
            block_stats = self.bank(features, warmup_eta=warmup_eta)
            features = block_stats["output"]
        elif self.merc_block is not None:
            block_stats = self.merc_block(features, warmup_eta=warmup_eta)
            features = block_stats["output"]
        out = self.head(features, labels=labels, warmup_eta=warmup_eta)
        out["features"] = features
        if block_stats is not None:
            out["block_stats"] = [block_stats]
            if "gate" in block_stats and torch.is_tensor(block_stats["gate"]):
                out["bank_gate_mean"] = block_stats["gate"].mean()
            elif "update_gate" in block_stats and torch.is_tensor(block_stats["update_gate"]):
                out["bank_gate_mean"] = block_stats["update_gate"].mean()
        return out


def _collect_scalar_chunks(out: Dict[str, Any], storage: Dict[str, list[torch.Tensor]], keys: tuple[str, ...]) -> None:
    for key in keys:
        value = out.get(key)
        if torch.is_tensor(value):
            storage.setdefault(key, []).append(value.detach().reshape(value.size(0), -1).mean(dim=-1))


@torch.no_grad()
def evaluate_model(model: CNNHeadModel, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []
    noise_parts: list[torch.Tensor] = []
    occlusion_parts: list[torch.Tensor] = []
    chunks: Dict[str, list[torch.Tensor]] = {}
    for batch in loader:
        image, label = _batch_to_image_label(batch, device)
        out = model(image, labels=label, warmup_eta=1.0)
        logits_parts.append(out["logits"].detach())
        labels_parts.append(label.detach())
        _collect_scalar_chunks(
            out,
            chunks,
            (
                "positive_logit",
                "hard_negative_logit",
                "margin",
                "positive_drive",
                "hard_negative_drive",
                "positive_resistance",
                "hard_negative_resistance",
                "sample_uncertainty",
                "ambiguity",
                "class_resistance",
            ),
        )
        noise = _meta(batch, "noise_level", device)
        occ = _meta(batch, "occlusion_level", device)
        if noise is not None:
            noise_parts.append(noise.detach())
        if occ is not None:
            occlusion_parts.append(occ.detach())
    logits = torch.cat(logits_parts, dim=0)
    labels = torch.cat(labels_parts, dim=0)
    predictions = logits.argmax(dim=-1)
    result: Dict[str, Any] = {
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
    for key, values in chunks.items():
        joined = torch.cat(values, dim=0)
        result[key] = joined.detach().cpu()
        result[f"{key}_mean"] = float(joined.float().mean().detach().cpu().item())
    positive_resistance = result.get("positive_resistance")
    if torch.is_tensor(positive_resistance):
        if noise_parts:
            noise = torch.cat(noise_parts, dim=0).detach().cpu()
            result["resistance_noise_corr"] = pearson_corr(positive_resistance, noise)
        if occlusion_parts:
            occ = torch.cat(occlusion_parts, dim=0).detach().cpu()
            result["resistance_occlusion_corr"] = pearson_corr(positive_resistance, occ)
    return result


def _summary(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
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
        f"{prefix}_resistance_noise_corr": metrics.get("resistance_noise_corr", float("nan")),
        f"{prefix}_resistance_occlusion_corr": metrics.get("resistance_occlusion_corr", float("nan")),
    }


def _not_run(args: argparse.Namespace, seed: int, model_name: str, loss_mode: str, reason: str) -> None:
    ExperimentLogger.not_run(
        run_id=f"e2e_{args.dataset}_{model_name}_{loss_mode}_seed{seed}",
        config={
            "mode": "head_ablation",
            "experiment_type": "end_to_end",
            "task_name": "image_classification",
            "model_name": model_name,
            "dataset_name": args.dataset,
            "seed": seed,
            "device": args.device,
            "loss_mode": loss_mode,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        reason=reason,
        root=args.runs_root,
    )


def _run_one(
    args: argparse.Namespace,
    seed: int,
    model_name: str,
    head_name: str,
    use_bank: bool,
    use_merc_block: bool,
    loss_mode: str,
    input_channels: int,
    num_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    steps: int,
    device: torch.device,
) -> None:
    set_seed(seed)
    model = CNNHeadModel(input_channels, args.feature_dim, num_classes, head_name, use_bank, use_merc_block=use_merc_block).to(device)
    pairwise_weight = args.pairwise_weight if loss_mode == "ce_pairwise" else 0.0
    if loss_mode == "ce_pairwise" and not has_prototypes(model.head):
        _not_run(args, seed, model_name, loss_mode, "pairwise prototype margin is not applicable")
        return
    run_id = f"e2e_{args.dataset}_{model_name}_{loss_mode}_seed{seed}"
    logger = ExperimentLogger(
        run_id=run_id,
        config={
            "mode": "head_ablation",
            "experiment_type": "end_to_end",
            "task_name": "image_classification",
            "model_name": model_name,
            "head_name": head_name,
            "dataset_name": args.dataset,
            "seed": seed,
            "device": str(device),
            "loss_mode": loss_mode,
            "steps": steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "pairwise_weight": pairwise_weight,
            "pairwise_margin": args.pairwise_margin,
            "use_eml_residual_bank": use_bank,
            "use_merc_block": use_merc_block,
        },
        root=args.runs_root,
    )
    model_info = logger.set_model_info(model, extra={"feature_dim": args.feature_dim, "num_classes": num_classes})
    optimizer = AdamW(model.parameters(), lr=args.lr)
    iterator = _cycle(train_loader)
    best_val_acc = 0.0
    best_step = 0
    eval_count = 0
    stale_evals = 0
    early_stop_triggered = False
    start = time.time()
    last_metrics: Dict[str, Any] = {}
    for step in range(1, steps + 1):
        model.train()
        batch = next(iterator)
        image, label = _batch_to_image_label(batch, device)
        warmup_eta = min(1.0, step / max(1, steps // 2))
        out = model(image, labels=label, warmup_eta=warmup_eta)
        ce = F.cross_entropy(out["logits"], label)
        pairwise = pairwise_prototype_loss(model.head, margin=args.pairwise_margin)
        loss = ce + pairwise_weight * pairwise
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        acc = classification_accuracy(out["logits"].detach(), label)
        metrics = {
            "step": step,
            "train_loss": float(loss.detach().cpu().item()),
            "train_accuracy": acc,
            "ce_loss": float(ce.detach().cpu().item()),
            "pairwise_loss": float(pairwise.detach().cpu().item()),
            "learning_rate": args.lr,
            "wall_clock_time_sec": time.time() - start,
        }
        if step % max(1, steps // 3) == 0 or step == steps:
            val_metrics = evaluate_model(model, val_loader, device)
            metrics.update(_summary("val", val_metrics))
            eval_count += 1
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = float(val_metrics["accuracy"])
                best_step = step
                stale_evals = 0
            else:
                stale_evals += 1
        logger.log_step(metrics, collect_eml_diagnostics(out))
        last_metrics = metrics
        if (
            args.early_stop_patience > 0
            and eval_count >= args.early_stop_min_evals
            and stale_evals >= args.early_stop_patience
        ):
            early_stop_triggered = True
            break
    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)
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
    logger.add_artifact("eval_predictions", str(predictions_path))
    logger.add_artifact("model_state", str(logger.run_dir / "model.pt"))
    torch.save(model.state_dict(), logger.run_dir / "model.pt")
    total_time = time.time() - start
    summary = {
        **_summary("val", val_metrics),
        **_summary("test", test_metrics),
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


def run(args: argparse.Namespace) -> None:
    if args.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    train_size, val_size, test_size, steps = _defaults(args)
    device = _device(args.device)
    for seed in args.seeds:
        set_seed(seed)
        try:
            train_set, val_set, test_set, input_channels, num_classes = _build_datasets(
                SimpleNamespace(
                    dataset=args.dataset,
                    data_dir=args.data_dir,
                    allow_download=args.allow_download,
                    train_size=train_size,
                    val_size=val_size,
                    test_size=test_size,
                    image_size=args.image_size,
                    seed=seed,
                )
            )
            train_loader = _loader(train_set, args, shuffle=True)
            val_loader = _loader(val_set, args, shuffle=False)
            test_loader = _loader(test_set, args, shuffle=False)
        except Exception as exc:
            reason = repr(exc)
            print(traceback.format_exc())
            specs = [(m, h, b, False) for (m, h, b) in MODEL_SPECS]
            if args.include_merc:
                specs.extend(MERC_MODEL_SPECS)
            for model_name, _head_name, _use_bank, _use_merc_block in specs:
                for loss_mode in ("ce", "ce_pairwise"):
                    _not_run(args, seed, model_name, loss_mode, reason)
            continue
        specs = [(m, h, b, False) for (m, h, b) in MODEL_SPECS]
        if args.include_merc:
            specs.extend(MERC_MODEL_SPECS)
        for model_name, head_name, use_bank, use_merc_block in specs:
            for loss_mode in ("ce", "ce_pairwise"):
                try:
                    _run_one(
                        args,
                        seed,
                        model_name,
                        head_name,
                        use_bank,
                        use_merc_block,
                        loss_mode,
                        input_channels,
                        num_classes,
                        train_loader,
                        val_loader,
                        test_loader,
                        steps,
                        device,
                    )
                except Exception as exc:
                    _not_run(args, seed, model_name, loss_mode, repr(exc))


def main() -> None:
    args = build_parser().parse_args()
    run(args)
    print(json.dumps({"status": "complete", "runs_root": args.runs_root}, sort_keys=True))


if __name__ == "__main__":
    main()
