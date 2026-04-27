from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eml_mnist.diagnostics import collect_eml_diagnostics
from eml_mnist.experiment_utils import ExperimentLogger, count_parameters, safe_torchvision_available
from eml_mnist.head_ablation import build_head, pairwise_prototype_loss
from eml_mnist.image_datasets import CIFARCorruptionDataset, SyntheticShapeEnergyDataset
from eml_mnist.metrics import (
    area_under_risk_coverage_curve,
    binary_auroc,
    brier_score,
    classification_accuracy,
    expected_calibration_error,
    negative_log_likelihood,
    pearson_corr,
    selective_risk_curve,
)
from eml_mnist.model import ConvBackbone
from eml_mnist.training import OptionalDatasetDependencyError, set_seed


HEADS = [
    "linear",
    "mlp",
    "cosine_prototype",
    "eml_no_ambiguity",
    "eml_centered_ambiguity",
    "eml_supervised_resistance",
    "merc_linear",
    "merc_energy",
]
CONDITIONS = ("clean", "noisy", "occluded")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EML uncertainty and selective classification benchmark")
    parser.add_argument("--dataset", choices=["synthetic_shape", "cifar10", "all"], default="synthetic_shape")
    parser.add_argument("--mode", choices=["smoke", "medium"], default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", default="~/dataset")
    parser.add_argument("--runs-root", default="reports/uncertainty_benchmark/runs")
    parser.add_argument("--report", default="reports/EML_UNCERTAINTY_BENCHMARK_REPORT.md")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--early-stop-min-evals", type=int, default=2)
    return parser


def _defaults(mode: str) -> Dict[str, int]:
    if mode == "medium":
        return {
            "train_size": 4096,
            "val_size": 1024,
            "test_size": 1024,
            "backbone_steps": 300,
            "head_steps": 250,
            "eval_interval": 25,
        }
    return {
        "train_size": 768,
        "val_size": 256,
        "test_size": 256,
        "backbone_steps": 40,
        "head_steps": 40,
        "eval_interval": 10,
    }


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _set_loader_strategy(num_workers: int) -> None:
    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")


def _loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )


def _batch_to_dict(batch: Any, device: torch.device) -> Dict[str, torch.Tensor]:
    if not isinstance(batch, dict):
        image, label = batch
        return {
            "image": image.to(device),
            "label": label.to(device).long(),
        }
    result: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if key == "label":
                result[key] = value.to(device).long()
            else:
                result[key] = value.to(device)
    return result


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _build_synthetic_splits(args: argparse.Namespace, seed: int, sizes: Dict[str, int]) -> tuple[Dataset, Dict[str, Dataset], int, int]:
    train = SyntheticShapeEnergyDataset(
        size=sizes["train_size"],
        image_size=args.image_size,
        seed=seed,
        target_type="shape",
        include_background_clutter=True,
        include_mask=False,
    )
    eval_sets = {
        "clean": SyntheticShapeEnergyDataset(
            size=sizes["test_size"],
            image_size=args.image_size,
            seed=seed + 10_000,
            target_type="shape",
            include_background_clutter=False,
            include_mask=False,
            forced_noise_name="low",
            forced_occlusion_name="none",
            forced_clutter_flag=0,
        ),
        "noisy": SyntheticShapeEnergyDataset(
            size=sizes["test_size"],
            image_size=args.image_size,
            seed=seed + 20_000,
            target_type="shape",
            include_background_clutter=False,
            include_mask=False,
            forced_noise_name="high",
            forced_occlusion_name="none",
            forced_clutter_flag=0,
        ),
        "occluded": SyntheticShapeEnergyDataset(
            size=sizes["test_size"],
            image_size=args.image_size,
            seed=seed + 30_000,
            target_type="shape",
            include_background_clutter=False,
            include_mask=False,
            forced_noise_name="low",
            forced_occlusion_name="partial",
            forced_clutter_flag=0,
        ),
    }
    return train, eval_sets, 3, train.num_classes


def _build_cifar_splits(args: argparse.Namespace, seed: int, sizes: Dict[str, int]) -> tuple[Dataset, Dict[str, Dataset], int, int]:
    ok, _version = safe_torchvision_available()
    if not ok:
        raise OptionalDatasetDependencyError("CIFAR-10 benchmark requires a working torchvision installation")
    from torchvision import datasets, transforms  # type: ignore

    transform = transforms.ToTensor()
    data_dir = str(Path(args.data_dir).expanduser())
    train_full = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=args.allow_download)
    test_full = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=args.allow_download)
    generator = torch.Generator().manual_seed(seed)
    train_size = min(sizes["train_size"], len(train_full))
    train_subset, _unused = random_split(train_full, [train_size, len(train_full) - train_size], generator=generator)
    val_size = min(sizes["val_size"], len(test_full) // 2)
    test_size = min(sizes["test_size"], len(test_full) - val_size)
    indices = torch.randperm(len(test_full), generator=generator).tolist()
    eval_base = {
        "clean": Subset(test_full, indices[:test_size]),
        "noisy": Subset(test_full, indices[test_size : test_size * 2]),
        "occluded": Subset(test_full, indices[test_size * 2 : test_size * 3]),
    }
    train = CIFARCorruptionDataset(train_subset, mode="mixed", seed=seed)
    eval_sets = {
        "clean": CIFARCorruptionDataset(eval_base["clean"], mode="clean", seed=seed + 10_000),
        "noisy": CIFARCorruptionDataset(eval_base["noisy"], mode="noisy", seed=seed + 20_000),
        "occluded": CIFARCorruptionDataset(eval_base["occluded"], mode="occluded", seed=seed + 30_000),
    }
    return train, eval_sets, 3, 10


def _build_dataset_splits(args: argparse.Namespace, dataset_name: str, seed: int, sizes: Dict[str, int]) -> tuple[Dataset, Dict[str, Dataset], int, int]:
    if dataset_name == "synthetic_shape":
        return _build_synthetic_splits(args, seed, sizes)
    if dataset_name == "cifar10":
        return _build_cifar_splits(args, seed, sizes)
    raise ValueError(f"unsupported dataset: {dataset_name}")


def _train_backbone(
    backbone: ConvBackbone,
    train_loader: DataLoader,
    num_classes: int,
    steps: int,
    device: torch.device,
) -> Dict[str, float]:
    classifier = torch.nn.Linear(backbone.projector[1].out_features, num_classes).to(device)
    optimizer = AdamW(list(backbone.parameters()) + list(classifier.parameters()), lr=1.0e-3)
    iterator = _cycle(train_loader)
    last_loss = 0.0
    last_acc = 0.0
    backbone.train()
    classifier.train()
    for _step in range(steps):
        batch = _batch_to_dict(next(iterator), device)
        features = backbone(batch["image"])
        logits = classifier(features)
        loss = F.cross_entropy(logits, batch["label"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        last_acc = classification_accuracy(logits.detach(), batch["label"])
    return {"feature_pretrain_loss": last_loss, "feature_pretrain_accuracy": last_acc}


@torch.no_grad()
def _extract_features(backbone: ConvBackbone, loader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    backbone.eval()
    collected: Dict[str, list[torch.Tensor]] = {
        "features": [],
        "labels": [],
        "noise_level": [],
        "occlusion_level": [],
        "resistance_target": [],
        "evidence_target": [],
    }
    for batch in loader:
        payload = _batch_to_dict(batch, device)
        collected["features"].append(backbone(payload["image"]).detach().cpu())
        collected["labels"].append(payload["label"].detach().cpu())
        for key in ("noise_level", "occlusion_level", "resistance_target", "evidence_target"):
            if key in payload:
                collected[key].append(payload[key].detach().float().cpu().reshape(-1))
    result = {
        "features": torch.cat(collected["features"], dim=0),
        "labels": torch.cat(collected["labels"], dim=0),
    }
    for key in ("noise_level", "occlusion_level", "resistance_target", "evidence_target"):
        if collected[key]:
            result[key] = torch.cat(collected[key], dim=0)
    return result


def _build_head_loss(
    head_name: str,
    head: torch.nn.Module,
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    resistance_target: torch.Tensor | None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    ce = F.cross_entropy(outputs["logits"], labels)
    pairwise = pairwise_prototype_loss(head, margin=0.0)
    total = ce
    metrics = {
        "ce_loss": float(ce.detach().cpu().item()),
        "pairwise_loss": float(pairwise.detach().cpu().item()),
        "resistance_supervision_loss": 0.0,
    }
    if head_name.startswith("eml_"):
        total = total + 0.05 * pairwise
    if head_name == "eml_supervised_resistance" and resistance_target is not None and torch.is_tensor(outputs.get("resistance_score")):
        target = resistance_target.to(device=outputs["resistance_score"].device, dtype=outputs["resistance_score"].dtype)
        resistance_loss = F.mse_loss(outputs["resistance_score"], target)
        total = total + 0.25 * resistance_loss
        metrics["resistance_supervision_loss"] = float(resistance_loss.detach().cpu().item())
    return total, metrics


def _uncertainty_from_outputs(outputs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, str]:
    if torch.is_tensor(outputs.get("resistance_score")):
        return outputs["resistance_score"].detach(), "positive_resistance"
    probs = torch.softmax(outputs["logits"], dim=-1)
    return (1.0 - probs.max(dim=-1).values).detach(), "one_minus_confidence"


@torch.no_grad()
def _evaluate_condition(
    head: torch.nn.Module,
    split: Dict[str, torch.Tensor],
    batch_size: int,
) -> Dict[str, Any]:
    device = next(head.parameters()).device
    features = split["features"]
    labels = split["labels"].long()
    logits_parts: list[torch.Tensor] = []
    uncertainty_parts: list[torch.Tensor] = []
    positive_resistance_parts: list[torch.Tensor] = []
    support_score_parts: list[torch.Tensor] = []
    conflict_score_parts: list[torch.Tensor] = []
    for start in range(0, features.size(0), batch_size):
        end = min(features.size(0), start + batch_size)
        batch_features = features[start:end].to(device=device, dtype=torch.float32)
        batch_labels = labels[start:end].to(device=device, dtype=torch.long)
        resistance_target = split.get("resistance_target")
        batch_target = (
            resistance_target[start:end].to(device=device, dtype=torch.float32)
            if torch.is_tensor(resistance_target)
            else None
        )
        outputs = head(batch_features, labels=batch_labels, warmup_eta=1.0, resistance_target=batch_target)
        logits_parts.append(outputs["logits"].detach().cpu())
        uncertainty, score_name = _uncertainty_from_outputs(outputs)
        uncertainty_parts.append(uncertainty.detach().cpu())
        if torch.is_tensor(outputs.get("positive_resistance")):
            positive_resistance_parts.append(outputs["positive_resistance"].detach().cpu())
        if torch.is_tensor(outputs.get("support_factors")):
            support_score_parts.append(outputs["support_factors"].detach().float().mean(dim=-1).cpu())
        if torch.is_tensor(outputs.get("conflict_factors")):
            conflict_score_parts.append(outputs["conflict_factors"].detach().float().mean(dim=-1).cpu())
    logits = torch.cat(logits_parts, dim=0)
    probs = torch.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values
    predictions = logits.argmax(dim=-1)
    labels_cpu = labels.cpu()
    correct = predictions.eq(labels_cpu)
    uncertainty = torch.cat(uncertainty_parts, dim=0)
    acceptance = -uncertainty
    result: Dict[str, Any] = {
        "accuracy": classification_accuracy(logits, labels_cpu),
        "nll": negative_log_likelihood(logits, labels_cpu),
        "ece": expected_calibration_error(logits, labels_cpu),
        "brier": brier_score(logits, labels_cpu),
        "confidence_mean": float(confidence.mean().item()),
        "uncertainty_mean": float(uncertainty.mean().item()),
        "selective_aurc": area_under_risk_coverage_curve(correct, acceptance),
        "score_name": score_name,
        "logits": logits,
        "labels": labels_cpu,
        "correct": correct,
        "uncertainty": uncertainty,
    }
    result.update(selective_risk_curve(correct, acceptance))
    if positive_resistance_parts:
        positive_resistance = torch.cat(positive_resistance_parts, dim=0)
        result["positive_resistance"] = positive_resistance
        result["positive_resistance_mean"] = float(positive_resistance.mean().item())
        if torch.is_tensor(split.get("noise_level")):
            result["resistance_noise_corr"] = pearson_corr(positive_resistance, split["noise_level"])
        if torch.is_tensor(split.get("occlusion_level")):
            result["resistance_occlusion_corr"] = pearson_corr(positive_resistance, split["occlusion_level"])
    if support_score_parts:
        support_score = torch.cat(support_score_parts, dim=0)
        result["support_score"] = support_score
        result["support_score_mean"] = float(support_score.mean().item())
        if torch.is_tensor(split.get("evidence_target")):
            result["support_evidence_corr"] = pearson_corr(support_score, split["evidence_target"])
    if conflict_score_parts:
        conflict_score = torch.cat(conflict_score_parts, dim=0)
        result["conflict_score"] = conflict_score
        result["conflict_score_mean"] = float(conflict_score.mean().item())
        if torch.is_tensor(split.get("resistance_target")):
            result["conflict_resistance_corr"] = pearson_corr(conflict_score, split["resistance_target"])
    return result


def _pooled_resistance_correlations(condition_metrics: Dict[str, Dict[str, Any]], eval_splits: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
    resistance_parts: list[torch.Tensor] = []
    noise_parts: list[torch.Tensor] = []
    occlusion_parts: list[torch.Tensor] = []
    support_parts: list[torch.Tensor] = []
    evidence_parts: list[torch.Tensor] = []
    conflict_parts: list[torch.Tensor] = []
    resistance_target_parts: list[torch.Tensor] = []
    for condition in CONDITIONS:
        metrics = condition_metrics[condition]
        split = eval_splits["clean"] if condition == "clean" else eval_splits[condition]
        resistance = metrics.get("positive_resistance")
        if not isinstance(resistance, torch.Tensor):
            continue
        resistance_parts.append(resistance)
        if torch.is_tensor(split.get("noise_level")):
            noise_parts.append(split["noise_level"].detach().cpu().reshape(-1))
        if torch.is_tensor(split.get("occlusion_level")):
            occlusion_parts.append(split["occlusion_level"].detach().cpu().reshape(-1))
        support = metrics.get("support_score")
        if isinstance(support, torch.Tensor) and torch.is_tensor(split.get("evidence_target")):
            support_parts.append(support)
            evidence_parts.append(split["evidence_target"].detach().cpu().reshape(-1))
        conflict = metrics.get("conflict_score")
        if isinstance(conflict, torch.Tensor) and torch.is_tensor(split.get("resistance_target")):
            conflict_parts.append(conflict)
            resistance_target_parts.append(split["resistance_target"].detach().cpu().reshape(-1))
    output: Dict[str, float] = {}
    if resistance_parts and noise_parts and len(resistance_parts) == len(noise_parts):
        output["pooled_resistance_noise_corr"] = pearson_corr(torch.cat(resistance_parts), torch.cat(noise_parts))
    if resistance_parts and occlusion_parts and len(resistance_parts) == len(occlusion_parts):
        output["pooled_resistance_occlusion_corr"] = pearson_corr(torch.cat(resistance_parts), torch.cat(occlusion_parts))
    if support_parts and evidence_parts and len(support_parts) == len(evidence_parts):
        output["pooled_support_evidence_corr"] = pearson_corr(torch.cat(support_parts), torch.cat(evidence_parts))
    if conflict_parts and resistance_target_parts and len(conflict_parts) == len(resistance_target_parts):
        output["pooled_conflict_resistance_corr"] = pearson_corr(torch.cat(conflict_parts), torch.cat(resistance_target_parts))
    return output


def _train_and_evaluate_head(
    args: argparse.Namespace,
    dataset_name: str,
    seed: int,
    head_name: str,
    train_split: Dict[str, torch.Tensor],
    eval_splits: Dict[str, Dict[str, torch.Tensor]],
    num_classes: int,
    feature_dim: int,
    sizes: Dict[str, int],
) -> Dict[str, Any]:
    device = _device(args.device)
    set_seed(seed)
    head = build_head(head_name, input_dim=feature_dim, num_classes=num_classes, hidden_dim=max(96, feature_dim * 2), temperature=0.25).to(device)
    logger = ExperimentLogger(
        run_id=f"uncertainty_{dataset_name}_{head_name}_seed{seed}",
        config={
            "mode": "uncertainty_benchmark",
            "experiment_type": "frozen_features_uncertainty",
            "task_name": "image_uncertainty_selective_classification",
            "model_name": head_name,
            "dataset_name": dataset_name,
            "seed": seed,
            "device": str(device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "train_size": int(train_split["features"].size(0)),
            "test_size": int(next(iter(eval_splits.values()))["features"].size(0)),
        },
        root=args.runs_root,
    )
    logger.set_model_info(head, extra={"feature_dim": feature_dim, "num_classes": num_classes})
    optimizer = AdamW(head.parameters(), lr=1.0e-3)
    start_time = time.time()
    train_features = train_split["features"].to(device).float()
    train_labels = train_split["labels"].to(device).long()
    train_resistance = train_split.get("resistance_target")
    if torch.is_tensor(train_resistance):
        train_resistance = train_resistance.to(device).float()
    best_val_acc = -1.0
    best_state: Dict[str, torch.Tensor] | None = None
    last_metrics: Dict[str, Any] = {}
    best_step = 0
    evals_without_improvement = 0
    eval_count = 0
    early_stop_triggered = False
    clean_features = eval_splits["clean"]["features"]
    val_count = max(1, min(clean_features.size(0) // 2, 256))
    val_split = {key: value[:val_count] for key, value in eval_splits["clean"].items()}
    clean_test_split = {key: value[val_count:] for key, value in eval_splits["clean"].items()}
    if clean_test_split["features"].size(0) == 0:
        clean_test_split = val_split

    for step in range(1, sizes["head_steps"] + 1):
        head.train()
        index = torch.randint(0, train_features.size(0), (args.batch_size,), device=device)
        batch_features = train_features[index]
        batch_labels = train_labels[index]
        batch_target = train_resistance[index] if torch.is_tensor(train_resistance) else None
        warmup_eta = min(1.0, step / max(1, sizes["head_steps"] // 2))
        outputs = head(batch_features, labels=batch_labels, warmup_eta=warmup_eta, resistance_target=batch_target)
        loss, extra_losses = _build_head_loss(head_name, head, outputs, batch_labels, batch_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics = {
            "step": step,
            "train_loss": float(loss.detach().cpu().item()),
            "train_accuracy": classification_accuracy(outputs["logits"].detach(), batch_labels),
            "learning_rate": 1.0e-3,
            "wall_clock_time_sec": time.time() - start_time,
            **extra_losses,
        }
        if step % sizes["eval_interval"] == 0 or step == sizes["head_steps"]:
            eval_count += 1
            val_metrics = _evaluate_condition(head, val_split, batch_size=max(64, args.batch_size))
            metrics["val_accuracy"] = val_metrics["accuracy"]
            metrics["val_ece"] = val_metrics["ece"]
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_step = step
                evals_without_improvement = 0
                best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            else:
                evals_without_improvement += 1
        logger.log_step(metrics, collect_eml_diagnostics(outputs))
        last_metrics = metrics
        if (
            eval_count >= args.early_stop_min_evals
            and args.early_stop_patience >= 0
            and evals_without_improvement >= args.early_stop_patience
        ):
            early_stop_triggered = True
            break

    if best_state is not None:
        head.load_state_dict(best_state)

    condition_metrics = {
        condition: _evaluate_condition(
            head,
            clean_test_split if condition == "clean" else split,
            batch_size=max(64, args.batch_size),
        )
        for condition, split in eval_splits.items()
    }
    clean_uncertainty = condition_metrics["clean"]["uncertainty"]
    summary: Dict[str, Any] = {
        "best_metric": best_val_acc,
        "final_metric": condition_metrics["clean"]["accuracy"],
        "final_train_loss": last_metrics.get("train_loss", float("nan")),
        "total_train_time_sec": time.time() - start_time,
        "best_step": best_step,
        "steps_run": last_metrics.get("step", 0),
        "early_stop_triggered": early_stop_triggered,
        "early_stop_patience": args.early_stop_patience,
    }
    for condition, metrics in condition_metrics.items():
        for key, value in metrics.items():
            if isinstance(value, (int, float, str)):
                summary[f"{condition}_{key}"] = value
    for corruption in ("noisy", "occluded"):
        corrupt_uncertainty = condition_metrics[corruption]["uncertainty"]
        labels = torch.cat(
            [
                torch.zeros(clean_uncertainty.size(0), dtype=torch.long),
                torch.ones(corrupt_uncertainty.size(0), dtype=torch.long),
            ],
            dim=0,
        )
        scores = torch.cat([clean_uncertainty, corrupt_uncertainty], dim=0)
        summary[f"clean_vs_{corruption}_auroc"] = binary_auroc(scores, labels)
    summary.update(_pooled_resistance_correlations(condition_metrics, {**eval_splits, "clean": clean_test_split}))
    predictions_path = logger.run_dir / "uncertainty_eval.pt"
    torch.save(
        {
            condition: {
                "logits": metrics["logits"],
                "labels": metrics["labels"],
                "correct": metrics["correct"],
                "uncertainty": metrics["uncertainty"],
            }
            for condition, metrics in condition_metrics.items()
        },
        predictions_path,
    )
    logger.add_artifact("eval_predictions", str(predictions_path))
    summary["condition_metrics_path"] = str(predictions_path)
    logger.finalize(summary=summary)
    return summary


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return "MISSING"
    if math.isnan(number):
        return "MISSING"
    return f"{number:.{digits}f}"


def generate_report(runs_root: str | Path, output: str | Path) -> Path:
    root = Path(runs_root)
    summary_path = root / "summary.csv"
    rows: list[Dict[str, Any]] = []
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                run_dir = Path(row.get("run_dir", ""))
                summary_json = run_dir / "summary.json"
                if summary_json.exists():
                    try:
                        row.update(json.loads(summary_json.read_text(encoding="utf-8")))
                    except Exception:
                        pass
                rows.append(row)

    grouped: Dict[tuple[str, str], list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "COMPLETED":
            grouped[(str(row.get("dataset_name", "")), str(row.get("model_name", "")))].append(row)

    lines = [
        "# EML Uncertainty and Resistance Benchmark",
        "",
        "## Scope",
        "",
        "- Frozen CNN features with head-only comparisons.",
        "- Baselines: `linear`, `mlp`, `cosine_prototype`.",
        "- EML heads: `eml_no_ambiguity`, `eml_centered_ambiguity`, `eml_supervised_resistance`.",
        "- MERC heads: `merc_linear`, `merc_energy`.",
        "- Corruption tasks: SyntheticShape clean/noisy/occluded; CIFAR clean/noisy/occluded when `torchvision` and data are available.",
        "- Synthetic label-noise ablation: NOT RUN in this report; it remains optional and should be added only with explicit label-noise controls.",
        "- Clean CIFAR accuracy claims are intentionally conservative.",
        "",
        "## Run Status",
        "",
        "| run_id | status | model | dataset | reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    if not rows:
        lines.append("| MISSING | MISSING | MISSING | MISSING | no runs found |")
    else:
        for row in rows:
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    row.get("run_id", ""),
                    row.get("status", ""),
                    row.get("model_name", ""),
                    row.get("dataset_name", ""),
                    row.get("reason", ""),
                )
            )

    for dataset_name in sorted({key[0] for key in grouped}):
        lines.extend(
            [
                "",
                f"## {dataset_name}",
                "",
                "| model | clean acc | noisy acc | occluded acc | clean ECE | clean Brier | clean selective AURC | clean->noisy AUROC | clean->occluded AUROC | resistance-noise corr | resistance-occlusion corr | support-evidence corr | conflict-resistance corr |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        dataset_models = {model for ds, model in grouped if ds == dataset_name}
        for model_name in sorted(dataset_models):
            bucket = grouped[(dataset_name, model_name)]
            def mean_of(key: str) -> float:
                values = []
                for row in bucket:
                    try:
                        values.append(float(row.get(key, "nan")))
                    except Exception:
                        pass
                return float(sum(values) / len(values)) if values else float("nan")

            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    model_name,
                    _fmt(mean_of("clean_accuracy")),
                    _fmt(mean_of("noisy_accuracy")),
                    _fmt(mean_of("occluded_accuracy")),
                    _fmt(mean_of("clean_ece")),
                    _fmt(mean_of("clean_brier")),
                    _fmt(mean_of("clean_selective_aurc")),
                    _fmt(mean_of("clean_vs_noisy_auroc")),
                    _fmt(mean_of("clean_vs_occluded_auroc")),
                    _fmt(mean_of("pooled_resistance_noise_corr")),
                    _fmt(mean_of("pooled_resistance_occlusion_corr")),
                    _fmt(mean_of("pooled_support_evidence_corr")),
                    _fmt(mean_of("pooled_conflict_resistance_corr")),
                )
            )

        lines.extend(
            [
                "",
                f"### {dataset_name} Detailed Runs",
                "",
                "| run_id | model | seed | best step | steps run | early stop | clean acc | clean ECE | clean AURC | noisy acc | occluded acc |",
                "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for model_name in sorted(dataset_models):
            for row in grouped[(dataset_name, model_name)]:
                lines.append(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                        row.get("run_id", ""),
                        model_name,
                        row.get("seed", ""),
                        _fmt(row.get("best_step"), 0),
                        _fmt(row.get("steps_run"), 0),
                        row.get("early_stop_triggered", ""),
                        _fmt(row.get("clean_accuracy")),
                        _fmt(row.get("clean_ece")),
                        _fmt(row.get("clean_selective_aurc")),
                        _fmt(row.get("noisy_accuracy")),
                        _fmt(row.get("occluded_accuracy")),
                    )
                )

    lines.extend(["", "## Conclusions", ""])
    for dataset_name in sorted({key[0] for key in grouped}):
        cosine = grouped.get((dataset_name, "cosine_prototype"), [])
        eml = grouped.get((dataset_name, "eml_centered_ambiguity"), [])
        if cosine and eml:
            cosine_acc = sum(float(row.get("clean_accuracy", "nan")) for row in cosine) / len(cosine)
            eml_acc = sum(float(row.get("clean_accuracy", "nan")) for row in eml) / len(eml)
            cosine_ece = sum(float(row.get("clean_ece", "nan")) for row in cosine) / len(cosine)
            eml_ece = sum(float(row.get("clean_ece", "nan")) for row in eml) / len(eml)
            cosine_aurc = sum(float(row.get("clean_selective_aurc", "nan")) for row in cosine) / len(cosine)
            eml_aurc = sum(float(row.get("clean_selective_aurc", "nan")) for row in eml) / len(eml)
            corr = grouped.get((dataset_name, "eml_supervised_resistance"), eml)
            noisy_corr = sum(float(row.get("pooled_resistance_noise_corr", "nan")) for row in corr) / len(corr)
            occ_corr = sum(float(row.get("pooled_resistance_occlusion_corr", "nan")) for row in corr) / len(corr)
            acc_claim = "supported in this benchmark only" if math.isfinite(eml_acc) and math.isfinite(cosine_acc) and eml_acc > cosine_acc else "not supported"
            cal_claim = "better" if math.isfinite(eml_ece) and math.isfinite(cosine_ece) and eml_ece < cosine_ece else "not better"
            sel_claim = "better" if math.isfinite(eml_aurc) and math.isfinite(cosine_aurc) and eml_aurc < cosine_aurc else "not better"
            corr_strength = max(abs(noisy_corr) if math.isfinite(noisy_corr) else 0.0, abs(occ_corr) if math.isfinite(occ_corr) else 0.0)
            corr_claim = "supported" if corr_strength >= 0.2 else "not supported or weak"
            lines.append(f"- `{dataset_name}` clean accuracy vs cosine: EML centered {_fmt(eml_acc)} vs cosine {_fmt(cosine_acc)}. Head advantage is {acc_claim}.")
            lines.append(f"- `{dataset_name}` calibration vs cosine: EML centered ECE {_fmt(eml_ece)} vs cosine {_fmt(cosine_ece)}. Calibration is {cal_claim}.")
            lines.append(f"- `{dataset_name}` selective prediction vs cosine: EML centered clean AURC {_fmt(eml_aurc)} vs cosine {_fmt(cosine_aurc)}. Selective prediction is {sel_claim}.")
            lines.append(f"- `{dataset_name}` resistance-correlation check: noise {_fmt(noisy_corr)}, occlusion {_fmt(occ_corr)}. Corruption correlation is {corr_claim}.")
        else:
            lines.append(f"- `{dataset_name}` has incomplete paired runs. EML head advantage is NOT SUPPORTED from this report.")
        merc_rows = grouped.get((dataset_name, "merc_linear"), []) + grouped.get((dataset_name, "merc_energy"), [])
        if merc_rows:
            support_vals = [float(row.get("pooled_support_evidence_corr", "nan")) for row in merc_rows]
            conflict_vals = [float(row.get("pooled_conflict_resistance_corr", "nan")) for row in merc_rows]
            support_vals = [value for value in support_vals if math.isfinite(value)]
            conflict_vals = [value for value in conflict_vals if math.isfinite(value)]
            support_text = _fmt(sum(support_vals) / len(support_vals)) if support_vals else "MISSING"
            conflict_text = _fmt(sum(conflict_vals) / len(conflict_vals)) if conflict_vals else "MISSING"
            lines.append(f"- `{dataset_name}` MERC support/conflict alignment: support-evidence {support_text}, conflict-resistance {conflict_text}. MERC alignment is not claimed when values are MISSING or weak.")
    if ("cifar10", "cosine_prototype") not in grouped or ("cifar10", "eml_centered_ambiguity") not in grouped:
        lines.append("- `cifar10` was NOT RUN here because `torchvision` is unavailable in the local environment.")
    lines.extend(
        [
            "",
            "- If the EML rows do not beat cosine on calibration or selective risk, the benchmark does not support an EML head advantage.",
            "",
            "## Raw Artifacts",
            "",
            f"- Runs root: `{root}`",
            f"- Summary CSV: `{summary_path}`",
        ]
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _run_dataset(args: argparse.Namespace, dataset_name: str, seed: int, sizes: Dict[str, int]) -> None:
    device = _device(args.device)
    set_seed(seed)
    train_set, eval_sets, input_channels, num_classes = _build_dataset_splits(args, dataset_name, seed, sizes)
    train_loader = _loader(train_set, args.batch_size, args.num_workers, shuffle=True)
    backbone = ConvBackbone(feature_dim=args.feature_dim, input_channels=input_channels).to(device)
    _train_backbone(backbone, train_loader, num_classes=num_classes, steps=sizes["backbone_steps"], device=device)
    train_features = _extract_features(backbone, _loader(train_set, args.batch_size, args.num_workers, shuffle=False), device)
    eval_features = {
        condition: _extract_features(backbone, _loader(dataset, args.batch_size, args.num_workers, shuffle=False), device)
        for condition, dataset in eval_sets.items()
    }
    for head_name in HEADS:
        _train_and_evaluate_head(
            args=args,
            dataset_name=dataset_name,
            seed=seed,
            head_name=head_name,
            train_split=train_features,
            eval_splits=eval_features,
            num_classes=num_classes,
            feature_dim=args.feature_dim,
            sizes=sizes,
        )


def run(args: argparse.Namespace) -> Path:
    _set_loader_strategy(args.num_workers)
    sizes = _defaults(args.mode)
    dataset_names = ["synthetic_shape", "cifar10"] if args.dataset == "all" else [args.dataset]
    for dataset_name in dataset_names:
        for seed in args.seeds:
            try:
                _run_dataset(args, dataset_name, seed, sizes)
            except Exception as exc:
                ExperimentLogger.not_run(
                    run_id=f"uncertainty_{dataset_name}_seed{seed}",
                    config={
                        "mode": "uncertainty_benchmark",
                        "experiment_type": "frozen_features_uncertainty",
                        "task_name": "image_uncertainty_selective_classification",
                        "model_name": "benchmark_bundle",
                        "dataset_name": dataset_name,
                        "seed": seed,
                        "device": args.device,
                    },
                    reason=repr(exc),
                    root=args.runs_root,
                )
    return generate_report(args.runs_root, args.report)


def main() -> None:
    report_path = run(build_parser().parse_args())
    print(json.dumps({"report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
