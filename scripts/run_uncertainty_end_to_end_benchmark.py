from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eml_mnist.experiment_utils import ExperimentLogger, safe_torchvision_available
from eml_mnist.head_ablation import build_head
from eml_mnist.image_datasets import CIFARCorruptionDataset, SyntheticShapeEnergyDataset
from eml_mnist.metrics import area_under_risk_coverage_curve, binary_auroc, classification_accuracy, expected_calibration_error, pearson_corr
from eml_mnist.model import ConvBackbone
from eml_mnist.training import OptionalDatasetDependencyError, set_seed


MODELS = (
    "cnn_linear",
    "cnn_mlp",
    "cnn_cosine_prototype",
    "cnn_eml_centered_ambiguity",
    "cnn_eml_supervised_resistance",
    "cnn_cosine_eml_aux_resistance",
    "cnn_merc_energy",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run end-to-end uncertainty benchmark")
    parser.add_argument("--dataset", choices=["synthetic_shape_uncertainty", "cifar10_corrupt"], default="synthetic_shape_uncertainty")
    parser.add_argument("--mode", choices=["smoke", "medium"], default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", default="~/dataset")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--runs-root", default="reports/uncertainty_end_to_end/runs")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--early-stop-min-evals", type=int, default=2)
    return parser


def _sizes(mode: str) -> dict[str, int]:
    return {"train": 512, "test": 256, "steps": 50, "feature": 64} if mode == "smoke" else {"train": 4096, "test": 1024, "steps": 300, "feature": 64}


def _loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, persistent_workers=num_workers > 0)


def _batch(batch, device: torch.device) -> Dict[str, torch.Tensor]:
    if not isinstance(batch, dict):
        image, label = batch
        return {"image": image.to(device), "label": label.to(device).long()}
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
    out["label"] = out["label"].long()
    return out


def _build_splits(args: argparse.Namespace, seed: int, sizes: dict[str, int]) -> tuple[Dataset, dict[str, Dataset], int]:
    if args.dataset == "synthetic_shape_uncertainty":
        train = SyntheticShapeEnergyDataset(sizes["train"], seed=seed, target_type="shape", include_mask=False)
        evals = {
            "clean": SyntheticShapeEnergyDataset(sizes["test"], seed=seed + 10_000, target_type="shape", include_mask=False, forced_noise_name="low", forced_occlusion_name="none", forced_clutter_flag=0),
            "noisy": SyntheticShapeEnergyDataset(sizes["test"], seed=seed + 20_000, target_type="shape", include_mask=False, forced_noise_name="high", forced_occlusion_name="none", forced_clutter_flag=0),
            "occluded": SyntheticShapeEnergyDataset(sizes["test"], seed=seed + 30_000, target_type="shape", include_mask=False, forced_noise_name="low", forced_occlusion_name="partial", forced_clutter_flag=0),
        }
        return train, evals, train.num_classes
    ok, _version = safe_torchvision_available()
    if not ok:
        raise OptionalDatasetDependencyError("CIFAR run requires torchvision")
    from torchvision import datasets, transforms  # type: ignore

    transform = transforms.ToTensor()
    data_dir = str(Path(args.data_dir).expanduser())
    train_full = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=args.allow_download)
    test_full = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=args.allow_download)
    generator = torch.Generator().manual_seed(seed)
    train_size = min(sizes["train"], len(train_full))
    train_subset, _unused = random_split(train_full, [train_size, len(train_full) - train_size], generator=generator)
    indices = torch.randperm(len(test_full), generator=generator).tolist()
    test_size = min(sizes["test"], len(test_full) // 3)
    evals = {
        "clean": CIFARCorruptionDataset(Subset(test_full, indices[:test_size]), mode="clean", seed=seed + 10_000),
        "noisy": CIFARCorruptionDataset(Subset(test_full, indices[test_size : 2 * test_size]), mode="noisy", seed=seed + 20_000),
        "occluded": CIFARCorruptionDataset(Subset(test_full, indices[2 * test_size : 3 * test_size]), mode="occluded", seed=seed + 30_000),
    }
    return CIFARCorruptionDataset(train_subset, mode="mixed", seed=seed), evals, 10


class EndToEndModel(nn.Module):
    def __init__(self, model_name: str, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.model_name = model_name
        self.backbone = ConvBackbone(feature_dim=feature_dim, input_channels=3)
        if model_name == "cnn_cosine_eml_aux_resistance":
            self.head = build_head("cosine_prototype", feature_dim, num_classes, hidden_dim=128)
            self.aux = build_head("eml_supervised_resistance", feature_dim, num_classes, hidden_dim=128)
        else:
            head_name = model_name.removeprefix("cnn_")
            self.head = build_head(head_name, feature_dim, num_classes, hidden_dim=128)
            self.aux = None

    def forward(self, image: torch.Tensor, labels: torch.Tensor | None = None, resistance_target: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        features = self.backbone(image)
        out = self.head(features, labels=labels, resistance_target=resistance_target)
        if self.aux is not None:
            aux = self.aux(features, labels=labels, resistance_target=resistance_target)
            out["aux_resistance_score"] = aux["resistance_score"]
            out["resistance_score"] = aux["resistance_score"]
            out["aux_resistance"] = aux["resistance"]
        return out


def _evaluate(model: EndToEndModel, evals: dict[str, Dataset], args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, float] = {}
    clean_risk = None
    for condition, dataset in evals.items():
        logits_parts = []
        labels_parts = []
        risk_parts = []
        noise_parts = []
        occ_parts = []
        with torch.no_grad():
            for batch in _loader(dataset, args.batch_size, args.num_workers, shuffle=False):
                payload = _batch(batch, device)
                out = model(payload["image"], labels=payload["label"], resistance_target=payload.get("resistance_target"))
                logits_parts.append(out["logits"].cpu())
                labels_parts.append(payload["label"].cpu())
                probs = torch.softmax(out["logits"], dim=-1)
                risk = out.get("resistance_score", 1.0 - probs.max(dim=-1).values)
                risk_parts.append(risk.detach().reshape(-1).cpu())
                if "noise_level" in payload:
                    noise_parts.append(payload["noise_level"].detach().reshape(-1).cpu())
                if "occlusion_level" in payload:
                    occ_parts.append(payload["occlusion_level"].detach().reshape(-1).cpu())
        logits = torch.cat(logits_parts)
        labels = torch.cat(labels_parts)
        risk = torch.cat(risk_parts)
        correct = logits.argmax(dim=-1).eq(labels)
        metrics[f"{condition}_accuracy"] = classification_accuracy(logits, labels)
        metrics[f"{condition}_ece"] = expected_calibration_error(logits, labels)
        metrics[f"{condition}_aurc"] = area_under_risk_coverage_curve(correct, -risk)
        if condition == "clean":
            clean_risk = risk
        elif clean_risk is not None:
            binary = torch.cat([torch.zeros(clean_risk.numel(), dtype=torch.long), torch.ones(risk.numel(), dtype=torch.long)])
            metrics[f"clean_vs_{condition}_auroc"] = binary_auroc(torch.cat([clean_risk, risk]), binary)
        if noise_parts:
            metrics[f"{condition}_resistance_noise_corr"] = pearson_corr(risk, torch.cat(noise_parts))
        if occ_parts:
            metrics[f"{condition}_resistance_occlusion_corr"] = pearson_corr(risk, torch.cat(occ_parts))
    return metrics


def _run_one(args: argparse.Namespace, model_name: str, seed: int) -> None:
    set_seed(seed)
    if args.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    sizes = _sizes(args.mode)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    try:
        train, evals, num_classes = _build_splits(args, seed, sizes)
    except Exception as exc:
        ExperimentLogger.not_run(
            run_id=f"uncertainty_e2e_{args.dataset}_{model_name}_seed{seed}",
            config={"model_name": model_name, "dataset_name": args.dataset, "seed": seed, "task_name": "end_to_end_uncertainty"},
            reason=repr(exc),
            root=args.runs_root,
        )
        return
    model = EndToEndModel(model_name, sizes["feature"], num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=1.0e-3)
    loader = _loader(train, args.batch_size, args.num_workers, shuffle=True)
    logger = ExperimentLogger(
        run_id=f"uncertainty_e2e_{args.dataset}_{model_name}_seed{seed}",
        config={"mode": args.mode, "task_name": "end_to_end_uncertainty", "model_name": model_name, "dataset_name": args.dataset, "seed": seed, "device": str(device)},
        root=args.runs_root,
    )
    start = time.time()
    iterator = iter(loader)
    max_steps = int(args.max_steps or sizes["steps"])
    best_metric = float("-inf")
    best_step = 0
    best_summary: Dict[str, float] | None = None
    stale_evals = 0
    eval_count = 0
    early_stop_triggered = False
    steps_run = 0
    for step in range(1, max_steps + 1):
        steps_run = step
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        payload = _batch(batch, device)
        out = model(payload["image"], labels=payload["label"], resistance_target=payload.get("resistance_target"))
        loss = F.cross_entropy(out["logits"], payload["label"])
        if model_name in {"cnn_eml_supervised_resistance", "cnn_cosine_eml_aux_resistance"} and "resistance_score" in out and "resistance_target" in payload:
            loss = loss + 0.25 * F.mse_loss(out["resistance_score"], payload["resistance_target"].float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {"step": step, "train_loss": float(loss.detach().cpu().item()), "wall_clock_time_sec": time.time() - start}
        if step % args.eval_interval == 0 or step == max_steps:
            current = _evaluate(model, evals, args, device)
            eval_count += 1
            metric = float(current.get("clean_accuracy", float("nan")))
            if metric > best_metric + 1.0e-8:
                best_metric = metric
                best_step = step
                best_summary = current
                stale_evals = 0
            else:
                stale_evals += 1
            row.update({f"eval_{key}": value for key, value in current.items()})
            if eval_count >= args.early_stop_min_evals and stale_evals >= args.early_stop_patience:
                early_stop_triggered = True
                logger.log_step(row, {})
                break
        logger.log_step(row, {})
    summary = best_summary or _evaluate(model, evals, args, device)
    summary.update(
        {
            "best_metric": best_metric if best_summary is not None else summary.get("clean_accuracy", float("nan")),
            "final_metric": summary.get("clean_accuracy", float("nan")),
            "best_step": best_step,
            "steps_run": steps_run,
            "early_stop_triggered": early_stop_triggered,
            "total_train_time_sec": time.time() - start,
        }
    )
    logger.finalize(summary=summary)


def _write_report(runs_root: Path) -> None:
    report = Path("reports/UNCERTAINTY_END_TO_END_REPORT.md")
    lines = ["# End-to-End Uncertainty Benchmark", "", "| model | dataset | seed | clean acc | noisy acc | occluded acc | noisy AUROC | occluded AUROC |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    summary = runs_root / "summary.csv"
    if summary.exists():
        with summary.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                metrics = json.loads(row.get("metrics_json") or "{}")
                lines.append(
                    f"| {row.get('model_name')} | {row.get('dataset_name')} | {row.get('seed')} | {metrics.get('clean_accuracy', 'MISSING')} | {metrics.get('noisy_accuracy', 'MISSING')} | {metrics.get('occluded_accuracy', 'MISSING')} | {metrics.get('clean_vs_noisy_auroc', 'MISSING')} | {metrics.get('clean_vs_occluded_auroc', 'MISSING')} |"
                )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    for seed in args.seeds:
        for model_name in args.models:
            _run_one(args, model_name, seed)
    _write_report(Path(args.runs_root))


if __name__ == "__main__":
    main()
