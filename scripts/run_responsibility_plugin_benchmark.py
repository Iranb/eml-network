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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eml_mnist.experiment_utils import ExperimentLogger
from eml_mnist.metrics import area_under_risk_coverage_curve, classification_accuracy, pearson_corr
from eml_mnist.primitives import EMLPrecisionUpdate, EMLResponsibility, EMLUnit
from eml_mnist.training import set_seed


MODULES = (
    "identity",
    "mlp_refinement",
    "sigmoid_eml_gate",
    "responsibility_no_null",
    "thresholded_null",
    "thresholded_null_precision",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EML responsibility/null plug-in benchmark")
    parser.add_argument("--mode", choices=["smoke", "medium"], default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--runs-root", default="reports/responsibility_plugin/runs")
    parser.add_argument("--modules", nargs="+", choices=MODULES, default=list(MODULES))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--early-stop-min-evals", type=int, default=2)
    return parser


def _sizes(mode: str) -> dict[str, int]:
    return {"train": 512, "test": 256, "steps": 50, "dim": 16} if mode == "smoke" else {"train": 4096, "test": 1024, "steps": 300, "dim": 32}


def _make_data(size: int, dim: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    clean = torch.randn(size, dim, generator=generator)
    labels = (clean[:, 0] + clean[:, 1] - 0.5 * clean[:, 2] > 0).long()
    severity = torch.rand(size, generator=generator)
    corrupted = clean + severity.unsqueeze(1) * torch.randn(size, dim, generator=generator) * 1.25
    return TensorDataset(corrupted.float(), clean.float(), labels, severity.float())


class PluginClassifier(nn.Module):
    def __init__(self, dim: int, module_name: str) -> None:
        super().__init__()
        self.module_name = module_name
        self.refine = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.drive = nn.Linear(dim, 2)
        self.resistance = nn.Linear(dim, 2)
        self.value = nn.Linear(dim, dim)
        self.eml = EMLUnit(dim=2, init_gamma=0.3, init_lambda=1.0)
        self.responsibility = EMLResponsibility(use_null=True, mode="thresholded_null", evidence_threshold=0.25)
        self.precision = EMLPrecisionUpdate(old_confidence_init=4.0, update_threshold=0.5)
        self.classifier = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        candidate = self.refine(x)
        diagnostics: Dict[str, torch.Tensor] = {}
        if self.module_name == "identity":
            refined = x
        elif self.module_name == "mlp_refinement":
            refined = x + 0.25 * candidate
        elif self.module_name == "sigmoid_eml_gate":
            drive = self.drive(x)
            resistance = F.softplus(self.resistance(x))
            energy = self.eml(drive, resistance).mean(dim=-1, keepdim=True)
            gate = torch.sigmoid(energy)
            refined = x + gate * (candidate - x)
            diagnostics.update({"drive": drive, "resistance": resistance, "energy": energy.squeeze(-1), "update_strength": gate.squeeze(-1)})
        else:
            drive = self.drive(x)
            resistance = F.softplus(self.resistance(x))
            energy = self.eml(drive, resistance)
            mode = "thresholded_null" if "thresholded" in self.module_name else "standard"
            resp = self.responsibility(energy, use_null="no_null" not in self.module_name, mode=mode)
            values = torch.stack([x, self.value(candidate)], dim=1)
            message = (resp["neighbor_weights"].unsqueeze(-1) * values).sum(dim=1)
            update_strength = resp["update_strength"]
            if self.module_name == "thresholded_null_precision":
                update = self.precision(x, message, energy.mean(dim=-1), update_strength=update_strength)
                refined = update["updated_state"]
                diagnostics.update({"update_gate": update["update_gate"].reshape(x.size(0), -1).mean(dim=-1)})
            else:
                refined = x + update_strength.unsqueeze(-1) * (message - x)
            diagnostics.update(
                {
                    "drive": drive,
                    "resistance": resistance,
                    "energy": energy,
                    "null_weight": resp["null_weight"] if torch.is_tensor(resp["null_weight"]) else torch.zeros_like(update_strength),
                    "update_strength": update_strength,
                    "responsibility_entropy": resp["entropy"],
                }
            )
        logits = self.classifier(refined)
        diagnostics["logits"] = logits
        return diagnostics


def _evaluate(model: PluginClassifier, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    logits_parts = []
    labels_parts = []
    severity_parts = []
    null_parts = []
    strength_parts = []
    with torch.no_grad():
        for corrupted, _clean, labels, severity in loader:
            out = model(corrupted.to(device))
            logits_parts.append(out["logits"].cpu())
            labels_parts.append(labels)
            severity_parts.append(severity)
            if "null_weight" in out:
                null_parts.append(out["null_weight"].detach().cpu().reshape(-1))
            if "update_strength" in out:
                strength_parts.append(out["update_strength"].detach().cpu().reshape(-1))
    logits = torch.cat(logits_parts)
    labels = torch.cat(labels_parts)
    severity = torch.cat(severity_parts)
    correct = logits.argmax(dim=-1).eq(labels)
    confidence = torch.softmax(logits, dim=-1).max(dim=-1).values
    metrics: Dict[str, float] = {
        "accuracy": classification_accuracy(logits, labels),
        "selective_aurc": area_under_risk_coverage_curve(correct, confidence),
    }
    if null_parts:
        null_weight = torch.cat(null_parts)
        metrics["null_weight_mean"] = float(null_weight.mean().item())
        metrics["null_severity_corr"] = pearson_corr(null_weight, severity)
    if strength_parts:
        strength = torch.cat(strength_parts)
        metrics["update_strength_mean"] = float(strength.mean().item())
        metrics["update_strength_severity_corr"] = pearson_corr(strength, severity)
    return metrics


def _run_one(args: argparse.Namespace, module_name: str, seed: int) -> None:
    set_seed(seed)
    sizes = _sizes(args.mode)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train = _make_data(sizes["train"], sizes["dim"], seed)
    test = _make_data(sizes["test"], sizes["dim"], seed + 10_000)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = PluginClassifier(sizes["dim"], module_name).to(device)
    optimizer = AdamW(model.parameters(), lr=1.0e-3)
    logger = ExperimentLogger(
        run_id=f"responsibility_plugin_{module_name}_seed{seed}",
        config={
            "mode": args.mode,
            "task_name": "responsibility_plugin",
            "model_name": module_name,
            "dataset_name": "corrupted_feature_probe",
            "seed": seed,
            "device": str(device),
        },
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
            corrupted, _clean, labels, _severity = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            corrupted, _clean, labels, _severity = next(iterator)
        out = model(corrupted.to(device))
        loss = F.cross_entropy(out["logits"], labels.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {"step": step, "train_loss": float(loss.detach().cpu().item()), "wall_clock_time_sec": time.time() - start}
        if step % args.eval_interval == 0 or step == max_steps:
            current = _evaluate(model, test_loader, device)
            eval_count += 1
            metric = float(current.get("accuracy", float("nan")))
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
    summary = best_summary or _evaluate(model, test_loader, device)
    summary.update(
        {
            "best_metric": best_metric if best_summary is not None else summary["accuracy"],
            "final_metric": summary["accuracy"],
            "best_step": best_step,
            "steps_run": steps_run,
            "early_stop_triggered": early_stop_triggered,
            "total_train_time_sec": time.time() - start,
        }
    )
    logger.finalize(summary=summary)


def _write_report(runs_root: Path) -> None:
    report = Path("reports/RESPONSIBILITY_PLUGIN_REPORT.md")
    lines = ["# Responsibility Plugin Benchmark", "", "| module | seed | accuracy | AURC | null mean | null severity corr | update severity corr |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    summary = runs_root / "summary.csv"
    if summary.exists():
        with summary.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                metrics = json.loads(row.get("metrics_json") or "{}")
                lines.append(
                    f"| {row.get('model_name')} | {row.get('seed')} | {metrics.get('accuracy', 'MISSING')} | {metrics.get('selective_aurc', 'MISSING')} | {metrics.get('null_weight_mean', 'MISSING')} | {metrics.get('null_severity_corr', 'MISSING')} | {metrics.get('update_strength_severity_corr', 'MISSING')} |"
                )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    for seed in args.seeds:
        for module_name in args.modules:
            _run_one(args, module_name, seed)
    _write_report(Path(args.runs_root))


if __name__ == "__main__":
    main()
