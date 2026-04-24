from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eml_mnist.experiment_utils import ExperimentLogger
from eml_mnist.training import resolve_device
from run_eml_validation_suite import run_cifar_medium


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CIFAR medium only after synthetic efficient EML passes")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-root", default="reports/cifar_medium/runs")
    parser.add_argument("--image-runs-root", default="reports/image_representation_ablation/runs")
    parser.add_argument("--output", default="reports/CIFAR_MEDIUM_REPORT.md")
    parser.add_argument("--data-dir", default="~/dataset/data")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--gate-threshold", type=float, default=0.8)
    parser.add_argument("--staged-hardening", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--responsibility-temp-start", type=float, default=2.0)
    parser.add_argument("--responsibility-temp-end", type=float, default=0.8)
    parser.add_argument("--ambiguity-warmup-steps", type=int, default=100)
    parser.add_argument("--null-threshold-start", type=float, default=1.0)
    parser.add_argument("--null-threshold-end", type=float, default=0.0)
    return parser


def _rows(path: Path) -> list[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _best_efficient(image_runs_root: str | Path) -> tuple[float, str]:
    rows = _rows(Path(image_runs_root) / "summary.csv")
    best = float("-inf")
    best_model = ""
    for row in rows:
        if row.get("status") != "COMPLETED":
            continue
        model_name = row.get("model_name", "")
        if "efficient" not in model_name.lower() and "EfficientEMLImageClassifier" not in model_name:
            continue
        try:
            value = float(row.get("best_metric", "nan"))
        except Exception:
            continue
        if value > best:
            best = value
            best_model = model_name
    if best == float("-inf"):
        return 0.0, "MISSING"
    return best, best_model


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "MISSING"


def generate_report(runs_root: str | Path, output: str | Path, gate_metric: float, gate_model: str, threshold: float) -> Path:
    rows = _rows(Path(runs_root) / "summary.csv")
    lines = [
        "# CIFAR Medium Report",
        "",
        "## Synthetic Gate",
        f"- Best efficient synthetic image result: `{gate_model}` at `{gate_metric:.4f}`",
        f"- Required threshold: `{threshold:.4f}`",
        "",
        "## CIFAR Runs",
        "| run_id | status | model | best metric | final metric | reason |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    if rows:
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.get("run_id", ""),
                        row.get("status", ""),
                        row.get("model_name", ""),
                        _fmt(row.get("best_metric", "")),
                        _fmt(row.get("final_metric", "")),
                        row.get("reason", ""),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| none | NOT RUN | none | MISSING | MISSING | no rows recorded |")
    lines.extend(["", "## Raw Artifacts"])
    for row in rows:
        lines.append(f"- `{row.get('run_id', '')}`: `{row.get('run_dir', '')}`")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    if args.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    Path(args.runs_root).mkdir(parents=True, exist_ok=True)
    gate_metric, gate_model = _best_efficient(args.image_runs_root)
    if gate_metric < args.gate_threshold:
        ExperimentLogger.not_run(
            "cifar_medium_synthetic_gate",
            {
                "mode": "cifar-medium",
                "task_name": "image_cifar",
                "model_name": "selected_image_models",
                "dataset_name": "CIFAR10",
                "seed": args.seed,
                "device": args.device,
                "num_workers": args.num_workers,
            },
            f"SKIPPED: best efficient synthetic image metric {gate_metric:.4f} from {gate_model} is below {args.gate_threshold:.4f}",
            root=args.runs_root,
        )
    else:
        args.mode = "cifar-medium"
        device = resolve_device(args.device)
        run_cifar_medium(args, device)
    report = generate_report(args.runs_root, args.output, gate_metric, gate_model, args.gate_threshold)
    print(report)


if __name__ == "__main__":
    main()
