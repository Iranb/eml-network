from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eml_mnist.experiment_utils import ExperimentLogger
from eml_mnist.training import resolve_device, set_seed
from run_eml_validation_suite import _build_cnn_model, _build_efficient_image, _build_old_image_model, _train_image_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic image representation ablations")
    parser.add_argument("--mode", choices=["smoke", "ablation"], default="smoke")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runs-root", default="reports/image_representation_ablation/runs")
    parser.add_argument("--output", default="reports/IMAGE_REPRESENTATION_ABLATION_REPORT.md")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--staged-hardening", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--responsibility-temp-start", type=float, default=2.0)
    parser.add_argument("--responsibility-temp-end", type=float, default=0.8)
    parser.add_argument("--ambiguity-warmup-steps", type=int, default=20)
    parser.add_argument("--null-threshold-start", type=float, default=1.0)
    parser.add_argument("--null-threshold-end", type=float, default=0.0)
    return parser


def _default_steps(args: argparse.Namespace) -> int:
    if args.steps > 0:
        return args.steps
    return 4 if args.mode == "smoke" else 60


def _run_args(args: argparse.Namespace, staged: bool) -> argparse.Namespace:
    child = copy.copy(args)
    child.steps = _default_steps(args)
    child.staged_hardening = staged or bool(args.staged_hardening)
    return child


def _safe_image_run(
    run_id: str,
    model_name: str,
    factory: Callable[[], torch.nn.Module],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    warmup_enabled: bool = True,
    staged: bool = False,
) -> None:
    child_args = _run_args(args, staged=staged)
    try:
        set_seed(seed)
        _train_image_model(
            run_id,
            model_name,
            factory(),
            child_args,
            device,
            seed,
            warmup_enabled=warmup_enabled,
        )
    except Exception as exc:
        trace = traceback.format_exc()
        logger = ExperimentLogger(
            run_id=run_id,
            config={
                "mode": args.mode,
                "task_name": "image_synthetic",
                "model_name": model_name,
                "dataset_name": "SyntheticShapeEnergyDataset",
                "seed": seed,
                "device": str(device),
                "num_workers": args.num_workers,
            },
            root=args.runs_root,
        )
        logger.set_model_info(extra={"num_params": 0, "trainable_params": 0})
        logger.log_text(trace)
        logger.finalize(summary={"error_trace": trace}, status="FAILED", reason=repr(exc))


def _rows(root: Path) -> list[Dict[str, str]]:
    summary = root / "summary.csv"
    if not summary.exists():
        return []
    with summary.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _metrics(row: Dict[str, str]) -> Dict[str, Any]:
    try:
        metrics = json.loads(row.get("metrics_json") or "{}")
    except Exception:
        metrics = {}
    run_dir = row.get("run_dir")
    if run_dir:
        try:
            summary = json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))
            diagnostics = summary.get("final_diagnostics") or {}
            if isinstance(diagnostics, dict):
                metrics.update({key: value for key, value in diagnostics.items() if key not in metrics})
        except Exception:
            pass
    return metrics


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "MISSING"


def generate_report(runs_root: str | Path, output: str | Path) -> Path:
    rows = _rows(Path(runs_root))
    completed = [row for row in rows if row.get("status") == "COMPLETED"]
    grouped: dict[str, list[Dict[str, str]]] = defaultdict(list)
    for row in completed:
        grouped[row.get("model_name", "")].append(row)
    lines = [
        "# Image Representation Ablation Report",
        "",
        "## Summary",
        f"- Completed runs: {len(completed)}",
        f"- Failed runs: {sum(1 for row in rows if row.get('status') == 'FAILED')}",
        f"- NOT RUN entries: {sum(1 for row in rows if row.get('status') == 'NOT RUN')}",
        "- Synthetic shape accuracy above `0.8` is the gate before making CIFAR claims for the efficient representation path.",
        "",
        "## Results",
        "| model | n | best accuracy | mean final accuracy | mean loss | mean time sec | null weight | update gate | ambiguity | attractor diversity | noise corr | occlusion corr |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    best_efficient = 0.0
    for model_name in sorted(grouped):
        items = grouped[model_name]
        metrics = [_metrics(item) for item in items]

        def avg(key: str) -> float:
            values = []
            for item in metrics:
                try:
                    values.append(float(item.get(key)))
                except Exception:
                    pass
            return sum(values) / len(values) if values else float("nan")

        best = max(float(item.get("best_metric", "nan")) for item in items)
        if "EfficientEMLImageClassifier" in model_name or "efficient" in model_name.lower():
            best_efficient = max(best_efficient, best)
        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    str(len(items)),
                    _fmt(best),
                    _fmt(avg("final_train_accuracy")),
                    _fmt(avg("final_train_loss")),
                    _fmt(avg("total_train_time_sec")),
                    _fmt(avg("null_weight_mean")),
                    _fmt(avg("update_gate_mean")),
                    _fmt(avg("ambiguity_mean")),
                    _fmt(avg("attractor_diversity")),
                    _fmt(avg("resistance_noise_corr")),
                    _fmt(avg("resistance_occlusion_corr")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Gate",
            f"- Best efficient synthetic result in this report: `{best_efficient:.4f}`",
            "- CIFAR medium should be skipped unless this value is at least `0.8`.",
            "",
            "## Missing Or Failed",
            "| run_id | status | model | reason |",
            "| --- | --- | --- | --- |",
        ]
    )
    missing = [row for row in rows if row.get("status") != "COMPLETED"]
    if missing:
        for row in missing:
            lines.append(f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('model_name', '')} | {row.get('reason', '')} |")
    else:
        lines.append("| none | none | none | none |")
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
    device = resolve_device(args.device)
    Path(args.runs_root).mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        specs = [
            ("cnn_eml_workers0", "cnn_eml_workers0", lambda: _build_cnn_model(args.image_size), True, False),
            ("pure_eml_workers0", "pure_eml_workers0", lambda: _build_old_image_model("pure_eml", args.image_size), True, False),
            ("efficient_baseline", "EfficientEMLImageClassifier_baseline", lambda: _build_efficient_image(4, 3, sensor_bypass=False), True, False),
            ("efficient_centered_ambiguity", "EfficientEMLImageClassifier_centered_ambiguity", lambda: _build_efficient_image(4, 3, sensor_bypass=False), True, False),
            ("efficient_thresholded_null", "EfficientEMLImageClassifier_thresholded_null", lambda: _build_efficient_image(4, 3, responsibility_mode="thresholded_null", sensor_bypass=False), True, False),
            ("efficient_precision_identity", "EfficientEMLImageClassifier_precision_identity", lambda: _build_efficient_image(4, 3, precision_old_confidence_init=5.0, sensor_bypass=False), True, False),
            ("efficient_combo", "EfficientEMLImageClassifier_combo", lambda: _build_efficient_image(4, 3, responsibility_mode="thresholded_null", precision_old_confidence_init=5.0, sensor_bypass=False), True, False),
            ("efficient_combo_no_composition", "EfficientEMLImageClassifier_no_composition", lambda: _build_efficient_image(4, 3, enable_composition=False, responsibility_mode="thresholded_null", precision_old_confidence_init=5.0, sensor_bypass=False), True, False),
            ("efficient_combo_no_attractor", "EfficientEMLImageClassifier_no_attractor", lambda: _build_efficient_image(1, 3, enable_attractor=False, responsibility_mode="thresholded_null", precision_old_confidence_init=5.0, sensor_bypass=False), True, False),
            ("efficient_combo_sensor_bypass", "EfficientEMLImageClassifier_sensor_bypass", lambda: _build_efficient_image(4, 3, responsibility_mode="thresholded_null", precision_old_confidence_init=5.0, sensor_bypass=True), True, False),
            ("efficient_combo_staged", "EfficientEMLImageClassifier_staged", lambda: _build_efficient_image(4, 3, responsibility_mode="thresholded_null", precision_old_confidence_init=5.0, sensor_bypass=False), True, True),
            ("efficient_combo_bypass_staged", "EfficientEMLImageClassifier_bypass_staged", lambda: _build_efficient_image(4, 3, responsibility_mode="thresholded_null", precision_old_confidence_init=5.0, sensor_bypass=True), True, True),
            ("head_without_ambiguity", "EfficientEMLImageClassifier_head_without_ambiguity", lambda: _build_efficient_image(4, 3, center_ambiguity=False, ambiguity_weight=0.0, schedule_ambiguity_weight=False, sensor_bypass=False), True, False),
        ]
        for name, model_name, factory, warmup, staged in specs:
            _safe_image_run(f"{name}_seed{seed}", model_name, factory, args, device, seed, warmup_enabled=warmup, staged=staged)
    report = generate_report(args.runs_root, args.output)
    print(report)


if __name__ == "__main__":
    main()
