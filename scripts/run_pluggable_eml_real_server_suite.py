from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-server pluggable EML uncertainty suite with early-stop reruns")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", default="/data16T/hyq/dataset/data")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mode", choices=["smoke", "medium"], default="medium")
    parser.add_argument("--datasets", nargs="+", choices=["synthetic_shape_uncertainty", "cifar10_corrupt"], default=["synthetic_shape_uncertainty", "cifar10_corrupt"])
    parser.add_argument("--runs-root", default="reports/pluggable_uncertainty_real/runs")
    parser.add_argument("--report", default="reports/EML_PLUGGABLE_PRIMITIVE_REPORT.md")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--backbone-steps", type=int, default=300)
    parser.add_argument("--head-steps", type=int, default=1200)
    parser.add_argument("--max-head-steps", type=int, default=2400)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--early-stop-min-evals", type=int, default=2)
    parser.add_argument("--heads", nargs="+", default=None)
    parser.add_argument("--skip-cifar", action="store_true")
    return parser


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _load_summary(runs_root: Path) -> list[dict[str, Any]]:
    summary_path = runs_root / "summary.csv"
    if not summary_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metrics = {}
            raw = row.get("metrics_json")
            if raw:
                try:
                    metrics = json.loads(raw)
                except Exception:
                    metrics = {}
            rows.append({**row, **metrics})
    return rows


def _pending_capped_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("dataset_name", "")), str(row.get("model_name", "")), str(row.get("seed", "")))
        grouped.setdefault(key, []).append(row)
    pending: list[dict[str, Any]] = []
    for entries in grouped.values():
        if any(str(row.get("early_stop_triggered", "")).lower() == "true" for row in entries):
            continue
        latest = entries[-1]
        if latest.get("status") == "COMPLETED" and str(latest.get("early_stop_triggered", "")).lower() == "false":
            pending.append(latest)
    return pending


def _dataset_for_runner(dataset_name: str) -> str:
    if dataset_name == "synthetic_shape":
        return "synthetic_shape_uncertainty"
    if dataset_name == "cifar10":
        return "cifar10_corrupt"
    return dataset_name


def _run_frozen(args: argparse.Namespace, dataset: str, seeds: list[int], heads: list[str] | None, head_steps: int) -> None:
    command = [
        sys.executable,
        "scripts/run_uncertainty_frozen_feature_benchmark.py",
        "--dataset",
        dataset,
        "--mode",
        args.mode,
        "--device",
        args.device,
        "--data-dir",
        args.data_dir,
        "--num-workers",
        str(args.num_workers),
        "--batch-size",
        str(args.batch_size),
        "--runs-root",
        args.runs_root,
        "--report",
        args.report,
        "--backbone-steps",
        str(args.backbone_steps),
        "--head-steps",
        str(head_steps),
        "--eval-interval",
        str(args.eval_interval),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--early-stop-min-evals",
        str(args.early_stop_min_evals),
        "--seeds",
        *[str(seed) for seed in seeds],
    ]
    if heads:
        command.extend(["--heads", *heads])
    _run(command)


def main() -> None:
    args = build_parser().parse_args()
    datasets = [dataset for dataset in args.datasets if not (args.skip_cifar and dataset == "cifar10_corrupt")]
    for dataset in datasets:
        _run_frozen(args, dataset=dataset, seeds=args.seeds, heads=args.heads, head_steps=args.head_steps)

    runs_root = Path(args.runs_root)
    head_steps = args.head_steps
    while True:
        pending = _pending_capped_rows(_load_summary(runs_root))
        if not pending or head_steps >= args.max_head_steps:
            break
        head_steps = min(args.max_head_steps, max(head_steps + args.eval_interval, head_steps * 2))
        for row in pending:
            dataset = _dataset_for_runner(str(row.get("dataset_name", "")))
            seed = int(row.get("seed", 0))
            head = str(row.get("model_name", ""))
            _run_frozen(args, dataset=dataset, seeds=[seed], heads=[head], head_steps=head_steps)

    pending = _pending_capped_rows(_load_summary(runs_root))
    if pending:
        print("WARNING: rows still did not early-stop after reruns:", flush=True)
        for row in pending:
            print(
                f"- {row.get('dataset_name')} {row.get('model_name')} seed={row.get('seed')} steps={row.get('steps_run')}",
                flush=True,
            )

    _run(
        [
            sys.executable,
            "scripts/generate_uncertainty_eml_report.py",
            "--runs-root",
            args.runs_root,
            "--output",
            args.report,
        ]
    )


if __name__ == "__main__":
    main()
