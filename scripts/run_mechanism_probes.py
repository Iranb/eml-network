from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eml_mnist.diagnostics import collect_eml_diagnostics
from eml_mnist.experiment_utils import ExperimentLogger
from eml_mnist.mechanism_probes import MECHANISM_NAMES, PROBE_NAMES, run_mechanism_probe
from eml_mnist.training import resolve_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EML mechanism probes")
    parser.add_argument("--mode", choices=["smoke", "ablation"], default="smoke")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runs-root", default="reports/mechanism_probes/runs")
    parser.add_argument("--output", default="reports/MECHANISM_PROBE_REPORT.md")
    return parser


def _run_one(
    probe_name: str,
    mechanism: str,
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    run_id = f"probe_{probe_name}_{mechanism}_seed{seed}"
    logger = ExperimentLogger(
        run_id=run_id,
        config={
            "mode": args.mode,
            "task_name": "mechanism_probe",
            "model_name": mechanism,
            "dataset_name": probe_name,
            "seed": seed,
            "device": str(device),
            "probe_name": probe_name,
            "mechanism": mechanism,
        },
        root=args.runs_root,
    )
    start = time.time()
    try:
        result = run_mechanism_probe(probe_name, mechanism, seed=seed, device=device)
        metrics = {
            "step": 1,
            "train_loss": 0.0,
            "final_metric": result["metrics"].get("accuracy", 0.0),
            "best_metric": result["metrics"].get("accuracy", 0.0),
            "wall_clock_time_sec": time.time() - start,
            "step_time_sec": time.time() - start,
            **result["metrics"],
        }
        diagnostics = {
            **collect_eml_diagnostics(result.get("outputs", {})),
            **result.get("diagnostics", {}),
        }
        logger.set_model_info(extra={"num_params": 0, "trainable_params": 0})
        logger.log_step(metrics, diagnostics)
        logger.finalize(
            summary={**metrics, "final_diagnostics": diagnostics, "total_train_time_sec": time.time() - start},
            model_info={"num_params": 0, "trainable_params": 0},
        )
    except Exception as exc:
        trace = traceback.format_exc()
        logger.set_model_info(extra={"num_params": 0, "trainable_params": 0})
        logger.log_text(trace)
        logger.finalize(summary={"error_trace": trace}, status="FAILED", reason=repr(exc))


def _rows(root: Path) -> list[Dict[str, str]]:
    path = root / "summary.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _metrics(row: Dict[str, str]) -> Dict[str, Any]:
    try:
        return json.loads(row.get("metrics_json") or "{}")
    except Exception:
        return {}


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "MISSING"


def generate_report(runs_root: str | Path, output: str | Path) -> Path:
    rows = _rows(Path(runs_root))
    completed = [row for row in rows if row.get("status") == "COMPLETED"]
    failed = [row for row in rows if row.get("status") == "FAILED"]
    not_run = [row for row in rows if row.get("status") == "NOT RUN"]
    grouped: dict[tuple[str, str], list[Dict[str, str]]] = defaultdict(list)
    for row in completed:
        grouped[(row.get("dataset_name", ""), row.get("model_name", ""))].append(row)

    lines = [
        "# Mechanism Probe Report",
        "",
        "## Executive Summary",
        f"- Completed runs: {len(completed)}",
        f"- Failed runs: {len(failed)}",
        f"- NOT RUN entries: {len(not_run)}",
        "- Probe results are diagnostic checks, not task-level model accuracy.",
        "",
        "## Probe Table",
        "| probe | mechanism | n | success rate | null weight | max responsibility | update gate | update norm | resistance-conflict corr | attractor diversity |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(grouped):
        probe_name, mechanism = key
        items = grouped[key]
        values = [_metrics(item) for item in items]
        def avg(metric: str) -> float:
            nums = []
            for item in values:
                try:
                    nums.append(float(item.get(metric)))
                except Exception:
                    pass
            return sum(nums) / len(nums) if nums else float("nan")

        lines.append(
            "| "
            + " | ".join(
                [
                    probe_name,
                    mechanism,
                    str(len(items)),
                    _fmt(avg("accuracy")),
                    _fmt(avg("null_weight")),
                    _fmt(avg("max_responsibility")),
                    _fmt(avg("update_gate")),
                    _fmt(avg("update_norm")),
                    _fmt(avg("resistance_conflict_correlation")),
                    _fmt(avg("attractor_diversity")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Missing Or Failed",
            "| run_id | status | reason |",
            "| --- | --- | --- |",
        ]
    )
    for row in failed + not_run:
        lines.append(f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('reason', '')} |")
    if not failed and not not_run:
        lines.append("| none | none | none |")
    lines.extend(["", "## Raw Artifacts"])
    for row in rows:
        lines.append(f"- `{row.get('run_id', '')}`: `{row.get('run_dir', '')}`")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    Path(args.runs_root).mkdir(parents=True, exist_ok=True)
    probes = PROBE_NAMES if args.mode == "ablation" else PROBE_NAMES
    mechanisms = MECHANISM_NAMES if args.mode == "ablation" else MECHANISM_NAMES
    for seed in args.seeds:
        set_seed(seed)
        for probe_name in probes:
            for mechanism in mechanisms:
                _run_one(probe_name, mechanism, seed, device, args)
    report = generate_report(args.runs_root, args.output)
    print(report)


if __name__ == "__main__":
    main()
