from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MERC synthetic evidence validation")
    parser.add_argument("--mode", choices=["smoke", "medium"], default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", default="~/dataset")
    parser.add_argument("--runs-root", default="reports/merc_synthetic_evidence/runs")
    parser.add_argument("--report", default="reports/MERC_SYNTHETIC_EVIDENCE_REPORT.md")
    return parser


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT)


def _render_report(runs_root: Path, report: Path) -> None:
    summary_path = runs_root / "summary.csv"
    lines = [
        "# MERC Synthetic Evidence Report",
        "",
        "| run_id | model | status | final metric | support evidence corr | conflict resistance corr |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                metrics = {}
                run_dir = Path(row.get("run_dir", ""))
                summary_json = run_dir / "summary.json"
                if summary_json.exists():
                    metrics = json.loads(summary_json.read_text(encoding="utf-8"))
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            row.get("run_id", ""),
                            row.get("model_name", ""),
                            row.get("status", ""),
                            f"{float(metrics.get('final_metric', float('nan'))):.4f}" if metrics.get("final_metric") is not None else "MISSING",
                            f"{float(metrics.get('test_support_evidence_corr', float('nan'))):.4f}" if metrics.get("test_support_evidence_corr") is not None else "MISSING",
                            f"{float(metrics.get('test_conflict_resistance_corr', float('nan'))):.4f}" if metrics.get("test_conflict_resistance_corr") is not None else "MISSING",
                        ]
                    )
                    + " |"
                )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    cmd = [
        sys.executable,
        "scripts/run_head_ablation.py",
        "--dataset",
        "synthetic_evidence",
        "--mode",
        args.mode,
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--data-dir",
        args.data_dir,
        "--runs-root",
        args.runs_root,
        "--include-merc",
        "--seeds",
        *[str(seed) for seed in args.seeds],
    ]
    _run(cmd)
    _render_report(Path(args.runs_root), Path(args.report))
    print(json.dumps({"status": "complete", "report": args.report, "runs_root": args.runs_root}, sort_keys=True))


if __name__ == "__main__":
    main()
