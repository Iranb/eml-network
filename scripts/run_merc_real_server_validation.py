from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MERC real-server validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", default="/data16T/hyq/dataset/data")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mode", choices=["smoke", "medium"], default="medium")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-index", default="")
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--early-stop-min-evals", type=int, default=2)
    return parser


def _run(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def _write_report(output: Path, commands: list[list[str]]) -> None:
    lines = [
        "# MERC Real Server Validation Report",
        "",
        "This report is generated from the actual executed validation commands.",
        "",
        "## Commands",
        "",
    ]
    for cmd in commands:
        lines.append(f"- `{' '.join(cmd)}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `reports/MERC_TOY_REPORT.md`",
            "- `reports/MERC_HEAD_ABLATION_REPORT.md`",
            "- `reports/MERC_END_TO_END_REPORT.md`",
            "- `reports/MERC_SYNTHETIC_EVIDENCE_REPORT.md`",
            "- `reports/MERC_MASTER_REPORT.md`",
            "",
            "## Conclusion",
            "",
            "- Read the generated subreports for measured results. This wrapper does not invent summary numbers.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    env = os.environ.copy()
    if args.gpu_index != "":
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    seeds = [str(seed) for seed in args.seeds]
    commands = [
        [
            sys.executable,
            "scripts/run_merc_toy_experiments.py",
            "--mode",
            args.mode,
            "--device",
            args.device,
            "--seeds",
            *seeds,
        ],
        [
            sys.executable,
            "scripts/run_head_ablation.py",
            "--dataset",
            "cifar10",
            "--mode",
            args.mode,
            "--seeds",
            *seeds,
            "--device",
            args.device,
            "--data-dir",
            args.data_dir,
            "--num-workers",
            str(args.num_workers),
            "--batch-size",
            str(args.batch_size),
            "--include-merc",
            "--early-stop-patience",
            str(args.early_stop_patience),
            "--early-stop-min-evals",
            str(args.early_stop_min_evals),
            "--runs-root",
            "reports/merc_head_ablation/runs",
        ],
        [
            sys.executable,
            "scripts/generate_head_ablation_report.py",
            "--runs-root",
            "reports/merc_head_ablation/runs",
            "--output",
            "reports/MERC_HEAD_ABLATION_REPORT.md",
        ],
        [
            sys.executable,
            "scripts/run_cnn_head_end_to_end_ablation.py",
            "--dataset",
            "cifar10",
            "--mode",
            args.mode,
            "--seeds",
            *seeds,
            "--device",
            args.device,
            "--data-dir",
            args.data_dir,
            "--num-workers",
            str(args.num_workers),
            "--batch-size",
            str(args.batch_size),
            "--include-merc",
            "--early-stop-patience",
            str(args.early_stop_patience),
            "--early-stop-min-evals",
            str(args.early_stop_min_evals),
            "--runs-root",
            "reports/merc_end_to_end/runs",
        ],
        [
            sys.executable,
            "scripts/generate_head_ablation_report.py",
            "--runs-root",
            "reports/merc_end_to_end/runs",
            "--output",
            "reports/MERC_END_TO_END_REPORT.md",
        ],
        [
            sys.executable,
            "scripts/run_merc_synthetic_evidence.py",
            "--mode",
            args.mode,
            "--device",
            args.device,
            "--seeds",
            *seeds,
            "--num-workers",
            str(args.num_workers),
            "--data-dir",
            args.data_dir,
        ],
        [
            sys.executable,
            "scripts/generate_merc_report.py",
        ],
    ]
    for cmd in commands:
        _run(cmd, env)
    _write_report(ROOT / "reports/MERC_REAL_SERVER_VALIDATION_REPORT.md", commands)
    print(
        json.dumps(
            {
                "status": "complete",
                "report": "reports/MERC_REAL_SERVER_VALIDATION_REPORT.md",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
