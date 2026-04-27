from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_eml_uncertainty_benchmark as benchmark


DATASET_ALIASES = {
    "synthetic_shape_uncertainty": "synthetic_shape",
    "cifar10_corrupt": "cifar10",
    "synthetic_shape": "synthetic_shape",
    "cifar10": "cifar10",
    "all": "all",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen-feature uncertainty/resistance benchmark")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_ALIASES),
        default="synthetic_shape_uncertainty",
    )
    parser.add_argument("--mode", choices=["smoke", "medium"], default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", default="~/dataset")
    parser.add_argument("--runs-root", default="reports/uncertainty_frozen/runs")
    parser.add_argument("--report", default="reports/EML_PLUGGABLE_PRIMITIVE_REPORT.md")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--heads", nargs="+", choices=benchmark.HEADS, default=None)
    parser.add_argument("--backbone-steps", type=int, default=None)
    parser.add_argument("--head-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--early-stop-min-evals", type=int, default=2)
    parser.add_argument(
        "--supervise-resistance",
        choices=["true", "false"],
        default="true",
        help="Kept for CLI compatibility; resistance supervision is controlled by the eml_supervised_resistance head.",
    )
    return parser


def _to_benchmark_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=DATASET_ALIASES[args.dataset],
        mode=args.mode,
        device=args.device,
        data_dir=args.data_dir,
        runs_root=args.runs_root,
        report=args.report,
        image_size=args.image_size,
        feature_dim=args.feature_dim,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        allow_download=args.allow_download,
        seeds=args.seeds,
        heads=args.heads,
        backbone_steps=args.backbone_steps,
        head_steps=args.head_steps,
        eval_interval=args.eval_interval,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_evals=args.early_stop_min_evals,
    )


def main() -> None:
    args = build_parser().parse_args()
    report_path = benchmark.run(_to_benchmark_args(args))
    print(json.dumps({"report": str(Path(report_path))}, sort_keys=True))


if __name__ == "__main__":
    main()
