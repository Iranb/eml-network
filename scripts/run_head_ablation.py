from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eml_mnist.experiment_utils import ExperimentLogger

from extract_cnn_features import extract_features
from train_head_on_frozen_features import train_head


HEADS = [
    "linear",
    "mlp",
    "cosine_prototype",
    "eml_no_ambiguity",
    "eml_raw_ambiguity",
    "eml_centered_ambiguity",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen-feature CNN head ablations")
    parser.add_argument("--dataset", choices=["synthetic_shape", "cifar10"], default="synthetic_shape")
    parser.add_argument("--mode", choices=["smoke", "medium", "full"], default="smoke")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", default="~/dataset")
    parser.add_argument("--runs-root", default="reports/head_ablation/runs")
    parser.add_argument("--features-root", default="reports/head_ablation/features")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--feature-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    return parser


def _defaults(args: argparse.Namespace) -> tuple[int, int, int, int, int]:
    if args.mode == "smoke":
        return 256, 128, 128, args.steps or 30, args.feature_steps or 20
    if args.mode == "medium":
        return 2048, 512, 512, args.steps or 300, args.feature_steps or 200
    return 8192, 2048, 2048, args.steps or 1000, args.feature_steps or 500


def _not_run(seed: int, head_name: str, args: argparse.Namespace, reason: str) -> None:
    ExperimentLogger.not_run(
        run_id=f"frozen_{args.dataset}_{head_name}_seed{seed}",
        config={
            "mode": "head_ablation",
            "experiment_type": "frozen_features",
            "task_name": "image_classification",
            "model_name": head_name,
            "dataset_name": args.dataset,
            "seed": seed,
            "device": args.device,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        reason=reason,
        root=args.runs_root,
    )


def _feature_args(args: argparse.Namespace, seed: int, train_size: int, val_size: int, test_size: int, feature_steps: int) -> SimpleNamespace:
    feature_dir = Path(args.features_root) / f"{args.dataset}_{args.mode}_seed{seed}"
    return SimpleNamespace(
        dataset=args.dataset,
        output_dir=str(feature_dir),
        data_dir=args.data_dir,
        device=args.device,
        seed=seed,
        image_size=args.image_size,
        feature_dim=args.feature_dim,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        batch_size=args.batch_size,
        steps=feature_steps,
        lr=args.lr,
        num_workers=args.num_workers,
        allow_download=args.allow_download,
        force=args.force_features,
    )


def run(args: argparse.Namespace) -> None:
    if args.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    train_size, val_size, test_size, head_steps, feature_steps = _defaults(args)
    for seed in args.seeds:
        try:
            feature_dir = extract_features(_feature_args(args, seed, train_size, val_size, test_size, feature_steps))
        except Exception as exc:
            reason = repr(exc)
            trace = traceback.format_exc()
            for head_name in HEADS:
                _not_run(seed, head_name, args, reason)
            print(trace)
            continue
        for head_name in HEADS:
            try:
                run_id = f"frozen_{args.dataset}_{head_name}_seed{seed}"
                train_head(
                    SimpleNamespace(
                        features_dir=str(feature_dir),
                        head=head_name,
                        runs_root=args.runs_root,
                        run_id=run_id,
                        dataset=args.dataset,
                        device=args.device,
                        seed=seed,
                        steps=head_steps,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        hidden_dim=max(96, args.feature_dim * 2),
                        temperature=0.25,
                        pairwise_weight=0.0,
                        pairwise_margin=0.0,
                        eval_interval=max(1, head_steps // 3),
                    )
                )
            except Exception as exc:
                _not_run(seed, head_name, args, repr(exc))


def main() -> None:
    args = build_parser().parse_args()
    run(args)
    print(json.dumps({"status": "complete", "runs_root": args.runs_root}, sort_keys=True))


if __name__ == "__main__":
    main()
