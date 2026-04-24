from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eml_mnist import EMLImageFieldClassifier, EfficientEMLImageClassifier, build_mnist_eml_model
from eml_mnist.training import build_classification_loaders, resolve_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare EML image models on real CIFAR-10")
    parser.add_argument("--data-dir", type=str, default="/data16T/hyq/dataset/data")
    parser.add_argument("--output-dir", type=str, default="outputs/real_cifar10_compare")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cnn_eml", "pure_eml_v2", "eml_image_field", "efficient_eml_image"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-batches", type=int, default=120)
    parser.add_argument("--max-eval-batches", type=int, default=0, help="0 means full eval split")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--field-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--amp", action="store_true")
    return parser


def make_model(name: str, args: argparse.Namespace) -> torch.nn.Module:
    if name == "eml_image_field":
        return EMLImageFieldClassifier(
            num_classes=10,
            input_channels=3,
            sensor_dim=args.field_dim,
            measurement_dim=args.field_dim,
            field_dim=args.field_dim,
            hidden_dim=args.hidden_dim,
            num_hypotheses=4,
            num_parent_hypotheses=4,
            num_attractors=4,
            representation_dim=args.field_dim,
            patch_size=5,
            patch_stride=4,
            local_window_size=3,
            parent_window_size=3,
            composition_region_size=2,
            clip_value=3.0,
            prototype_temperature=0.25,
            enable_parent_consensus=True,
        )
    if name == "efficient_eml_image":
        return EfficientEMLImageClassifier(
            num_classes=10,
            input_channels=3,
            state_dim=args.field_dim,
            hidden_dim=args.hidden_dim,
            num_hypotheses=8,
            num_attractors=4,
            representation_dim=args.field_dim,
            patch_stride=4,
            local_window_size=3,
            composition_region_size=2,
            clip_value=3.0,
        )

    config: Dict[str, Any] = {
        "model_name": name,
        "num_classes": 10,
        "image_size": 32,
        "input_channels": 3,
        "feature_dim": 96,
        "hidden_dim": args.hidden_dim,
        "bank_dim": 96,
        "bank_blocks": 3 if name == "pure_eml_v2" else 2,
        "patch_size": 4,
        "patch_stride": 2,
        "clip_value": 3.0,
        "dropout": 0.1,
        "prototype_temperature": 0.25,
        "local_window_size": 3,
        "merge_every": 2,
    }
    return build_mnist_eml_model(config)


def tensor_stats(value: torch.Tensor) -> Dict[str, float]:
    value = value.detach().float()
    return {"mean": float(value.mean().cpu()), "std": float(value.std(unbiased=False).cpu())}


def extract_diagnostics(outputs: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key in ("drive", "resistance", "energy"):
        value = outputs.get(key)
        if torch.is_tensor(value):
            stats = tensor_stats(value)
            metrics[f"{key}_mean"] = stats["mean"]
            metrics[f"{key}_std"] = stats["std"]
    activation = outputs.get("attractor_activation")
    if torch.is_tensor(activation):
        stats = tensor_stats(activation)
        metrics["activation_mean"] = stats["mean"]
        metrics["activation_std"] = stats["std"]
    diagnostics = outputs.get("diagnostics")
    if isinstance(diagnostics, dict):
        for key, value in diagnostics.items():
            if torch.is_tensor(value) and value.numel() == 1:
                metrics[f"diag_{key}"] = float(value.detach().float().cpu())
    return metrics


def merge_diag(total: Dict[str, float], diag: Dict[str, float], count: int) -> None:
    for key, value in diag.items():
        if math.isfinite(value):
            total[key] = total.get(key, 0.0) + value * count


def finalize_diag(total: Dict[str, float], samples: int) -> Dict[str, float]:
    if samples <= 0:
        return {}
    return {key: value / samples for key, value in sorted(total.items())}


def run_phase(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    max_batches: int,
    total_train_steps: int,
    step_offset: int,
) -> Dict[str, Any]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    diag_total: Dict[str, float] = {}
    steps = 0

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)
        global_step = step_offset + batch_index
        warmup_denominator = max(1.0, total_train_steps * args.warmup_epochs)
        warmup_eta = min(1.0, float(global_step + 1) / warmup_denominator)

        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
            ):
                outputs = model(images, warmup_eta=warmup_eta)
                logits = outputs["logits"]
                loss = F.cross_entropy(logits, targets)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.detach().argmax(dim=1) == targets).sum().cpu())
        total_seen += batch_size
        merge_diag(diag_total, extract_diagnostics(outputs), batch_size)
        steps += 1

    return {
        "loss": total_loss / max(1, total_seen),
        "acc": total_correct / max(1, total_seen),
        "samples": total_seen,
        "batches": steps,
        "diagnostics": finalize_diag(diag_total, total_seen),
    }


def write_reports(output_dir: Path, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "results": rows}, handle, indent=2)

    csv_fields = [
        "model",
        "best_eval_acc",
        "final_eval_acc",
        "final_train_acc",
        "final_train_loss",
        "final_eval_loss",
        "epochs",
        "train_batches_per_epoch",
        "eval_batches",
        "train_samples_last_epoch",
        "eval_samples",
        "seconds",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in csv_fields})

    lines = [
        "# CIFAR-10 real-data model comparison",
        "",
        f"Data dir: `{args.data_dir}`",
        f"Epochs: `{args.epochs}`",
        f"Max train batches per epoch: `{args.max_train_batches}`",
        f"Max eval batches: `{args.max_eval_batches or 'full'}`",
        "",
        "| model | best eval acc | final eval acc | final train acc | seconds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.1f} |".format(
                row["model"],
                row["best_eval_acc"],
                row["final_eval_acc"],
                row["final_train_acc"],
                row["seconds"],
            )
        )
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"cifar10_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, eval_loader = build_classification_loaders(
        dataset_name="cifar10",
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,
    )
    effective_train_batches = args.max_train_batches or len(train_loader)
    effective_train_batches = min(effective_train_batches, len(train_loader))
    total_train_steps = max(1, effective_train_batches * args.epochs)

    print(
        "dataset=cifar10 "
        f"data_dir={args.data_dir} train_batches={len(train_loader)} eval_batches={len(eval_loader)}"
    )
    print(f"output_dir={output_dir}")

    rows: List[Dict[str, Any]] = []
    for model_index, model_name in enumerate(args.models):
        print(f"\nmodel={model_name} start")
        set_seed(args.seed + model_index)
        model = make_model(model_name, args).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
        best_eval_acc = 0.0
        history: List[Dict[str, Any]] = []
        start_time = time.time()
        step_offset = 0

        for epoch_index in range(args.epochs):
            train_metrics = run_phase(
                model,
                train_loader,
                device,
                optimizer,
                scaler,
                args,
                args.max_train_batches,
                total_train_steps,
                step_offset,
            )
            step_offset += train_metrics["batches"]
            with torch.no_grad():
                eval_metrics = run_phase(
                    model,
                    eval_loader,
                    device,
                    None,
                    scaler,
                    args,
                    args.max_eval_batches,
                    total_train_steps,
                    step_offset,
                )
            best_eval_acc = max(best_eval_acc, eval_metrics["acc"])
            record = {
                "epoch": epoch_index + 1,
                "train": train_metrics,
                "eval": eval_metrics,
                "best_eval_acc": best_eval_acc,
            }
            history.append(record)
            print(
                f"model={model_name} epoch={epoch_index + 1}/{args.epochs} "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
                f"eval_loss={eval_metrics['loss']:.4f} eval_acc={eval_metrics['acc']:.4f} "
                f"best_acc={best_eval_acc:.4f}"
            )

        seconds = time.time() - start_time
        final_train = history[-1]["train"]
        final_eval = history[-1]["eval"]
        row = {
            "model": model_name,
            "best_eval_acc": best_eval_acc,
            "final_eval_acc": final_eval["acc"],
            "final_train_acc": final_train["acc"],
            "final_train_loss": final_train["loss"],
            "final_eval_loss": final_eval["loss"],
            "epochs": args.epochs,
            "train_batches_per_epoch": final_train["batches"],
            "eval_batches": final_eval["batches"],
            "train_samples_last_epoch": final_train["samples"],
            "eval_samples": final_eval["samples"],
            "seconds": seconds,
            "history": history,
        }
        rows.append(row)
        (output_dir / f"{model_name}_history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        write_reports(output_dir, rows, args)
        print(f"model={model_name} done best_eval_acc={best_eval_acc:.4f} seconds={seconds:.1f}")
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_reports(output_dir, rows, args)
    print("\ncomparison complete")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
