import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from eml_mnist import build_mnist_eml_model
from eml_mnist.training import (
    AverageMeter,
    build_classification_loaders,
    compute_entropy_weight,
    compute_loss_bundle,
    compute_warmup_eta,
    ensure_dir,
    get_dataset_spec,
    move_batch_to_device,
    resolve_device,
    save_json,
    set_seed,
)


METRIC_NAMES = [
    "loss",
    "ce",
    "pairwise",
    "resistance",
    "energy",
    "prototype_diversity",
    "activation_budget",
    "entropy",
    "acc",
    "mean_gate",
    "sample_uncertainty_mean",
    "mean_uncertainty",
    "drive_pos_mean",
    "drive_hard_neg_mean",
    "resistance_pos_mean",
    "resistance_hard_neg_mean",
    "energy_pos_mean",
    "energy_hard_neg_mean",
    "margin_mean",
    "class_radius_mean",
    "eml_gamma",
    "eml_lambda",
]

EPOCH_DIAGNOSTIC_NAMES = [
    "drive_pos_mean",
    "drive_hard_neg_mean",
    "resistance_pos_mean",
    "resistance_hard_neg_mean",
    "energy_pos_mean",
    "energy_hard_neg_mean",
    "margin_mean",
    "mean_gate",
    "sample_uncertainty_mean",
    "class_radius_mean",
    "eml_gamma",
    "eml_lambda",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an EML-based image classifier")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/eml_mnist")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument(
        "--model-name",
        type=str,
        default="cnn_eml",
        choices=["cnn_eml", "cnn_eml_stage", "pure_eml", "pure_eml_v2"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--entropy-decay-epochs", type=int, default=10)
    parser.add_argument("--pairwise-weight", type=float, default=0.2)
    parser.add_argument("--resistance-weight", type=float, default=0.05)
    parser.add_argument("--energy-weight", type=float, default=1e-4)
    parser.add_argument("--prototype-diversity-weight", type=float, default=0.01)
    parser.add_argument("--activation-budget-weight", type=float, default=0.0)
    parser.add_argument("--activation-budget-target", type=float, default=0.35)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--pairwise-margin", type=float, default=0.2)
    parser.add_argument("--resistance-margin", type=float, default=0.5)
    parser.add_argument("--energy-margin", type=float, default=6.0)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--input-channels", type=int, default=None)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--bank-dim", type=int, default=128)
    parser.add_argument("--bank-blocks", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--patch-stride", type=int, default=None)
    parser.add_argument("--clip-value", type=float, default=3.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--prototype-temperature", type=float, default=0.25)
    parser.add_argument("--local-window-size", type=int, default=3)
    parser.add_argument("--merge-every", type=int, default=2)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-min-epochs", type=int, default=30)
    return parser


def apply_dataset_defaults(config: Dict[str, Any]) -> None:
    spec = get_dataset_spec(config["dataset"])
    if config["num_classes"] is None:
        config["num_classes"] = spec["num_classes"]
    if config["image_size"] is None:
        config["image_size"] = spec["image_size"]
    if config["input_channels"] is None:
        config["input_channels"] = spec["input_channels"]
    if config["patch_size"] is None:
        config["patch_size"] = spec["default_patch_size"]
    if config["patch_stride"] is None:
        config["patch_stride"] = max(1, spec["default_patch_size"] // 2)


def apply_model_defaults(config: Dict[str, Any]) -> None:
    if config["model_name"] != "pure_eml_v2":
        return

    if config["bank_blocks"] == 2:
        config["bank_blocks"] = 6
    if config["warmup_epochs"] == 5:
        config["warmup_epochs"] = 20
    if config["label_smoothing"] == 0.05:
        config["label_smoothing"] = 0.1
    if config["pairwise_weight"] == 0.2:
        config["pairwise_weight"] = 0.05
    if config["resistance_weight"] == 0.05:
        config["resistance_weight"] = 0.01
    if config["energy_weight"] == 1e-4:
        config["energy_weight"] = 2e-5


def linear_ramp(epoch_index: int, total_epochs: int, start_frac: float, end_frac: float) -> float:
    if start_frac >= end_frac:
        return 1.0
    progress = epoch_index / max(1, total_epochs - 1)
    if progress <= start_frac:
        return 0.0
    if progress >= end_frac:
        return 1.0
    return (progress - start_frac) / (end_frac - start_frac)


def resolve_loss_weights(config: Dict[str, Any], epoch_index: int) -> Dict[str, float]:
    weights = {
        "pairwise_weight": config["pairwise_weight"],
        "resistance_weight": config["resistance_weight"],
        "energy_weight": config["energy_weight"],
        "prototype_diversity_weight": config["prototype_diversity_weight"],
        "activation_budget_weight": config["activation_budget_weight"],
        "entropy_weight": compute_entropy_weight(
            base_weight=config["entropy_weight"],
            epoch_index=epoch_index,
            total_epochs=config["epochs"],
            decay_epochs=config["entropy_decay_epochs"],
        ),
    }
    if config["model_name"] != "pure_eml_v2":
        return weights

    total_epochs = config["epochs"]
    weights["pairwise_weight"] *= linear_ramp(epoch_index, total_epochs, start_frac=0.15, end_frac=0.40)
    weights["resistance_weight"] *= linear_ramp(epoch_index, total_epochs, start_frac=0.30, end_frac=0.65)
    weights["energy_weight"] *= linear_ramp(epoch_index, total_epochs, start_frac=0.25, end_frac=0.55)
    return weights


def format_epoch_diagnostics(prefix: str, metrics: Dict[str, float]) -> str:
    return " ".join(f"{prefix}_{name}={metrics[name]:.4f}" for name in EPOCH_DIAGNOSTIC_NAMES if name in metrics)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_accuracy: float,
    config: Dict[str, Any],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "config": config,
        },
        path,
    )


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: AdamW,
    device: torch.device,
    epoch_index: int,
    config: Dict[str, Any],
) -> Dict[str, float]:
    model.train()
    meters = {name: AverageMeter() for name in METRIC_NAMES}

    progress_bar = tqdm(loader, desc=f"train {epoch_index + 1}/{config['epochs']}", leave=False)
    steps_per_epoch = len(loader)

    for step_index, batch in enumerate(progress_bar):
        images, targets = move_batch_to_device(batch, device)
        warmup_eta = compute_warmup_eta(epoch_index, step_index, steps_per_epoch, config["warmup_epochs"])
        effective_weights = resolve_loss_weights(config, epoch_index)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images, warmup_eta=warmup_eta)
        losses = compute_loss_bundle(
            outputs=outputs,
            targets=targets,
            label_smoothing=config["label_smoothing"],
            pairwise_weight=effective_weights["pairwise_weight"],
            resistance_weight=effective_weights["resistance_weight"],
            energy_weight=effective_weights["energy_weight"],
            entropy_weight=effective_weights["entropy_weight"],
            prototype_diversity_weight=effective_weights["prototype_diversity_weight"],
            pairwise_margin=config["pairwise_margin"],
            resistance_margin=config["resistance_margin"],
            energy_margin=config["energy_margin"],
            activation_budget_weight=effective_weights["activation_budget_weight"],
            activation_budget_target=config["activation_budget_target"],
        )
        losses["loss"].backward()
        grad_norm = clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()

        batch_size = images.size(0)
        for name, meter in meters.items():
            meter.update(losses[name].item(), batch_size)

        if (step_index + 1) % config["log_interval"] == 0 or (step_index + 1) == steps_per_epoch:
            progress_bar.set_postfix(
                loss=f"{meters['loss'].avg:.4f}",
                acc=f"{meters['acc'].avg:.4f}",
                eta=f"{warmup_eta:.2f}",
                gn=f"{float(grad_norm):.2f}",
            )

    return {name: meter.avg for name, meter in meters.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch_index: int,
    config: Dict[str, Any],
) -> Dict[str, float]:
    model.eval()
    meters = {name: AverageMeter() for name in METRIC_NAMES}
    effective_weights = resolve_loss_weights(config, epoch_index=epoch_index)

    for batch in tqdm(loader, desc="eval", leave=False):
        images, targets = move_batch_to_device(batch, device)
        outputs = model(images, warmup_eta=1.0)
        losses = compute_loss_bundle(
            outputs=outputs,
            targets=targets,
            label_smoothing=0.0,
            pairwise_weight=effective_weights["pairwise_weight"],
            resistance_weight=effective_weights["resistance_weight"],
            energy_weight=effective_weights["energy_weight"],
            entropy_weight=0.0,
            prototype_diversity_weight=effective_weights["prototype_diversity_weight"],
            pairwise_margin=config["pairwise_margin"],
            resistance_margin=config["resistance_margin"],
            energy_margin=config["energy_margin"],
            activation_budget_weight=effective_weights["activation_budget_weight"],
            activation_budget_target=config["activation_budget_target"],
        )
        batch_size = images.size(0)
        for name, meter in meters.items():
            meter.update(losses[name].item(), batch_size)

    return {name: meter.avg for name, meter in meters.items()}


def maybe_resume(
    path: str,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    device: torch.device,
) -> Dict[str, Any]:
    if not path:
        return {"start_epoch": 0, "best_accuracy": 0.0}

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return {
        "start_epoch": int(checkpoint["epoch"]) + 1,
        "best_accuracy": float(checkpoint.get("best_accuracy", 0.0)),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = vars(args)
    apply_dataset_defaults(config)
    apply_model_defaults(config)

    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    save_json(output_dir / "config.json", config)

    train_loader, test_loader = build_classification_loaders(
        dataset_name=config["dataset"],
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_mnist_eml_model(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    resume_info = maybe_resume(args.resume, model, optimizer, scheduler, device)
    start_epoch = resume_info["start_epoch"]
    best_accuracy = resume_info["best_accuracy"]
    best_epoch = start_epoch if best_accuracy > 0.0 else 0
    epochs_without_improvement = 0
    stopped_early = False

    history: List[Dict[str, Any]] = []
    start_time = time.time()

    for epoch_index in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch_index=epoch_index,
            config=config,
        )
        val_metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            epoch_index=epoch_index,
            config=config,
        )
        scheduler.step()

        epoch_record = {
            "epoch": epoch_index + 1,
            "train": train_metrics,
            "eval": val_metrics,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_record)

        current_accuracy = val_metrics["acc"]
        is_best = current_accuracy > (best_accuracy + config["early_stop_min_delta"])
        if is_best:
            best_accuracy = current_accuracy
            best_epoch = epoch_index + 1
            epochs_without_improvement = 0
        elif (
            config["early_stop_patience"] > 0
            and (epoch_index + 1) >= config["early_stop_min_epochs"]
        ):
            epochs_without_improvement += 1

        save_checkpoint(
            path=output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch_index,
            best_accuracy=best_accuracy,
            config=config,
        )
        if is_best:
            save_checkpoint(
                path=output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_index,
                best_accuracy=best_accuracy,
                config=config,
            )

        save_json(
            output_dir / "history.json",
            {
                "epochs": history,
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "stopped_early": stopped_early,
            },
        )
        print(
            f"epoch={epoch_index + 1} "
            f"dataset={config['dataset']} "
            f"model={config['model_name']} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"eval_loss={val_metrics['loss']:.4f} "
            f"eval_acc={val_metrics['acc']:.4f} "
            f"best_acc={best_accuracy:.4f} "
            f"best_epoch={best_epoch} "
            f"no_improve={epochs_without_improvement} "
            f"{format_epoch_diagnostics('train', train_metrics)} "
            f"{format_epoch_diagnostics('eval', val_metrics)}"
        )

        if (
            config["early_stop_patience"] > 0
            and (epoch_index + 1) >= config["early_stop_min_epochs"]
            and epochs_without_improvement >= config["early_stop_patience"]
        ):
            stopped_early = True
            save_json(
                output_dir / "history.json",
                {
                    "epochs": history,
                    "best_accuracy": best_accuracy,
                    "best_epoch": best_epoch,
                    "stopped_early": stopped_early,
                },
            )
            print(
                f"early_stop epoch={epoch_index + 1} "
                f"best_epoch={best_epoch} "
                f"best_acc={best_accuracy:.4f} "
                f"patience={config['early_stop_patience']}"
            )
            break

    elapsed = time.time() - start_time
    save_json(
        output_dir / "summary.json",
        {
            "dataset": config["dataset"],
            "model_name": config["model_name"],
            "best_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "epochs_trained": len(history),
            "configured_epochs": config["epochs"],
            "stopped_early": stopped_early,
            "epochs_without_improvement": epochs_without_improvement,
            "elapsed_seconds": elapsed,
            "device": str(device),
        },
    )
    print(f"training finished in {elapsed:.1f}s, best_acc={best_accuracy:.4f}")


if __name__ == "__main__":
    main()
