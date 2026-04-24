import argparse
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from eml_mnist import EMLFoundationCore
from eml_mnist.toy_datasets import ToyPatchRankingDataset
from eml_mnist.toy_training import (
    AverageMeter,
    classification_accuracy,
    compute_warmup_eta,
    ensure_dir,
    iter_batches,
    move_batch_to_device,
    parse_slot_layout,
    resolve_device,
    save_json,
    scalar,
    set_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the EML foundation core on ToyPatchRankingDataset")
    parser.add_argument("--output-dir", type=str, default="./outputs/toy_patch_rank")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--event-dim", type=int, default=8)
    parser.add_argument("--slot-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--representation-dim", type=int, default=16)
    parser.add_argument("--patch-dim", type=int, default=10)
    parser.add_argument("--num-patches", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--slot-layout", type=str, default="goal:1,file:2,tool:2,memory:1")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser


def run_epoch(
    model: EMLFoundationCore,
    loader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
    warmup_steps: int,
    global_step: int,
    max_batches: int | None,
) -> Tuple[Dict[str, float], int]:
    is_train = optimizer is not None
    model.train(is_train)
    meters = {name: AverageMeter() for name in ["loss", "acc"]}

    for _, batch in iter_batches(loader, max_batches=max_batches):
        batch = move_batch_to_device(batch, device)
        eta = compute_warmup_eta(global_step, warmup_steps) if is_train else 1.0

        with torch.set_grad_enabled(is_train):
            outputs = model(
                event=batch["event"],
                candidate_patches=batch["candidate_patches"],
                warmup_eta=eta,
            )
            logits = outputs["patch_rank"]["patch_score"]
            loss = F.cross_entropy(logits, batch["patch_target"])
            acc = classification_accuracy(logits, batch["patch_target"])

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                global_step += 1

        batch_size = batch["event"].size(0)
        meters["loss"].update(scalar(loss), batch_size)
        meters["acc"].update(acc, batch_size)

    return {name: meter.avg for name, meter in meters.items()}, global_step


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    train_dataset = ToyPatchRankingDataset(
        size=args.train_size,
        event_dim=args.event_dim,
        patch_dim=args.patch_dim,
        num_patches=args.num_patches,
        seed=args.seed,
    )
    val_dataset = ToyPatchRankingDataset(
        size=args.val_size,
        event_dim=args.event_dim,
        patch_dim=args.patch_dim,
        num_patches=args.num_patches,
        seed=args.seed + 1,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = EMLFoundationCore(
        slot_dim=args.slot_dim,
        event_dim=args.event_dim,
        hidden_dim=args.hidden_dim,
        slot_layout=parse_slot_layout(args.slot_layout),
        num_layers=args.num_layers,
        top_k=args.top_k,
        representation_dim=args.representation_dim,
        patch_dim=args.patch_dim,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    max_train_batches = args.max_train_batches or None
    max_val_batches = args.max_val_batches or None
    best_val_loss = float("inf")
    history = []
    global_step = 0

    for epoch in range(args.epochs):
        train_metrics, global_step = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            warmup_steps=args.warmup_steps,
            global_step=global_step,
            max_batches=max_train_batches,
        )
        val_metrics, global_step = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            warmup_steps=args.warmup_steps,
            global_step=global_step,
            max_batches=max_val_batches,
        )
        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": vars(args),
                    "best_val_loss": best_val_loss,
                    "history": history,
                },
                output_dir / "best.pt",
            )

    save_json(
        output_dir / "metrics.json",
        {
            "task": "toy_patch_rank",
            "config": vars(args),
            "history": history,
            "best_val_loss": best_val_loss,
        },
    )


if __name__ == "__main__":
    main()
