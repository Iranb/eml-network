import argparse
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from eml_mnist import EMLSlotGraphLayer
from eml_mnist.toy_datasets import ToyStateTransitionDataset
from eml_mnist.toy_training import (
    AverageMeter,
    compute_warmup_eta,
    ensure_dir,
    iter_batches,
    move_batch_to_device,
    resolve_device,
    save_json,
    scalar,
    set_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train EMLSlotGraphLayer on ToyStateTransitionDataset")
    parser.add_argument("--output-dir", type=str, default="./outputs/toy_state_transition")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--event-dim", type=int, default=8)
    parser.add_argument("--slot-dim", type=int, default=8)
    parser.add_argument("--num-slots", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--router-weight", type=float, default=0.1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser


def _target_active_mask(target_topk_indices: torch.Tensor, num_slots: int) -> torch.Tensor:
    target = torch.zeros(target_topk_indices.size(0), num_slots, device=target_topk_indices.device)
    target.scatter_(1, target_topk_indices, 1.0)
    return target


def _router_topk_hit(predicted: torch.Tensor, target: torch.Tensor) -> float:
    intersection = 0.0
    total = 0.0
    for pred_row, target_row in zip(predicted.tolist(), target.tolist()):
        pred_set = set(pred_row)
        target_set = set(target_row)
        intersection += len(pred_set & target_set)
        total += len(target_set)
    return 0.0 if total == 0 else intersection / total


def run_epoch(
    model: EMLSlotGraphLayer,
    loader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
    warmup_steps: int,
    global_step: int,
    router_weight: float,
    max_batches: int | None,
) -> Tuple[Dict[str, float], int]:
    is_train = optimizer is not None
    model.train(is_train)
    meters = {name: AverageMeter() for name in ["loss", "transition_mse", "router_bce", "router_hit"]}

    for _, batch in iter_batches(loader, max_batches=max_batches):
        batch = move_batch_to_device(batch, device)
        eta = compute_warmup_eta(global_step, warmup_steps) if is_train else 1.0

        with torch.set_grad_enabled(is_train):
            outputs = model(
                event=batch["event"],
                slot_states=batch["slot_states"],
                slot_mask=batch["slot_mask"],
                warmup_eta=eta,
            )
            transition_mse = F.mse_loss(outputs["slot_states"], batch["next_slot_states"])
            target_active = _target_active_mask(batch["target_topk_indices"], batch["slot_states"].size(1))
            router_bce = F.binary_cross_entropy(outputs["router"]["gate"], target_active)
            loss = transition_mse + router_weight * router_bce

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                global_step += 1

        batch_size = batch["event"].size(0)
        meters["loss"].update(scalar(loss), batch_size)
        meters["transition_mse"].update(scalar(transition_mse), batch_size)
        meters["router_bce"].update(scalar(router_bce), batch_size)
        meters["router_hit"].update(
            _router_topk_hit(outputs["active_indices"].detach().cpu(), batch["target_topk_indices"].detach().cpu()),
            batch_size,
        )

    return {name: meter.avg for name, meter in meters.items()}, global_step


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    train_dataset = ToyStateTransitionDataset(
        size=args.train_size,
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        event_dim=args.event_dim,
        top_k=args.top_k,
        seed=args.seed,
    )
    val_dataset = ToyStateTransitionDataset(
        size=args.val_size,
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        event_dim=args.event_dim,
        top_k=args.top_k,
        seed=args.seed + 1,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = EMLSlotGraphLayer(
        slot_dim=args.slot_dim,
        event_dim=args.event_dim,
        hidden_dim=args.hidden_dim,
        top_k=args.top_k,
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
            router_weight=args.router_weight,
            max_batches=max_train_batches,
        )
        val_metrics, global_step = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            warmup_steps=args.warmup_steps,
            global_step=global_step,
            router_weight=args.router_weight,
            max_batches=max_val_batches,
        )
        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_metrics['loss']:.4f} train_hit={train_metrics['router_hit']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_hit={val_metrics['router_hit']:.4f}"
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
            "task": "toy_state_transition",
            "config": vars(args),
            "history": history,
            "best_val_loss": best_val_loss,
        },
    )


if __name__ == "__main__":
    main()
