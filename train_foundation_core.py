import argparse
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from eml_mnist import EMLFoundationCore
from eml_mnist.toy_datasets import ToyFoundationDataset
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
    parser = argparse.ArgumentParser(description="Train the general EML foundation core on ToyFoundationDataset")
    parser.add_argument("--output-dir", type=str, default="./outputs/toy_foundation_core")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=75)
    parser.add_argument("--event-dim", type=int, default=8)
    parser.add_argument("--slot-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--representation-dim", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--patch-dim", type=int, default=10)
    parser.add_argument("--local-query-dim", type=int, default=6)
    parser.add_argument("--reconstruction-dim", type=int, default=8)
    parser.add_argument("--num-actions", type=int, default=5)
    parser.add_argument("--num-patches", type=int, default=6)
    parser.add_argument("--num-queries", type=int, default=4)
    parser.add_argument("--num-risk-outputs", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--slot-layout", type=str, default="goal:1,file:2,tool:2,memory:1,risk:1,patch:2")
    parser.add_argument("--risk-weight", type=float, default=0.5)
    parser.add_argument("--reconstruction-weight", type=float, default=0.5)
    parser.add_argument("--novelty-weight", type=float, default=0.25)
    parser.add_argument("--enable-prototype-novelty", action="store_true")
    parser.add_argument("--num-novelty-prototypes", type=int, default=8)
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
    risk_weight: float,
    reconstruction_weight: float,
    novelty_weight: float,
    use_novelty: bool,
    max_batches: int | None,
) -> Tuple[Dict[str, float], int]:
    is_train = optimizer is not None
    model.train(is_train)
    metric_names = [
        "loss",
        "action_loss",
        "patch_loss",
        "risk_loss",
        "reconstruction_loss",
        "action_acc",
        "patch_acc",
    ]
    if use_novelty:
        metric_names.append("novelty_loss")
    meters = {name: AverageMeter() for name in metric_names}

    for _, batch in iter_batches(loader, max_batches=max_batches):
        batch = move_batch_to_device(batch, device)
        eta = compute_warmup_eta(global_step, warmup_steps) if is_train else 1.0

        with torch.set_grad_enabled(is_train):
            outputs = model(
                event=batch["event"],
                candidate_actions=batch["candidate_actions"],
                candidate_patches=batch["candidate_patches"],
                local_queries=batch["local_queries"],
                warmup_eta=eta,
            )

            action_logits = outputs["action"]["action_score"]
            patch_logits = outputs["patch_rank"]["patch_score"]
            risk_logits = outputs["risk_resistance"]["risk_score"]
            reconstruction = outputs["local_reconstruction"]["reconstruction"]

            action_loss = F.cross_entropy(action_logits, batch["action_target"])
            patch_loss = F.cross_entropy(patch_logits, batch["patch_target"])
            risk_loss = F.binary_cross_entropy_with_logits(risk_logits, batch["risk_target"])
            reconstruction_loss = F.mse_loss(reconstruction, batch["reconstruction_target"])
            loss = action_loss + patch_loss + risk_weight * risk_loss + reconstruction_weight * reconstruction_loss

            novelty_loss = None
            if use_novelty:
                novelty_logits = outputs["prototype_novelty"]["novelty_score"]
                novelty_loss = F.binary_cross_entropy_with_logits(novelty_logits, batch["novelty_target"])
                loss = loss + novelty_weight * novelty_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                global_step += 1

        batch_size = batch["event"].size(0)
        meters["loss"].update(scalar(loss), batch_size)
        meters["action_loss"].update(scalar(action_loss), batch_size)
        meters["patch_loss"].update(scalar(patch_loss), batch_size)
        meters["risk_loss"].update(scalar(risk_loss), batch_size)
        meters["reconstruction_loss"].update(scalar(reconstruction_loss), batch_size)
        meters["action_acc"].update(classification_accuracy(action_logits, batch["action_target"]), batch_size)
        meters["patch_acc"].update(classification_accuracy(patch_logits, batch["patch_target"]), batch_size)
        if use_novelty and novelty_loss is not None:
            meters["novelty_loss"].update(scalar(novelty_loss), batch_size)

    return {name: meter.avg for name, meter in meters.items()}, global_step


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    dataset_kwargs = {
        "event_dim": args.event_dim,
        "action_dim": args.action_dim,
        "patch_dim": args.patch_dim,
        "local_query_dim": args.local_query_dim,
        "reconstruction_dim": args.reconstruction_dim,
        "num_actions": args.num_actions,
        "num_patches": args.num_patches,
        "num_queries": args.num_queries,
        "num_risk_outputs": args.num_risk_outputs,
        "num_novelty_prototypes": args.num_novelty_prototypes,
    }
    train_dataset = ToyFoundationDataset(size=args.train_size, seed=args.seed, **dataset_kwargs)
    val_dataset = ToyFoundationDataset(size=args.val_size, seed=args.seed + 1, **dataset_kwargs)
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
        action_dim=args.action_dim,
        patch_dim=args.patch_dim,
        local_query_dim=args.local_query_dim,
        reconstruction_dim=args.reconstruction_dim,
        num_risk_outputs=args.num_risk_outputs,
        local_num_queries=args.num_queries,
        enable_prototype_novelty=args.enable_prototype_novelty,
        num_novelty_prototypes=args.num_novelty_prototypes,
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
            risk_weight=args.risk_weight,
            reconstruction_weight=args.reconstruction_weight,
            novelty_weight=args.novelty_weight,
            use_novelty=args.enable_prototype_novelty,
            max_batches=max_train_batches,
        )
        val_metrics, global_step = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            warmup_steps=args.warmup_steps,
            global_step=global_step,
            risk_weight=args.risk_weight,
            reconstruction_weight=args.reconstruction_weight,
            novelty_weight=args.novelty_weight,
            use_novelty=args.enable_prototype_novelty,
            max_batches=max_val_batches,
        )
        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_metrics['loss']:.4f} train_action_acc={train_metrics['action_acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_action_acc={val_metrics['action_acc']:.4f} "
            f"val_patch_acc={val_metrics['patch_acc']:.4f}"
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
            "task": "toy_foundation_core",
            "config": vars(args),
            "history": history,
            "best_val_loss": best_val_loss,
        },
    )


if __name__ == "__main__":
    main()
