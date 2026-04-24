import argparse
import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from eml_mnist import EMLImageFieldClassifier, SyntheticShapeEnergyDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an EML image-field classifier on SyntheticShapeEnergyDataset")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--target-type", type=str, default="shape", choices=["shape", "color", "combo"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="./outputs/eml_image_field")
    parser.add_argument("--activation-budget-weight", type=float, default=0.02)
    parser.add_argument("--energy-penalty-weight", type=float, default=0.01)
    parser.add_argument("--energy-margin", type=float, default=3.0)
    parser.add_argument("--resistance-supervision-weight", type=float, default=0.10)
    parser.add_argument("--attractor-diversity-weight", type=float, default=0.01)
    return parser


def cycle(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def scalar(value: object) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    if value is None:
        return 0.0
    return float(value)


def batch_corrcoef(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.detach().to(dtype=torch.float32)
    y = y.detach().to(dtype=torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.std(unbiased=False) * y.std(unbiased=False)
    if float(denom) < 1.0e-6:
        return x.new_zeros(())
    return (x * y).mean() / denom


def attractor_diversity_loss(attractor_states: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(attractor_states, dim=-1)
    cosine = normalized @ normalized.transpose(1, 2)
    eye = torch.eye(cosine.size(-1), device=cosine.device, dtype=torch.bool).unsqueeze(0)
    off_diagonal = cosine.masked_select(~eye)
    if off_diagonal.numel() == 0:
        return attractor_states.new_zeros(())
    return off_diagonal.square().mean()


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    dataset = SyntheticShapeEnergyDataset(
        size=args.train_size,
        image_size=args.image_size,
        seed=args.seed,
        target_type=args.target_type,
        include_background_clutter=True,
        include_mask=True,
    )
    if args.num_classes is not None and args.num_classes != dataset.num_classes:
        raise ValueError(
            f"--num-classes={args.num_classes} does not match target_type={args.target_type} ({dataset.num_classes})"
        )
    num_classes = args.num_classes or dataset.num_classes

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    iterator = cycle(loader)

    model = EMLImageFieldClassifier(
        num_classes=num_classes,
        input_channels=3,
        sensor_dim=24,
        measurement_dim=24,
        field_dim=24,
        hidden_dim=48,
        num_hypotheses=4,
        num_parent_hypotheses=4,
        num_attractors=4,
        representation_dim=24,
        patch_size=5,
        patch_stride=4,
        local_window_size=3,
        parent_window_size=3,
        composition_region_size=2,
        clip_value=3.0,
        prototype_temperature=0.25,
        enable_parent_consensus=True,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        batch = move_batch(next(iterator), device)
        warmup_eta = min(1.0, step / max(1, args.steps // 4))
        outputs = model(batch["image"], warmup_eta=warmup_eta)

        logits = outputs["logits"]
        class_loss = F.cross_entropy(logits, batch["label"])
        budget_loss = outputs["diagnostics"]["budget_loss"]
        energy_penalty = F.relu(logits.abs() - args.energy_margin).square().mean()
        sample_resistance = outputs["resistance"].mean(dim=-1)
        resistance_supervision = F.mse_loss(sample_resistance, batch["resistance_target"])
        diversity_loss = attractor_diversity_loss(outputs["attractor_states"])

        total_loss = class_loss
        total_loss = total_loss + args.activation_budget_weight * budget_loss
        total_loss = total_loss + args.energy_penalty_weight * energy_penalty
        total_loss = total_loss + args.resistance_supervision_weight * resistance_supervision
        total_loss = total_loss + args.attractor_diversity_weight * diversity_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if step % args.log_interval == 0 or step == args.steps:
            stats = outputs["diagnostics"]["stats"]
            accuracy = (logits.argmax(dim=-1) == batch["label"]).to(dtype=torch.float32).mean()
            resistance_corr = batch_corrcoef(sample_resistance, batch["resistance_target"])
            print(
                f"step={step} "
                f"loss={total_loss.item():.4f} "
                f"accuracy={accuracy.item():.4f} "
                f"drive_mean={outputs['drive'].mean().item():.4f} drive_std={outputs['drive'].std(unbiased=False).item():.4f} "
                f"resistance_mean={outputs['resistance'].mean().item():.4f} resistance_std={outputs['resistance'].std(unbiased=False).item():.4f} "
                f"energy_mean={outputs['energy'].mean().item():.4f} energy_std={outputs['energy'].std(unbiased=False).item():.4f} "
                f"activation_rate={scalar(stats.get('parent_activation_mean')):.4f} "
                f"support_mean={scalar(stats.get('parent_support_mean')):.4f} "
                f"conflict_mean={scalar(stats.get('parent_conflict_mean')):.4f} "
                f"attractor_activation_mean={scalar(stats.get('attractor_activation_mean')):.4f} "
                f"resistance_corr={resistance_corr.item():.4f}"
            )

    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(args),
        },
        output_dir / "last.pt",
    )


if __name__ == "__main__":
    main()
