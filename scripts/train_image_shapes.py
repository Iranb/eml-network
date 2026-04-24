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

from eml_mnist import EMLFoundationCore, PureEMLImageBackbone, SyntheticShapeDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an EML image classifier on SyntheticShapeDataset")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="./outputs/image_shapes")
    parser.add_argument("--include-combo", action="store_true")
    return parser


def cycle(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def scalar(value: object) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    if value is None:
        return 0.0
    return float(value)


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    dataset = SyntheticShapeDataset(
        size=args.train_size,
        image_size=args.image_size,
        seed=args.seed,
        include_combo_label=args.include_combo,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    iterator = cycle(loader)

    image_head_specs = {"shape": 5, "color": 4}
    if args.include_combo:
        image_head_specs["combo"] = 20

    image_backbone = PureEMLImageBackbone(
        image_size=args.image_size,
        input_channels=3,
        feature_dim=24,
        event_dim=24,
        hidden_dim=48,
        bank_dim=48,
        num_layers=2,
        patch_size=4,
        patch_stride=4,
        local_window_size=3,
        merge_every=2,
        dropout=0.0,
    ).to(device)
    model = EMLFoundationCore(
        slot_dim=24,
        event_dim=24,
        hidden_dim=48,
        slot_layout={"goal": 1, "image": 3, "memory": 2, "risk": 1},
        num_layers=2,
        top_k=3,
        representation_dim=24,
        local_query_dim=24,
        reconstruction_dim=24,
        image_head_specs=image_head_specs,
        enable_action_head=False,
        enable_patch_rank_head=False,
    ).to(device)
    optimizer = AdamW(list(image_backbone.parameters()) + list(model.parameters()), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        batch = move_batch(next(iterator), device)
        warmup_eta = min(1.0, step / max(1, args.steps // 4))
        image_backbone_out = image_backbone(batch["image"], warmup_eta=warmup_eta)
        outputs = model(image_backbone_outputs=image_backbone_out, warmup_eta=warmup_eta)
        image_heads = outputs["image_heads"]

        shape_logits = image_heads["shape"]["logits"]
        color_logits = image_heads["color"]["logits"]
        shape_loss = F.cross_entropy(shape_logits, batch["shape_label"])
        color_loss = F.cross_entropy(color_logits, batch["color_label"])
        total_loss = shape_loss + color_loss

        if args.include_combo:
            combo_logits = image_heads["combo"]["logits"]
            combo_loss = F.cross_entropy(combo_logits, batch["combo_label"])
            total_loss = total_loss + combo_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if step % args.log_interval == 0 or step == args.steps:
            shape_acc = (shape_logits.argmax(dim=-1) == batch["shape_label"]).float().mean().item()
            stats = outputs["diagnostics"]["stats"]
            representation_modality = outputs.get("representation_modality", {})
            print(
                f"step={step} "
                f"loss={total_loss.item():.4f} "
                f"accuracy={shape_acc:.4f} "
                f"drive_mean={stats.get('drive_mean', 0.0):.4f} drive_std={stats.get('drive_std', 0.0):.4f} "
                f"resistance_mean={stats.get('resistance_mean', 0.0):.4f} resistance_std={stats.get('resistance_std', 0.0):.4f} "
                f"energy_mean={stats.get('energy_mean', 0.0):.4f} energy_std={stats.get('energy_std', 0.0):.4f} "
                f"gate_activation_rate={stats.get('gate_activation_rate', 0.0):.4f} "
                f"active_route_strength_mean={stats.get('active_route_strength_mean', 0.0):.4f} "
                f"graph_gate_mass_mean={stats.get('gate_mass_mean', 0.0):.4f} "
                f"image_slot_readout_weight={scalar(representation_modality.get('image_slot_readout_weight_mean')):.4f} "
                f"injected_image_slots={outputs['modality_injection']['injected_image_slots']}"
            )

    torch.save(
        {
            "backbone": image_backbone.state_dict(),
            "model": model.state_dict(),
            "config": vars(args),
        },
        output_dir / "last.pt",
    )


if __name__ == "__main__":
    main()
