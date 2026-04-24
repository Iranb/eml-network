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

from eml_mnist import CharVocabulary, EMLFoundationCore, SyntheticShapeEnergyDataset, SyntheticTextEnergyDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a combined EML field foundation smoke model")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-train-size", type=int, default=512)
    parser.add_argument("--text-train-size", type=int, default=512)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="./outputs/eml_field_foundation")
    parser.add_argument("--activation-budget-weight", type=float, default=0.01)
    parser.add_argument("--energy-penalty-weight", type=float, default=0.005)
    parser.add_argument("--energy-margin", type=float, default=4.0)
    parser.add_argument("--resistance-supervision-weight", type=float, default=0.02)
    parser.add_argument("--attractor-diversity-weight", type=float, default=0.005)
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


def tensor_stats(*values: torch.Tensor) -> tuple[float, float]:
    tensors = [value.detach().to(dtype=torch.float32).reshape(-1) for value in values if torch.is_tensor(value)]
    if not tensors:
        return 0.0, 0.0
    merged = torch.cat(tensors, dim=0)
    return float(merged.mean().item()), float(merged.std(unbiased=False).item())


def masked_char_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    correct = ((predictions == targets) & mask).to(dtype=torch.float32).sum()
    total = mask.to(dtype=torch.float32).sum().clamp_min(1.0)
    return float((correct / total).item())


def masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_fp32 = mask.to(dtype=torch.float32)
    loss = (x - y).square() * mask_fp32
    return loss.sum() / mask_fp32.sum().clamp_min(1.0)


def attractor_diversity_loss(attractor_states: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(attractor_states, dim=-1)
    cosine = normalized @ normalized.transpose(1, 2)
    eye = torch.eye(cosine.size(-1), device=cosine.device, dtype=torch.bool).unsqueeze(0)
    off_diagonal = cosine.masked_select(~eye)
    if off_diagonal.numel() == 0:
        return attractor_states.new_zeros(())
    return off_diagonal.square().mean()


def build_model(vocab_size: int, image_classes: int) -> EMLFoundationCore:
    return EMLFoundationCore(
        slot_dim=24,
        event_dim=24,
        hidden_dim=48,
        slot_layout={"goal": 1, "image": 3, "text": 3, "memory": 2},
        num_layers=2,
        top_k=3,
        representation_dim=24,
        local_query_dim=24,
        reconstruction_dim=24,
        image_head_specs={"shape": image_classes},
        text_vocab_size=vocab_size,
        text_embed_dim=24,
        text_feature_dim=24,
        text_hidden_dim=48,
        enable_text_generation_head=True,
        enable_action_head=False,
        enable_patch_rank_head=False,
        enable_image_field_encoder=True,
        enable_text_field_encoder=True,
        image_field_config={
            "input_channels": 3,
            "sensor_dim": 24,
            "measurement_dim": 24,
            "field_dim": 24,
            "hidden_dim": 48,
            "num_hypotheses": 4,
            "num_parent_hypotheses": 4,
            "num_attractors": 4,
            "representation_dim": 24,
            "patch_stride": 4,
        },
        text_field_config={
            "embed_dim": 24,
            "sensor_dim": 24,
            "measurement_dim": 24,
            "field_dim": 24,
            "hidden_dim": 48,
            "num_hypotheses": 4,
            "num_chunk_hypotheses": 4,
            "num_attractors": 4,
            "representation_dim": 24,
            "causal_window_size": 5,
            "chunk_size": 4,
            "chunk_window_size": 3,
        },
    )


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    vocab = CharVocabulary()
    image_dataset = SyntheticShapeEnergyDataset(
        size=args.image_train_size,
        image_size=args.image_size,
        seed=args.seed,
        target_type="shape",
        include_background_clutter=True,
        include_mask=True,
    )
    text_dataset = SyntheticTextEnergyDataset(
        size=args.text_train_size,
        seq_len=args.seq_len,
        vocab=vocab,
        seed=args.seed,
    )
    image_loader = DataLoader(image_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    text_loader = DataLoader(text_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    image_iterator = cycle(image_loader)
    text_iterator = cycle(text_loader)

    model = build_model(vocab_size=len(vocab), image_classes=image_dataset.num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        image_batch = move_batch(next(image_iterator), device)
        text_batch = move_batch(next(text_iterator), device)
        warmup_eta = min(1.0, step / max(1, args.steps // 4))

        image_out = model(images=image_batch["image"], use_field_path=True, warmup_eta=warmup_eta)
        text_out = model(
            text_tokens=text_batch["input_ids"],
            text_padding_mask=text_batch["input_mask"],
            use_field_path=True,
            warmup_eta=warmup_eta,
        )

        image_logits = image_out["image_heads"]["shape"]["logits"]
        text_logits = text_out["text_generation"]["logits"]
        field_text_logits = text_out["text_field_generation"]["logits"]
        image_loss = F.cross_entropy(image_logits, image_batch["label"])
        text_loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            text_batch["target_ids"].reshape(-1),
            ignore_index=vocab.pad_id,
        )
        field_text_loss = F.cross_entropy(
            field_text_logits.reshape(-1, field_text_logits.size(-1)),
            text_batch["target_ids"].reshape(-1),
            ignore_index=vocab.pad_id,
        )

        image_budget = image_out["image_field"]["diagnostics"]["budget_loss"]
        text_budget = text_out["text_field"]["diagnostics"]["budget_loss"]
        image_energy_penalty = F.relu(image_logits.abs() - args.energy_margin).square().mean()
        text_energy_penalty = F.relu(text_logits.abs() - args.energy_margin).square().mean()
        image_resistance = image_out["image_heads"]["shape"]["resistance"].mean(dim=-1)
        text_resistance = text_out["text_field"]["local_hypotheses"]["resistance"].mean(dim=-1)
        image_resistance_loss = F.mse_loss(image_resistance, image_batch["resistance_target"])
        text_resistance_loss = masked_mse(text_resistance, text_batch["resistance_target"], text_batch["input_mask"])
        diversity_loss = attractor_diversity_loss(image_out["image_field"]["attractor_states"])
        diversity_loss = diversity_loss + attractor_diversity_loss(text_out["text_field"]["attractor_states"])

        total_loss = image_loss + text_loss + 0.25 * field_text_loss
        total_loss = total_loss + args.activation_budget_weight * (image_budget + text_budget)
        total_loss = total_loss + args.energy_penalty_weight * (image_energy_penalty + text_energy_penalty)
        total_loss = total_loss + args.resistance_supervision_weight * (image_resistance_loss + text_resistance_loss)
        total_loss = total_loss + args.attractor_diversity_weight * diversity_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if step % args.log_interval == 0 or step == args.steps:
            image_accuracy = (image_logits.argmax(dim=-1) == image_batch["label"]).to(dtype=torch.float32).mean()
            text_valid = text_batch["target_ids"] != vocab.pad_id
            text_accuracy = masked_char_accuracy(text_logits, text_batch["target_ids"], text_valid)
            drive_mean, drive_std = tensor_stats(image_out["image_heads"]["shape"]["drive"], text_out["text_generation"]["drive"])
            resistance_mean, resistance_std = tensor_stats(
                image_out["image_heads"]["shape"]["resistance"],
                text_out["text_generation"]["resistance"],
            )
            energy_mean, energy_std = tensor_stats(image_out["image_heads"]["shape"]["energy"], text_out["text_generation"]["energy"])
            image_stats = image_out["diagnostics"]["stats"]
            text_stats = text_out["diagnostics"]["stats"]
            activation_rate = 0.5 * (
                scalar(image_stats.get("gate_mean")) + scalar(text_stats.get("gate_mean"))
            )
            image_injection_norm = scalar(image_out["attractor_injection"]["image"]["injection_norm"])
            text_injection_norm = scalar(text_out["attractor_injection"]["text"]["injection_norm"])
            graph_gate_mass = 0.5 * (
                scalar(image_stats.get("gate_mass_mean")) + scalar(text_stats.get("gate_mass_mean"))
            )
            route_strength = 0.5 * (
                scalar(image_stats.get("active_route_strength_mean")) + scalar(text_stats.get("active_route_strength_mean"))
            )
            print(
                f"step={step} "
                f"image_loss={image_loss.item():.4f} "
                f"text_loss={text_loss.item():.4f} "
                f"total_loss={total_loss.item():.4f} "
                f"image_accuracy={image_accuracy.item():.4f} "
                f"text_char_accuracy={text_accuracy:.4f} "
                f"drive_mean={drive_mean:.4f} drive_std={drive_std:.4f} "
                f"resistance_mean={resistance_mean:.4f} resistance_std={resistance_std:.4f} "
                f"energy_mean={energy_mean:.4f} energy_std={energy_std:.4f} "
                f"activation_rate={activation_rate:.4f} "
                f"attractor_injection_norm={0.5 * (image_injection_norm + text_injection_norm):.4f} "
                f"graph_gate_mass_mean={graph_gate_mass:.4f} "
                f"active_route_strength_mean={route_strength:.4f}"
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
