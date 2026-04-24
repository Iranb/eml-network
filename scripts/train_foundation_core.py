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

from eml_mnist import (
    CharVocabulary,
    EMLFoundationCore,
    EMLTextBackbone,
    PureEMLImageBackbone,
    SyntheticGrammarDataset,
    SyntheticShapeDataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a mixed image+text EML foundation core smoke run")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--text-max-length", type=int, default=48)
    parser.add_argument("--image-train-size", type=int, default=512)
    parser.add_argument("--text-train-size", type=int, default=512)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="./outputs/foundation_core")
    return parser


def cycle(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def masked_char_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    correct = ((predictions == targets) & mask).float().sum()
    total = mask.float().sum().clamp_min(1.0)
    return float((correct / total).item())


def merge_stats(image_stats: dict[str, float], text_stats: dict[str, float]) -> dict[str, float]:
    merged = {}
    keys = set(image_stats) | set(text_stats)
    for key in keys:
        merged[key] = 0.5 * (image_stats.get(key, 0.0) + text_stats.get(key, 0.0))
    return merged


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

    image_dataset = SyntheticShapeDataset(size=args.image_train_size, image_size=args.image_size, seed=args.seed)
    image_loader = DataLoader(image_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    image_iter = cycle(image_loader)

    vocab = CharVocabulary()
    text_dataset = SyntheticGrammarDataset(
        size=args.text_train_size,
        vocab=vocab,
        max_length=args.text_max_length,
        seed=args.seed + 1,
    )
    text_loader = DataLoader(text_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    text_iter = cycle(text_loader)

    image_backbone = PureEMLImageBackbone(
        image_size=args.image_size,
        input_channels=3,
        feature_dim=32,
        event_dim=32,
        hidden_dim=64,
        bank_dim=64,
        num_layers=3,
        patch_size=4,
        patch_stride=4,
        local_window_size=3,
        merge_every=2,
        dropout=0.0,
    ).to(device)
    text_backbone = EMLTextBackbone(
        vocab_size=len(vocab),
        embed_dim=32,
        feature_dim=32,
        event_dim=32,
        hidden_dim=64,
        bank_dim=64,
        num_layers=3,
        pad_id=vocab.pad_id,
        dropout=0.0,
    ).to(device)
    model = EMLFoundationCore(
        slot_dim=32,
        event_dim=32,
        hidden_dim=64,
        slot_layout={"goal": 1, "image": 2, "text": 2, "memory": 2, "risk": 1},
        num_layers=3,
        top_k=4,
        representation_dim=32,
        local_query_dim=32,
        reconstruction_dim=32,
        image_head_specs={"shape": 5, "color": 4},
        text_vocab_size=len(vocab),
        text_feature_dim=32,
        enable_text_generation_head=True,
        enable_action_head=False,
        enable_patch_rank_head=False,
        enable_prototype_novelty=False,
    ).to(device)
    optimizer = AdamW(
        list(image_backbone.parameters()) + list(text_backbone.parameters()) + list(model.parameters()),
        lr=args.lr,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_injection_diff_norm = 0.0
    text_injection_diff_norm = 0.0

    for step in range(1, args.steps + 1):
        warmup_eta = min(1.0, step / max(1, args.steps // 4))
        image_batch = move_batch(next(image_iter), device)
        text_batch = move_batch(next(text_iter), device)

        image_backbone_out = image_backbone(image_batch["image"], warmup_eta=warmup_eta)
        image_out = model(image_backbone_outputs=image_backbone_out, warmup_eta=warmup_eta)
        if step == 1:
            with torch.no_grad():
                image_without_injection = model(
                    image_backbone_outputs=image_backbone_out,
                    warmup_eta=warmup_eta,
                    inject_modality_slots=False,
                )
                image_injection_diff_norm = float(
                    (image_out["representation"].detach() - image_without_injection["representation"]).norm(dim=-1).mean().item()
                )
        shape_logits = image_out["image_heads"]["shape"]["logits"]
        color_logits = image_out["image_heads"]["color"]["logits"]
        image_loss = F.cross_entropy(shape_logits, image_batch["shape_label"]) + F.cross_entropy(
            color_logits, image_batch["color_label"]
        )
        image_acc = (shape_logits.argmax(dim=-1) == image_batch["shape_label"]).float().mean().item()

        text_backbone_out = text_backbone(
            input_ids=text_batch["input_ids"],
            padding_mask=text_batch["input_mask"],
            warmup_eta=warmup_eta,
        )
        text_out = model(
            text_backbone_outputs=text_backbone_out,
            warmup_eta=warmup_eta,
        )
        if step == 1:
            with torch.no_grad():
                text_without_injection = model(
                    text_backbone_outputs=text_backbone_out,
                    warmup_eta=warmup_eta,
                    inject_modality_slots=False,
                )
                text_injection_diff_norm = float(
                    (text_out["representation"].detach() - text_without_injection["representation"]).norm(dim=-1).mean().item()
                )
        text_logits = text_out["text_generation"]["logits"]
        text_loss = F.cross_entropy(
            text_logits.view(-1, text_logits.size(-1)),
            text_batch["target_ids"].view(-1),
            ignore_index=vocab.pad_id,
        )
        text_acc = masked_char_accuracy(text_logits, text_batch["target_ids"], text_batch["input_mask"])

        total_loss = image_loss + text_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if step % args.log_interval == 0 or step == args.steps:
            stats = merge_stats(image_out["diagnostics"]["stats"], text_out["diagnostics"]["stats"])
            image_modality = image_out.get("representation_modality", {})
            text_modality = text_out.get("representation_modality", {})
            print(
                f"step={step} "
                f"image_loss={image_loss.item():.4f} text_loss={text_loss.item():.4f} total_loss={total_loss.item():.4f} "
                f"image_accuracy={image_acc:.4f} text_char_accuracy={text_acc:.4f} "
                f"drive_mean={stats.get('drive_mean', 0.0):.4f} drive_std={stats.get('drive_std', 0.0):.4f} "
                f"resistance_mean={stats.get('resistance_mean', 0.0):.4f} resistance_std={stats.get('resistance_std', 0.0):.4f} "
                f"energy_mean={stats.get('energy_mean', 0.0):.4f} energy_std={stats.get('energy_std', 0.0):.4f} "
                f"gate_activation_rate={stats.get('gate_activation_rate', 0.0):.4f} "
                f"active_route_strength_mean={stats.get('active_route_strength_mean', 0.0):.4f} "
                f"graph_gate_mass_mean={stats.get('gate_mass_mean', 0.0):.4f} "
                f"image_slot_readout_weight={scalar(image_modality.get('image_slot_readout_weight_mean')):.4f} "
                f"text_slot_readout_weight={scalar(text_modality.get('text_slot_readout_weight_mean')):.4f} "
                f"image_injection_diff_norm={image_injection_diff_norm:.4f} "
                f"text_injection_diff_norm={text_injection_diff_norm:.4f}"
            )

    torch.save(
        {
            "image_backbone": image_backbone.state_dict(),
            "text_backbone": text_backbone.state_dict(),
            "model": model.state_dict(),
            "config": vars(args),
        },
        output_dir / "last.pt",
    )


if __name__ == "__main__":
    main()
