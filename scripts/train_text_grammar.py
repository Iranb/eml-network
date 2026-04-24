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

from eml_mnist import CharVocabulary, EMLFoundationCore, EMLTextBackbone, SyntheticGrammarDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an EML text model on SyntheticGrammarDataset")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="./outputs/text_grammar")
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

    vocab = CharVocabulary()
    dataset = SyntheticGrammarDataset(
        size=args.train_size,
        vocab=vocab,
        max_length=args.max_length,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    iterator = cycle(loader)

    text_backbone = EMLTextBackbone(
        vocab_size=len(vocab),
        embed_dim=24,
        feature_dim=24,
        event_dim=24,
        hidden_dim=48,
        bank_dim=48,
        num_layers=2,
        pad_id=vocab.pad_id,
        dropout=0.0,
    ).to(device)
    model = EMLFoundationCore(
        slot_dim=24,
        event_dim=24,
        hidden_dim=48,
        slot_layout={"goal": 1, "text": 3, "memory": 2, "risk": 1},
        num_layers=2,
        top_k=3,
        representation_dim=24,
        local_query_dim=24,
        reconstruction_dim=24,
        enable_text_generation_head=True,
        text_vocab_size=len(vocab),
        text_feature_dim=24,
        enable_action_head=False,
        enable_patch_rank_head=False,
    ).to(device)
    optimizer = AdamW(list(text_backbone.parameters()) + list(model.parameters()), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        batch = move_batch(next(iterator), device)
        warmup_eta = min(1.0, step / max(1, args.steps // 4))
        text_backbone_out = text_backbone(
            input_ids=batch["input_ids"],
            padding_mask=batch["input_mask"],
            warmup_eta=warmup_eta,
        )
        outputs = model(
            text_backbone_outputs=text_backbone_out,
            warmup_eta=warmup_eta,
        )
        logits = outputs["text_generation"]["logits"]
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            batch["target_ids"].view(-1),
            ignore_index=vocab.pad_id,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0 or step == args.steps:
            char_acc = masked_char_accuracy(logits, batch["target_ids"], batch["input_mask"])
            stats = outputs["diagnostics"]["stats"]
            representation_modality = outputs.get("representation_modality", {})
            print(
                f"step={step} "
                f"loss={loss.item():.4f} "
                f"char_accuracy={char_acc:.4f} "
                f"drive_mean={stats.get('drive_mean', 0.0):.4f} drive_std={stats.get('drive_std', 0.0):.4f} "
                f"resistance_mean={stats.get('resistance_mean', 0.0):.4f} resistance_std={stats.get('resistance_std', 0.0):.4f} "
                f"energy_mean={stats.get('energy_mean', 0.0):.4f} energy_std={stats.get('energy_std', 0.0):.4f} "
                f"gate_activation_rate={stats.get('gate_activation_rate', 0.0):.4f} "
                f"active_route_strength_mean={stats.get('active_route_strength_mean', 0.0):.4f} "
                f"graph_gate_mass_mean={stats.get('gate_mass_mean', 0.0):.4f} "
                f"text_slot_readout_weight={scalar(representation_modality.get('text_slot_readout_weight_mean')):.4f} "
                f"injected_text_slots={outputs['modality_injection']['injected_text_slots']}"
            )

    torch.save(
        {
            "backbone": text_backbone.state_dict(),
            "model": model.state_dict(),
            "config": vars(args),
        },
        output_dir / "last.pt",
    )


if __name__ == "__main__":
    main()
