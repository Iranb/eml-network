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

from eml_mnist import CharVocabulary, EMLTextFieldEncoder, EMLTextFieldGenerationHead, SyntheticTextEnergyDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an EML text-field generator on SyntheticTextEnergyDataset")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument(
        "--task-type",
        type=str,
        default="mixed",
        choices=["mixed", "brackets", "repeat", "copy", "reverse", "kv", "dsl"],
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="./outputs/eml_text_field")
    parser.add_argument("--activation-budget-weight", type=float, default=0.02)
    parser.add_argument("--energy-penalty-weight", type=float, default=0.01)
    parser.add_argument("--energy-margin", type=float, default=4.0)
    parser.add_argument("--resistance-supervision-weight", type=float, default=0.05)
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


def masked_corrcoef(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if mask.sum() < 2:
        return x.new_zeros(())
    x = x.detach().to(dtype=torch.float32)[mask]
    y = y.detach().to(dtype=torch.float32)[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = x.std(unbiased=False) * y.std(unbiased=False)
    if float(denom) < 1.0e-6:
        return x.new_zeros(())
    return (x * y).mean() / denom


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


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    vocab = CharVocabulary()
    dataset = SyntheticTextEnergyDataset(
        size=args.train_size,
        seq_len=args.seq_len,
        vocab=vocab,
        seed=args.seed,
        task_type=args.task_type,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    iterator = cycle(loader)

    encoder = EMLTextFieldEncoder(
        vocab_size=len(vocab),
        embed_dim=24,
        sensor_dim=24,
        measurement_dim=24,
        field_dim=24,
        hidden_dim=48,
        num_hypotheses=4,
        num_chunk_hypotheses=4,
        num_attractors=4,
        representation_dim=24,
        pad_id=vocab.pad_id,
        causal_window_size=5,
        chunk_size=4,
        chunk_window_size=3,
    ).to(device)
    head = EMLTextFieldGenerationHead(
        state_dim=24,
        vocab_size=len(vocab),
        hidden_dim=48,
        prototype_temperature=0.5,
    ).to(device)
    optimizer = AdamW(list(encoder.parameters()) + list(head.parameters()), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        batch = move_batch(next(iterator), device)
        warmup_eta = min(1.0, step / max(1, args.steps // 4))
        encoder_out = encoder(batch["input_ids"], padding_mask=batch["input_mask"], warmup_eta=warmup_eta)
        head_out = head(encoder_out["sequence_states"], padding_mask=batch["input_mask"], warmup_eta=warmup_eta)

        logits = head_out["logits"]
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch["target_ids"].reshape(-1),
            ignore_index=vocab.pad_id,
        )
        budget_loss = encoder_out["diagnostics"]["budget_loss"]
        energy_penalty = F.relu(logits.abs() - args.energy_margin).square().mean()
        sequence_resistance = encoder_out["local_hypotheses"]["resistance"].mean(dim=-1)
        resistance_supervision = masked_mse(sequence_resistance, batch["resistance_target"], batch["input_mask"])
        diversity_loss = attractor_diversity_loss(encoder_out["attractor_states"])

        total_loss = token_loss
        total_loss = total_loss + args.activation_budget_weight * budget_loss
        total_loss = total_loss + args.energy_penalty_weight * energy_penalty
        total_loss = total_loss + args.resistance_supervision_weight * resistance_supervision
        total_loss = total_loss + args.attractor_diversity_weight * diversity_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if step % args.log_interval == 0 or step == args.steps:
            valid = batch["target_ids"] != vocab.pad_id
            accuracy = ((logits.argmax(dim=-1) == batch["target_ids"]) & valid).to(dtype=torch.float32).sum()
            accuracy = accuracy / valid.to(dtype=torch.float32).sum().clamp_min(1.0)
            stats = encoder_out["diagnostics"]["stats"]
            resistance_corr = masked_corrcoef(sequence_resistance, batch["resistance_target"], batch["input_mask"])
            print(
                f"step={step} "
                f"next_token_loss={total_loss.item():.4f} "
                f"accuracy={accuracy.item():.4f} "
                f"drive_mean={head_out['drive'].mean().item():.4f} drive_std={head_out['drive'].std(unbiased=False).item():.4f} "
                f"resistance_mean={head_out['resistance'].mean().item():.4f} resistance_std={head_out['resistance'].std(unbiased=False).item():.4f} "
                f"energy_mean={head_out['energy'].mean().item():.4f} energy_std={head_out['energy'].std(unbiased=False).item():.4f} "
                f"activation_rate={scalar(stats.get('local_activation_mean')):.4f} "
                f"support_mean={scalar(stats.get('chunk_support_mean')):.4f} "
                f"conflict_mean={scalar(stats.get('chunk_conflict_mean')):.4f} "
                f"chunk_attractor_activation={scalar(stats.get('attractor_activation_mean')):.4f} "
                f"corruption_resistance_corr={resistance_corr.item():.4f}"
            )

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "config": vars(args),
        },
        output_dir / "last.pt",
    )


if __name__ == "__main__":
    main()
