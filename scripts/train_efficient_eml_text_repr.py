from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eml_mnist import CharVocabulary, EfficientEMLTextEncoder, EfficientEMLTextGenerationHead, SyntheticTextEnergyDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train efficient EML text representation smoke model")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--vocab-type", type=str, default="char")
    parser.add_argument("--task-type", type=str, default="mixed")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _mean_diag(outputs: Dict[str, Any], key: str) -> float:
    diagnostics = outputs.get("diagnostics", {})
    value = diagnostics.get(key)
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu())
    return 0.0


def main() -> None:
    args = build_parser().parse_args()
    del args.vocab_type
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    vocab = CharVocabulary()
    dataset = SyntheticTextEnergyDataset(
        size=max(args.steps * args.batch_size, 128),
        seq_len=args.seq_len,
        vocab=vocab,
        seed=args.seed,
        task_type=args.task_type,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    batches = itertools.cycle(loader)
    encoder = EfficientEMLTextEncoder(
        vocab_size=len(vocab),
        embed_dim=32,
        state_dim=32,
        hidden_dim=64,
        num_hypotheses=4,
        num_attractors=4,
        representation_dim=32,
        causal_window_size=8,
        chunk_size=4,
        pad_id=vocab.pad_id,
    ).to(device)
    head = EfficientEMLTextGenerationHead(state_dim=32, vocab_size=len(vocab), hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=args.lr)
    start = time.time()

    for step in range(1, args.steps + 1):
        batch = next(batches)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        mask = batch["input_mask"].to(device)
        warmup_eta = min(1.0, step / max(1, args.steps // 2))
        encoded = encoder(input_ids, padding_mask=mask, warmup_eta=warmup_eta)
        outputs = head(encoded["sequence_states"], padding_mask=mask, warmup_eta=warmup_eta)
        token_loss = F.cross_entropy(outputs["logits"].reshape(-1, len(vocab)), target_ids.reshape(-1), ignore_index=vocab.pad_id)
        corruption_target = batch["resistance_target"].to(device)
        resistance_supervision = F.mse_loss(outputs["resistance"].mean(dim=-1), corruption_target)
        loss = token_loss + 0.01 * resistance_supervision
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.0)
        optimizer.step()

        if step == 1 or step == args.steps or step % max(1, args.steps // 5) == 0:
            elapsed = max(1.0e-6, time.time() - start)
            predictions = outputs["logits"].argmax(dim=-1)
            correct = ((predictions == target_ids) & mask).float().sum()
            total = mask.float().sum().clamp_min(1.0)
            accuracy = (correct / total).item()
            tokens_per_sec = step * args.batch_size * args.seq_len / elapsed
            print(
                f"step={step} loss={loss.item():.4f} token_acc={accuracy:.4f} "
                f"drive={outputs['drive'].mean().item():.4f}/{outputs['drive'].std(unbiased=False).item():.4f} "
                f"resistance={outputs['resistance'].mean().item():.4f}/{outputs['resistance'].std(unbiased=False).item():.4f} "
                f"energy={outputs['energy'].mean().item():.4f}/{outputs['energy'].std(unbiased=False).item():.4f} "
                f"null={_mean_diag(encoded, 'null_weight_mean'):.4f} "
                f"update={_mean_diag(encoded, 'update_strength_mean'):.4f} "
                f"entropy={_mean_diag(encoded, 'responsibility_entropy_mean'):.4f} "
                f"attractor={encoded['attractor_activation'].mean().item():.4f} "
                f"tokens_per_sec={tokens_per_sec:.1f}"
            )


if __name__ == "__main__":
    main()
