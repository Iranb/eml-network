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

from eml_mnist import EMLFoundationCore, SyntheticShapeEnergyDataset, SyntheticTextEnergyDataset
from eml_mnist.text_codecs import CharVocabulary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train efficient EML foundation representation smoke model")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    return parser


def _mean_tensor(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu())
    return 0.0


def _graph_stat(outputs: Dict[str, Any], key: str) -> float:
    graph = outputs.get("graph_layers", [])
    values = []
    for layer in graph:
        message = layer.get("message_passing", {})
        value = message.get(key)
        if torch.is_tensor(value):
            values.append(value.detach().float().reshape(-1))
        route = layer.get("active_route_strength") if key == "active_route_strength" else None
        if torch.is_tensor(route):
            values.append(route.detach().float().reshape(-1))
    if not values:
        return 0.0
    return float(torch.cat(values).mean().cpu())


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    vocab = CharVocabulary()
    image_dataset = SyntheticShapeEnergyDataset(size=max(args.steps * args.batch_size, 128), image_size=args.image_size, seed=args.seed)
    text_dataset = SyntheticTextEnergyDataset(size=max(args.steps * args.batch_size, 128), seq_len=args.seq_len, vocab=vocab, seed=args.seed + 100)
    image_loader = itertools.cycle(DataLoader(image_dataset, batch_size=args.batch_size, shuffle=True))
    text_loader = itertools.cycle(DataLoader(text_dataset, batch_size=args.batch_size, shuffle=True))
    model = EMLFoundationCore(
        slot_dim=32,
        event_dim=32,
        hidden_dim=64,
        slot_layout={"goal": 1, "image": 4, "text": 4, "memory": 2},
        num_layers=1,
        top_k=4,
        representation_dim=32,
        local_query_dim=32,
        reconstruction_dim=32,
        image_input_channels=3,
        image_head_specs={"shape": 5},
        text_vocab_size=len(vocab),
        text_embed_dim=32,
        text_feature_dim=32,
        text_hidden_dim=64,
        enable_text_generation_head=True,
        enable_action_head=False,
        enable_patch_rank_head=False,
        enable_efficient_image_encoder=True,
        enable_efficient_text_encoder=True,
        image_repr_config={
            "state_dim": 32,
            "hidden_dim": 64,
            "num_hypotheses": 4,
            "num_attractors": 4,
            "representation_dim": 32,
            "patch_stride": 4,
        },
        text_repr_config={
            "embed_dim": 32,
            "state_dim": 32,
            "hidden_dim": 64,
            "num_hypotheses": 4,
            "num_attractors": 4,
            "representation_dim": 32,
            "causal_window_size": 8,
            "chunk_size": 4,
            "pad_id": vocab.pad_id,
        },
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = time.time()

    for step in range(1, args.steps + 1):
        image_batch = next(image_loader)
        text_batch = next(text_loader)
        images = image_batch["image"].to(device)
        labels = image_batch["label"].to(device)
        input_ids = text_batch["input_ids"].to(device)
        target_ids = text_batch["target_ids"].to(device)
        mask = text_batch["input_mask"].to(device)
        warmup_eta = min(1.0, step / max(1, args.steps // 2))

        image_out = model(images=images, use_efficient_repr_path=True, warmup_eta=warmup_eta)
        text_out = model(text_tokens=input_ids, text_padding_mask=mask, use_efficient_repr_path=True, warmup_eta=warmup_eta)
        image_logits = image_out["image_heads"]["shape"]["logits"]
        text_logits = text_out["efficient_text_generation"]["logits"]
        image_loss = F.cross_entropy(image_logits, labels)
        text_loss = F.cross_entropy(text_logits.reshape(-1, len(vocab)), target_ids.reshape(-1), ignore_index=vocab.pad_id)
        loss = image_loss + text_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step == args.steps or step % max(1, args.steps // 5) == 0:
            elapsed = max(1.0e-6, time.time() - start)
            image_acc = (image_logits.argmax(dim=-1) == labels).float().mean().item()
            text_pred = text_logits.argmax(dim=-1)
            text_acc = (((text_pred == target_ids) & mask).float().sum() / mask.float().sum().clamp_min(1.0)).item()
            image_injection = image_out["attractor_injection"]["efficient_image"]["injection_norm"]
            text_injection = text_out["attractor_injection"]["efficient_text"]["injection_norm"]
            print(
                f"step={step} total_loss={loss.item():.4f} image_loss={image_loss.item():.4f} text_loss={text_loss.item():.4f} "
                f"image_acc={image_acc:.4f} text_acc={text_acc:.4f} "
                f"injection_norm={(image_injection + text_injection) * 0.5:.4f} "
                f"graph_update_strength={_graph_stat(text_out, 'update_strength'):.4f} "
                f"active_route_strength={_graph_stat(text_out, 'active_route_strength'):.4f} "
                f"step_time={(time.time() - start) / step:.4f}"
            )


if __name__ == "__main__":
    main()
