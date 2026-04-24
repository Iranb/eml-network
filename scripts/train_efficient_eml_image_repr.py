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

from eml_mnist import EfficientEMLImageClassifier, SyntheticShapeEnergyDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train efficient EML image representation smoke model")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=5)
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
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dataset = SyntheticShapeEnergyDataset(size=max(args.steps * args.batch_size, 128), image_size=args.image_size, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    batches = itertools.cycle(loader)
    model = EfficientEMLImageClassifier(
        num_classes=args.num_classes,
        state_dim=32,
        hidden_dim=64,
        num_hypotheses=4,
        num_attractors=4,
        representation_dim=32,
        patch_stride=4,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = time.time()

    for step in range(1, args.steps + 1):
        batch = next(batches)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        warmup_eta = min(1.0, step / max(1, args.steps // 2))
        outputs = model(images, warmup_eta=warmup_eta)
        ce = F.cross_entropy(outputs["logits"], labels)
        energy_penalty = F.relu(outputs["energy"].abs() - 8.0).square().mean()
        loss = ce + 1.0e-4 * energy_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step == args.steps or step % max(1, args.steps // 5) == 0:
            elapsed = max(1.0e-6, time.time() - start)
            accuracy = (outputs["logits"].argmax(dim=-1) == labels).float().mean().item()
            examples_per_sec = step * args.batch_size / elapsed
            print(
                f"step={step} loss={loss.item():.4f} acc={accuracy:.4f} "
                f"drive={outputs['drive'].mean().item():.4f}/{outputs['drive'].std(unbiased=False).item():.4f} "
                f"resistance={outputs['resistance'].mean().item():.4f}/{outputs['resistance'].std(unbiased=False).item():.4f} "
                f"energy={outputs['energy'].mean().item():.4f}/{outputs['energy'].std(unbiased=False).item():.4f} "
                f"null={_mean_diag(outputs, 'null_weight_mean'):.4f} "
                f"update={_mean_diag(outputs, 'update_strength_mean'):.4f} "
                f"entropy={_mean_diag(outputs, 'responsibility_entropy_mean'):.4f} "
                f"diversity={_mean_diag(outputs, 'attractor_diversity_penalty'):.4f} "
                f"message_norm={_mean_diag(outputs, 'message_norm_mean'):.4f} "
                f"update_norm={_mean_diag(outputs, 'update_norm_mean'):.4f} "
                f"examples_per_sec={examples_per_sec:.1f}"
            )


if __name__ == "__main__":
    main()
