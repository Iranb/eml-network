from __future__ import annotations

import math
from typing import Dict

import torch


def classification_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    predictions = logits.argmax(dim=-1)
    return float((predictions == targets).float().mean().detach().cpu().item())


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    if logits.numel() == 0:
        return 0.0
    k = min(k, logits.size(-1))
    topk = logits.topk(k=k, dim=-1).indices
    return float((topk == targets.unsqueeze(-1)).any(dim=-1).float().mean().detach().cpu().item())


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    predictions = logits.argmax(dim=-1)
    valid = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask.bool()
    if valid.sum().item() == 0:
        return 0.0
    return float((predictions[valid] == targets[valid]).float().mean().detach().cpu().item())


def perplexity(loss: float) -> float:
    if not math.isfinite(loss):
        return float("nan")
    return float(math.exp(min(loss, 50.0)))


def bits_per_token(loss: float) -> float:
    if not math.isfinite(loss):
        return float("nan")
    return float(loss / math.log(2.0))


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    if x.numel() != y.numel() or x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) <= 1.0e-12:
        return float("nan")
    return float((x @ y / denom).cpu().item())


def finite_summary(values: Dict[str, torch.Tensor]) -> Dict[str, float]:
    total = 0
    bad = 0
    for value in values.values():
        if not torch.is_tensor(value):
            continue
        total += value.numel()
        bad += int((~torch.isfinite(value)).sum().detach().cpu().item())
    return {"finite_total": float(total), "nan_inf_count": float(bad)}


__all__ = [
    "bits_per_token",
    "classification_accuracy",
    "finite_summary",
    "pearson_corr",
    "perplexity",
    "token_accuracy",
    "topk_accuracy",
]
