from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


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


def negative_log_likelihood(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    return float(F.cross_entropy(logits, targets).detach().cpu().item())


def brier_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    probs = torch.softmax(logits, dim=-1)
    one_hot = F.one_hot(targets, num_classes=logits.size(-1)).to(dtype=probs.dtype)
    return float((probs - one_hot).square().sum(dim=-1).mean().detach().cpu().item())


def expected_calibration_error(logits: torch.Tensor, targets: torch.Tensor, num_bins: int = 15) -> float:
    if logits.numel() == 0:
        return 0.0
    probs = torch.softmax(logits, dim=-1)
    confidence, predictions = probs.max(dim=-1)
    correct = predictions.eq(targets).to(dtype=probs.dtype)
    ece = logits.new_tensor(0.0)
    boundaries = torch.linspace(0.0, 1.0, num_bins + 1, device=logits.device, dtype=probs.dtype)
    for index in range(num_bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        if index == 0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            bin_confidence = confidence[mask].mean()
            bin_accuracy = correct[mask].mean()
            ece = ece + mask.to(dtype=probs.dtype).mean() * (bin_confidence - bin_accuracy).abs()
    return float(ece.detach().cpu().item())


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
    "brier_score",
    "bits_per_token",
    "classification_accuracy",
    "expected_calibration_error",
    "finite_summary",
    "negative_log_likelihood",
    "pearson_corr",
    "perplexity",
    "token_accuracy",
    "topk_accuracy",
]
