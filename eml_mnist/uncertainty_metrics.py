from __future__ import annotations

import math
from typing import Dict

import torch

from .metrics import (
    area_under_risk_coverage_curve,
    binary_auroc,
    brier_score,
    classification_accuracy,
    expected_calibration_error,
    negative_log_likelihood,
    pearson_corr,
    selective_risk_at_coverage,
)


def binary_auprc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().float().reshape(-1)
    labels = labels.detach().reshape(-1).to(dtype=torch.long)
    if scores.numel() != labels.numel() or scores.numel() == 0:
        return float("nan")
    positives = labels == 1
    if int(positives.sum().item()) == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_labels = positives[order].float()
    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1.0 - sorted_labels, dim=0)
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / positives.sum().float().clamp_min(1.0)
    recall = torch.cat([recall.new_tensor([0.0]), recall])
    precision = torch.cat([precision.new_tensor([1.0]), precision])
    return float(torch.trapz(precision, recall).detach().cpu().item())


def risk_coverage_curve(
    correct: torch.Tensor,
    risk_score: torch.Tensor,
    coverages: tuple[float, ...] = (0.50, 0.70, 0.80, 0.90, 0.95, 1.00),
) -> Dict[str, float]:
    acceptance = -risk_score.detach().float().reshape(-1)
    return {
        f"selective_risk_at_{int(round(coverage * 100)):02d}_coverage": selective_risk_at_coverage(
            correct, acceptance, coverage
        )
        for coverage in coverages
    }


def aurc(correct: torch.Tensor, risk_score: torch.Tensor, steps: int = 50) -> float:
    acceptance = -risk_score.detach().float().reshape(-1)
    return area_under_risk_coverage_curve(correct, acceptance, steps=steps)


def excess_aurc(correct: torch.Tensor, risk_score: torch.Tensor, steps: int = 50) -> float:
    observed = aurc(correct, risk_score, steps=steps)
    if not math.isfinite(observed):
        return float("nan")
    errors = 1.0 - correct.detach().float().reshape(-1)
    base_risk = float(errors.mean().cpu().item()) if errors.numel() else float("nan")
    if not math.isfinite(base_risk):
        return float("nan")
    return max(0.0, observed - base_risk * 0.5)


def margin_statistics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    if logits.ndim != 2 or targets.numel() == 0:
        return {
            "positive_logit_mean": float("nan"),
            "hard_negative_logit_mean": float("nan"),
            "margin_mean": float("nan"),
        }
    targets = targets.long().reshape(-1)
    positive = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    mask = torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).bool()
    hard_negative = logits.masked_fill(mask, float("-inf")).max(dim=1).values
    margin = positive - hard_negative
    return {
        "positive_logit_mean": float(positive.mean().detach().cpu().item()),
        "hard_negative_logit_mean": float(hard_negative.mean().detach().cpu().item()),
        "margin_mean": float(margin.mean().detach().cpu().item()),
    }


def classification_uncertainty_summary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    risk_score: torch.Tensor | None = None,
    corruption_labels: torch.Tensor | None = None,
) -> Dict[str, float]:
    probs = torch.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values
    risk = (1.0 - confidence) if risk_score is None else risk_score.detach().float().reshape(-1)
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(targets.long().reshape(-1))
    metrics: Dict[str, float] = {
        "accuracy": classification_accuracy(logits, targets.long()),
        "nll": negative_log_likelihood(logits, targets.long()),
        "ece": expected_calibration_error(logits, targets.long()),
        "brier": brier_score(logits, targets.long()),
        "aurc": aurc(correct, risk),
        "e_aurc": excess_aurc(correct, risk),
    }
    metrics.update(risk_coverage_curve(correct, risk))
    metrics.update(margin_statistics(logits, targets.long()))
    if corruption_labels is not None:
        labels = corruption_labels.long().reshape(-1)
        metrics["corruption_auroc"] = binary_auroc(risk, labels)
        metrics["corruption_auprc"] = binary_auprc(risk, labels)
    return metrics


def correlation_metrics(
    risk_or_resistance: torch.Tensor,
    *,
    noise_level: torch.Tensor | None = None,
    occlusion_level: torch.Tensor | None = None,
    severity: torch.Tensor | None = None,
    uncertainty: torch.Tensor | None = None,
) -> Dict[str, float]:
    values = risk_or_resistance.detach().float().reshape(-1)
    out: Dict[str, float] = {}
    if noise_level is not None:
        out["resistance_noise_corr"] = pearson_corr(values, noise_level)
    if occlusion_level is not None:
        out["resistance_occlusion_corr"] = pearson_corr(values, occlusion_level)
    if severity is not None:
        out["resistance_severity_corr"] = pearson_corr(values, severity)
    if uncertainty is not None and severity is not None:
        out["uncertainty_severity_corr"] = pearson_corr(uncertainty, severity)
    return out


__all__ = [
    "aurc",
    "binary_auprc",
    "binary_auroc",
    "classification_uncertainty_summary",
    "correlation_metrics",
    "excess_aurc",
    "margin_statistics",
    "risk_coverage_curve",
]
