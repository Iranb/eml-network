from __future__ import annotations

import math

import torch

from eml_mnist.uncertainty_metrics import (
    binary_auprc,
    classification_uncertainty_summary,
    correlation_metrics,
    margin_statistics,
)


def test_uncertainty_metric_helpers_are_finite() -> None:
    logits = torch.tensor([[4.0, 0.0], [0.2, 1.0], [2.0, 0.5]])
    labels = torch.tensor([0, 1, 1])
    summary = classification_uncertainty_summary(logits, labels)
    margins = margin_statistics(logits, labels)

    assert 0.0 <= summary["accuracy"] <= 1.0
    assert math.isfinite(summary["nll"])
    assert math.isfinite(summary["ece"])
    assert math.isfinite(summary["aurc"])
    assert math.isfinite(margins["margin_mean"])


def test_auprc_and_correlations_handle_degenerate_inputs() -> None:
    scores = torch.tensor([0.1, 0.3, 0.8, 0.9])
    labels = torch.tensor([0, 0, 1, 1])
    assert binary_auprc(scores, labels) > 0.9

    corr = correlation_metrics(torch.ones(4), noise_level=torch.arange(4.0), occlusion_level=torch.arange(4.0))
    assert math.isnan(corr["resistance_noise_corr"])
