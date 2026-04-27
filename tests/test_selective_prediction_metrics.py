from __future__ import annotations

import torch

from eml_mnist.uncertainty_metrics import aurc, excess_aurc, risk_coverage_curve


def test_risk_coverage_curve_orders_by_confidence() -> None:
    correct = torch.tensor([True, True, False, False])
    risk = torch.tensor([0.1, 0.2, 0.8, 0.9])
    curve = risk_coverage_curve(correct, risk)

    assert curve["selective_risk_at_50_coverage"] <= curve["selective_risk_at_100_coverage"]
    assert 0.0 <= aurc(correct, risk, steps=8) <= 1.0
    assert excess_aurc(correct, risk, steps=8) >= 0.0
