from __future__ import annotations

import torch

from eml_mnist.primitives import EMLResponsibility


def test_thresholded_null_increases_when_evidence_is_weak() -> None:
    responsibility = EMLResponsibility(mode="thresholded_null", evidence_threshold=0.5, temperature=0.25)
    weak = responsibility(torch.full((4, 3), -2.0))
    strong = responsibility(torch.tensor([[-2.0, 3.0, -2.0]]).repeat(4, 1))

    assert torch.all(weak["null_weight"] > 0.7)
    assert torch.all(strong["neighbor_weights"][:, 1] > 0.7)
    assert torch.isfinite(weak["neighbor_weights"]).all()


def test_masked_responsibility_has_zero_masked_weight() -> None:
    responsibility = EMLResponsibility(mode="thresholded_null", evidence_threshold=0.0)
    out = responsibility(torch.tensor([[2.0, 3.0, 4.0]]), mask=torch.tensor([[True, False, True]]))

    assert out["neighbor_weights"][0, 1].item() == 0.0
    assert out["neighbor_weights"].sum().item() <= 1.0
