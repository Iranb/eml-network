from __future__ import annotations

import torch

from eml_mnist import EMLResponsibility, EMLScore, EMLUnit


def test_eml_score_has_energy_alias() -> None:
    score = EMLScore(dim=4)
    drive = torch.randn(2, 4)
    resistance = torch.randn(2, 4)

    out = score(drive, resistance, warmup_eta=0.5)

    assert "energy" in out
    assert torch.allclose(out["score"], out["energy"])


def test_eml_unit_finite_gradients_and_fp32_island() -> None:
    unit = EMLUnit(dim=3)
    drive = torch.randn(4, 3, dtype=torch.float16, requires_grad=True)
    resistance = torch.randn(4, 3, dtype=torch.float16, requires_grad=True)

    out = unit.compute(drive, resistance, warmup_eta=1.0)
    loss = out["energy"].float().sum()
    loss.backward()

    assert out["internal_dtype"] == torch.float32
    assert torch.isfinite(out["energy"]).all()
    assert drive.grad is not None and torch.isfinite(drive.grad).all()
    assert resistance.grad is not None and torch.isfinite(resistance.grad).all()


def test_eml_responsibility_weights_are_finite_and_bounded_with_null() -> None:
    responsibility = EMLResponsibility(use_null=True)
    energy = torch.randn(2, 5)
    mask = torch.tensor([[True, True, False, True, False], [False, False, False, False, False]])

    out = responsibility(energy, mask=mask)
    weights = out["neighbor_weights"]

    assert torch.isfinite(weights).all()
    assert torch.isfinite(out["null_weight"]).all()
    assert torch.isfinite(out["logits"]).all()
    assert out["logits"].shape == (2, 6)
    assert (weights.sum(dim=-1) <= 1.0 + 1.0e-6).all()
    assert torch.allclose(weights[1], torch.zeros_like(weights[1]))


def test_null_weight_increases_when_all_energies_are_low() -> None:
    responsibility = EMLResponsibility(use_null=True)
    high = responsibility(torch.full((1, 4), 4.0))["null_weight"]
    low = responsibility(torch.full((1, 4), -12.0))["null_weight"]

    assert low.item() > high.item()
    assert low.item() > 0.9


def test_thresholded_null_rejects_low_evidence() -> None:
    responsibility = EMLResponsibility(mode="thresholded_null", use_null=True, evidence_threshold=0.0)
    out = responsibility(torch.full((2, 5), -4.0))

    assert out["null_weight"].min().item() > 0.7
    assert out["neighbor_weights"].sum(dim=-1).max().item() < 0.3


def test_thresholded_null_selects_high_evidence() -> None:
    responsibility = EMLResponsibility(mode="thresholded_null", use_null=True, evidence_threshold=0.0)
    energy = torch.tensor([[-2.0, -1.0, 5.0, -3.0]])
    out = responsibility(energy)

    assert out["neighbor_weights"][0, 2].item() > 0.7
    assert out["null_weight"].item() < 0.3
