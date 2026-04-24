from __future__ import annotations

import torch

from eml_mnist import EMLPrecisionUpdate


def test_eml_precision_update_gate_in_bounds() -> None:
    update = EMLPrecisionUpdate()
    state = torch.randn(2, 5, 8)
    candidate = torch.randn(2, 5, 8)
    new_energy = torch.randn(2, 5, 1)
    old_confidence = torch.randn(2, 5, 1)

    out = update(state, candidate, new_energy, old_confidence)

    assert out["state"].shape == state.shape
    assert out["updated_state"].shape == state.shape
    assert torch.isfinite(out["state"]).all()
    assert (out["update_gate"] >= 0.0).all()
    assert (out["update_gate"] <= 1.0).all()
    assert torch.isfinite(out["new_precision"]).all()
    assert torch.isfinite(out["old_precision"]).all()


def test_eml_precision_update_sigmoid_mode_compatibility() -> None:
    update = EMLPrecisionUpdate(mode="sigmoid")
    state = torch.zeros(1, 3, 4)
    candidate = torch.ones(1, 3, 4)
    new_energy = torch.full((1, 3, 1), -20.0)

    out = update(state, candidate, new_energy)

    assert torch.isfinite(out["state"]).all()
    assert out["update_gate"].max().item() < 1.0e-3


def test_eml_precision_update_preserves_state_when_new_energy_is_low() -> None:
    update = EMLPrecisionUpdate()
    state = torch.zeros(1, 2, 4)
    candidate = torch.ones(1, 2, 4)
    new_energy = torch.full((1, 2, 1), -50.0)
    old_confidence = torch.full((1, 2, 1), 50.0)

    out = update(state, candidate, new_energy, old_confidence)

    assert out["update_gate"].max().item() < 1.0e-4
    assert torch.allclose(out["updated_state"], state, atol=1.0e-3)


def test_eml_precision_update_changes_state_when_new_energy_is_high() -> None:
    update = EMLPrecisionUpdate()
    state = torch.zeros(1, 2, 4)
    candidate = torch.ones(1, 2, 4)
    new_energy = torch.full((1, 2, 1), 50.0)
    old_confidence = torch.full((1, 2, 1), -50.0)

    out = update(state, candidate, new_energy, old_confidence)

    assert out["update_gate"].min().item() > 0.99
    assert torch.allclose(out["updated_state"], candidate, atol=1.0e-3)


def test_precision_update_identity_init_gate_is_small() -> None:
    update = EMLPrecisionUpdate(old_confidence_init=5.0)
    state = torch.zeros(2, 3, 4)
    candidate = torch.ones_like(state)
    new_energy = torch.zeros(2, 3, 1)

    out = update(state, candidate, new_energy)

    gate_mean = out["update_gate"].mean().item()
    assert 0.05 <= gate_mean <= 0.15
    assert torch.allclose(out["updated_state"], state, atol=0.35)
