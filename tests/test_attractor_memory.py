from __future__ import annotations

from typing import Any

import torch

from eml_mnist.field import EMLAttractorMemory, EMLFieldReadout


def _assert_finite(value: Any) -> None:
    if torch.is_tensor(value):
        if value.dtype.is_floating_point or value.dtype.is_complex:
            assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for child in value.values():
            _assert_finite(child)


def test_eml_attractor_memory_output_shape() -> None:
    memory = EMLAttractorMemory(field_dim=8, num_attractors=4, hidden_dim=16)
    hypothesis_state = torch.randn(2, 5, 3, 8)
    activation = torch.sigmoid(torch.randn(2, 5, 3))

    out = memory(hypothesis_state, activation=activation, warmup_eta=0.5)

    assert out["attractor_states"].shape == (2, 4, 8)
    assert out["attractor_drive"].shape == (2, 4)
    assert out["attractor_resistance"].shape == (2, 4)
    assert out["attractor_energy"].shape == (2, 4)
    assert out["attractor_activation"].shape == (2, 4)
    assert out["update_gate"].shape == (2, 4, 8)
    _assert_finite(out)


def test_eml_field_readout_output_shape() -> None:
    memory = EMLAttractorMemory(field_dim=8, num_attractors=4, hidden_dim=16)
    readout = EMLFieldReadout(field_dim=8, hidden_dim=16, representation_dim=6)
    hypothesis_state = torch.randn(2, 5, 3, 8)
    activation = torch.sigmoid(torch.randn(2, 5, 3))
    memory_out = memory(hypothesis_state, activation=activation, warmup_eta=0.5)

    out = readout(
        memory_out["attractor_states"],
        attractor_activation=memory_out["attractor_activation"],
        warmup_eta=0.5,
    )

    assert out["representation"].shape == (2, 6)
    assert out["weights"].shape == (2, 4)
    assert out["drive"].shape == (2, 4)
    assert out["resistance"].shape == (2, 4)
    assert out["energy"].shape == (2, 4)
    assert torch.allclose(out["weights"].sum(dim=1), torch.ones(2), atol=1.0e-5)
    _assert_finite(out)


def test_attractor_memory_no_nan_inf_with_flat_parent_states() -> None:
    memory = EMLAttractorMemory(field_dim=8, num_attractors=3, hidden_dim=16)
    parent_state = torch.randn(2, 4, 8)
    parent_activation = torch.sigmoid(torch.randn(2, 4))

    out = memory(parent_state, activation=parent_activation)

    assert out["attractor_states"].shape == (2, 3, 8)
    _assert_finite(out)
