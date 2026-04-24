from __future__ import annotations

from typing import Any

import torch

from eml_mnist.field import (
    EMLCompositionField,
    EMLConsensusField,
    EMLHypothesisCompetition,
    EMLHypothesisField,
    EMLSensor,
)


def _assert_finite(value: Any) -> None:
    if torch.is_tensor(value):
        if value.dtype.is_floating_point or value.dtype.is_complex:
            assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for child in value.values():
            _assert_finite(child)


def _hypothesis_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    field = EMLHypothesisField(measurement_dim=6, field_dim=8, num_hypotheses=3, hidden_dim=16)
    measurements = torch.randn(2, 16, 6)
    out = field(measurements, warmup_eta=0.5)
    return out["hypothesis_state"], out["activation"], out["drive"], out["resistance"]


def test_eml_sensor_output_shapes() -> None:
    sensor = EMLSensor(input_dim=5, measurement_dim=7, seed_dim=4, hidden_dim=12, position_dim=2)
    x = torch.randn(2, 6, 5)
    position_features = torch.randn(6, 2)

    out = sensor(x, position_features=position_features)

    assert out["measurement"].shape == (2, 6, 7)
    assert out["drive_seed"].shape == (2, 6, 4)
    assert out["resistance_seed"].shape == (2, 6, 4)
    assert out["position_features"].shape == (2, 6, 2)
    _assert_finite(out)


def test_eml_hypothesis_field_output_shapes_and_finite_values() -> None:
    field = EMLHypothesisField(measurement_dim=6, field_dim=8, num_hypotheses=3, hidden_dim=16)
    measurements = torch.randn(2, 5, 6)

    out = field(measurements, warmup_eta=0.4)

    assert out["hypothesis_state"].shape == (2, 5, 3, 8)
    assert out["drive"].shape == (2, 5, 3)
    assert out["resistance"].shape == (2, 5, 3)
    assert out["energy"].shape == (2, 5, 3)
    assert out["activation"].shape == (2, 5, 3)
    _assert_finite(out)


def test_eml_hypothesis_competition_finite_activations() -> None:
    competition = EMLHypothesisCompetition(temperature=0.75, competition_strength=0.2, top_k=2)
    energy = torch.randn(2, 5, 4)
    resistance = torch.rand(2, 5, 4)

    out = competition(energy=energy, resistance=resistance)

    assert out["activation"].shape == (2, 5, 4)
    assert out["topk_mask"].shape == (2, 5, 4)
    _assert_finite(out)


def test_eml_consensus_field_image_mode_shape_and_finite_values() -> None:
    hypothesis_state, activation, drive, resistance = _hypothesis_inputs()
    consensus = EMLConsensusField(field_dim=8, hidden_dim=16, num_hypotheses=3, mode="image", window_size=3)

    out = consensus(
        hypothesis_state=hypothesis_state,
        activation=activation,
        drive=drive,
        resistance=resistance,
        image_shape=(4, 4),
        warmup_eta=0.5,
    )

    assert out["support"].shape == (2, 16, 3)
    assert out["conflict"].shape == (2, 16, 3)
    assert out["energy"].shape == (2, 16, 3)
    assert out["activation"].shape == (2, 16, 3)
    assert out["gate_mass"].shape == (2, 16, 3)
    _assert_finite(out)


def test_eml_consensus_field_text_causal_mode_shape_and_finite_values() -> None:
    hypothesis_state = torch.randn(2, 6, 3, 8)
    activation = torch.sigmoid(torch.randn(2, 6, 3))
    consensus = EMLConsensusField(field_dim=8, hidden_dim=16, num_hypotheses=3, mode="text", window_size=3)

    out = consensus(hypothesis_state=hypothesis_state, activation=activation, warmup_eta=0.5)

    assert out["support"].shape == (2, 6, 3)
    assert out["conflict"].shape == (2, 6, 3)
    assert out["energy"].shape == (2, 6, 3)
    assert out["activation"].shape == (2, 6, 3)
    _assert_finite(out)


def test_eml_consensus_field_text_mode_does_not_use_future_positions() -> None:
    torch.manual_seed(3)
    hypothesis_state = torch.randn(1, 6, 2, 5)
    activation = torch.sigmoid(torch.randn(1, 6, 2))
    consensus = EMLConsensusField(field_dim=5, hidden_dim=10, num_hypotheses=2, mode="text", window_size=3)

    baseline = consensus(hypothesis_state=hypothesis_state, activation=activation)
    changed = hypothesis_state.clone()
    changed[:, 5] = changed[:, 5] + 100.0
    changed_out = consensus(hypothesis_state=changed, activation=activation)

    assert torch.allclose(baseline["support"][:, :3], changed_out["support"][:, :3], atol=1.0e-6)
    assert torch.allclose(baseline["conflict"][:, :3], changed_out["conflict"][:, :3], atol=1.0e-6)


def test_eml_composition_field_output_shape() -> None:
    hypothesis_state, activation, _, _ = _hypothesis_inputs()
    composition = EMLCompositionField(
        field_dim=8,
        hidden_dim=16,
        mode="image",
        region_size=2,
        num_parent_hypotheses=2,
    )

    out = composition(hypothesis_state=hypothesis_state, activation=activation, image_shape=(4, 4), warmup_eta=0.5)

    assert out["parent_state"].shape == (2, 4, 2, 8)
    assert out["parent_drive"].shape == (2, 4, 2)
    assert out["parent_resistance"].shape == (2, 4, 2)
    assert out["parent_energy"].shape == (2, 4, 2)
    assert out["parent_activation"].shape == (2, 4, 2)
    _assert_finite(out)


def test_field_modules_no_nan_inf() -> None:
    hypothesis_state, activation, drive, resistance = _hypothesis_inputs()
    consensus = EMLConsensusField(field_dim=8, hidden_dim=16, num_hypotheses=3, mode="image", window_size=3)
    composition = EMLCompositionField(field_dim=8, hidden_dim=16, mode="image", region_size=2)

    consensus_out = consensus(hypothesis_state, activation, drive=drive, resistance=resistance, image_shape=(4, 4))
    composition_out = composition(hypothesis_state, activation, image_shape=(4, 4))

    _assert_finite(consensus_out)
    _assert_finite(composition_out)
