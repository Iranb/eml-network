from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from eml_mnist import EMLImageFieldClassifier, EMLImageFieldEncoder, SyntheticShapeEnergyDataset


def _assert_finite(value: Any) -> None:
    if torch.is_tensor(value):
        if value.dtype.is_floating_point or value.dtype.is_complex:
            assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for child in value.values():
            _assert_finite(child)


def _sample_batch(batch_size: int = 4, image_size: int = 32, target_type: str = "shape") -> dict[str, torch.Tensor]:
    dataset = SyntheticShapeEnergyDataset(
        size=batch_size,
        image_size=image_size,
        seed=7,
        target_type=target_type,
        include_background_clutter=True,
        include_mask=True,
    )
    batch = [dataset[index] for index in range(batch_size)]
    return {
        key: torch.stack([sample[key] for sample in batch], dim=0) if torch.is_tensor(batch[0][key]) else torch.tensor([sample[key] for sample in batch])
        for key in batch[0]
    }


def test_synthetic_shape_energy_dataset_works_offline() -> None:
    dataset = SyntheticShapeEnergyDataset(size=8, image_size=32, seed=0, target_type="shape", include_mask=True)
    sample = dataset[0]

    assert sample["image"].shape == (3, 32, 32)
    assert sample["mask"].shape == (32, 32)
    assert 0 <= int(sample["label"]) < dataset.num_classes
    assert torch.isfinite(sample["noise_level"])
    assert torch.isfinite(sample["occlusion_level"])
    assert torch.isfinite(sample["resistance_target"])


def test_eml_image_field_encoder_forward_pass_works() -> None:
    batch = _sample_batch(batch_size=3)
    encoder = EMLImageFieldEncoder(
        input_channels=3,
        sensor_dim=24,
        measurement_dim=24,
        field_dim=24,
        hidden_dim=48,
        num_hypotheses=4,
        num_parent_hypotheses=4,
        num_attractors=4,
        representation_dim=24,
    )

    out = encoder(batch["image"], warmup_eta=0.5)

    assert out["representation"].shape == (3, 24)
    assert out["attractor_states"].shape == (3, 4, 24)
    assert out["attractor_activation"].shape == (3, 4)
    assert out["local_hypotheses"]["state"].ndim == 4
    assert out["parent_hypotheses"]["state"].ndim == 4
    _assert_finite(out)


def test_eml_image_field_classifier_forward_pass_works() -> None:
    batch = _sample_batch(batch_size=3)
    classifier = EMLImageFieldClassifier(
        num_classes=5,
        input_channels=3,
        sensor_dim=24,
        measurement_dim=24,
        field_dim=24,
        hidden_dim=48,
        num_hypotheses=4,
        num_parent_hypotheses=4,
        num_attractors=4,
        representation_dim=24,
    )

    out = classifier(batch["image"], warmup_eta=0.5)

    assert out["logits"].shape == (3, 5)
    assert out["probs"].shape == (3, 5)
    assert out["representation"].shape == (3, 24)
    assert out["ambiguity"].shape == (3, 5)
    _assert_finite(out)


def test_output_logits_shape_is_correct() -> None:
    batch = _sample_batch(batch_size=2, target_type="combo")
    classifier = EMLImageFieldClassifier(num_classes=20, representation_dim=24, field_dim=24, measurement_dim=24, sensor_dim=24)

    out = classifier(batch["image"], warmup_eta=0.5)

    assert out["logits"].shape == (2, 20)


def test_attractor_states_are_finite() -> None:
    batch = _sample_batch(batch_size=2)
    encoder = EMLImageFieldEncoder(representation_dim=24, field_dim=24, measurement_dim=24, sensor_dim=24)
    out = encoder(batch["image"], warmup_eta=0.5)

    assert torch.isfinite(out["attractor_states"]).all()
    assert torch.isfinite(out["attractor_activation"]).all()


def test_diagnostics_contain_drive_resistance_energy_activation() -> None:
    batch = _sample_batch(batch_size=2)
    encoder = EMLImageFieldEncoder(representation_dim=24, field_dim=24, measurement_dim=24, sensor_dim=24)
    out = encoder(batch["image"], warmup_eta=0.5)

    diagnostics = out["diagnostics"]
    for section in ("local", "parent", "attractor"):
        assert section in diagnostics
    for key in ("drive", "resistance", "energy", "activation"):
        assert key in diagnostics["local"]
        assert key in diagnostics["parent"]
        assert key in diagnostics["attractor"]


def test_tiny_training_step_produces_finite_loss() -> None:
    batch = _sample_batch(batch_size=4)
    classifier = EMLImageFieldClassifier(
        num_classes=5,
        input_channels=3,
        sensor_dim=24,
        measurement_dim=24,
        field_dim=24,
        hidden_dim=48,
        num_hypotheses=4,
        num_parent_hypotheses=4,
        num_attractors=4,
        representation_dim=24,
    )
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)

    out = classifier(batch["image"], warmup_eta=0.5)
    loss = F.cross_entropy(out["logits"], batch["label"])
    loss = loss + 0.02 * out["diagnostics"]["budget_loss"]
    loss = loss + 0.01 * out["attractor_states"].float().square().mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_no_forbidden_modules_in_image_field_models() -> None:
    modules = nn.ModuleList(
        [
            EMLImageFieldEncoder(representation_dim=24, field_dim=24, measurement_dim=24, sensor_dim=24),
            EMLImageFieldClassifier(num_classes=5, representation_dim=24, field_dim=24, measurement_dim=24, sensor_dim=24),
        ]
    )

    assert not any(isinstance(module, nn.MultiheadAttention) for module in modules.modules())
    assert not any("transformer" in module.__class__.__name__.lower() for module in modules.modules())
