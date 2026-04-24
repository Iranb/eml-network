from __future__ import annotations

import torch
import torch.nn.functional as F

from eml_mnist import EfficientEMLImageClassifier, EfficientEMLImageEncoder, SyntheticShapeEnergyDataset


def _images(batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = SyntheticShapeEnergyDataset(size=batch_size, image_size=32, seed=11)
    images = torch.stack([dataset[index]["image"] for index in range(batch_size)], dim=0)
    labels = torch.tensor([int(dataset[index]["label"]) for index in range(batch_size)], dtype=torch.long)
    return images, labels


def test_efficient_eml_image_encoder_forward() -> None:
    images, _ = _images()
    model = EfficientEMLImageEncoder(
        state_dim=24,
        hidden_dim=48,
        num_hypotheses=4,
        num_attractors=3,
        representation_dim=24,
        patch_stride=4,
    )

    out = model(images, warmup_eta=0.5)

    assert out["representation"].shape == (2, 24)
    assert out["local_states"].ndim == 3
    assert out["parent_states"].ndim == 3
    assert out["attractor_states"].shape[:2] == (2, 3)
    assert torch.isfinite(out["representation"]).all()
    for key in ("drive_mean", "resistance_mean", "energy_mean", "null_weight_mean", "update_strength_mean"):
        assert key in out["diagnostics"]


def test_efficient_eml_image_classifier_logits_and_tiny_step() -> None:
    images, labels = _images()
    model = EfficientEMLImageClassifier(
        num_classes=5,
        state_dim=24,
        hidden_dim=48,
        num_hypotheses=4,
        num_attractors=3,
        representation_dim=24,
        patch_stride=4,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    out = model(images, warmup_eta=0.5)
    loss = F.cross_entropy(out["logits"], labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert out["logits"].shape == (2, 5)
    assert torch.isfinite(loss)
    assert torch.isfinite(out["attractor_states"]).all()


def test_efficient_eml_image_encoder_ablation_switches() -> None:
    images, _ = _images()
    model = EfficientEMLImageEncoder(
        state_dim=24,
        hidden_dim=48,
        num_hypotheses=4,
        num_attractors=1,
        representation_dim=24,
        patch_stride=4,
        enable_composition=False,
        enable_attractor=False,
        sensor_bypass=True,
    )

    out = model(images, warmup_eta=0.5)

    assert out["representation"].shape == (2, 24)
    assert out["diagnostics"]["num_attractors"].item() == 0
    assert "sensor_bypass_alpha" in out["diagnostics"]
    assert torch.isfinite(out["representation"]).all()
