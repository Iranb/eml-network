from __future__ import annotations

import torch

from eml_mnist import PureEMLImageBackbone, SyntheticShapeDataset


def test_synthetic_shape_dataset_works_offline() -> None:
    dataset = SyntheticShapeDataset(size=8, image_size=32, seed=0)
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 32)
    assert 0 <= int(sample["shape_label"]) < 5
    assert 0 <= int(sample["color_label"]) < 4


def test_image_backbone_forward_returns_finite_outputs() -> None:
    dataset = SyntheticShapeDataset(size=4, image_size=32, seed=1)
    images = torch.stack([dataset[index]["image"] for index in range(4)], dim=0)

    backbone = PureEMLImageBackbone(
        image_size=32,
        input_channels=3,
        feature_dim=24,
        event_dim=24,
        hidden_dim=48,
        bank_dim=48,
        num_layers=2,
        patch_size=4,
        patch_stride=4,
        local_window_size=3,
        merge_every=2,
        dropout=0.0,
    )
    outputs = backbone(images, warmup_eta=0.5)

    assert outputs["event"].shape == (4, 24)
    assert outputs["pooled_representation"].shape == (4, 24)
    assert outputs["token_features"].ndim == 3
    assert outputs["global_slot_features"].shape[:2] == (4, 4)
    assert outputs["pool_weights"].ndim == 2
    assert torch.isfinite(outputs["event"]).all()
    assert torch.isfinite(outputs["token_features"]).all()
