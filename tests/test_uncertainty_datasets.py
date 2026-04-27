from __future__ import annotations

import torch

from eml_mnist.uncertainty_datasets import (
    CIFARCorruptionWrapper,
    SyntheticShapeUncertaintyDataset,
    TextCorruptionDataset,
)


class _TinyImageDataset:
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int):
        return torch.full((3, 16, 16), float(index) / 3.0), index % 2


def test_synthetic_shape_uncertainty_fields() -> None:
    dataset = SyntheticShapeUncertaintyDataset(size=4, image_size=16, seed=3, corruption_type="gaussian_noise")
    sample = dataset[0]

    assert sample["image"].shape == (3, 16, 16)
    assert "clean_image" in sample
    assert int(sample["label"]) == int(sample["clean_label"])
    assert float(sample["noise_level"]) > 0.0
    assert 0.0 <= float(sample["resistance_target"]) <= 1.0


def test_cifar_corruption_wrapper_fields() -> None:
    sample = CIFARCorruptionWrapper(_TinyImageDataset(), corruption_type="cutout_occlusion", seed=1)[0]

    assert sample["image"].shape == (3, 16, 16)
    assert float(sample["occlusion_level"]) > 0.0
    assert float(sample["is_corrupted"]) == 1.0
    assert 0.0 <= float(sample["resistance_target"]) <= 1.0


def test_text_corruption_dataset_fields() -> None:
    dataset = TextCorruptionDataset(size=3, seq_len=16, seed=5, corruption_type="random_token", corruption_prob=0.5)
    sample = dataset[0]

    assert sample["input_ids"].shape == sample["target_ids"].shape
    assert sample["corruption_mask"].dtype == torch.bool
    assert "resistance_target" in sample
    assert sample["resistance_target"].shape == sample["input_ids"].shape
