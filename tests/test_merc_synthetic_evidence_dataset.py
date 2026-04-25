from __future__ import annotations

import torch

from eml_mnist.image_datasets import SyntheticShapeEvidenceDataset


def test_synthetic_shape_evidence_dataset_fields() -> None:
    dataset = SyntheticShapeEvidenceDataset(size=4, image_size=32, seed=0)
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 32)
    assert int(sample["label"]) in (0, 1)
    for key in (
        "evidence_target",
        "resistance_target",
        "color_present",
        "shape_present",
        "texture_present",
        "position_condition",
        "contradictory_feature",
    ):
        assert key in sample
        assert torch.is_tensor(sample[key])
