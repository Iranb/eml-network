from __future__ import annotations

import torch

from eml_mnist.image_datasets import CIFARCorruptionDataset, SyntheticShapeEnergyDataset
from eml_mnist.metrics import area_under_risk_coverage_curve, binary_auroc, selective_risk_curve


class _ToyTensorDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        image = torch.full((3, 8, 8), float(index) / 4.0)
        return image, index % 2


def test_synthetic_shape_dataset_forced_corruption_modes() -> None:
    clean_dataset = SyntheticShapeEnergyDataset(
        size=4,
        image_size=16,
        seed=0,
        forced_noise_name="low",
        forced_occlusion_name="none",
        forced_clutter_flag=0,
        include_background_clutter=False,
        include_mask=False,
    )
    occluded_dataset = SyntheticShapeEnergyDataset(
        size=16,
        image_size=16,
        seed=0,
        forced_noise_name="low",
        forced_occlusion_name="partial",
        forced_clutter_flag=0,
        include_background_clutter=False,
        include_mask=False,
    )
    clean = clean_dataset[0]

    assert float(clean["noise_level"]) <= 0.25
    assert float(clean["occlusion_level"]) == 0.0
    assert any(float(occluded_dataset[index]["occlusion_level"]) > 0.0 for index in range(len(occluded_dataset)))


def test_cifar_corruption_dataset_modes_emit_metadata() -> None:
    base = _ToyTensorDataset()
    clean = CIFARCorruptionDataset(base, mode="clean", seed=0)[0]
    noisy = CIFARCorruptionDataset(base, mode="noisy", seed=0)[0]
    occluded = CIFARCorruptionDataset(base, mode="occluded", seed=0)[0]

    assert float(clean["noise_level"]) == 0.0
    assert float(clean["occlusion_level"]) == 0.0
    assert float(noisy["noise_level"]) > 0.0
    assert float(occluded["occlusion_level"]) > 0.0


def test_uncertainty_metrics_behave_sensibly() -> None:
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0, 0, 1, 1])
    correct = torch.tensor([False, True, True, True])
    acceptance = torch.tensor([0.1, 0.4, 0.8, 0.9])

    auc = binary_auroc(scores, labels)
    selective = selective_risk_curve(correct, acceptance)
    aurc = area_under_risk_coverage_curve(correct, acceptance, steps=10)

    assert auc > 0.9
    assert selective["risk_at_50_coverage"] <= selective["risk_at_100_coverage"]
    assert 0.0 <= aurc <= 1.0
