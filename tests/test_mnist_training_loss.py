from __future__ import annotations

import torch
import torch.nn.functional as F

from eml_mnist.training import compute_loss_bundle


def test_loss_bundle_uses_pairwise_ranking_clipped_energy_and_prototype_diversity() -> None:
    logits = torch.tensor([[0.5, 2.0, 5.5], [3.5, -1.0, 2.0]], requires_grad=True)
    drive = torch.tensor([[0.2, 1.2, 1.8], [1.7, 0.1, 1.0]], requires_grad=True)
    resistance = torch.tensor([[1.0, 0.2, 0.7], [0.4, 1.1, 0.8]], requires_grad=True)
    prototypes = torch.tensor(
        [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0]],
        requires_grad=True,
    )
    targets = torch.tensor([1, 0])
    pairwise_margin = 0.25
    resistance_margin = 0.4
    energy_margin = 3.0

    losses = compute_loss_bundle(
        outputs={
            "logits": logits,
            "drive": drive,
            "resistance": resistance,
            "probs": torch.softmax(logits, dim=-1),
            "sample_uncertainty": torch.ones(2, 1) * 0.3,
            "class_radius": torch.ones(2, 3) * 0.2,
            "prototypes": prototypes,
            "eml_gamma": torch.tensor(0.1),
            "eml_lambda": torch.tensor(1.0),
            "block_stats": [{"gate": torch.ones(2, 3) * 0.6}],
        },
        targets=targets,
        label_smoothing=0.0,
        pairwise_weight=1.0,
        resistance_weight=1.0,
        energy_weight=1.0,
        entropy_weight=0.0,
        prototype_diversity_weight=1.0,
        pairwise_margin=pairwise_margin,
        resistance_margin=resistance_margin,
        energy_margin=energy_margin,
    )

    target_mask = F.one_hot(targets, num_classes=3).bool()
    positive_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    hardest_negative, hard_neg_indices = logits.masked_fill(target_mask, float("-inf")).max(dim=1)
    expected_pairwise = F.softplus(hardest_negative - positive_logits + pairwise_margin).mean()
    expected_resistance = F.softplus(
        resistance.gather(1, targets.unsqueeze(1)).squeeze(1)
        - resistance.gather(1, hard_neg_indices.unsqueeze(1)).squeeze(1)
        + resistance_margin
    ).mean()
    expected_energy = F.relu(logits.abs() - energy_margin).square().mean()
    normalized = F.normalize(prototypes, dim=-1)
    cosine = normalized @ normalized.t()
    expected_diversity = cosine.masked_select(~torch.eye(3, dtype=torch.bool)).square().mean()

    assert torch.allclose(losses["pairwise"], expected_pairwise)
    assert torch.allclose(losses["resistance"], expected_resistance)
    assert torch.allclose(losses["energy"], expected_energy)
    assert torch.allclose(losses["prototype_diversity"], expected_diversity)
    assert torch.allclose(losses["margin_mean"], (positive_logits - hardest_negative).mean())
    assert torch.allclose(losses["sample_uncertainty_mean"], torch.tensor(0.3))
    assert torch.allclose(losses["class_radius_mean"], torch.tensor(0.2))
    assert torch.isfinite(losses["loss"])

    losses["loss"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert resistance.grad is not None and torch.isfinite(resistance.grad).all()
    assert prototypes.grad is not None and torch.isfinite(prototypes.grad).all()
