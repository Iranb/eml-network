from __future__ import annotations

import torch

from eml_mnist import ClassificationHead, RepresentationHead


def test_classification_head_ambiguity() -> None:
    head = ClassificationHead(input_dim=8, num_classes=4, hidden_dim=16)
    representation = torch.randn(3, 8)

    out = head(representation, warmup_eta=0.5)

    assert out["logits"].shape == (3, 4)
    assert out["probs"].shape == (3, 4)
    assert out["ambiguity"].shape == (3, 4)
    assert out["weighted_ambiguity"].shape == (3, 4)
    assert out["ambiguity_weight"].shape == (3, 4)
    assert out["class_resistance"].shape == (3, 4)
    assert torch.allclose(
        out["resistance"],
        out["weighted_ambiguity"] + out["class_resistance"] + out["sample_uncertainty"],
        atol=1.0e-6,
    )
    assert torch.allclose(out["ambiguity_weight"], torch.full_like(out["ambiguity_weight"], 0.5))
    assert torch.isfinite(out["prototype_diversity_penalty"])
    assert torch.isfinite(out["logits"]).all()


def test_classification_head_legacy_ambiguity_flag() -> None:
    head = ClassificationHead(
        input_dim=8,
        num_classes=4,
        hidden_dim=16,
        center_ambiguity=False,
        schedule_ambiguity_weight=False,
    )
    representation = torch.randn(2, 8)

    out = head(representation, warmup_eta=0.25)

    assert torch.allclose(out["ambiguity_weight"], torch.ones_like(out["ambiguity_weight"]))
    assert torch.isfinite(out["ambiguity"]).all()


def test_representation_weights_sum_to_one_over_valid_slots() -> None:
    head = RepresentationHead(slot_dim=8, hidden_dim=16, representation_dim=8)
    slot_states = torch.randn(2, 5, 8)
    slot_mask = torch.tensor([[True, True, False, True, False], [False, True, True, True, False]])

    out = head(slot_states, slot_mask=slot_mask, warmup_eta=0.5)

    assert out["weights"].shape == (2, 5)
    assert torch.allclose(out["weights"].sum(dim=1), torch.ones(2), atol=1.0e-5)
    assert torch.all(out["weights"][~slot_mask] == 0)
    assert out["top_indices"].shape[0] == 2
