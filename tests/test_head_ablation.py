from __future__ import annotations

import torch

from eml_mnist.head_ablation import HEADS, EMLPrototypeHeadCenteredAmbiguity, build_head


def test_head_ablation_heads_forward_shapes() -> None:
    features = torch.randn(4, 16)
    labels = torch.tensor([0, 1, 2, 1])
    for name in HEADS:
        head = build_head(name, input_dim=16, num_classes=3, hidden_dim=32)
        out = head(features, labels=labels, warmup_eta=0.5)
        assert out["logits"].shape == (4, 3)
        assert out["probs"].shape == (4, 3)
        assert out["margin"].shape == (4,)
        assert torch.isfinite(out["logits"]).all()
        assert torch.isfinite(out["probs"]).all()


def test_centered_eml_head_returns_drive_resistance_diagnostics() -> None:
    head = EMLPrototypeHeadCenteredAmbiguity(input_dim=12, num_classes=5, hidden_dim=24)
    features = torch.randn(3, 12)
    labels = torch.tensor([0, 1, 4])

    out = head(features, labels=labels, warmup_eta=0.25)

    assert out["drive"].shape == (3, 5)
    assert out["resistance"].shape == (3, 5)
    assert out["similarity"].shape == (3, 5)
    assert out["ambiguity"].shape == (3, 5)
    assert out["sample_uncertainty"].shape == (3, 1)
    assert torch.allclose(out["ambiguity_weight"], torch.full_like(out["ambiguity_weight"], 0.25))
    assert torch.isfinite(out["positive_drive"]).all()
    assert torch.isfinite(out["hard_negative_resistance"]).all()
