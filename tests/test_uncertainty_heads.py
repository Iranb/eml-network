from __future__ import annotations

import torch

from eml_mnist.uncertainty_heads import build_uncertainty_head


def test_uncertainty_heads_share_output_api() -> None:
    x = torch.randn(5, 12)
    labels = torch.tensor([0, 1, 2, 1, 0])
    resistance_target = torch.rand(5)

    for name in ["linear", "mlp", "cosine_prototype", "eml_no_ambiguity", "eml_centered_ambiguity", "eml_supervised_resistance"]:
        head = build_uncertainty_head(name, input_dim=12, num_classes=3, hidden_dim=16)
        out = head(x, labels=labels, resistance_target=resistance_target)

        assert out["logits"].shape == (5, 3)
        assert out["probs"].shape == (5, 3)
        assert out["confidence"].shape == (5,)
        assert out["uncertainty"].shape == (5,)
        assert out["risk_score"].shape == (5,)
        assert torch.isfinite(out["logits"]).all()
        assert torch.isfinite(out["risk_score"]).all()


def test_eml_supervised_resistance_exposes_loss() -> None:
    head = build_uncertainty_head("eml_supervised_resistance", input_dim=8, num_classes=2, hidden_dim=12)
    out = head(torch.randn(4, 8), labels=torch.tensor([0, 1, 0, 1]), resistance_target=torch.rand(4))

    assert "resistance_loss" in out
    assert torch.isfinite(out["resistance_loss"])
