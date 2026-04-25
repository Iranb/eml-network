from __future__ import annotations

import torch

from eml_mnist.head_ablation import build_head


def test_merc_head_logits_shape() -> None:
    head = build_head("merc_linear", input_dim=32, num_classes=5, hidden_dim=48)
    x = torch.randn(7, 32)
    y = torch.randint(0, 5, (7,))
    out = head(x, labels=y)
    assert out["logits"].shape == (7, 5)
    assert torch.isfinite(out["logits"]).all()
    assert "support_factors" in out
    assert "conflict_factors" in out


def test_old_heads_still_work() -> None:
    for head_name in ("linear", "mlp", "cosine_prototype", "eml_centered_ambiguity"):
        head = build_head(head_name, input_dim=16, num_classes=3, hidden_dim=24)
        x = torch.randn(4, 16)
        y = torch.randint(0, 3, (4,))
        out = head(x, labels=y)
        assert out["logits"].shape == (4, 3)
