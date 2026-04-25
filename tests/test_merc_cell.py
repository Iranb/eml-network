from __future__ import annotations

import torch

from eml_mnist.merc import MERCCell, MERCResidualBlock


def test_merc_cell_forward_shape_and_finite() -> None:
    cell = MERCCell(input_dim=16, output_dim=12, hidden_dim=24)
    x = torch.randn(5, 16, requires_grad=True)
    out = cell(x)
    assert out["output"].shape == (5, 12)
    assert torch.isfinite(out["output"]).all()
    assert (out["support_factors"] > 0).all()
    assert (out["conflict_factors"] >= 0).all()
    assert torch.isfinite(out["log_support"]).all()
    loss = out["output"].square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_merc_residual_block_update_gate_range() -> None:
    block = MERCResidualBlock(input_dim=10, hidden_dim=20)
    x = torch.randn(4, 10)
    out = block(x)
    assert out["output"].shape == (4, 10)
    assert torch.isfinite(out["output"]).all()
    assert ((out["update_gate"] >= 0.0) & (out["update_gate"] <= 1.0)).all()
