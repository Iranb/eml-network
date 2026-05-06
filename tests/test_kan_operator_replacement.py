from __future__ import annotations

import torch
import torch.nn.functional as F

from eml_mnist.eml_edge_network import EMLEdgeFunctionNetwork
from eml_mnist.kan_replacement import LinearSplineKANLayer, LinearSplineKANNetwork


def test_linear_spline_kan_layer_forward_is_finite() -> None:
    layer = LinearSplineKANLayer(in_features=3, out_features=2, grid_size=9)
    x = torch.randn(4, 3)

    out = layer(x)

    assert out["output"].shape == (4, 2)
    assert out["edge_output"].shape == (4, 3, 2)
    assert out["basis"].shape == (4, 3, 9)
    assert torch.isfinite(out["output"]).all()


def test_spline_kan_and_semL_replacement_train_step() -> None:
    x = torch.rand(8, 4) * 2.0 - 1.0
    target = (torch.sin(torch.pi * x[:, 0]) + x[:, 1].square()).unsqueeze(-1)
    models = [
        LinearSplineKANNetwork([4, 8, 1], grid_size=9),
        EMLEdgeFunctionNetwork([4, 8, 1]),
    ]

    for model in models:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        out = model(x, warmup_eta=0.5) if isinstance(model, EMLEdgeFunctionNetwork) else model(x)
        loss = F.mse_loss(out["output"], target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        assert out["output"].shape == (8, 1)
        assert torch.isfinite(loss)
