from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_widths(widths: Iterable[int]) -> list[int]:
    values = [int(value) for value in widths]
    if len(values) < 2:
        raise ValueError("at least input and output widths are required")
    if any(value <= 0 for value in values):
        raise ValueError("all widths must be positive")
    return values


def _stats(prefix: str, value: torch.Tensor) -> Dict[str, torch.Tensor]:
    detached = value.detach().float()
    return {
        f"{prefix}_mean": detached.mean(),
        f"{prefix}_std": detached.std(unbiased=False),
    }


class LinearSplineKANLayer(nn.Module):
    """KAN-style edge-function layer with degree-1 B-spline basis functions.

    This is a compact PyTorch baseline for operator-replacement tests. Each
    source-destination edge owns a univariate spline plus a SiLU residual, and
    each destination node sums incoming edge outputs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 17,
        grid_range: float = 2.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        if grid_range <= 0.0:
            raise ValueError("grid_range must be positive")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.grid_range = float(grid_range)

        grid = torch.linspace(-self.grid_range, self.grid_range, self.grid_size, dtype=torch.float32)
        self.register_buffer("grid", grid)
        self.register_buffer("grid_spacing", torch.tensor(float(grid[1] - grid[0]), dtype=torch.float32))

        edge_shape = (self.in_features, self.out_features)
        self.base_scale = nn.Parameter(torch.empty(edge_shape))
        self.spline_weight = nn.Parameter(torch.empty(self.in_features, self.out_features, self.grid_size))
        self.output_bias = nn.Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan = 1.0 / math.sqrt(float(self.in_features))
        nn.init.normal_(self.base_scale, mean=0.0, std=fan)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1 * fan)

    def _linear_spline_basis(self, values: torch.Tensor) -> torch.Tensor:
        values_fp32 = values.float().unsqueeze(-1)
        grid = self.grid.to(device=values.device, dtype=torch.float32)
        spacing = self.grid_spacing.to(device=values.device, dtype=torch.float32)
        basis = (1.0 - (values_fp32 - grid).abs() / spacing).clamp_min(0.0)
        return basis.to(dtype=values.dtype)

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        if inputs.ndim < 2 or inputs.size(-1) != self.in_features:
            raise ValueError("inputs must have shape [..., in_features]")

        original_shape = inputs.shape[:-1]
        flat_inputs = inputs.reshape(-1, self.in_features)
        basis = self._linear_spline_basis(flat_inputs)
        spline_output = torch.einsum("big,iog->bio", basis, self.spline_weight)
        base_output = F.silu(flat_inputs).unsqueeze(-1) * self.base_scale.unsqueeze(0)
        edge_output = base_output + spline_output
        output = edge_output.sum(dim=1) / math.sqrt(float(self.in_features))
        output = self.dropout(output + self.output_bias)

        return {
            "output": output.reshape(*original_shape, self.out_features),
            "edge_output": edge_output.reshape(*original_shape, self.in_features, self.out_features),
            "basis": basis.reshape(*original_shape, self.in_features, self.grid_size),
        }


class LinearSplineKANNetwork(nn.Module):
    """Stacked KAN-style network using degree-1 spline edge operators."""

    def __init__(
        self,
        widths: Sequence[int],
        grid_size: int = 17,
        grid_range: float = 2.5,
        dropout: float = 0.0,
        hidden_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        values = _as_widths(widths)
        self.widths = values
        self.layers = nn.ModuleList(
            [
                LinearSplineKANLayer(
                    values[index],
                    values[index + 1],
                    grid_size=grid_size,
                    grid_range=grid_range,
                    dropout=dropout if index < len(values) - 2 else 0.0,
                )
                for index in range(len(values) - 1)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(values[index + 1]) for index in range(len(values) - 2)])
        self.hidden_layer_norm = bool(hidden_layer_norm)

    def forward(self, inputs: torch.Tensor) -> Dict[str, Any]:
        x = inputs
        layer_outputs: list[Dict[str, torch.Tensor]] = []
        for index, layer in enumerate(self.layers):
            layer_out = layer(x)
            x = layer_out["output"]
            if self.hidden_layer_norm and index < len(self.layers) - 1:
                x = self.norms[index](x)
            layer_out["output"] = x
            layer_outputs.append(layer_out)

        diagnostics: Dict[str, torch.Tensor] = {}
        if layer_outputs:
            diagnostics.update(_stats("spline_edge_output", layer_outputs[-1]["edge_output"]))
            diagnostics.update(_stats("spline_basis", layer_outputs[-1]["basis"]))
        return {
            "output": x,
            "layers": layer_outputs,
            "diagnostics": diagnostics,
            "edge_output": layer_outputs[-1]["edge_output"] if layer_outputs else torch.empty(0, device=inputs.device),
        }


__all__ = [
    "LinearSplineKANLayer",
    "LinearSplineKANNetwork",
]
