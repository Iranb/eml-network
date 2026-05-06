from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import EMLUnit, _reset_linear, inverse_softplus


def _as_list(widths: Iterable[int]) -> list[int]:
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


class EMLEdgeFunctionLayer(nn.Module):
    """KAN-style edge-function layer using stable sEML instead of splines.

    A KAN layer is a matrix of learnable univariate edge functions whose
    outputs are summed at each destination node. This layer keeps that outer
    structure, but each edge function is a residual base activation plus a
    stable EML drive/resistance term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        clip_value: float = 3.0,
        init_gamma: float = 0.1,
        init_lambda: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.edge_dim = self.in_features * self.out_features

        edge_shape = (self.in_features, self.out_features)
        self.drive_scale = nn.Parameter(torch.empty(edge_shape))
        self.drive_bias = nn.Parameter(torch.zeros(edge_shape))
        self.resistance_center = nn.Parameter(torch.zeros(edge_shape))
        self.raw_resistance_scale = nn.Parameter(torch.empty(edge_shape))
        self.raw_resistance_floor = nn.Parameter(
            torch.full(edge_shape, inverse_softplus(0.05), dtype=torch.float32)
        )
        self.base_scale = nn.Parameter(torch.empty(edge_shape))
        self.eml_scale = nn.Parameter(torch.empty(edge_shape))
        self.output_bias = nn.Parameter(torch.zeros(out_features))
        self.eml = EMLUnit(
            dim=self.edge_dim,
            clip_value=clip_value,
            init_gamma=init_gamma,
            init_lambda=init_lambda,
            init_bias=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan = 1.0 / math.sqrt(float(self.in_features))
        nn.init.normal_(self.drive_scale, mean=0.0, std=fan)
        nn.init.normal_(self.base_scale, mean=0.0, std=fan)
        nn.init.normal_(self.eml_scale, mean=0.0, std=0.1 * fan)
        nn.init.constant_(self.raw_resistance_scale, inverse_softplus(0.2))

    def forward(
        self,
        inputs: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if inputs.ndim < 2 or inputs.size(-1) != self.in_features:
            raise ValueError("inputs must have shape [..., in_features]")

        original_shape = inputs.shape[:-1]
        flat_inputs = inputs.reshape(-1, self.in_features)
        edge_input = flat_inputs.unsqueeze(-1)
        drive = edge_input * self.drive_scale.unsqueeze(0) + self.drive_bias.unsqueeze(0)

        resistance_distance = edge_input - self.resistance_center.unsqueeze(0)
        resistance_scale = F.softplus(self.raw_resistance_scale).unsqueeze(0)
        resistance_floor = F.softplus(self.raw_resistance_floor).unsqueeze(0)
        resistance = resistance_distance.square() * resistance_scale + resistance_floor

        energy = self.eml(
            drive.reshape(flat_inputs.size(0), self.edge_dim),
            resistance.reshape(flat_inputs.size(0), self.edge_dim),
            warmup_eta=warmup_eta,
        ).reshape(flat_inputs.size(0), self.in_features, self.out_features)

        base = F.silu(edge_input) * self.base_scale.unsqueeze(0)
        edge_output = base + self.eml_scale.unsqueeze(0) * energy
        output = edge_output.sum(dim=1) / math.sqrt(float(self.in_features))
        output = self.dropout(output + self.output_bias)
        output = output.reshape(*original_shape, self.out_features)

        return {
            "output": output,
            "drive": drive.reshape(*original_shape, self.in_features, self.out_features),
            "resistance": resistance.reshape(*original_shape, self.in_features, self.out_features),
            "energy": energy.reshape(*original_shape, self.in_features, self.out_features),
            "edge_output": edge_output.reshape(*original_shape, self.in_features, self.out_features),
        }


class EMLEdgeFunctionNetwork(nn.Module):
    """Stacked KAN-style sEML edge-function layers."""

    def __init__(
        self,
        widths: Sequence[int],
        clip_value: float = 3.0,
        dropout: float = 0.0,
        final_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        values = _as_list(widths)
        self.widths = values
        self.layers = nn.ModuleList(
            [
                EMLEdgeFunctionLayer(
                    in_features=values[index],
                    out_features=values[index + 1],
                    clip_value=clip_value,
                    dropout=dropout if index < len(values) - 2 else 0.0,
                )
                for index in range(len(values) - 1)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(values[index + 1])
                for index in range(len(values) - 1)
                if index < len(values) - 2 or final_layer_norm
            ]
        )
        self.final_layer_norm = bool(final_layer_norm)

    def forward(
        self,
        inputs: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, Any]:
        x = inputs
        layer_outputs: list[Dict[str, torch.Tensor]] = []
        norm_index = 0
        for index, layer in enumerate(self.layers):
            layer_out = layer(x, warmup_eta=warmup_eta)
            x = layer_out["output"]
            should_norm = index < len(self.layers) - 1 or self.final_layer_norm
            if should_norm:
                x = self.norms[norm_index](x)
                norm_index += 1
            layer_out["output"] = x
            layer_outputs.append(layer_out)
        diagnostics: Dict[str, torch.Tensor] = {}
        if layer_outputs:
            last = layer_outputs[-1]
            diagnostics.update(_stats("edge_drive", last["drive"]))
            diagnostics.update(_stats("edge_resistance", last["resistance"]))
            diagnostics.update(_stats("edge_energy", last["energy"]))
            diagnostics.update(_stats("edge_output", last["edge_output"]))
        return {
            "output": x,
            "layers": layer_outputs,
            "diagnostics": diagnostics,
            "drive": layer_outputs[-1]["drive"] if layer_outputs else torch.empty(0, device=inputs.device),
            "resistance": layer_outputs[-1]["resistance"] if layer_outputs else torch.empty(0, device=inputs.device),
            "energy": layer_outputs[-1]["energy"] if layer_outputs else torch.empty(0, device=inputs.device),
        }


class EMLEdgeImageClassifier(nn.Module):
    """Small image classifier with a CNN sensor and sEML edge-function readout."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        state_dim: int = 32,
        edge_width: int = 32,
        clip_value: float = 3.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes <= 0 or input_channels <= 0 or state_dim <= 0 or edge_width <= 0:
            raise ValueError("invalid image classifier dimensions")
        self.num_classes = int(num_classes)
        sensor_hidden = max(16, state_dim // 2)
        self.sensor = nn.Sequential(
            nn.Conv2d(input_channels, sensor_hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, sensor_hidden),
            nn.GELU(),
            nn.Conv2d(sensor_hidden, sensor_hidden, kernel_size=3, padding=1, groups=sensor_hidden),
            nn.Conv2d(sensor_hidden, state_dim, kernel_size=1),
            nn.GroupNorm(1, state_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.pre_norm = nn.LayerNorm(state_dim)
        self.edge_net = EMLEdgeFunctionNetwork(
            [state_dim, edge_width, num_classes],
            clip_value=clip_value,
            dropout=dropout,
            final_layer_norm=False,
        )
        for module in self.sensor.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        images: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, Any]:
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width]")
        sensor_features = self.sensor(images).flatten(1)
        representation = self.pre_norm(sensor_features)
        edge_out = self.edge_net(representation, warmup_eta=warmup_eta)
        logits = edge_out["output"]
        probs = torch.softmax(logits, dim=-1)
        diagnostics = dict(edge_out["diagnostics"])
        diagnostics["sensor_norm"] = sensor_features.detach().float().norm(dim=-1).mean()
        return {
            "logits": logits,
            "probs": probs,
            "representation": representation,
            "sensor_features": sensor_features,
            "diagnostics": diagnostics,
            "edge": edge_out,
            "drive": edge_out["drive"],
            "resistance": edge_out["resistance"],
            "energy": edge_out["energy"],
        }


class EMLEdgeTextLM(nn.Module):
    """Causal local text LM with sEML edge-function token readout."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        state_dim: int = 32,
        edge_width: int = 48,
        clip_value: float = 3.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or state_dim <= 0 or edge_width <= 0:
            raise ValueError("invalid text LM dimensions")
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.embedding = nn.Embedding(vocab_size, state_dim, padding_idx=pad_id)
        self.conv1 = nn.Conv1d(state_dim, state_dim, kernel_size=3)
        self.conv2 = nn.Conv1d(state_dim, state_dim, kernel_size=3)
        self.norm = nn.LayerNorm(state_dim)
        self.edge_net = EMLEdgeFunctionNetwork(
            [state_dim, edge_width, vocab_size],
            clip_value=clip_value,
            dropout=dropout,
            final_layer_norm=False,
        )
        _reset_linear(self.conv1)
        _reset_linear(self.conv2)

    def _causal_conv(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        return conv(F.pad(x, (conv.kernel_size[0] - 1, 0)))

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if padding_mask is None:
            padding_mask = input_ids != self.pad_id
        padding_mask = padding_mask.bool()
        x = self.embedding(input_ids).transpose(1, 2)
        x = F.gelu(self._causal_conv(x, self.conv1))
        x = F.gelu(self._causal_conv(x, self.conv2)).transpose(1, 2)
        sequence_states = torch.where(padding_mask.unsqueeze(-1), self.norm(x), torch.zeros_like(x))
        edge_out = self.edge_net(sequence_states, warmup_eta=warmup_eta)
        logits = edge_out["output"]
        logits = torch.where(padding_mask.unsqueeze(-1), logits, torch.zeros_like(logits))
        diagnostics = dict(edge_out["diagnostics"])
        diagnostics["valid_token_rate"] = padding_mask.float().mean().detach()
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
            "sequence_states": sequence_states,
            "padding_mask": padding_mask,
            "diagnostics": diagnostics,
            "edge": edge_out,
            "drive": edge_out["drive"],
            "resistance": edge_out["resistance"],
            "energy": edge_out["energy"],
        }


__all__ = [
    "EMLEdgeFunctionLayer",
    "EMLEdgeFunctionNetwork",
    "EMLEdgeImageClassifier",
    "EMLEdgeTextLM",
]
