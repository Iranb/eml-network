from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import EMLActivationBudget, EMLUnit, EMLUpdateGate, _reset_linear


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, final_tanh: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        ]
        if final_tanh:
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _reset_linear(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _infer_image_shape(num_positions: int, image_shape: tuple[int, int] | None) -> tuple[int, int]:
    if image_shape is not None:
        height, width = image_shape
        if height <= 0 or width <= 0 or height * width != num_positions:
            raise ValueError("image_shape must multiply to the number of positions")
        return height, width

    side = int(math.sqrt(num_positions))
    if side * side != num_positions:
        raise ValueError("image_shape is required when positions do not form a square")
    return side, side


def _as_position_features(
    position_features: torch.Tensor | None,
    batch_size: int,
    num_positions: int,
    position_dim: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if position_dim <= 0:
        return reference.new_zeros(batch_size, num_positions, 0)
    if position_features is None:
        return reference.new_zeros(batch_size, num_positions, position_dim)
    if position_features.ndim == 2:
        if position_features.shape != (num_positions, position_dim):
            raise ValueError("position_features must have shape [positions, position_dim]")
        return position_features.to(device=reference.device, dtype=reference.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    if position_features.ndim == 3:
        if position_features.shape != (batch_size, num_positions, position_dim):
            raise ValueError("position_features must have shape [batch, positions, position_dim]")
        return position_features.to(device=reference.device, dtype=reference.dtype)
    raise ValueError("position_features must be rank 2 or 3")


class EMLSensor(nn.Module):
    """Convert local features into measurements and drive/resistance seeds."""

    def __init__(
        self,
        input_dim: int,
        measurement_dim: int,
        seed_dim: int | None = None,
        hidden_dim: int | None = None,
        position_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or measurement_dim <= 0:
            raise ValueError("input_dim and measurement_dim must be positive")
        if seed_dim is not None and seed_dim <= 0:
            raise ValueError("seed_dim must be positive when provided")
        if position_dim < 0:
            raise ValueError("position_dim must be non-negative")

        self.input_dim = input_dim
        self.measurement_dim = measurement_dim
        self.seed_dim = seed_dim or measurement_dim
        self.position_dim = position_dim
        hidden = hidden_dim or max(input_dim + position_dim, measurement_dim, self.seed_dim)
        joint_dim = input_dim + position_dim

        self.norm = nn.LayerNorm(joint_dim)
        self.measurement = nn.Sequential(
            nn.Linear(joint_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, measurement_dim),
        )
        self.drive_seed = nn.Linear(joint_dim, self.seed_dim)
        self.resistance_seed = nn.Linear(joint_dim, self.seed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _reset_linear(module)

    def forward(
        self,
        x: torch.Tensor,
        position_features: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if x.ndim != 3 or x.size(-1) != self.input_dim:
            raise ValueError("x must have shape [batch, positions, input_dim]")

        batch_size, num_positions, _ = x.shape
        positions = _as_position_features(position_features, batch_size, num_positions, self.position_dim, x)
        joint = torch.cat([x, positions], dim=-1)
        normalized = self.norm(joint)
        out = {
            "measurement": self.measurement(normalized),
            "drive_seed": self.drive_seed(normalized),
            "resistance_seed": self.resistance_seed(normalized),
        }
        if self.position_dim > 0:
            out["position_features"] = positions
        return out


class EMLHypothesisField(nn.Module):
    """Maintain local hypotheses and score them as drive/resistance energy."""

    def __init__(
        self,
        measurement_dim: int,
        field_dim: int,
        num_hypotheses: int,
        hidden_dim: int | None = None,
        clip_value: float = 3.0,
        activation_temperature: float = 1.0,
        prototype_temperature: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if measurement_dim <= 0 or field_dim <= 0 or num_hypotheses <= 0:
            raise ValueError("measurement_dim, field_dim, and num_hypotheses must be positive")
        if prototype_temperature <= 0.0:
            raise ValueError("prototype_temperature must be positive")

        self.measurement_dim = measurement_dim
        self.field_dim = field_dim
        self.num_hypotheses = num_hypotheses
        self.prototype_temperature = float(prototype_temperature)
        hidden = hidden_dim or max(measurement_dim, field_dim)

        self.measurement_norm = nn.LayerNorm(measurement_dim)
        self.measurement_proj = nn.Sequential(
            nn.Linear(measurement_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, field_dim),
        )
        self.hypothesis_prototypes = nn.Parameter(torch.empty(num_hypotheses, field_dim))
        joint_dim = field_dim * 4
        self.joint_norm = nn.LayerNorm(joint_dim)
        self.state_net = _MLP(joint_dim, hidden, field_dim, final_tanh=True)
        self.drive_net = _MLP(joint_dim, hidden, 1)
        self.resistance_net = _MLP(joint_dim, hidden, 1)
        self.uncertainty_net = _MLP(measurement_dim, hidden, 1)
        self.state_norm = nn.LayerNorm(field_dim)
        self.eml = EMLUnit(dim=1, clip_value=clip_value, init_bias=0.0)
        self.activation_budget = EMLActivationBudget(temperature=activation_temperature)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.measurement_proj.modules():
            if isinstance(module, nn.Linear):
                _reset_linear(module)
        nn.init.normal_(self.hypothesis_prototypes, mean=0.0, std=0.02)

    def _ambiguity(self, resonance: torch.Tensor) -> torch.Tensor:
        if self.num_hypotheses == 1:
            return torch.zeros_like(resonance)
        eye = torch.eye(self.num_hypotheses, device=resonance.device, dtype=torch.bool)
        expanded = resonance.unsqueeze(-2).expand(*resonance.shape[:-1], self.num_hypotheses, self.num_hypotheses)
        competing = expanded.masked_fill(eye, float("-inf")).logsumexp(dim=-1)
        return F.softplus(competing - resonance)

    def forward(
        self,
        measurements: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if measurements.ndim != 3 or measurements.size(-1) != self.measurement_dim:
            raise ValueError("measurements must have shape [batch, positions, measurement_dim]")

        normalized_measurements = self.measurement_norm(measurements)
        measurement_state = self.measurement_proj(normalized_measurements)
        batch_size, num_positions, _ = measurement_state.shape

        prototypes = self.hypothesis_prototypes.view(1, 1, self.num_hypotheses, self.field_dim)
        measured = measurement_state.unsqueeze(2).expand(batch_size, num_positions, self.num_hypotheses, -1)
        prototype_features = prototypes.expand(batch_size, num_positions, -1, -1)
        joint = self.joint_norm(
            torch.cat(
                [
                    measured,
                    prototype_features,
                    measured * prototype_features,
                    measured - prototype_features,
                ],
                dim=-1,
            )
        )

        normalized_state = F.normalize(measurement_state, dim=-1)
        normalized_prototypes = F.normalize(self.hypothesis_prototypes, dim=-1)
        resonance = torch.sum(
            normalized_state.unsqueeze(2) * normalized_prototypes.view(1, 1, self.num_hypotheses, self.field_dim),
            dim=-1,
        ) / self.prototype_temperature
        uncertainty = F.softplus(self.uncertainty_net(normalized_measurements)).squeeze(-1).unsqueeze(-1)
        ambiguity = self._ambiguity(resonance)

        hypothesis_state = self.state_norm(self.state_net(joint))
        drive = self.drive_net(joint).squeeze(-1) + resonance
        resistance = F.softplus(self.resistance_net(joint).squeeze(-1)) + uncertainty + ambiguity
        energy = self.eml(drive.unsqueeze(-1), resistance.unsqueeze(-1), warmup_eta=warmup_eta).squeeze(-1)
        activation_out = self.activation_budget(energy)

        return {
            "hypothesis_state": hypothesis_state,
            "drive": drive,
            "resistance": resistance,
            "energy": energy,
            "activation": activation_out["activation"],
            "budget_loss": activation_out["budget_loss"],
            "entropy": activation_out["entropy"],
            "active_rate": activation_out["active_rate"],
            "resonance": resonance,
            "uncertainty": uncertainty.expand_as(resistance),
            "ambiguity": ambiguity,
        }


class EMLHypothesisCompetition(nn.Module):
    """Refine local hypothesis activations while keeping early co-winners possible."""

    def __init__(
        self,
        temperature: float = 1.0,
        competition_strength: float = 0.25,
        target_rate: float | None = None,
        budget_weight: float = 1.0,
        soft_sparse: bool = False,
        top_k: int | None = None,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= competition_strength <= 1.0:
            raise ValueError("competition_strength must be in [0, 1]")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.temperature = float(temperature)
        self.competition_strength = float(competition_strength)
        self.eps = float(eps)
        self.activation_budget = EMLActivationBudget(
            temperature=temperature,
            target_rate=target_rate,
            budget_weight=budget_weight,
            soft_sparse=soft_sparse,
            top_k=top_k,
        )

    def forward(
        self,
        energy: torch.Tensor,
        activation: torch.Tensor | None = None,
        resistance: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        budget_out = self.activation_budget(energy, mask=mask)
        refined = budget_out["activation"] if activation is None else activation
        if refined.shape != energy.shape:
            raise ValueError("activation must match energy shape")
        if resistance is not None:
            if resistance.shape != energy.shape:
                raise ValueError("resistance must match energy shape")
            refined = refined * torch.sigmoid(-resistance.to(dtype=torch.float32)).to(dtype=refined.dtype)

        if self.competition_strength > 0.0:
            logits = energy.to(dtype=torch.float32) / self.temperature
            if mask is not None:
                logits = logits.masked_fill(~mask.bool(), float("-inf"))
            normalized = torch.softmax(logits, dim=-1).to(dtype=refined.dtype)
            if mask is not None:
                normalized = normalized.masked_fill(~mask.bool(), 0.0)
            refined = (1.0 - self.competition_strength) * refined + self.competition_strength * normalized
        if mask is not None:
            refined = refined.masked_fill(~mask.bool(), 0.0)

        refined_fp32 = refined.to(dtype=torch.float32)
        mass = refined_fp32.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        distribution = (refined_fp32 / mass).clamp(self.eps, 1.0)
        entropy = -(distribution * distribution.log()).sum(dim=-1).mean()
        active_rate = refined_fp32.mean()

        return {
            "activation": refined,
            "budget_loss": budget_out["budget_loss"],
            "entropy": entropy,
            "active_rate": active_rate,
            "hard_active_rate": (refined_fp32 > 0.5).to(dtype=torch.float32).mean(),
            "topk_mask": budget_out["topk_mask"],
        }


class EMLConsensusField(nn.Module):
    """Propagate local support and conflict across neighboring hypotheses."""

    def __init__(
        self,
        field_dim: int,
        hidden_dim: int,
        num_hypotheses: int,
        mode: str = "image",
        window_size: int = 3,
        relative_dim: int = 8,
        clip_value: float = 3.0,
        activation_temperature: float = 1.0,
        gate_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if field_dim <= 0 or hidden_dim <= 0 or num_hypotheses <= 0:
            raise ValueError("field_dim, hidden_dim, and num_hypotheses must be positive")
        if mode not in {"image", "text"}:
            raise ValueError("mode must be 'image' or 'text'")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if mode == "image" and window_size % 2 == 0:
            raise ValueError("image mode window_size must be odd")
        if relative_dim <= 0:
            raise ValueError("relative_dim must be positive")
        if gate_eps <= 0.0:
            raise ValueError("gate_eps must be positive")

        self.field_dim = field_dim
        self.num_hypotheses = num_hypotheses
        self.mode = mode
        self.window_size = window_size
        self.gate_eps = float(gate_eps)
        self.neighbor_count = window_size * window_size if mode == "image" else window_size
        self.relative = nn.Parameter(torch.empty(self.neighbor_count, relative_dim))
        joint_dim = field_dim * 3 + relative_dim
        self.joint_norm = nn.LayerNorm(joint_dim)
        self.support_net = _MLP(joint_dim, hidden_dim, 1)
        self.conflict_net = _MLP(joint_dim, hidden_dim, 1)
        self.eml = EMLUnit(dim=1, clip_value=clip_value, init_bias=-0.1)
        self.activation_budget = EMLActivationBudget(temperature=activation_temperature)
        nn.init.normal_(self.relative, mean=0.0, std=0.02)

    def _image_neighbors(
        self,
        hypothesis_state: torch.Tensor,
        activation: torch.Tensor,
        image_shape: tuple[int, int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_positions, num_hypotheses, field_dim = hypothesis_state.shape
        height, width = _infer_image_shape(num_positions, image_shape)
        state_grid = hypothesis_state.view(batch_size, height, width, num_hypotheses, field_dim)
        state_grid = state_grid.permute(0, 3, 4, 1, 2).reshape(batch_size, num_hypotheses * field_dim, height, width)
        state_windows = F.unfold(state_grid, kernel_size=self.window_size, padding=self.window_size // 2)
        state_windows = state_windows.view(
            batch_size,
            num_hypotheses,
            field_dim,
            self.neighbor_count,
            num_positions,
        ).permute(0, 4, 3, 1, 2)

        activation_grid = activation.view(batch_size, height, width, num_hypotheses).permute(0, 3, 1, 2)
        activation_windows = F.unfold(activation_grid, kernel_size=self.window_size, padding=self.window_size // 2)
        activation_windows = activation_windows.view(
            batch_size,
            num_hypotheses,
            self.neighbor_count,
            num_positions,
        ).permute(0, 3, 2, 1)

        valid_grid = torch.ones(batch_size, 1, height, width, device=hypothesis_state.device, dtype=hypothesis_state.dtype)
        valid_windows = F.unfold(valid_grid, kernel_size=self.window_size, padding=self.window_size // 2)
        valid_windows = valid_windows.transpose(1, 2).to(dtype=torch.bool)
        return state_windows, activation_windows, valid_windows

    def _text_neighbors(
        self,
        hypothesis_state: torch.Tensor,
        activation: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_positions, _, _ = hypothesis_state.shape
        padded_state = F.pad(hypothesis_state, (0, 0, 0, 0, self.window_size - 1, 0))
        state_windows = torch.stack(
            [padded_state[:, offset : offset + num_positions] for offset in range(self.window_size)],
            dim=2,
        )
        padded_activation = F.pad(activation, (0, 0, self.window_size - 1, 0))
        activation_windows = torch.stack(
            [padded_activation[:, offset : offset + num_positions] for offset in range(self.window_size)],
            dim=2,
        )

        distances = torch.arange(self.window_size - 1, -1, -1, device=hypothesis_state.device)
        positions = torch.arange(num_positions, device=hypothesis_state.device).unsqueeze(-1)
        valid_windows = (positions >= distances.unsqueeze(0)).unsqueeze(0).expand(batch_size, -1, -1)
        if padding_mask is not None:
            if padding_mask.shape != (batch_size, num_positions):
                raise ValueError("padding_mask must have shape [batch, positions]")
            padded_mask = F.pad(padding_mask.bool(), (self.window_size - 1, 0), value=False)
            source_mask = torch.stack(
                [padded_mask[:, offset : offset + num_positions] for offset in range(self.window_size)],
                dim=2,
            )
            valid_windows = valid_windows & source_mask & padding_mask.bool().unsqueeze(-1)
        return state_windows, activation_windows, valid_windows

    def forward(
        self,
        hypothesis_state: torch.Tensor,
        activation: torch.Tensor,
        drive: torch.Tensor | None = None,
        resistance: torch.Tensor | None = None,
        image_shape: tuple[int, int] | None = None,
        padding_mask: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if hypothesis_state.ndim != 4 or hypothesis_state.size(-1) != self.field_dim:
            raise ValueError("hypothesis_state must have shape [batch, positions, hypotheses, field_dim]")
        if hypothesis_state.size(2) != self.num_hypotheses:
            raise ValueError("hypothesis_state hypothesis count does not match this field")
        if activation.shape != hypothesis_state.shape[:3]:
            raise ValueError("activation must have shape [batch, positions, hypotheses]")
        if drive is not None and drive.shape != activation.shape:
            raise ValueError("drive must match activation shape")
        if resistance is not None and resistance.shape != activation.shape:
            raise ValueError("resistance must match activation shape")

        if self.mode == "image":
            neighbor_state, neighbor_activation, neighbor_mask = self._image_neighbors(hypothesis_state, activation, image_shape)
        else:
            neighbor_state, neighbor_activation, neighbor_mask = self._text_neighbors(hypothesis_state, activation, padding_mask)

        target = hypothesis_state.unsqueeze(3).unsqueeze(4)
        source = neighbor_state.unsqueeze(2)
        target = target.expand(-1, -1, -1, self.neighbor_count, self.num_hypotheses, -1)
        source = source.expand(-1, -1, self.num_hypotheses, -1, -1, -1)
        relative = self.relative.view(1, 1, 1, self.neighbor_count, 1, -1)
        relative = relative.expand(*source.shape[:-1], -1)
        joint = self.joint_norm(torch.cat([target, source, target - source, relative], dim=-1))

        support_value = F.softplus(self.support_net(joint).squeeze(-1))
        conflict_value = F.softplus(self.conflict_net(joint).squeeze(-1))
        source_gate = neighbor_activation.to(dtype=torch.float32).unsqueeze(2)
        source_gate = source_gate * neighbor_mask.unsqueeze(-1).unsqueeze(2).to(dtype=torch.float32)
        gate_mass = source_gate.sum(dim=(3, 4)).clamp_min(self.gate_eps)

        support = (support_value * source_gate).sum(dim=(3, 4)) / gate_mass
        conflict = (conflict_value * source_gate).sum(dim=(3, 4)) / gate_mass
        propagated_drive = support if drive is None else drive + support
        propagated_resistance = conflict if resistance is None else resistance + conflict
        energy = self.eml(
            propagated_drive.unsqueeze(-1),
            propagated_resistance.unsqueeze(-1),
            warmup_eta=warmup_eta,
        ).squeeze(-1)
        activation_out = self.activation_budget(energy)

        return {
            "support": support,
            "conflict": conflict,
            "drive": propagated_drive,
            "resistance": propagated_resistance,
            "energy": energy,
            "activation": activation_out["activation"],
            "gate_mass": gate_mass.expand_as(support),
            "budget_loss": activation_out["budget_loss"],
            "entropy": activation_out["entropy"],
            "active_rate": activation_out["active_rate"],
        }


class EMLCompositionField(nn.Module):
    """Fold consistent child hypotheses into parent hypotheses."""

    def __init__(
        self,
        field_dim: int,
        hidden_dim: int,
        mode: str = "image",
        region_size: int = 2,
        chunk_size: int = 2,
        num_parent_hypotheses: int | None = None,
        clip_value: float = 3.0,
        activation_temperature: float = 1.0,
        gate_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if field_dim <= 0 or hidden_dim <= 0:
            raise ValueError("field_dim and hidden_dim must be positive")
        if mode not in {"image", "text"}:
            raise ValueError("mode must be 'image' or 'text'")
        if region_size <= 0 or chunk_size <= 0:
            raise ValueError("region_size and chunk_size must be positive")
        if num_parent_hypotheses is not None and num_parent_hypotheses <= 0:
            raise ValueError("num_parent_hypotheses must be positive when provided")
        if gate_eps <= 0.0:
            raise ValueError("gate_eps must be positive")

        self.field_dim = field_dim
        self.mode = mode
        self.region_size = region_size
        self.chunk_size = chunk_size
        self.num_parent_hypotheses = num_parent_hypotheses
        self.gate_eps = float(gate_eps)
        self.parent_prototypes = (
            nn.Parameter(torch.empty(num_parent_hypotheses, field_dim)) if num_parent_hypotheses is not None else None
        )
        joint_dim = field_dim * 4
        self.joint_norm = nn.LayerNorm(joint_dim)
        self.state_net = _MLP(joint_dim, hidden_dim, field_dim, final_tanh=True)
        self.drive_net = _MLP(joint_dim, hidden_dim, 1)
        self.resistance_net = _MLP(joint_dim, hidden_dim, 1)
        self.state_norm = nn.LayerNorm(field_dim)
        self.eml = EMLUnit(dim=1, clip_value=clip_value, init_bias=0.0)
        self.activation_budget = EMLActivationBudget(temperature=activation_temperature)
        if self.parent_prototypes is not None:
            nn.init.normal_(self.parent_prototypes, mean=0.0, std=0.02)

    def _image_regions(
        self,
        hypothesis_state: torch.Tensor,
        activation: torch.Tensor,
        image_shape: tuple[int, int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_positions, num_hypotheses, field_dim = hypothesis_state.shape
        height, width = _infer_image_shape(num_positions, image_shape)
        pad_h = (self.region_size - height % self.region_size) % self.region_size
        pad_w = (self.region_size - width % self.region_size) % self.region_size
        padded_height = height + pad_h
        padded_width = width + pad_w
        region_count = self.region_size * self.region_size

        state_grid = hypothesis_state.view(batch_size, height, width, num_hypotheses, field_dim)
        state_grid = state_grid.permute(0, 3, 4, 1, 2).reshape(batch_size, num_hypotheses * field_dim, height, width)
        state_grid = F.pad(state_grid, (0, pad_w, 0, pad_h))
        state_regions = F.unfold(state_grid, kernel_size=self.region_size, stride=self.region_size)
        parent_positions = (padded_height // self.region_size) * (padded_width // self.region_size)
        state_regions = state_regions.view(
            batch_size,
            num_hypotheses,
            field_dim,
            region_count,
            parent_positions,
        ).permute(0, 4, 3, 1, 2)

        activation_grid = activation.view(batch_size, height, width, num_hypotheses).permute(0, 3, 1, 2)
        activation_grid = F.pad(activation_grid, (0, pad_w, 0, pad_h))
        activation_regions = F.unfold(activation_grid, kernel_size=self.region_size, stride=self.region_size)
        activation_regions = activation_regions.view(
            batch_size,
            num_hypotheses,
            region_count,
            parent_positions,
        ).permute(0, 3, 2, 1)

        valid_grid = torch.ones(batch_size, 1, height, width, device=hypothesis_state.device, dtype=hypothesis_state.dtype)
        valid_grid = F.pad(valid_grid, (0, pad_w, 0, pad_h))
        valid_regions = F.unfold(valid_grid, kernel_size=self.region_size, stride=self.region_size)
        valid_regions = valid_regions.transpose(1, 2).to(dtype=torch.bool)
        return state_regions, activation_regions, valid_regions

    def _text_chunks(
        self,
        hypothesis_state: torch.Tensor,
        activation: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_positions, num_hypotheses, field_dim = hypothesis_state.shape
        pad_len = (self.chunk_size - num_positions % self.chunk_size) % self.chunk_size
        padded_state = F.pad(hypothesis_state, (0, 0, 0, 0, 0, pad_len))
        padded_activation = F.pad(activation, (0, 0, 0, pad_len))
        parent_positions = (num_positions + pad_len) // self.chunk_size
        state_chunks = padded_state.view(batch_size, parent_positions, self.chunk_size, num_hypotheses, field_dim)
        activation_chunks = padded_activation.view(batch_size, parent_positions, self.chunk_size, num_hypotheses)

        valid = torch.ones(batch_size, num_positions, device=hypothesis_state.device, dtype=torch.bool)
        if padding_mask is not None:
            if padding_mask.shape != (batch_size, num_positions):
                raise ValueError("padding_mask must have shape [batch, positions]")
            valid = valid & padding_mask.bool()
        valid = F.pad(valid, (0, pad_len), value=False).view(batch_size, parent_positions, self.chunk_size)
        return state_chunks, activation_chunks, valid

    def forward(
        self,
        hypothesis_state: torch.Tensor,
        activation: torch.Tensor,
        image_shape: tuple[int, int] | None = None,
        padding_mask: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        if hypothesis_state.ndim != 4 or hypothesis_state.size(-1) != self.field_dim:
            raise ValueError("hypothesis_state must have shape [batch, positions, hypotheses, field_dim]")
        if activation.shape != hypothesis_state.shape[:3]:
            raise ValueError("activation must have shape [batch, positions, hypotheses]")

        if self.mode == "image":
            child_state, child_activation, child_mask = self._image_regions(hypothesis_state, activation, image_shape)
        else:
            child_state, child_activation, child_mask = self._text_chunks(hypothesis_state, activation, padding_mask)

        masked_activation = child_activation.to(dtype=torch.float32) * child_mask.unsqueeze(-1).to(dtype=torch.float32)
        if self.parent_prototypes is None:
            child_mass = masked_activation.sum(dim=2).clamp_min(self.gate_eps)
            child_summary = (child_state * masked_activation.unsqueeze(-1)).sum(dim=2) / child_mass.unsqueeze(-1)
            global_summary = child_summary.mean(dim=2, keepdim=True).expand_as(child_summary)
            variance = (
                (child_state - child_summary.unsqueeze(2)).square() * masked_activation.unsqueeze(-1)
            ).sum(dim=2) / child_mass.unsqueeze(-1)
            disagreement = variance.mean(dim=-1)
            expected = child_mask.to(dtype=torch.float32).sum(dim=2, keepdim=True).clamp_min(1.0)
            missing = (1.0 - (child_mass / expected).clamp(max=1.0)).to(dtype=child_summary.dtype)
            joint = torch.cat(
                [
                    child_summary,
                    global_summary,
                    child_summary * global_summary,
                    child_summary - global_summary,
                ],
                dim=-1,
            )
            mass_signal = (child_mass / expected).to(dtype=child_summary.dtype)
        else:
            child_mass = masked_activation.sum(dim=(2, 3)).clamp_min(self.gate_eps)
            child_summary = (child_state * masked_activation.unsqueeze(-1)).sum(dim=(2, 3)) / child_mass.unsqueeze(-1)
            prototypes = self.parent_prototypes.view(1, 1, self.num_parent_hypotheses, self.field_dim)
            child_summary = child_summary.unsqueeze(2).expand(-1, -1, self.num_parent_hypotheses, -1)
            prototype_features = prototypes.expand(child_summary.size(0), child_summary.size(1), -1, -1)
            variance = (
                (child_state.unsqueeze(3) - child_summary.unsqueeze(2).unsqueeze(4)).square()
                * masked_activation.unsqueeze(-1).unsqueeze(3)
            ).sum(dim=(2, 4)) / child_mass.view(child_mass.size(0), child_mass.size(1), 1, 1)
            disagreement = variance.mean(dim=-1)
            expected = (
                child_mask.to(dtype=torch.float32).sum(dim=2) * child_activation.size(-1)
            ).clamp_min(1.0)
            missing = (1.0 - (child_mass / expected).clamp(max=1.0)).unsqueeze(-1)
            missing = missing.expand(-1, -1, self.num_parent_hypotheses).to(dtype=child_summary.dtype)
            joint = torch.cat(
                [
                    child_summary,
                    prototype_features,
                    child_summary * prototype_features,
                    child_summary - prototype_features,
                ],
                dim=-1,
            )
            mass_signal = (child_mass / expected).unsqueeze(-1).expand_as(missing).to(dtype=child_summary.dtype)

        normalized_joint = self.joint_norm(joint)
        parent_state = self.state_norm(self.state_net(normalized_joint))
        parent_drive = self.drive_net(normalized_joint).squeeze(-1) + mass_signal
        parent_resistance = F.softplus(self.resistance_net(normalized_joint).squeeze(-1)) + missing + disagreement
        parent_energy = self.eml(parent_drive.unsqueeze(-1), parent_resistance.unsqueeze(-1), warmup_eta=warmup_eta).squeeze(-1)
        activation_out = self.activation_budget(parent_energy)

        return {
            "parent_state": parent_state,
            "parent_drive": parent_drive,
            "parent_resistance": parent_resistance,
            "parent_energy": parent_energy,
            "parent_activation": activation_out["activation"],
            "diagnostics": {
                "child_mass": child_mass,
                "missing_resistance": missing,
                "disagreement": disagreement,
                "budget_loss": activation_out["budget_loss"],
                "entropy": activation_out["entropy"],
                "active_rate": activation_out["active_rate"],
            },
        }


class EMLAttractorMemory(nn.Module):
    """Maintain global attractor states updated by activated hypotheses."""

    def __init__(
        self,
        field_dim: int,
        num_attractors: int,
        hidden_dim: int,
        clip_value: float = 3.0,
        activation_temperature: float = 1.0,
        gate_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if field_dim <= 0 or num_attractors <= 0 or hidden_dim <= 0:
            raise ValueError("field_dim, num_attractors, and hidden_dim must be positive")
        if gate_eps <= 0.0:
            raise ValueError("gate_eps must be positive")

        self.field_dim = field_dim
        self.num_attractors = num_attractors
        self.gate_eps = float(gate_eps)
        self.attractor_states = nn.Parameter(torch.empty(num_attractors, field_dim))
        joint_dim = field_dim * 4
        self.joint_norm = nn.LayerNorm(joint_dim)
        self.candidate_net = _MLP(joint_dim, hidden_dim, field_dim, final_tanh=True)
        self.drive_net = _MLP(joint_dim, hidden_dim, field_dim)
        self.resistance_net = _MLP(joint_dim, hidden_dim, field_dim)
        self.score_drive = _MLP(field_dim, hidden_dim, 1)
        self.score_resistance = _MLP(field_dim, hidden_dim, 1)
        self.update_gate = EMLUpdateGate(dim=field_dim, clip_value=clip_value, init_bias=-0.5)
        self.score_eml = EMLUnit(dim=1, clip_value=clip_value, init_bias=0.0)
        self.activation_budget = EMLActivationBudget(temperature=activation_temperature)
        self.state_norm = nn.LayerNorm(field_dim)
        nn.init.normal_(self.attractor_states, mean=0.0, std=0.02)

    def _flatten_evidence(
        self,
        hypothesis_state: torch.Tensor,
        activation: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hypothesis_state.ndim == 4:
            batch_size, positions, hypotheses, field_dim = hypothesis_state.shape
            if field_dim != self.field_dim:
                raise ValueError("hypothesis_state field_dim does not match this memory")
            flat_state = hypothesis_state.reshape(batch_size, positions * hypotheses, field_dim)
            if activation is None:
                flat_activation = torch.ones(batch_size, positions * hypotheses, device=hypothesis_state.device)
            elif activation.shape == hypothesis_state.shape[:3]:
                flat_activation = activation.reshape(batch_size, positions * hypotheses)
            else:
                raise ValueError("activation must have shape [batch, positions, hypotheses]")
            return flat_state, flat_activation

        if hypothesis_state.ndim == 3:
            batch_size, items, field_dim = hypothesis_state.shape
            if field_dim != self.field_dim:
                raise ValueError("hypothesis_state field_dim does not match this memory")
            if activation is None:
                flat_activation = torch.ones(batch_size, items, device=hypothesis_state.device)
            elif activation.shape == hypothesis_state.shape[:2]:
                flat_activation = activation
            else:
                raise ValueError("activation must have shape [batch, items]")
            return hypothesis_state, flat_activation
        raise ValueError("hypothesis_state must be rank 3 or 4")

    def forward(
        self,
        hypothesis_state: torch.Tensor,
        activation: torch.Tensor | None = None,
        previous_states: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        flat_state, flat_activation = self._flatten_evidence(hypothesis_state, activation)
        flat_activation = flat_activation.to(device=flat_state.device, dtype=flat_state.dtype).clamp_min(0.0)
        evidence_mass = flat_activation.sum(dim=1, keepdim=True).clamp_min(self.gate_eps)
        evidence = (flat_state * flat_activation.unsqueeze(-1)).sum(dim=1) / evidence_mass

        batch_size = flat_state.size(0)
        if previous_states is None:
            current = self.attractor_states.unsqueeze(0).expand(batch_size, -1, -1)
        elif previous_states.ndim == 2:
            if previous_states.shape != (self.num_attractors, self.field_dim):
                raise ValueError("previous_states must have shape [num_attractors, field_dim]")
            current = previous_states.unsqueeze(0).expand(batch_size, -1, -1)
        elif previous_states.shape == (batch_size, self.num_attractors, self.field_dim):
            current = previous_states
        else:
            raise ValueError("previous_states shape does not match this memory")

        evidence_features = evidence.unsqueeze(1).expand(batch_size, self.num_attractors, self.field_dim)
        joint = self.joint_norm(
            torch.cat(
                [
                    current,
                    evidence_features,
                    current * evidence_features,
                    evidence_features - current,
                ],
                dim=-1,
            )
        )
        candidate = self.candidate_net(joint)
        update_drive = self.drive_net(joint)
        update_resistance = F.softplus(self.resistance_net(joint))
        gate_out = self.update_gate(update_drive, update_resistance, warmup_eta=warmup_eta)
        gate = gate_out["gate"]
        attractor_states = self.state_norm(current + gate * (candidate - current))

        attractor_drive = self.score_drive(attractor_states).squeeze(-1)
        attractor_resistance = F.softplus(self.score_resistance(attractor_states).squeeze(-1))
        attractor_energy = self.score_eml(
            attractor_drive.unsqueeze(-1),
            attractor_resistance.unsqueeze(-1),
            warmup_eta=warmup_eta,
        ).squeeze(-1)
        activation_out = self.activation_budget(attractor_energy)

        return {
            "attractor_states": attractor_states,
            "attractor_drive": attractor_drive,
            "attractor_resistance": attractor_resistance,
            "attractor_energy": attractor_energy,
            "attractor_activation": activation_out["activation"],
            "update_gate": gate,
            "evidence": evidence,
            "evidence_mass": evidence_mass.squeeze(-1),
            "budget_loss": activation_out["budget_loss"],
        }


class EMLFieldReadout(nn.Module):
    """Read a global representation from attractor states with sEML scoring."""

    def __init__(
        self,
        field_dim: int,
        hidden_dim: int,
        representation_dim: int | None = None,
        clip_value: float = 3.0,
        temperature: float = 1.0,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if field_dim <= 0 or hidden_dim <= 0:
            raise ValueError("field_dim and hidden_dim must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.field_dim = field_dim
        self.representation_dim = representation_dim or field_dim
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.norm = nn.LayerNorm(field_dim)
        self.drive_net = _MLP(field_dim, hidden_dim, 1)
        self.resistance_net = _MLP(field_dim, hidden_dim, 1)
        self.value_proj = nn.Linear(field_dim, self.representation_dim)
        self.eml = EMLUnit(dim=1, clip_value=clip_value, init_bias=0.0)
        _reset_linear(self.value_proj)

    def forward(
        self,
        attractor_states: torch.Tensor,
        attractor_activation: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if attractor_states.ndim != 3 or attractor_states.size(-1) != self.field_dim:
            raise ValueError("attractor_states must have shape [batch, attractors, field_dim]")
        if attractor_activation is not None and attractor_activation.shape != attractor_states.shape[:2]:
            raise ValueError("attractor_activation must have shape [batch, attractors]")

        normalized = self.norm(attractor_states)
        drive = self.drive_net(normalized).squeeze(-1)
        resistance = F.softplus(self.resistance_net(normalized).squeeze(-1))
        energy = self.eml(drive.unsqueeze(-1), resistance.unsqueeze(-1), warmup_eta=warmup_eta).squeeze(-1)
        logits = energy.to(dtype=torch.float32) / self.temperature
        if attractor_activation is not None:
            activation_prior = attractor_activation.to(dtype=torch.float32).clamp_min(self.eps)
            logits = logits + activation_prior.log()
        weights = torch.softmax(logits, dim=1).to(dtype=attractor_states.dtype)
        representation = torch.sum(weights.unsqueeze(-1) * self.value_proj(normalized), dim=1)

        return {
            "representation": representation,
            "weights": weights,
            "drive": drive,
            "resistance": resistance,
            "energy": energy,
        }


__all__ = [
    "EMLAttractorMemory",
    "EMLCompositionField",
    "EMLConsensusField",
    "EMLFieldReadout",
    "EMLHypothesisCompetition",
    "EMLHypothesisField",
    "EMLSensor",
]
