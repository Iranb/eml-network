from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import EMLPrecisionUpdate, EMLResponsibility, EMLUnit, _reset_linear


def _reset_module(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            _reset_linear(child)


def _stats(prefix: str, value: torch.Tensor) -> Dict[str, torch.Tensor]:
    value = value.detach().float()
    return {
        f"{prefix}_mean": value.mean(),
        f"{prefix}_std": value.std(unbiased=False),
    }


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
        _reset_module(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EMLLocalEvidenceEncoder(nn.Module):
    """Convert local features into measurement, drive seeds, and resistance seeds."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        num_hypotheses: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or state_dim <= 0 or num_hypotheses <= 0 or hidden_dim <= 0:
            raise ValueError("all dimensions must be positive")
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.num_hypotheses = num_hypotheses
        self.norm = nn.LayerNorm(input_dim)
        self.measurement = _MLP(input_dim, hidden_dim, state_dim)
        self.drive_seed = _MLP(input_dim, hidden_dim, num_hypotheses)
        self.resistance_seed = _MLP(input_dim, hidden_dim, num_hypotheses)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 3 or x.size(-1) != self.input_dim:
            raise ValueError("x must have shape [batch, positions, input_dim]")
        normalized = self.norm(x)
        measurement = self.measurement(normalized)
        return {
            "measurement": measurement,
            "drive_seed": self.drive_seed(normalized),
            "resistance_seed": F.softplus(self.resistance_seed(normalized)),
        }


class EMLSupportConflictKernel(nn.Module):
    """Local support/conflict scorer for target-neighbor pairs."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_hypotheses: int,
        relative_dim: int,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or hidden_dim <= 0 or num_hypotheses <= 0 or relative_dim <= 0:
            raise ValueError("all dimensions must be positive")
        self.state_dim = state_dim
        self.num_hypotheses = num_hypotheses
        self.relative_dim = relative_dim
        input_dim = state_dim * 4 + relative_dim
        self.norm = nn.LayerNorm(input_dim)
        self.support = _MLP(input_dim, hidden_dim, num_hypotheses)
        self.conflict = _MLP(input_dim, hidden_dim, num_hypotheses)

    def forward(
        self,
        target: torch.Tensor,
        neighbor: torch.Tensor,
        relative_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if target.shape != neighbor.shape or target.ndim != 4:
            raise ValueError("target and neighbor must have shape [batch, positions, window, state_dim]")
        if target.size(-1) != self.state_dim:
            raise ValueError("target last dimension does not match state_dim")
        if relative_features.ndim == 2:
            relative_features = relative_features.view(1, 1, relative_features.size(0), relative_features.size(1))
            relative_features = relative_features.expand(target.size(0), target.size(1), -1, -1)
        elif relative_features.ndim == 4:
            if relative_features.shape[:3] != target.shape[:3]:
                raise ValueError("relative_features batch/position/window shape must match target")
        else:
            raise ValueError("relative_features must have shape [window, relative_dim] or [batch, positions, window, relative_dim]")
        if relative_features.size(-1) != self.relative_dim:
            raise ValueError("relative_features last dimension does not match relative_dim")

        joint = torch.cat([target, neighbor, target * neighbor, target - neighbor, relative_features], dim=-1)
        joint = self.norm(joint)
        support = self.support(joint)
        conflict = F.softplus(self.conflict(joint))
        return {"support": support, "conflict": conflict, "value": neighbor}


class EMLResponsibilityPropagation(nn.Module):
    """Local EML responsibility propagation for image or causal text states."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_hypotheses: int = 16,
        mode: str = "image",
        window_size: int = 3,
        clip_value: float = 3.0,
        responsibility_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in {"image", "text"}:
            raise ValueError("mode must be image or text")
        if state_dim <= 0 or hidden_dim <= 0 or num_hypotheses <= 0 or window_size <= 0:
            raise ValueError("invalid propagation dimensions")
        if mode == "image" and window_size % 2 == 0:
            raise ValueError("image window_size must be odd")
        self.state_dim = state_dim
        self.num_hypotheses = num_hypotheses
        self.mode = mode
        self.window_size = window_size
        self.relative_dim = 3 if mode == "image" else 2
        self.pre_norm = nn.LayerNorm(state_dim)
        self.kernel = EMLSupportConflictKernel(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_hypotheses=num_hypotheses,
            relative_dim=self.relative_dim,
        )
        self.energy = EMLUnit(dim=num_hypotheses, clip_value=clip_value, init_bias=-0.2)
        self.responsibility = EMLResponsibility(
            temperature=responsibility_temperature,
            use_null=True,
            null_logit=0.0,
        )
        self.value_proj = nn.Linear(state_dim, state_dim)
        self.candidate = _MLP(state_dim * 3, hidden_dim, state_dim, final_tanh=True)
        self.old_confidence = _MLP(state_dim, hidden_dim, 1)
        self.precision_update = EMLPrecisionUpdate(mode="precision")
        self.out_norm = nn.LayerNorm(state_dim)
        _reset_linear(self.value_proj)

    def _image_relative(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        half = self.window_size // 2
        coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        denom = max(1, half)
        yy = yy.reshape(-1) / denom
        xx = xx.reshape(-1) / denom
        dist = torch.sqrt(yy.square() + xx.square())
        return torch.stack([yy, xx, dist], dim=-1)

    def _text_relative(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        distances = torch.arange(self.window_size - 1, -1, -1, device=device, dtype=dtype)
        normalized = distances / max(1, self.window_size - 1)
        is_current = (distances == 0).to(dtype=dtype)
        return torch.stack([normalized, is_current], dim=-1)

    def _image_neighbors(
        self,
        states: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, positions, channels = states.shape
        height, width = image_shape
        if positions != height * width:
            raise ValueError("image_shape does not match number of positions")
        padding = self.window_size // 2
        grid = states.view(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()
        patches = F.unfold(grid, kernel_size=self.window_size, padding=padding)
        neighbors = patches.transpose(1, 2).contiguous().view(batch_size, positions, channels, self.window_size * self.window_size)
        neighbors = neighbors.permute(0, 1, 3, 2).contiguous()
        ones = torch.ones(batch_size, 1, height, width, device=states.device, dtype=states.dtype)
        mask = F.unfold(ones, kernel_size=self.window_size, padding=padding)
        mask = mask.transpose(1, 2).contiguous().view(batch_size, positions, self.window_size * self.window_size) > 0.0
        relative = self._image_relative(states.device, states.dtype)
        return neighbors, mask, relative

    def _text_neighbors(
        self,
        states: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = states.shape
        padded_states = F.pad(states, (0, 0, self.window_size - 1, 0))
        neighbors = torch.stack(
            [padded_states[:, offset : offset + seq_len, :] for offset in range(self.window_size)],
            dim=2,
        )
        distances = torch.arange(self.window_size - 1, -1, -1, device=states.device)
        positions = torch.arange(seq_len, device=states.device).unsqueeze(-1)
        mask = positions >= distances.unsqueeze(0)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        if padding_mask is not None:
            padded_mask = F.pad(padding_mask.bool(), (self.window_size - 1, 0), value=False)
            source_mask = torch.stack(
                [padded_mask[:, offset : offset + seq_len] for offset in range(self.window_size)],
                dim=2,
            )
            mask = mask & source_mask & padding_mask.bool().unsqueeze(-1)
        relative = self._text_relative(states.device, states.dtype)
        return neighbors, mask, relative

    def forward(
        self,
        states: torch.Tensor,
        drive_seed: torch.Tensor | None = None,
        resistance_seed: torch.Tensor | None = None,
        image_shape: tuple[int, int] | None = None,
        padding_mask: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if states.ndim != 3 or states.size(-1) != self.state_dim:
            raise ValueError("states must have shape [batch, positions, state_dim]")
        if drive_seed is not None and drive_seed.shape != states.shape[:2] + (self.num_hypotheses,):
            raise ValueError("drive_seed must have shape [batch, positions, num_hypotheses]")
        if resistance_seed is not None and resistance_seed.shape != states.shape[:2] + (self.num_hypotheses,):
            raise ValueError("resistance_seed must have shape [batch, positions, num_hypotheses]")

        normalized = self.pre_norm(states)
        if self.mode == "image":
            if image_shape is None:
                raise ValueError("image_shape is required for image propagation")
            neighbors, edge_mask, relative = self._image_neighbors(normalized, image_shape)
        else:
            neighbors, edge_mask, relative = self._text_neighbors(normalized, padding_mask)

        target = normalized.unsqueeze(2).expand_as(neighbors)
        support_conflict = self.kernel(target, neighbors, relative)
        drive = support_conflict["support"]
        resistance = support_conflict["conflict"]
        if drive_seed is not None:
            drive = drive + drive_seed.unsqueeze(2)
        if resistance_seed is not None:
            resistance = resistance + resistance_seed.unsqueeze(2)

        energy_heads = self.energy(drive, resistance, warmup_eta=warmup_eta)
        edge_energy = energy_heads.mean(dim=-1)
        responsibility_out = self.responsibility(edge_energy, mask=edge_mask)
        weights = responsibility_out["neighbor_weights"]
        values = self.value_proj(neighbors)
        message = (weights.unsqueeze(-1) * values).sum(dim=2)
        update_strength = responsibility_out["update_strength"]
        message = message * update_strength.unsqueeze(-1)

        candidate_input = torch.cat([normalized, message, normalized * message], dim=-1)
        candidate = self.candidate(candidate_input)
        old_confidence = self.old_confidence(normalized)
        new_energy = (weights * edge_energy).sum(dim=-1, keepdim=True)
        update_out = self.precision_update(
            state=states,
            candidate=candidate,
            new_energy=new_energy,
            old_confidence=old_confidence,
            update_strength=update_strength,
        )
        state_new = self.out_norm(update_out["state"])
        if padding_mask is not None and self.mode == "text":
            state_new = torch.where(padding_mask.unsqueeze(-1), state_new, states)

        diagnostics: Dict[str, torch.Tensor] = {}
        diagnostics.update(_stats("drive", drive))
        diagnostics.update(_stats("resistance", resistance))
        diagnostics.update(_stats("energy", energy_heads))
        diagnostics.update(_stats("null_weight", responsibility_out["null_weight_fp32"]))
        diagnostics.update(_stats("update_strength", responsibility_out["update_strength_fp32"]))
        diagnostics.update(_stats("responsibility_entropy", responsibility_out["entropy"]))
        diagnostics.update(_stats("message_norm", message.norm(dim=-1)))
        diagnostics.update(_stats("update_norm", (state_new - states).norm(dim=-1)))
        return {
            "state": state_new,
            "state_new": state_new,
            "message": message,
            "responsibility_weights": weights,
            "neighbor_weights": weights,
            "null_weight": responsibility_out["null_weight"],
            "update_strength": update_strength,
            "drive": drive,
            "resistance": resistance,
            "energy": energy_heads,
            "edge_energy": edge_energy,
            "edge_mask": edge_mask,
            "update_gate": update_out["update_gate"],
            "diagnostics": diagnostics,
        }


class EMLComposition(nn.Module):
    """Compose local states into parent states using EML consistency."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        mode: str = "image",
        region_size: int = 2,
        clip_value: float = 3.0,
    ) -> None:
        super().__init__()
        if mode not in {"image", "text"}:
            raise ValueError("mode must be image or text")
        if state_dim <= 0 or hidden_dim <= 0 or region_size <= 0:
            raise ValueError("invalid composition dimensions")
        self.state_dim = state_dim
        self.mode = mode
        self.region_size = region_size
        self.child_norm = nn.LayerNorm(state_dim)
        self.parent_seed = nn.Parameter(torch.empty(1, 1, state_dim))
        self.drive_net = _MLP(state_dim * 3, hidden_dim, 1)
        self.resistance_net = _MLP(state_dim * 3, hidden_dim, 1)
        self.value_proj = nn.Linear(state_dim, state_dim)
        self.energy = EMLUnit(dim=1, clip_value=clip_value, init_bias=-0.1)
        self.responsibility = EMLResponsibility(use_null=True, null_logit=0.0)
        self.precision_update = EMLPrecisionUpdate(mode="precision")
        self.out_norm = nn.LayerNorm(state_dim)
        nn.init.normal_(self.parent_seed, mean=0.0, std=0.02)
        _reset_linear(self.value_proj)

    def _image_groups(
        self,
        states: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        batch_size, positions, channels = states.shape
        height, width = image_shape
        if positions != height * width:
            raise ValueError("image_shape does not match number of positions")
        parent_h = max(1, height // self.region_size)
        parent_w = max(1, width // self.region_size)
        keep_h = parent_h * self.region_size
        keep_w = parent_w * self.region_size
        grid = states.view(batch_size, height, width, channels)[:, :keep_h, :keep_w, :]
        groups = grid.view(batch_size, parent_h, self.region_size, parent_w, self.region_size, channels)
        groups = groups.permute(0, 1, 3, 2, 4, 5).contiguous()
        groups = groups.view(batch_size, parent_h * parent_w, self.region_size * self.region_size, channels)
        mask = torch.ones(groups.shape[:3], device=states.device, dtype=torch.bool)
        return groups, mask, (parent_h, parent_w)

    def _text_groups(
        self,
        states: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        batch_size, seq_len, channels = states.shape
        chunks = (seq_len + self.region_size - 1) // self.region_size
        padded_len = chunks * self.region_size
        pad_len = padded_len - seq_len
        if pad_len:
            states = F.pad(states, (0, 0, 0, pad_len))
            if padding_mask is None:
                mask = torch.ones(batch_size, seq_len, device=states.device, dtype=torch.bool)
            else:
                mask = padding_mask.bool()
            mask = F.pad(mask, (0, pad_len), value=False)
        else:
            mask = torch.ones(batch_size, seq_len, device=states.device, dtype=torch.bool) if padding_mask is None else padding_mask.bool()
        groups = states.view(batch_size, chunks, self.region_size, channels)
        group_mask = mask.view(batch_size, chunks, self.region_size)
        return groups, group_mask, (chunks, 1)

    def forward(
        self,
        states: torch.Tensor,
        image_shape: tuple[int, int] | None = None,
        padding_mask: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor | tuple[int, int] | Dict[str, torch.Tensor]]:
        if states.ndim != 3 or states.size(-1) != self.state_dim:
            raise ValueError("states must have shape [batch, positions, state_dim]")
        if self.mode == "image":
            if image_shape is None:
                raise ValueError("image_shape is required for image composition")
            groups, child_mask, parent_shape = self._image_groups(states, image_shape)
        else:
            groups, child_mask, parent_shape = self._text_groups(states, padding_mask)

        normalized = self.child_norm(groups)
        mask_f = child_mask.unsqueeze(-1).to(dtype=states.dtype)
        group_mean = (normalized * mask_f).sum(dim=2, keepdim=True) / mask_f.sum(dim=2, keepdim=True).clamp_min(1.0)
        joint = torch.cat([normalized, group_mean.expand_as(normalized), normalized * group_mean.expand_as(normalized)], dim=-1)
        drive = self.drive_net(joint)
        variance = (normalized - group_mean).square().mean(dim=-1, keepdim=True)
        resistance = F.softplus(self.resistance_net(joint)) + variance
        energy = self.energy(drive, resistance, warmup_eta=warmup_eta).squeeze(-1)
        responsibility_out = self.responsibility(energy, mask=child_mask)
        weights = responsibility_out["neighbor_weights"]
        candidate = (weights.unsqueeze(-1) * self.value_proj(normalized)).sum(dim=2)
        old_state = self.parent_seed.expand(states.size(0), candidate.size(1), -1)
        new_energy = (weights * energy).sum(dim=-1, keepdim=True)
        update_out = self.precision_update(
            state=old_state,
            candidate=candidate,
            new_energy=new_energy,
            old_confidence=torch.zeros_like(new_energy),
            update_strength=responsibility_out["update_strength"],
        )
        parent_state = self.out_norm(update_out["state"])
        diagnostics: Dict[str, torch.Tensor] = {}
        diagnostics.update(_stats("parent_drive", drive))
        diagnostics.update(_stats("parent_resistance", resistance))
        diagnostics.update(_stats("parent_energy", energy))
        diagnostics.update(_stats("parent_activation", responsibility_out["update_strength_fp32"]))
        return {
            "parent_state": parent_state,
            "parent_states": parent_state,
            "parent_drive": drive.squeeze(-1),
            "parent_resistance": resistance.squeeze(-1),
            "parent_energy": energy,
            "parent_activation": responsibility_out["update_strength"],
            "parent_shape": parent_shape,
            "child_weights": weights,
            "null_weight": responsibility_out["null_weight"],
            "diagnostics": diagnostics,
        }


class EMLAttractorMemory(nn.Module):
    """Small fixed attractor memory with EML responsibility assignment."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_attractors: int = 4,
        clip_value: float = 3.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or hidden_dim <= 0 or num_attractors <= 0:
            raise ValueError("invalid attractor dimensions")
        self.state_dim = state_dim
        self.num_attractors = num_attractors
        self.attractors = nn.Parameter(torch.empty(num_attractors, state_dim))
        self.state_norm = nn.LayerNorm(state_dim)
        self.attractor_norm = nn.LayerNorm(state_dim)
        self.drive_net = _MLP(state_dim * 4, hidden_dim, 1)
        self.resistance_net = _MLP(state_dim * 4, hidden_dim, 1)
        self.value_proj = nn.Linear(state_dim, state_dim)
        self.energy = EMLUnit(dim=1, clip_value=clip_value, init_bias=-0.1)
        self.responsibility = EMLResponsibility(use_null=True, null_logit=0.0)
        self.old_confidence = _MLP(state_dim, hidden_dim, 1)
        self.precision_update = EMLPrecisionUpdate(mode="precision")
        self.out_norm = nn.LayerNorm(state_dim)
        nn.init.normal_(self.attractors, mean=0.0, std=0.02)
        _reset_linear(self.value_proj)

    def forward(
        self,
        states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        if states.ndim != 3 or states.size(-1) != self.state_dim:
            raise ValueError("states must have shape [batch, positions, state_dim]")
        batch_size, positions, _ = states.shape
        normalized_states = self.state_norm(states)
        attractors = self.attractors.unsqueeze(0).expand(batch_size, -1, -1)
        normalized_attractors = self.attractor_norm(attractors)
        target = normalized_attractors.unsqueeze(2).expand(-1, -1, positions, -1)
        evidence = normalized_states.unsqueeze(1).expand(-1, self.num_attractors, -1, -1)
        joint = torch.cat([target, evidence, target * evidence, target - evidence], dim=-1)
        drive = self.drive_net(joint)
        resistance = F.softplus(self.resistance_net(joint))
        energy = self.energy(drive, resistance, warmup_eta=warmup_eta).squeeze(-1)
        mask = None if padding_mask is None else padding_mask.bool().unsqueeze(1).expand(-1, self.num_attractors, -1)
        responsibility_out = self.responsibility(energy, mask=mask)
        weights = responsibility_out["neighbor_weights"]
        values = self.value_proj(normalized_states)
        message = torch.einsum("ban,bnc->bac", weights, values)
        new_energy = (weights * energy).sum(dim=-1, keepdim=True)
        old_confidence = self.old_confidence(normalized_attractors)
        update_out = self.precision_update(
            state=attractors,
            candidate=message,
            new_energy=new_energy,
            old_confidence=old_confidence,
            update_strength=responsibility_out["update_strength"],
        )
        attractor_states = self.out_norm(update_out["state"])
        normalized = F.normalize(attractor_states, dim=-1)
        diversity = normalized @ normalized.transpose(1, 2)
        off_diag = diversity.masked_select(~torch.eye(self.num_attractors, device=states.device, dtype=torch.bool).unsqueeze(0))
        diversity_penalty = off_diag.square().mean() if off_diag.numel() else states.new_tensor(0.0)
        diagnostics: Dict[str, torch.Tensor] = {}
        diagnostics.update(_stats("attractor_drive", drive))
        diagnostics.update(_stats("attractor_resistance", resistance))
        diagnostics.update(_stats("attractor_energy", energy))
        diagnostics.update(_stats("attractor_null_weight", responsibility_out["null_weight_fp32"]))
        diagnostics.update(_stats("attractor_update_strength", responsibility_out["update_strength_fp32"]))
        diagnostics["attractor_diversity_penalty"] = diversity_penalty
        return {
            "attractor_states": attractor_states,
            "attractor_weights": weights,
            "null_weight": responsibility_out["null_weight"],
            "update_strength": responsibility_out["update_strength"],
            "drive": drive.squeeze(-1),
            "resistance": resistance.squeeze(-1),
            "energy": energy,
            "message": message,
            "update_gate": update_out["update_gate"],
            "diagnostics": diagnostics,
        }


class EMLRepresentationReadout(nn.Module):
    """Read a global representation from attractor states."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        representation_dim: int | None = None,
        clip_value: float = 3.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or hidden_dim <= 0:
            raise ValueError("state_dim and hidden_dim must be positive")
        self.state_dim = state_dim
        self.representation_dim = representation_dim or state_dim
        self.norm = nn.LayerNorm(state_dim)
        self.drive_net = _MLP(state_dim, hidden_dim, 1)
        self.resistance_net = _MLP(state_dim, hidden_dim, 1)
        self.value_proj = nn.Linear(state_dim, self.representation_dim)
        self.energy = EMLUnit(dim=1, clip_value=clip_value, init_bias=0.0)
        self.responsibility = EMLResponsibility(use_null=False)
        _reset_linear(self.value_proj)

    def forward(
        self,
        attractor_states: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if attractor_states.ndim != 3 or attractor_states.size(-1) != self.state_dim:
            raise ValueError("attractor_states must have shape [batch, attractors, state_dim]")
        normalized = self.norm(attractor_states)
        drive = self.drive_net(normalized)
        resistance = F.softplus(self.resistance_net(normalized))
        energy = self.energy(drive, resistance, warmup_eta=warmup_eta).squeeze(-1)
        responsibility_out = self.responsibility(energy)
        weights = responsibility_out["neighbor_weights"]
        representation = (weights.unsqueeze(-1) * self.value_proj(normalized)).sum(dim=1)
        diagnostics: Dict[str, torch.Tensor] = {}
        diagnostics.update(_stats("readout_drive", drive))
        diagnostics.update(_stats("readout_resistance", resistance))
        diagnostics.update(_stats("readout_energy", energy))
        diagnostics.update(_stats("readout_weight", weights))
        return {
            "representation": representation,
            "readout_weights": weights,
            "weights": weights,
            "drive": drive.squeeze(-1),
            "resistance": resistance.squeeze(-1),
            "energy": energy,
            "diagnostics": diagnostics,
        }


__all__ = [
    "EMLAttractorMemory",
    "EMLComposition",
    "EMLLocalEvidenceEncoder",
    "EMLRepresentationReadout",
    "EMLResponsibilityPropagation",
    "EMLSupportConflictKernel",
]
