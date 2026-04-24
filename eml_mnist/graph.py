from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict

import torch
import torch.nn as nn

from .primitives import EMLGate, EMLMessageGate, EMLUpdateGate, _reset_linear


def _normalize_slot_layout(slot_layout: Mapping[str, int] | Sequence[str]) -> dict[str, int]:
    if isinstance(slot_layout, Mapping):
        layout = {str(name): int(count) for name, count in slot_layout.items()}
    else:
        layout = {str(name): 1 for name in slot_layout}
    if not layout:
        raise ValueError("slot_layout must define at least one slot type")
    if any(count <= 0 for count in layout.values()):
        raise ValueError("slot counts must be positive")
    return layout


def _gather_along_slots(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_index = indices.unsqueeze(-1).expand(*indices.shape, values.size(-1))
    return values.gather(1, gather_index)


def _scatter_along_slots(base: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    scatter_index = indices.unsqueeze(-1).expand_as(updates)
    scattered = base.clone()
    scattered.scatter_(1, scatter_index, updates)
    return scattered


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


class SlotBank(nn.Module):
    """Typed slot memory with learnable slot states and type embeddings."""

    def __init__(
        self,
        slot_dim: int,
        slot_layout: Mapping[str, int] | Sequence[str],
        init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        if slot_dim <= 0:
            raise ValueError("slot_dim must be positive")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")

        layout = _normalize_slot_layout(slot_layout)
        self.slot_dim = slot_dim
        self.slot_layout = layout
        self.slot_type_names = tuple(layout.keys())
        self.num_slot_types = len(self.slot_type_names)
        self.num_slots = sum(layout.values())

        slot_type_ids = []
        for type_id, type_name in enumerate(self.slot_type_names):
            slot_type_ids.extend([type_id] * layout[type_name])
        self.register_buffer("slot_type_ids", torch.tensor(slot_type_ids, dtype=torch.long))

        self.slot_states = nn.Parameter(torch.empty(self.num_slots, slot_dim))
        self.type_embeddings = nn.Embedding(self.num_slot_types, slot_dim)
        self.init_scale = init_scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.slot_states, mean=0.0, std=self.init_scale)
        nn.init.normal_(self.type_embeddings.weight, mean=0.0, std=self.init_scale)

    def compose(self, slot_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        if slot_states.ndim != 3:
            raise ValueError("slot_states must have shape [batch, num_slots, slot_dim]")
        if slot_states.size(1) != self.num_slots or slot_states.size(2) != self.slot_dim:
            raise ValueError("slot_states shape does not match this SlotBank")

        batch_size = slot_states.size(0)
        type_ids = self.slot_type_ids.to(device=slot_states.device).unsqueeze(0).expand(batch_size, -1)
        type_features = self.type_embeddings(type_ids)
        typed_states = slot_states + type_features
        slot_mask = torch.ones(batch_size, self.num_slots, device=slot_states.device, dtype=torch.bool)
        return {
            "slot_states": slot_states,
            "typed_states": typed_states,
            "type_ids": type_ids,
            "type_features": type_features,
            "slot_mask": slot_mask,
        }

    def forward(
        self,
        batch_size: int | None = None,
        slot_states: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if slot_states is None:
            if batch_size is None:
                raise ValueError("batch_size is required when slot_states is not provided")
            slot_states = self.slot_states.unsqueeze(0).expand(batch_size, -1, -1)
        elif batch_size is not None and slot_states.size(0) != batch_size:
            raise ValueError("batch_size does not match slot_states")
        return self.compose(slot_states)

    def gather(self, bank: Dict[str, torch.Tensor], indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        if indices.ndim != 2:
            raise ValueError("indices must have shape [batch, k]")
        return {
            "indices": indices,
            "slot_states": _gather_along_slots(bank["slot_states"], indices),
            "typed_states": _gather_along_slots(bank["typed_states"], indices),
            "type_features": _gather_along_slots(bank["type_features"], indices),
            "type_ids": bank["type_ids"].gather(1, indices),
            "slot_mask": _gather_along_slots(bank["slot_mask"].unsqueeze(-1).float(), indices).squeeze(-1).bool(),
        }

    def scatter(
        self,
        slot_states: torch.Tensor,
        indices: torch.Tensor,
        updates: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.compose(_scatter_along_slots(slot_states, indices, updates))


class EMLSparseRouter(nn.Module):
    """Route an event to the top-k active typed slots using sEML gating."""

    def __init__(
        self,
        slot_dim: int,
        event_dim: int,
        hidden_dim: int,
        top_k: int,
        clip_value: float = 3.0,
        gate_bias: float = -1.0,
    ) -> None:
        super().__init__()
        if slot_dim <= 0 or event_dim <= 0 or hidden_dim <= 0:
            raise ValueError("slot_dim, event_dim, and hidden_dim must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self.slot_dim = slot_dim
        self.event_dim = event_dim
        self.top_k = top_k

        self.event_proj = nn.Linear(event_dim, slot_dim)
        self.slot_proj = nn.Linear(slot_dim, slot_dim)
        self.norm = nn.LayerNorm(slot_dim * 3)
        self.drive_net = _MLP(slot_dim * 3, hidden_dim, 1)
        self.resistance_net = _MLP(slot_dim * 3, hidden_dim, 1)
        self.gate = EMLGate(dim=1, clip_value=clip_value, init_bias=gate_bias)
        _reset_linear(self.event_proj)
        _reset_linear(self.slot_proj)

    def forward(
        self,
        event: torch.Tensor,
        slot_states: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
        top_k: int | None = None,
        slot_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if event.ndim != 2:
            raise ValueError("event must have shape [batch, event_dim]")
        if slot_states.ndim != 3:
            raise ValueError("slot_states must have shape [batch, num_slots, slot_dim]")
        if event.size(0) != slot_states.size(0):
            raise ValueError("event and slot_states batch sizes must match")
        if slot_states.size(-1) != self.slot_dim:
            raise ValueError("slot_states last dimension does not match slot_dim")

        batch_size, num_slots, _ = slot_states.shape
        effective_top_k = min(top_k or self.top_k, num_slots)

        event_features = self.event_proj(event).unsqueeze(1).expand(batch_size, num_slots, -1)
        slot_features = self.slot_proj(slot_states)
        interaction = event_features * slot_features
        joint = self.norm(torch.cat([event_features, slot_features, interaction], dim=-1))

        drive = self.drive_net(joint)
        resistance = self.resistance_net(joint)
        gate_out = self.gate(drive, resistance, warmup_eta=warmup_eta)
        gate = gate_out["gate"].squeeze(-1)
        energy = gate_out["energy"].squeeze(-1)
        drive = gate_out["drive"].squeeze(-1)
        resistance = gate_out["resistance"].squeeze(-1)

        if slot_mask is not None:
            if slot_mask.shape != gate.shape:
                raise ValueError("slot_mask must have shape [batch, num_slots]")
            masked_gate = gate.masked_fill(~slot_mask, -1.0)
        else:
            masked_gate = gate

        topk_scores, topk_indices = torch.topk(masked_gate, k=effective_top_k, dim=-1, largest=True, sorted=True)
        active_mask = torch.zeros_like(gate, dtype=torch.bool)
        active_mask.scatter_(1, topk_indices, True)
        if slot_mask is not None:
            active_mask &= slot_mask

        active_states = _gather_along_slots(slot_states, topk_indices)

        return {
            "energy": energy,
            "gate": gate,
            "drive": drive,
            "resistance": resistance,
            "topk_indices": topk_indices,
            "topk_scores": topk_scores,
            "active_mask": active_mask,
            "active_states": active_states,
        }


class EMLMessagePassing(nn.Module):
    """Sparse sEML-gated message passing over only the active slot subgraph."""

    def __init__(
        self,
        slot_dim: int,
        event_dim: int,
        hidden_dim: int,
        clip_value: float = 3.0,
        gate_bias: float = -0.5,
        gate_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if slot_dim <= 0 or event_dim <= 0 or hidden_dim <= 0:
            raise ValueError("slot_dim, event_dim, and hidden_dim must be positive")
        if gate_eps <= 0.0:
            raise ValueError("gate_eps must be positive")

        self.slot_dim = slot_dim
        self.event_dim = event_dim
        self.gate_eps = float(gate_eps)

        self.event_proj = nn.Linear(event_dim, slot_dim)
        self.source_proj = nn.Linear(slot_dim, slot_dim)
        self.target_proj = nn.Linear(slot_dim, slot_dim)
        self.value_proj = nn.Linear(slot_dim, slot_dim)
        self.norm = nn.LayerNorm(slot_dim * 6)
        self.drive_net = _MLP(slot_dim * 6, hidden_dim, 1)
        self.resistance_net = _MLP(slot_dim * 6, hidden_dim, 1)
        self.gate = EMLMessageGate(dim=1, clip_value=clip_value, init_bias=gate_bias)
        _reset_linear(self.event_proj)
        _reset_linear(self.source_proj)
        _reset_linear(self.target_proj)
        _reset_linear(self.value_proj)

    def forward(
        self,
        active_slot_states: torch.Tensor,
        event: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
        active_indices: torch.Tensor | None = None,
        active_type_features: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if active_slot_states.ndim != 3:
            raise ValueError("active_slot_states must have shape [batch, active_slots, slot_dim]")
        if event.ndim != 2:
            raise ValueError("event must have shape [batch, event_dim]")
        if active_slot_states.size(0) != event.size(0):
            raise ValueError("batch sizes must match")
        if active_slot_states.size(-1) != self.slot_dim:
            raise ValueError("active_slot_states last dimension does not match slot_dim")

        batch_size, active_slots, _ = active_slot_states.shape
        if active_indices is None:
            active_indices = torch.arange(active_slots, device=active_slot_states.device).unsqueeze(0).expand(batch_size, -1)
        if active_type_features is None:
            active_type_features = torch.zeros_like(active_slot_states)

        if edge_mask is None:
            eye = torch.eye(active_slots, device=active_slot_states.device, dtype=torch.bool)
            edge_mask = (~eye).unsqueeze(0).expand(batch_size, -1, -1)
        elif edge_mask.shape != (batch_size, active_slots, active_slots):
            raise ValueError("edge_mask must have shape [batch, active_slots, active_slots]")
        else:
            edge_mask = edge_mask.bool()

        event_features = self.event_proj(event).view(batch_size, 1, 1, self.slot_dim).expand(batch_size, active_slots, active_slots, -1)
        source_states = self.source_proj(active_slot_states).unsqueeze(1).expand(batch_size, active_slots, active_slots, -1)
        target_states = self.target_proj(active_slot_states).unsqueeze(2).expand(batch_size, active_slots, active_slots, -1)
        source_types = active_type_features.unsqueeze(1).expand(batch_size, active_slots, active_slots, -1)
        target_types = active_type_features.unsqueeze(2).expand(batch_size, active_slots, active_slots, -1)
        relation = source_states - target_states

        joint = self.norm(torch.cat([target_states, source_states, event_features, relation, target_types, source_types], dim=-1))
        drive = self.drive_net(joint)
        resistance = self.resistance_net(joint)
        gate_out = self.gate(drive, resistance, warmup_eta=warmup_eta)
        gate = gate_out["gate"].squeeze(-1)
        gate = gate.masked_fill(~edge_mask, 0.0)

        values = self.value_proj(active_slot_states)
        messages = gate.unsqueeze(-1) * values.unsqueeze(1)
        gate_mass = gate.sum(dim=2, keepdim=True).clamp_min(self.gate_eps)
        aggregated_messages = messages.sum(dim=2) / gate_mass

        target_indices = active_indices.unsqueeze(2).expand(-1, -1, active_slots)
        source_indices = active_indices.unsqueeze(1).expand(-1, active_slots, -1)

        return {
            "drive": gate_out["drive"].squeeze(-1),
            "resistance": gate_out["resistance"].squeeze(-1),
            "energy": gate_out["energy"].squeeze(-1),
            "gate": gate,
            "gate_mass": gate_mass,
            "edge_mask": edge_mask,
            "messages": messages,
            "aggregated_messages": aggregated_messages,
            "active_indices": active_indices,
            "target_indices": target_indices,
            "source_indices": source_indices,
        }


class EMLStateUpdateCell(nn.Module):
    """Recurrent slot update cell with an sEML update gate."""

    def __init__(
        self,
        slot_dim: int,
        event_dim: int,
        hidden_dim: int,
        message_dim: int | None = None,
        clip_value: float = 3.0,
        gate_bias: float = -1.0,
    ) -> None:
        super().__init__()
        if slot_dim <= 0 or event_dim <= 0 or hidden_dim <= 0:
            raise ValueError("slot_dim, event_dim, and hidden_dim must be positive")

        self.slot_dim = slot_dim
        self.event_dim = event_dim
        self.message_dim = message_dim or slot_dim
        input_dim = slot_dim + self.message_dim + event_dim

        self.norm = nn.LayerNorm(input_dim)
        self.candidate = _MLP(input_dim, hidden_dim, slot_dim, final_tanh=True)
        self.drive_net = _MLP(input_dim, hidden_dim, slot_dim)
        self.resistance_net = _MLP(input_dim, hidden_dim, slot_dim)
        self.update_gate = EMLUpdateGate(dim=slot_dim, clip_value=clip_value, init_bias=gate_bias)
        self.out_norm = nn.LayerNorm(slot_dim)

    def forward(
        self,
        slot_states: torch.Tensor,
        message: torch.Tensor,
        event: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
        update_gate_scale: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if slot_states.ndim != 3 or message.ndim != 3:
            raise ValueError("slot_states and message must have shape [batch, slots, dim]")
        if event.ndim != 2:
            raise ValueError("event must have shape [batch, event_dim]")
        if slot_states.size(0) != message.size(0) or slot_states.size(0) != event.size(0):
            raise ValueError("batch sizes must match")
        if slot_states.size(1) != message.size(1):
            raise ValueError("slot_states and message must have the same number of slots")
        if slot_states.size(2) != self.slot_dim or message.size(2) != self.message_dim:
            raise ValueError("slot_states/message dimensions do not match this EMLStateUpdateCell")

        event_features = event.unsqueeze(1).expand(slot_states.size(0), slot_states.size(1), -1)
        joint = self.norm(torch.cat([slot_states, message, event_features], dim=-1))
        candidate = self.candidate(joint)
        drive = self.drive_net(joint)
        resistance = self.resistance_net(joint)
        gate_out = self.update_gate(drive, resistance, warmup_eta=warmup_eta)
        update_gate = gate_out["gate"]
        if update_gate_scale is not None:
            if update_gate_scale.ndim == 2:
                update_gate_scale = update_gate_scale.unsqueeze(-1)
            if update_gate_scale.shape != update_gate.shape[:2] + (1,):
                raise ValueError("update_gate_scale must have shape [batch, slots] or [batch, slots, 1]")
            update_gate = update_gate * update_gate_scale.to(device=update_gate.device, dtype=update_gate.dtype)
        updated_slot_states = self.out_norm(slot_states + update_gate * (candidate - slot_states))

        return {
            "slot_states": updated_slot_states,
            "candidate": candidate,
            "drive": gate_out["drive"],
            "resistance": gate_out["resistance"],
            "energy": gate_out["energy"],
            "update_gate": update_gate,
        }


class EMLUpdateCell(EMLStateUpdateCell):
    """Backward-compatible alias for the recurrent slot update cell."""


class EMLSlotGraphLayer(nn.Module):
    """Route -> sparse message pass -> recurrent update -> scatter back."""

    def __init__(
        self,
        slot_dim: int,
        event_dim: int,
        hidden_dim: int,
        top_k: int,
        clip_value: float = 3.0,
        router_bias: float = -1.0,
        message_bias: float = -0.5,
        update_bias: float = -1.0,
        modulate_messages_by_route: bool = True,
        modulate_updates_by_route: bool = False,
    ) -> None:
        super().__init__()
        self.modulate_messages_by_route = modulate_messages_by_route
        self.modulate_updates_by_route = modulate_updates_by_route
        self.router = EMLSparseRouter(
            slot_dim=slot_dim,
            event_dim=event_dim,
            hidden_dim=hidden_dim,
            top_k=top_k,
            clip_value=clip_value,
            gate_bias=router_bias,
        )
        self.message_passing = EMLMessagePassing(
            slot_dim=slot_dim,
            event_dim=event_dim,
            hidden_dim=hidden_dim,
            clip_value=clip_value,
            gate_bias=message_bias,
        )
        self.state_update = EMLStateUpdateCell(
            slot_dim=slot_dim,
            event_dim=event_dim,
            hidden_dim=hidden_dim,
            message_dim=slot_dim,
            clip_value=clip_value,
            gate_bias=update_bias,
        )

    def forward(
        self,
        event: torch.Tensor,
        slot_states: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
        type_features: torch.Tensor | None = None,
        slot_mask: torch.Tensor | None = None,
        top_k: int | None = None,
    ) -> Dict[str, Any]:
        if slot_states.ndim != 3:
            raise ValueError("slot_states must have shape [batch, num_slots, slot_dim]")
        if event.ndim != 2:
            raise ValueError("event must have shape [batch, event_dim]")
        if slot_states.size(0) != event.size(0):
            raise ValueError("batch sizes must match")

        if type_features is None:
            type_features = torch.zeros_like(slot_states)
        elif type_features.shape != slot_states.shape:
            raise ValueError("type_features must match slot_states shape")

        typed_states = slot_states + type_features
        router_out = self.router(
            event=event,
            slot_states=typed_states,
            warmup_eta=warmup_eta,
            top_k=top_k,
            slot_mask=slot_mask,
        )
        active_indices = router_out["topk_indices"]
        active_slot_states = _gather_along_slots(slot_states, active_indices)
        active_type_features = _gather_along_slots(type_features, active_indices)
        active_typed_states = active_slot_states + active_type_features

        message_out = self.message_passing(
            active_slot_states=active_typed_states,
            active_type_features=active_type_features,
            active_indices=active_indices,
            event=event,
            warmup_eta=warmup_eta,
        )
        active_route_strength = router_out["gate"].gather(1, active_indices)
        routed_message = message_out["aggregated_messages"]
        if self.modulate_messages_by_route:
            routed_message = routed_message * active_route_strength.unsqueeze(-1)
        update_out = self.state_update(
            slot_states=active_slot_states,
            message=routed_message,
            event=event,
            warmup_eta=warmup_eta,
            update_gate_scale=active_route_strength if self.modulate_updates_by_route else None,
        )

        updated_slot_states = _scatter_along_slots(slot_states, active_indices, update_out["slot_states"])
        updated_typed_states = updated_slot_states + type_features

        return {
            "slot_states": updated_slot_states,
            "typed_states": updated_typed_states,
            "type_features": type_features,
            "active_indices": active_indices,
            "active_mask": router_out["active_mask"],
            "active_route_strength": active_route_strength,
            "active_slot_states_before": active_slot_states,
            "active_slot_states_after": update_out["slot_states"],
            "router": router_out,
            "message_passing": message_out,
            "state_update": update_out,
        }


__all__ = [
    "EMLMessagePassing",
    "EMLSparseRouter",
    "EMLStateUpdateCell",
    "EMLUpdateCell",
    "EMLSlotGraphLayer",
    "SlotBank",
]
