from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from eml_mnist.graph import EMLMessagePassing, EMLSlotGraphLayer


def test_graph_message_gate_normalization() -> None:
    message_passing = EMLMessagePassing(slot_dim=8, event_dim=8, hidden_dim=16)
    active_slot_states = torch.randn(2, 4, 8)
    event = torch.randn(2, 8)

    out = message_passing(active_slot_states=active_slot_states, event=event, warmup_eta=0.5)

    assert out["aggregated_messages"].shape == (2, 4, 8)
    assert out["gate_mass"].shape == (2, 4, 1)
    assert torch.isfinite(out["aggregated_messages"]).all()
    assert torch.isfinite(out["gate_mass"]).all()


def test_message_aggregation_no_nan_when_all_edges_masked() -> None:
    message_passing = EMLMessagePassing(slot_dim=8, event_dim=8, hidden_dim=16)
    active_slot_states = torch.randn(2, 3, 8)
    event = torch.randn(2, 8)
    edge_mask = torch.zeros(2, 3, 3, dtype=torch.bool)

    out = message_passing(
        active_slot_states=active_slot_states,
        event=event,
        edge_mask=edge_mask,
        warmup_eta=0.5,
    )

    assert torch.isfinite(out["aggregated_messages"]).all()
    assert torch.allclose(out["aggregated_messages"], torch.zeros_like(out["aggregated_messages"]))
    assert torch.isfinite(out["gate_mass"]).all()


def test_changing_top_k_does_not_explosively_change_message_scale() -> None:
    message_passing = EMLMessagePassing(slot_dim=8, event_dim=8, hidden_dim=16)
    active_slot_states = torch.randn(2, 6, 8)
    event = torch.randn(2, 8)

    small = message_passing(active_slot_states=active_slot_states[:, :2], event=event, warmup_eta=0.5)
    large = message_passing(active_slot_states=active_slot_states, event=event, warmup_eta=0.5)

    small_scale = small["aggregated_messages"].norm(dim=-1).mean().clamp_min(1.0e-6)
    large_scale = large["aggregated_messages"].norm(dim=-1).mean()
    assert large_scale < small_scale * 10.0


class _FixedRouter(nn.Module):
    def __init__(self, gate_value: float) -> None:
        super().__init__()
        self.gate_value = gate_value

    def forward(self, event: torch.Tensor, slot_states: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        batch_size, num_slots, slot_dim = slot_states.shape
        active_indices = torch.tensor([0, 1], device=slot_states.device).unsqueeze(0).expand(batch_size, -1)
        gate = torch.full((batch_size, num_slots), self.gate_value, device=slot_states.device)
        active_mask = torch.zeros(batch_size, num_slots, device=slot_states.device, dtype=torch.bool)
        active_mask.scatter_(1, active_indices, True)
        return {
            "energy": gate,
            "gate": gate,
            "drive": gate,
            "resistance": gate,
            "topk_indices": active_indices,
            "topk_scores": gate.gather(1, active_indices),
            "active_mask": active_mask,
            "active_states": torch.zeros(batch_size, 2, slot_dim, device=slot_states.device),
        }


class _FixedMessagePassing(nn.Module):
    def forward(self, active_slot_states: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        batch_size, active_slots, slot_dim = active_slot_states.shape
        message = torch.ones(batch_size, active_slots, slot_dim, device=active_slot_states.device)
        gate = torch.ones(batch_size, active_slots, active_slots, device=active_slot_states.device)
        return {
            "aggregated_messages": message,
            "gate": gate,
            "gate_mass": gate.sum(dim=2, keepdim=True),
        }


class _AdditiveUpdate(nn.Module):
    def forward(
        self,
        slot_states: torch.Tensor,
        message: torch.Tensor,
        event: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
        update_gate_scale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del event, warmup_eta, update_gate_scale
        return {
            "slot_states": slot_states + message,
            "candidate": slot_states + message,
            "drive": message,
            "resistance": message,
            "energy": message,
            "update_gate": torch.ones_like(message),
        }


def _fixed_route_layer(gate_value: float) -> EMLSlotGraphLayer:
    layer = EMLSlotGraphLayer(slot_dim=4, event_dim=4, hidden_dim=8, top_k=2)
    layer.router = _FixedRouter(gate_value)
    layer.message_passing = _FixedMessagePassing()
    layer.state_update = _AdditiveUpdate()
    return layer


def test_router_strength_modulates_update() -> None:
    slot_states = torch.zeros(2, 4, 4)
    event = torch.randn(2, 4)

    low = _fixed_route_layer(0.1)(event=event, slot_states=slot_states)
    high = _fixed_route_layer(0.9)(event=event, slot_states=slot_states)

    low_update = (low["active_slot_states_after"] - low["active_slot_states_before"]).norm()
    high_update = (high["active_slot_states_after"] - high["active_slot_states_before"]).norm()
    assert low["active_route_strength"].shape == (2, 2)
    assert high_update > low_update * 5.0
