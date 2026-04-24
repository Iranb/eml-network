from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from eml_mnist.graph import EMLMessagePassing, EMLSlotGraphLayer


def test_message_scale_stable_when_active_count_changes() -> None:
    torch.manual_seed(0)
    message_passing = EMLMessagePassing(slot_dim=8, event_dim=8, hidden_dim=16)
    active_slot_states = torch.randn(2, 6, 8)
    event = torch.randn(2, 8)

    two_slots = message_passing(active_slot_states=active_slot_states[:, :2], event=event, warmup_eta=0.5)
    six_slots = message_passing(active_slot_states=active_slot_states, event=event, warmup_eta=0.5)

    two_scale = two_slots["aggregated_messages"].norm(dim=-1).mean().clamp_min(1.0e-6)
    six_scale = six_slots["aggregated_messages"].norm(dim=-1).mean()
    assert six_scale < two_scale * 10.0
    assert torch.isfinite(six_slots["gate_mass"]).all()


def test_no_nan_when_edges_are_masked() -> None:
    message_passing = EMLMessagePassing(slot_dim=8, event_dim=8, hidden_dim=16)
    active_slot_states = torch.randn(2, 3, 8)
    event = torch.randn(2, 8)
    edge_mask = torch.zeros(2, 3, 3, dtype=torch.bool)

    out = message_passing(active_slot_states=active_slot_states, event=event, edge_mask=edge_mask, warmup_eta=0.5)

    assert torch.isfinite(out["aggregated_messages"]).all()
    assert torch.allclose(out["aggregated_messages"], torch.zeros_like(out["aggregated_messages"]))


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
        gate = torch.ones(batch_size, active_slots, active_slots, device=active_slot_states.device)
        return {
            "aggregated_messages": torch.ones(batch_size, active_slots, slot_dim, device=active_slot_states.device),
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


def test_low_route_gate_reduces_update_norm() -> None:
    slot_states = torch.zeros(2, 4, 4)
    event = torch.randn(2, 4)

    low = EMLSlotGraphLayer(slot_dim=4, event_dim=4, hidden_dim=8, top_k=2)
    high = EMLSlotGraphLayer(slot_dim=4, event_dim=4, hidden_dim=8, top_k=2)
    for layer, gate_value in ((low, 0.1), (high, 0.9)):
        layer.router = _FixedRouter(gate_value)
        layer.message_passing = _FixedMessagePassing()
        layer.state_update = _AdditiveUpdate()

    low_out = low(event=event, slot_states=slot_states)
    high_out = high(event=event, slot_states=slot_states)

    low_update = (low_out["active_slot_states_after"] - low_out["active_slot_states_before"]).norm()
    high_update = (high_out["active_slot_states_after"] - high_out["active_slot_states_before"]).norm()
    assert high_update > low_update * 5.0
    assert low_out["active_route_strength"].shape == (2, 2)
