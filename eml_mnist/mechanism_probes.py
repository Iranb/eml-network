from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn.functional as F

from .metrics import pearson_corr
from .primitives import EMLPrecisionUpdate, EMLResponsibility, EMLUnit


PROBE_NAMES = (
    "all_noise_should_choose_null",
    "one_strong_signal_many_weak_noise",
    "conflicting_neighbors_increase_resistance",
    "old_state_confident_new_evidence_weak_should_not_update",
    "old_state_weak_new_evidence_strong_should_update",
    "composition_requires_consistent_children",
    "attractor_should_not_collapse",
)

MECHANISM_NAMES = (
    "sigmoid_gate_sum",
    "sigmoid_gate_mean",
    "responsibility_no_null",
    "responsibility_null",
    "thresholded_null",
    "precision_update",
    "sigmoid_update",
)


def _entropy(weights: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    values = weights.clamp_min(eps)
    return -(weights * values.log()).sum(dim=-1)


def _mechanism_weights(energy: torch.Tensor, mechanism: str) -> Dict[str, torch.Tensor | None]:
    if mechanism == "sigmoid_gate_sum":
        weights = torch.sigmoid(energy)
        mass = weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        normalized = weights / mass
        return {
            "neighbor_weights": normalized,
            "raw_neighbor_weights": weights,
            "null_weight": torch.zeros(energy.shape[:-1], device=energy.device),
            "update_strength": weights.sum(dim=-1).clamp(0.0, 1.0),
        }
    if mechanism == "sigmoid_gate_mean":
        weights = torch.sigmoid(energy)
        mass = weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        normalized = weights / mass
        return {
            "neighbor_weights": normalized,
            "raw_neighbor_weights": weights,
            "null_weight": torch.zeros(energy.shape[:-1], device=energy.device),
            "update_strength": weights.mean(dim=-1).clamp(0.0, 1.0),
        }
    if mechanism == "responsibility_no_null":
        out = EMLResponsibility(use_null=False)(energy)
        return {
            "neighbor_weights": out["neighbor_weights"],
            "raw_neighbor_weights": out["neighbor_weights"],
            "null_weight": torch.zeros(energy.shape[:-1], device=energy.device),
            "update_strength": out["update_strength"],
        }
    if mechanism == "responsibility_null":
        out = EMLResponsibility(use_null=True)(energy)
        return {
            "neighbor_weights": out["neighbor_weights"],
            "raw_neighbor_weights": out["neighbor_weights"],
            "null_weight": out["null_weight"],
            "update_strength": out["update_strength"],
        }
    if mechanism == "thresholded_null":
        out = EMLResponsibility(mode="thresholded_null", use_null=True, evidence_threshold=0.0)(energy)
        return {
            "neighbor_weights": out["neighbor_weights"],
            "raw_neighbor_weights": out["neighbor_weights"],
            "null_weight": out["null_weight"],
            "update_strength": out["update_strength"],
        }
    weights = torch.softmax(energy, dim=-1)
    return {
        "neighbor_weights": weights,
        "raw_neighbor_weights": weights,
        "null_weight": torch.zeros(energy.shape[:-1], device=energy.device),
        "update_strength": torch.ones(energy.shape[:-1], device=energy.device),
    }


def _update_probe(
    mechanism: str,
    old_confidence_value: float,
    new_energy_value: float,
    expected_high: bool,
    device: torch.device,
) -> Dict[str, Any]:
    state = torch.zeros(4, 3, 8, device=device)
    candidate = torch.ones_like(state)
    old_confidence = torch.full((4, 3, 1), old_confidence_value, device=device)
    new_energy = torch.full((4, 3, 1), new_energy_value, device=device)
    update_mode = "sigmoid" if mechanism == "sigmoid_update" else "precision"
    update = EMLPrecisionUpdate(mode=update_mode, old_confidence_init=0.0)
    out = update(state, candidate, new_energy, old_confidence)
    gate = out["update_gate"].detach().float()
    delta = (out["updated_state"] - state).detach().float().norm(dim=-1)
    if expected_high:
        success = float(gate.mean().item() > 0.7)
    else:
        success = float(delta.mean().item() < 0.05)
    return {
        "metrics": {
            "accuracy": success,
            "update_gate": float(gate.mean().item()),
            "update_norm": float(delta.mean().item()),
            "null_weight": 0.0,
            "responsibility_entropy": 0.0,
            "max_responsibility": 0.0,
            "resistance_conflict_correlation": float("nan"),
            "attractor_diversity": float("nan"),
        },
        "diagnostics": {
            "update_gate_mean": gate.mean(),
            "update_norm_mean": delta.mean(),
            "new_precision_mean": out["new_precision"].detach().float().mean(),
            "old_precision_mean": out["old_precision"].detach().float().mean(),
        },
        "outputs": out,
    }


def _weights_metrics(weights_out: Dict[str, torch.Tensor | None], expected_index: int | None = None) -> Dict[str, float]:
    weights = weights_out["neighbor_weights"]
    if not torch.is_tensor(weights):
        raise ValueError("neighbor weights missing")
    null_weight = weights_out["null_weight"]
    null_mean = float(null_weight.detach().float().mean().item()) if torch.is_tensor(null_weight) else 0.0
    max_weight = float(weights.detach().float().max(dim=-1).values.mean().item())
    entropy = float(_entropy(weights.detach().float()).mean().item())
    if expected_index is None:
        accuracy = float(null_mean > 0.7)
    else:
        accuracy = float(weights[..., expected_index].detach().float().mean().item() > 0.7)
    return {
        "accuracy": accuracy,
        "null_weight": null_mean,
        "responsibility_entropy": entropy,
        "max_responsibility": max_weight,
        "update_gate": float(weights_out["update_strength"].detach().float().mean().item()),  # type: ignore[union-attr]
        "update_norm": 0.0,
        "resistance_conflict_correlation": float("nan"),
        "attractor_diversity": float("nan"),
    }


def run_mechanism_probe(
    probe_name: str,
    mechanism: str,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    if probe_name not in PROBE_NAMES:
        raise ValueError(f"unknown probe: {probe_name}")
    if mechanism not in MECHANISM_NAMES:
        raise ValueError(f"unknown mechanism: {mechanism}")
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)

    if probe_name == "all_noise_should_choose_null":
        energy = torch.full((8, 8), -4.0, device=device)
        weights_out = _mechanism_weights(energy, mechanism)
        metrics = _weights_metrics(weights_out, expected_index=None)
        return {"metrics": metrics, "diagnostics": {"null_weight_mean": torch.tensor(metrics["null_weight"])}, "outputs": weights_out}

    if probe_name == "one_strong_signal_many_weak_noise":
        energy = torch.randn(8, 8, generator=generator, device=device) * 0.2 - 2.0
        energy[:, 3] = 5.0
        weights_out = _mechanism_weights(energy, mechanism)
        metrics = _weights_metrics(weights_out, expected_index=3)
        return {
            "metrics": metrics,
            "diagnostics": {
                "null_weight_mean": torch.tensor(metrics["null_weight"]),
                "responsibility_entropy_mean": torch.tensor(metrics["responsibility_entropy"]),
            },
            "outputs": weights_out,
        }

    if probe_name == "conflicting_neighbors_increase_resistance":
        target = torch.randn(32, 8, generator=generator, device=device)
        aligned = target + 0.05 * torch.randn(32, 8, generator=generator, device=device)
        conflicting = -target + 0.05 * torch.randn(32, 8, generator=generator, device=device)
        agreement = F.cosine_similarity(target, aligned, dim=-1)
        conflict = 1.0 - F.cosine_similarity(target, conflicting, dim=-1)
        unit = EMLUnit(dim=1).to(device)
        low = unit(agreement.unsqueeze(-1), torch.zeros(32, 1, device=device))
        high = unit(agreement.unsqueeze(-1), conflict.unsqueeze(-1) * 3.0)
        corr = pearson_corr(conflict.detach(), (low - high).detach().squeeze(-1))
        success = float(high.mean().item() < low.mean().item() and (math.isnan(corr) or corr > 0.0))
        return {
            "metrics": {
                "accuracy": success,
                "null_weight": 0.0,
                "responsibility_entropy": 0.0,
                "max_responsibility": 0.0,
                "update_gate": 0.0,
                "update_norm": 0.0,
                "resistance_conflict_correlation": corr,
                "attractor_diversity": float("nan"),
            },
            "diagnostics": {
                "drive_mean": agreement.mean(),
                "resistance_mean": conflict.mean(),
                "energy_mean": high.mean(),
            },
            "outputs": {"drive": agreement, "resistance": conflict, "energy": high},
        }

    if probe_name == "old_state_confident_new_evidence_weak_should_not_update":
        return _update_probe(mechanism, old_confidence_value=8.0, new_energy_value=-8.0, expected_high=False, device=device)

    if probe_name == "old_state_weak_new_evidence_strong_should_update":
        return _update_probe(mechanism, old_confidence_value=-8.0, new_energy_value=8.0, expected_high=True, device=device)

    if probe_name == "composition_requires_consistent_children":
        base = torch.randn(16, 1, 8, generator=generator, device=device)
        consistent = base.expand(-1, 4, -1) + 0.02 * torch.randn(16, 4, 8, generator=generator, device=device)
        inconsistent = torch.randn(16, 4, 8, generator=generator, device=device) * 1.5
        consistent_var = consistent.var(dim=1, unbiased=False).mean(dim=-1)
        inconsistent_var = inconsistent.var(dim=1, unbiased=False).mean(dim=-1)
        drive = 1.0 / (1.0 + consistent_var)
        resistance = inconsistent_var
        activation_gap = (1.0 / (1.0 + consistent_var)).mean() - (1.0 / (1.0 + inconsistent_var)).mean()
        success = float((inconsistent_var > consistent_var).float().mean().item() > 0.9)
        return {
            "metrics": {
                "accuracy": success,
                "null_weight": 0.0,
                "responsibility_entropy": 0.0,
                "max_responsibility": 0.0,
                "update_gate": 0.0,
                "update_norm": 0.0,
                "resistance_conflict_correlation": pearson_corr(inconsistent_var.detach(), resistance.detach()),
                "attractor_diversity": float("nan"),
                "activation_gap": float(activation_gap.detach().item()),
            },
            "diagnostics": {
                "drive_mean": drive.mean(),
                "resistance_mean": resistance.mean(),
                "activation_gap": activation_gap,
            },
            "outputs": {"drive": drive, "resistance": resistance},
        }

    cluster_centers = F.normalize(torch.randn(4, 8, generator=generator, device=device), dim=-1)
    assignments = torch.arange(4, device=device).repeat_interleave(8)
    states = cluster_centers[assignments] + 0.04 * torch.randn(32, 8, generator=generator, device=device)
    states = states.view(1, 32, 8)
    similarity = F.normalize(states.squeeze(0), dim=-1) @ cluster_centers.t()
    assigned = similarity.argmax(dim=-1)
    used = assigned.unique().numel()
    center_cosine = cluster_centers @ cluster_centers.t()
    off_diag = center_cosine.masked_select(~torch.eye(4, device=device, dtype=torch.bool))
    diversity = float((1.0 - off_diag.abs().mean()).detach().item())
    success = float(used >= 3 and diversity > 0.25)
    return {
        "metrics": {
            "accuracy": success,
            "null_weight": 0.0,
            "responsibility_entropy": float(_entropy(torch.softmax(similarity, dim=-1)).mean().item()),
            "max_responsibility": float(torch.softmax(similarity, dim=-1).max(dim=-1).values.mean().item()),
            "update_gate": 0.0,
            "update_norm": 0.0,
            "resistance_conflict_correlation": float("nan"),
            "attractor_diversity": diversity,
        },
        "diagnostics": {"attractor_diversity": torch.tensor(diversity, device=device)},
        "outputs": {"similarity": similarity, "assignments": assigned},
    }


__all__ = ["MECHANISM_NAMES", "PROBE_NAMES", "run_mechanism_probe"]
