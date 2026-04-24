from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict

import torch
import torch.nn.functional as F


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def _tensor_stats(value: torch.Tensor) -> Dict[str, float]:
    detached = value.detach().float()
    if detached.numel() == 0:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": _scalar(detached.mean()),
        "std": _scalar(detached.std(unbiased=False)),
    }


def flatten_nested_metrics(values: Mapping[str, Any], prefix: str = "", max_depth: int = 8) -> Dict[str, float | str | int]:
    """Flatten nested scalar-like metrics into a single row."""

    flat: Dict[str, float | str | int] = {}

    def visit(item: Any, key_prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        if torch.is_tensor(item):
            detached = item.detach()
            if detached.numel() == 1:
                flat[key_prefix] = _scalar(detached)
            else:
                stats = _tensor_stats(detached)
                flat[f"{key_prefix}_mean"] = stats["mean"]
                flat[f"{key_prefix}_std"] = stats["std"]
            return
        if isinstance(item, (int, float, str)):
            flat[key_prefix] = item
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                next_key = str(key) if not key_prefix else f"{key_prefix}.{key}"
                visit(child, next_key, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                next_key = f"{key_prefix}.{index}" if key_prefix else str(index)
                visit(child, next_key, depth + 1)

    visit(values, prefix, 0)
    return {key: value for key, value in flat.items() if key}


def attractor_diversity(attractor_states: torch.Tensor) -> float:
    if attractor_states.ndim != 3 or attractor_states.size(1) < 2:
        return 0.0
    normalized = F.normalize(attractor_states.detach().float(), dim=-1)
    cosine = normalized @ normalized.transpose(1, 2)
    mask = ~torch.eye(cosine.size(1), device=cosine.device, dtype=torch.bool).unsqueeze(0)
    values = cosine.masked_select(mask)
    return _scalar(values.square().mean()) if values.numel() else 0.0


def collect_eml_diagnostics(outputs: Mapping[str, Any]) -> Dict[str, float]:
    """Collect common EML diagnostics from nested model outputs."""

    collected: dict[str, list[torch.Tensor]] = {
        "drive": [],
        "resistance": [],
        "energy": [],
        "gate": [],
        "responsibility_entropy": [],
        "null_weight": [],
        "update_strength": [],
        "update_gate": [],
        "new_precision": [],
        "old_precision": [],
        "message_norm": [],
        "update_norm": [],
        "active_route_strength": [],
        "ambiguity": [],
        "ambiguity_weight": [],
        "sample_uncertainty": [],
        "prototype_diversity_penalty": [],
        "attractor_activation": [],
    }
    diversity_values: list[float] = []

    def visit(item: Any, key_name: str = "") -> None:
        if torch.is_tensor(item):
            if key_name in collected:
                collected[key_name].append(item.detach().float().reshape(-1))
            if key_name == "attractor_states":
                diversity_values.append(attractor_diversity(item))
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, str(key))
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child, key_name)

    visit(outputs)

    diagnostics: Dict[str, float] = {}
    for name, tensors in collected.items():
        if not tensors:
            continue
        values = torch.cat(tensors, dim=0)
        stats = _tensor_stats(values)
        diagnostics[f"{name}_mean"] = stats["mean"]
        diagnostics[f"{name}_std"] = stats["std"]

    if "gate_mean" in diagnostics:
        diagnostics["gate_activation_rate"] = diagnostics["gate_mean"]
    if diversity_values:
        diagnostics["attractor_diversity"] = float(sum(diversity_values) / len(diversity_values))
    return diagnostics


__all__ = ["attractor_diversity", "collect_eml_diagnostics", "flatten_nested_metrics"]
