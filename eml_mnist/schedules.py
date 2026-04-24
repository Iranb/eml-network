from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class StagedHardeningConfig:
    warmup_steps: int = 100
    responsibility_temp_start: float = 2.0
    responsibility_temp_end: float = 0.8
    ambiguity_warmup_steps: int = 100
    null_threshold_start: float = 1.0
    null_threshold_end: float = 0.0
    attractor_entropy_start: float = 0.1
    attractor_entropy_end: float = 0.0
    precision_threshold_start: float = 1.0
    precision_threshold_end: float = 0.0
    schedule: str = "linear"


def _as_config(config: StagedHardeningConfig | Mapping[str, Any] | None) -> StagedHardeningConfig:
    if config is None:
        return StagedHardeningConfig()
    if isinstance(config, StagedHardeningConfig):
        return config
    allowed = StagedHardeningConfig.__dataclass_fields__.keys()
    values = {key: value for key, value in config.items() if key in allowed}
    return StagedHardeningConfig(**values)


def _progress(step: int, total_steps: int, warmup_steps: int, schedule: str) -> float:
    horizon = max(1, min(max(1, int(total_steps)), max(1, int(warmup_steps))))
    value = min(1.0, max(0.0, float(step) / float(horizon)))
    if schedule == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * value)
    if schedule != "linear":
        raise ValueError("schedule must be 'linear' or 'cosine'")
    return value


def _lerp(start: float, end: float, amount: float) -> float:
    return float(start + amount * (end - start))


def get_staged_hardening_values(
    step: int,
    total_steps: int,
    config: StagedHardeningConfig | Mapping[str, Any] | None = None,
) -> dict[str, float]:
    cfg = _as_config(config)
    step = max(0, int(step))
    total_steps = max(1, int(total_steps))
    warmup_p = _progress(step, total_steps, cfg.warmup_steps, cfg.schedule)
    ambiguity_p = _progress(step, total_steps, cfg.ambiguity_warmup_steps, cfg.schedule)
    full_p = _progress(step, total_steps, total_steps, cfg.schedule)
    return {
        "warmup_eta": warmup_p,
        "responsibility_temperature": _lerp(
            cfg.responsibility_temp_start,
            cfg.responsibility_temp_end,
            full_p,
        ),
        "ambiguity_weight": ambiguity_p,
        "null_threshold": _lerp(cfg.null_threshold_start, cfg.null_threshold_end, full_p),
        "attractor_entropy_weight": _lerp(cfg.attractor_entropy_start, cfg.attractor_entropy_end, full_p),
        "precision_update_threshold": _lerp(
            cfg.precision_threshold_start,
            cfg.precision_threshold_end,
            full_p,
        ),
    }


__all__ = ["StagedHardeningConfig", "get_staged_hardening_values"]
