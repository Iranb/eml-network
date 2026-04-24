from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import EMLUnit, inverse_softplus


def _init_linear(module: nn.Linear, std: float = 0.02) -> None:
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class _ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, final_zero: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.reset_parameters(final_zero=final_zero)

    def reset_parameters(self, final_zero: bool = False) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module, std=0.02)
        if final_zero:
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def _masked_hard_negative(values: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2:
        raise ValueError("values must have shape [batch, classes]")
    mask = F.one_hot(labels, num_classes=values.size(-1)).bool()
    hard_values, hard_indices = values.masked_fill(mask, float("-inf")).max(dim=-1)
    return hard_values, hard_indices


def _margin_diagnostics(out: Dict[str, torch.Tensor], labels: torch.Tensor | None) -> Dict[str, torch.Tensor]:
    logits = out["logits"]
    if labels is None:
        return {}
    positive_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    hard_negative_logits, hard_indices = _masked_hard_negative(logits, labels)
    diagnostics: Dict[str, torch.Tensor] = {
        "positive_logit": positive_logits,
        "hard_negative_logit": hard_negative_logits,
        "margin": positive_logits - hard_negative_logits,
        "positive_logit_mean": positive_logits.mean(),
        "hard_negative_logit_mean": hard_negative_logits.mean(),
        "margin_mean": (positive_logits - hard_negative_logits).mean(),
        "hard_negative_index": hard_indices,
    }
    for name in ("drive", "resistance"):
        value = out.get(name)
        if torch.is_tensor(value) and value.shape == logits.shape:
            positive = value.gather(1, labels.unsqueeze(1)).squeeze(1)
            hard_negative = value.gather(1, hard_indices.unsqueeze(1)).squeeze(1)
            diagnostics[f"positive_{name}"] = positive
            diagnostics[f"hard_negative_{name}"] = hard_negative
            diagnostics[f"positive_{name}_mean"] = positive.mean()
            diagnostics[f"hard_negative_{name}_mean"] = hard_negative.mean()
    return diagnostics


def prototype_diversity_penalty(prototypes: torch.Tensor) -> torch.Tensor:
    if prototypes.size(0) <= 1:
        return prototypes.new_tensor(0.0)
    normalized = F.normalize(prototypes, dim=-1)
    cosine = normalized @ normalized.t()
    off_diagonal = cosine.masked_select(~torch.eye(cosine.size(0), device=cosine.device, dtype=torch.bool))
    return off_diagonal.square().mean()


class LinearHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, num_classes)
        _init_linear(self.linear)

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
        resistance_target: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del warmup_eta, resistance_target
        logits = self.linear(z)
        out = {"logits": logits, "probs": _probs(logits)}
        out.update(_margin_diagnostics(out, labels))
        return out


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(64, input_dim * 2)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
        resistance_target: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del warmup_eta, resistance_target
        logits = self.net(z)
        out = {"logits": logits, "probs": _probs(logits)}
        out.update(_margin_diagnostics(out, labels))
        return out


class CosinePrototypeHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        temperature: float = 0.25,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.temperature = float(temperature)
        self.prototypes = nn.Parameter(torch.empty(num_classes, input_dim))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.05)

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
        resistance_target: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del warmup_eta, resistance_target
        normalized_z = F.normalize(z, dim=-1)
        normalized_prototypes = F.normalize(self.prototypes, dim=-1)
        similarity = normalized_z @ normalized_prototypes.t()
        logits = similarity / self.temperature
        out = {
            "logits": logits,
            "probs": _probs(logits),
            "similarity": similarity,
            "prototypes": self.prototypes,
            "prototype_diversity_penalty": prototype_diversity_penalty(self.prototypes),
        }
        out.update(_margin_diagnostics(out, labels))
        return out


class _BaseEMLPrototypeHead(nn.Module):
    ambiguity_mode = "none"

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int | None = None,
        temperature: float = 0.25,
        clip_value: float = 3.0,
        ambiguity_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        hidden_dim = hidden_dim or max(64, input_dim * 2)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.temperature = float(temperature)
        self.ambiguity_weight = float(ambiguity_weight)
        self.prototypes = nn.Parameter(torch.empty(num_classes, input_dim))
        self.drive_residual = _ResidualMLP(input_dim, hidden_dim, num_classes, final_zero=True)
        self.uncertainty_head = _ResidualMLP(input_dim, hidden_dim, 1, final_zero=True)
        self.raw_class_resistance = nn.Parameter(
            torch.full((num_classes,), inverse_softplus(0.2), dtype=torch.float32)
        )
        self.resistance_supervision_scale = 0.0
        self.eml = EMLUnit(dim=num_classes, clip_value=clip_value, init_gamma=0.1, init_lambda=1.0, init_bias=0.0)
        nn.init.normal_(self.prototypes, mean=0.0, std=0.05)

    def _ambiguity(self, scaled_similarity: torch.Tensor, warmup_eta: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_classes <= 1 or self.ambiguity_mode == "none":
            ambiguity = torch.zeros_like(scaled_similarity)
            return ambiguity, torch.zeros_like(scaled_similarity)
        eye = torch.eye(self.num_classes, device=scaled_similarity.device, dtype=torch.bool).unsqueeze(0)
        expanded = scaled_similarity.unsqueeze(1).expand(-1, self.num_classes, -1)
        ambiguity = torch.logsumexp(expanded.masked_fill(eye, float("-inf")), dim=-1)
        if self.ambiguity_mode == "centered":
            ambiguity = ambiguity - math.log(self.num_classes - 1)
        if self.ambiguity_mode == "centered":
            if torch.is_tensor(warmup_eta):
                eta = warmup_eta.to(device=scaled_similarity.device, dtype=scaled_similarity.dtype).clamp(0.0, 1.0)
            else:
                eta = torch.tensor(float(warmup_eta), device=scaled_similarity.device, dtype=scaled_similarity.dtype).clamp(0.0, 1.0)
            weight = eta * self.ambiguity_weight
        else:
            weight = torch.tensor(self.ambiguity_weight, device=scaled_similarity.device, dtype=scaled_similarity.dtype)
        return ambiguity, torch.ones_like(ambiguity) * weight

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
        resistance_target: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        normalized_z = F.normalize(z, dim=-1)
        normalized_prototypes = F.normalize(self.prototypes, dim=-1)
        similarity = normalized_z @ normalized_prototypes.t()
        scaled_similarity = similarity / self.temperature
        drive = scaled_similarity + self.drive_residual(z)
        ambiguity, ambiguity_weight = self._ambiguity(scaled_similarity, warmup_eta)
        class_resistance = F.softplus(self.raw_class_resistance).unsqueeze(0).expand_as(drive)
        sample_uncertainty = F.softplus(self.uncertainty_head(z))
        sample_uncertainty_expanded = sample_uncertainty.expand_as(drive)
        weighted_ambiguity = ambiguity_weight * ambiguity
        resistance = weighted_ambiguity + class_resistance + sample_uncertainty_expanded
        eml_out = self.eml.compute(drive, resistance, warmup_eta=warmup_eta)
        logits = eml_out["energy"]
        out = {
            "logits": logits,
            "energy": logits,
            "probs": _probs(logits),
            "drive": drive,
            "resistance": resistance,
            "similarity": similarity,
            "ambiguity": ambiguity,
            "weighted_ambiguity": weighted_ambiguity,
            "ambiguity_weight": ambiguity_weight,
            "sample_uncertainty": sample_uncertainty,
            "class_resistance": class_resistance,
            "class_radius": class_resistance,
            "prototypes": self.prototypes,
            "prototype_diversity_penalty": prototype_diversity_penalty(self.prototypes),
            "eml_gamma": eml_out["gamma_fp32"],
            "eml_lambda": eml_out["lambda_fp32"],
        }
        out.update(_margin_diagnostics(out, labels))
        if labels is not None and torch.is_tensor(out.get("positive_resistance")):
            positive_resistance = out["positive_resistance"]
        else:
            positive_resistance = resistance.mean(dim=-1)
        out["resistance_score"] = positive_resistance
        if resistance_target is not None:
            out["resistance_target"] = resistance_target.reshape(-1)
            out["resistance_supervision_error"] = positive_resistance - resistance_target.reshape(-1).to(
                device=positive_resistance.device,
                dtype=positive_resistance.dtype,
            )
        return out


class EMLPrototypeHeadNoAmbiguity(_BaseEMLPrototypeHead):
    ambiguity_mode = "none"


class EMLPrototypeHeadRawAmbiguity(_BaseEMLPrototypeHead):
    ambiguity_mode = "raw"


class EMLPrototypeHeadCenteredAmbiguity(_BaseEMLPrototypeHead):
    ambiguity_mode = "centered"


class EMLPrototypeHeadSupervisedResistance(EMLPrototypeHeadCenteredAmbiguity):
    def __init__(self, *args: Any, resistance_supervision_scale: float = 1.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.resistance_supervision_scale = float(resistance_supervision_scale)


HEADS = {
    "linear": LinearHead,
    "mlp": MLPHead,
    "cosine_prototype": CosinePrototypeHead,
    "eml_no_ambiguity": EMLPrototypeHeadNoAmbiguity,
    "eml_raw_ambiguity": EMLPrototypeHeadRawAmbiguity,
    "eml_centered_ambiguity": EMLPrototypeHeadCenteredAmbiguity,
    "eml_supervised_resistance": EMLPrototypeHeadSupervisedResistance,
}


def build_head(
    head_name: str,
    input_dim: int,
    num_classes: int,
    hidden_dim: int | None = None,
    temperature: float = 0.25,
) -> nn.Module:
    if head_name not in HEADS:
        raise ValueError(f"unknown head: {head_name}")
    cls = HEADS[head_name]
    if head_name == "linear":
        return cls(input_dim=input_dim, num_classes=num_classes)  # type: ignore[misc]
    if head_name == "mlp":
        return cls(input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim)  # type: ignore[misc]
    if head_name == "cosine_prototype":
        return cls(input_dim=input_dim, num_classes=num_classes, temperature=temperature)  # type: ignore[misc]
    return cls(input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim, temperature=temperature)  # type: ignore[misc]


def has_prototypes(module: nn.Module) -> bool:
    return torch.is_tensor(getattr(module, "prototypes", None))


def pairwise_prototype_loss(module: nn.Module, margin: float = 0.0) -> torch.Tensor:
    prototypes = getattr(module, "prototypes", None)
    if not torch.is_tensor(prototypes) or prototypes.size(0) <= 1:
        device = next(module.parameters()).device
        return torch.tensor(0.0, device=device)
    normalized = F.normalize(prototypes, dim=-1)
    cosine = normalized @ normalized.t()
    off_diagonal = cosine.masked_select(~torch.eye(cosine.size(0), device=cosine.device, dtype=torch.bool))
    return F.relu(off_diagonal - margin).square().mean()


__all__ = [
    "CosinePrototypeHead",
    "EMLPrototypeHeadCenteredAmbiguity",
    "EMLPrototypeHeadNoAmbiguity",
    "EMLPrototypeHeadRawAmbiguity",
    "EMLPrototypeHeadSupervisedResistance",
    "HEADS",
    "LinearHead",
    "MLPHead",
    "build_head",
    "has_prototypes",
    "pairwise_prototype_loss",
    "prototype_diversity_penalty",
]
