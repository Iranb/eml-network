from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head_ablation import build_head


def _add_confidence_fields(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    logits = outputs["logits"]
    probs = torch.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values
    uncertainty = 1.0 - confidence
    risk_score = outputs.get("resistance_score")
    if not torch.is_tensor(risk_score):
        risk_score = uncertainty
    outputs["probs"] = probs
    outputs["confidence"] = confidence
    outputs["uncertainty"] = uncertainty
    outputs["risk_score"] = risk_score.reshape(-1)
    outputs["corruption_score"] = outputs["risk_score"]
    return outputs


class _WrappedUncertaintyHead(nn.Module):
    head_name: str

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int | None = None,
        temperature: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.head = build_head(
            self.head_name,
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            temperature=temperature,
        )

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
        resistance_target: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = _add_confidence_fields(
            self.head(z, labels=labels, warmup_eta=warmup_eta, resistance_target=resistance_target)
        )
        if resistance_target is not None and "resistance_score" in outputs:
            outputs["resistance_loss"] = F.mse_loss(
                outputs["resistance_score"].reshape(-1),
                resistance_target.to(device=outputs["resistance_score"].device, dtype=outputs["resistance_score"].dtype).reshape(-1),
            )
        return outputs


class LinearClassifierWithConfidence(_WrappedUncertaintyHead):
    head_name = "linear"


class MLPClassifierWithConfidence(_WrappedUncertaintyHead):
    head_name = "mlp"


class CosinePrototypeWithConfidence(_WrappedUncertaintyHead):
    head_name = "cosine_prototype"


class EMLUncertaintyHead(_WrappedUncertaintyHead):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        mode: str = "eml_centered_ambiguity",
        hidden_dim: int | None = None,
        temperature: float = 0.25,
    ) -> None:
        if mode not in {"eml_no_ambiguity", "eml_centered_ambiguity", "eml_supervised_resistance"}:
            raise ValueError("unsupported EML uncertainty mode")
        self.head_name = mode
        super().__init__(input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim, temperature=temperature)


class MERCUncertaintyHead(_WrappedUncertaintyHead):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        mode: str = "merc_linear",
        hidden_dim: int | None = None,
        temperature: float = 0.25,
    ) -> None:
        if mode not in {"merc_linear", "merc_energy"}:
            raise ValueError("unsupported MERC uncertainty mode")
        self.head_name = mode
        super().__init__(input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim, temperature=temperature)


def build_uncertainty_head(
    name: str,
    input_dim: int,
    num_classes: int,
    hidden_dim: int | None = None,
    temperature: float = 0.25,
) -> nn.Module:
    if name == "linear":
        return LinearClassifierWithConfidence(input_dim, num_classes, hidden_dim, temperature)
    if name == "mlp":
        return MLPClassifierWithConfidence(input_dim, num_classes, hidden_dim, temperature)
    if name == "cosine_prototype":
        return CosinePrototypeWithConfidence(input_dim, num_classes, hidden_dim, temperature)
    if name.startswith("eml_"):
        return EMLUncertaintyHead(input_dim, num_classes, name, hidden_dim, temperature)
    if name.startswith("merc_"):
        return MERCUncertaintyHead(input_dim, num_classes, name, hidden_dim, temperature)
    raise ValueError(f"unknown uncertainty head: {name}")


__all__ = [
    "CosinePrototypeWithConfidence",
    "EMLUncertaintyHead",
    "LinearClassifierWithConfidence",
    "MERCUncertaintyHead",
    "MLPClassifierWithConfidence",
    "build_uncertainty_head",
]
