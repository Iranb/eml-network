from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import EMLUnit, inverse_softplus


def _init_linear(module: nn.Linear, std: float = 0.02, zero: bool = False) -> None:
    if zero:
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class SupportFactorBank(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_support_factors: int,
        summary_dim: int = 4,
        eps: float = 1.0e-4,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, num_support_factors)
        self.summary = nn.Linear(4, summary_dim)
        self.eps = float(eps)
        _init_linear(self.proj)
        _init_linear(self.summary)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        normalized = self.norm(x)
        raw_support = self.proj(normalized)
        support_factors = F.softplus(raw_support.float()) + self.eps
        log_support = torch.log(support_factors)
        support_mean = support_factors.mean(dim=-1, keepdim=True)
        support_std = support_factors.std(dim=-1, keepdim=True, unbiased=False)
        summary_seed = torch.cat([support_mean, support_std, log_support.mean(dim=-1, keepdim=True), log_support.max(dim=-1, keepdim=True).values], dim=-1)
        support_summary = self.summary(summary_seed.to(dtype=x.dtype))
        return {
            "support_factors": support_factors.to(dtype=x.dtype),
            "log_support": log_support.to(dtype=x.dtype),
            "support_summary": support_summary,
        }


class ConflictFactorBank(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_conflict_factors: int,
        summary_dim: int = 4,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, num_conflict_factors)
        self.summary = nn.Linear(4, summary_dim)
        _init_linear(self.proj)
        _init_linear(self.summary)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        normalized = self.norm(x)
        raw_conflict = self.proj(normalized)
        conflict_factors = F.softplus(raw_conflict.float())
        conflict_mean = conflict_factors.mean(dim=-1, keepdim=True)
        conflict_std = conflict_factors.std(dim=-1, keepdim=True, unbiased=False)
        summary_seed = torch.cat(
            [
                conflict_mean,
                conflict_std,
                conflict_factors.max(dim=-1, keepdim=True).values,
                conflict_factors.min(dim=-1, keepdim=True).values,
            ],
            dim=-1,
        )
        conflict_summary = self.summary(summary_seed.to(dtype=x.dtype))
        return {
            "conflict_factors": conflict_factors.to(dtype=x.dtype),
            "conflict_summary": conflict_summary,
        }


class MERCCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_support_factors: int = 4,
        num_conflict_factors: int = 4,
        support_scale: float = 1.0,
        clip_value: float = 3.0,
        init_gamma: float = 0.3,
        init_lambda: float = 1.0,
        init_bias: float = 0.0,
        old_confidence_init: float = 4.0,
        precision_threshold: float = 1.0,
        activation_temperature: float = 1.0,
        include_energy_feature: bool = True,
        include_precision_feature: bool = True,
        include_resistance_feature: bool = True,
        support_eps: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if num_support_factors <= 0 or num_conflict_factors <= 0:
            raise ValueError("MERC requires positive support/conflict factor counts")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_support_factors = num_support_factors
        self.num_conflict_factors = num_conflict_factors
        self.support_bank = SupportFactorBank(input_dim, num_support_factors, summary_dim=min(hidden_dim, 8), eps=support_eps)
        self.conflict_bank = ConflictFactorBank(input_dim, num_conflict_factors, summary_dim=min(hidden_dim, 8))
        self.value_net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.uncertainty_net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.energy_proj = nn.Linear(1, hidden_dim)
        self.precision_proj = nn.Linear(1, hidden_dim)
        self.resistance_proj = nn.Linear(1, hidden_dim)
        self.support_summary_proj = nn.Linear(min(hidden_dim, 8), hidden_dim)
        self.conflict_summary_proj = nn.Linear(min(hidden_dim, 8), hidden_dim)
        feature_dim = output_dim
        feature_dim += int(include_energy_feature) * hidden_dim
        feature_dim += int(include_precision_feature) * hidden_dim
        feature_dim += int(include_resistance_feature) * hidden_dim
        feature_dim += hidden_dim
        feature_dim += hidden_dim
        self.output_proj = nn.Linear(feature_dim, output_dim)
        self.support_logits = nn.Parameter(torch.zeros(num_support_factors))
        self.conflict_weights = nn.Parameter(torch.ones(num_conflict_factors))
        self.drive_bias = nn.Parameter(torch.zeros(1))
        self.bias_resistance = nn.Parameter(torch.zeros(1))
        self.old_confidence = nn.Parameter(torch.full((1,), float(old_confidence_init)))
        self.update_threshold = nn.Parameter(torch.full((1,), float(precision_threshold)))
        self.activation_temperature = float(activation_temperature)
        self.include_energy_feature = include_energy_feature
        self.include_precision_feature = include_precision_feature
        self.include_resistance_feature = include_resistance_feature
        self.support_scale = float(support_scale)
        self.eml = EMLUnit(
            dim=1,
            clip_value=clip_value,
            init_gamma=init_gamma,
            init_lambda=init_lambda,
            init_bias=init_bias,
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)

    def forward(self, x: torch.Tensor, warmup_eta: float | torch.Tensor = 1.0) -> Dict[str, torch.Tensor]:
        support_out = self.support_bank(x)
        conflict_out = self.conflict_bank(x)
        support_factors = support_out["support_factors"]
        log_support = support_out["log_support"]
        support_weights = torch.softmax(self.support_logits.float(), dim=0).to(device=x.device, dtype=x.dtype)
        drive = self.drive_bias.to(device=x.device, dtype=x.dtype)
        drive = drive + self.support_scale * (log_support * support_weights).sum(dim=-1, keepdim=True) / math.sqrt(self.num_support_factors)

        conflict_factors = conflict_out["conflict_factors"]
        conflict_weights = F.softplus(self.conflict_weights).to(device=x.device, dtype=x.dtype)
        weighted_conflict = (conflict_factors * conflict_weights).sum(dim=-1, keepdim=True) / math.sqrt(self.num_conflict_factors)
        uncertainty = F.softplus(self.uncertainty_net(x).float()).to(dtype=x.dtype)
        resistance = self.bias_resistance.to(device=x.device, dtype=x.dtype) + weighted_conflict + uncertainty

        energy = self.eml(drive, resistance, warmup_eta=warmup_eta)
        activation = torch.sigmoid(energy / self.activation_temperature)
        precision = F.softplus((energy - self.update_threshold.to(device=x.device, dtype=x.dtype)).float()).to(dtype=x.dtype)
        value = self.value_net(x)

        feature_parts = [activation * value]
        if self.include_energy_feature:
            feature_parts.append(self.energy_proj(energy))
        if self.include_precision_feature:
            feature_parts.append(self.precision_proj(precision))
        if self.include_resistance_feature:
            feature_parts.append(self.resistance_proj(resistance))
        feature_parts.append(self.support_summary_proj(support_out["support_summary"]))
        feature_parts.append(self.conflict_summary_proj(conflict_out["conflict_summary"]))
        output = self.output_proj(torch.cat(feature_parts, dim=-1))
        return {
            "output": output,
            "drive": drive,
            "resistance": resistance,
            "energy": energy,
            "activation": activation,
            "precision": precision,
            "support_factors": support_factors,
            "conflict_factors": conflict_factors,
            "log_support": log_support,
            "support_summary": support_out["support_summary"],
            "conflict_summary": conflict_out["conflict_summary"],
            "uncertainty": uncertainty,
        }


class MERCResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        num_support_factors: int = 4,
        num_conflict_factors: int = 4,
        init_gamma: float = 0.3,
        old_confidence_init: float = 4.0,
        update_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.cell = MERCCell(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_support_factors=num_support_factors,
            num_conflict_factors=num_conflict_factors,
            init_gamma=init_gamma,
            old_confidence_init=old_confidence_init,
            precision_threshold=update_threshold,
        )
        self.candidate_net = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.old_confidence = nn.Parameter(torch.full((output_dim,), float(old_confidence_init)))
        self.norm = nn.LayerNorm(output_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)

    def forward(self, x: torch.Tensor, warmup_eta: float | torch.Tensor = 1.0) -> Dict[str, torch.Tensor]:
        cell_out = self.cell(x, warmup_eta=warmup_eta)
        candidate = self.candidate_net(cell_out["output"])
        old_precision = F.softplus(self.old_confidence).to(device=x.device, dtype=x.dtype)
        new_precision = F.softplus((cell_out["energy"] - self.cell.update_threshold.to(device=x.device, dtype=x.dtype)).float()).to(dtype=x.dtype)
        update_gate = new_precision / (new_precision + old_precision.mean())
        updated = self.norm(x + update_gate * (candidate - x))
        return {
            "output": updated,
            "candidate": candidate,
            "update_gate": update_gate,
            "old_precision": old_precision,
            "new_precision": new_precision,
            **cell_out,
        }


class MERCHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_support_factors: int = 4,
        num_conflict_factors: int = 4,
        head_mode: str = "linear_readout",
        init_gamma: float = 0.3,
        old_confidence_init: float = 4.0,
        update_threshold: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if head_mode not in {"linear_readout", "eml_class_energy"}:
            raise ValueError("unsupported MERC head mode")
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.head_mode = head_mode
        self.temperature = float(temperature)
        self.block = MERCResidualBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_support_factors=num_support_factors,
            num_conflict_factors=num_conflict_factors,
            init_gamma=init_gamma,
            old_confidence_init=old_confidence_init,
            update_threshold=update_threshold,
        )
        self.readout = nn.Linear(input_dim, num_classes)
        self.class_drive = nn.Linear(input_dim, num_classes)
        self.class_resistance = nn.Linear(input_dim, num_classes)
        self.class_eml = EMLUnit(dim=num_classes, clip_value=3.0, init_gamma=init_gamma, init_lambda=1.0, init_bias=0.0)
        _init_linear(self.readout)
        _init_linear(self.class_drive)
        _init_linear(self.class_resistance)

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
        resistance_target: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        block_out = self.block(z, warmup_eta=warmup_eta)
        features = block_out["output"]
        if self.head_mode == "linear_readout":
            logits = self.readout(features)
            drive = self.class_drive(features)
            resistance = F.softplus(self.class_resistance(features))
            energy = logits
        else:
            drive = self.class_drive(features)
            resistance = F.softplus(self.class_resistance(features))
            energy = self.class_eml(drive, resistance, warmup_eta=warmup_eta)
            logits = energy
        probs = torch.softmax(logits, dim=-1)
        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probs": probs,
            "drive": drive,
            "resistance": resistance,
            "energy": energy,
            "activation": block_out["activation"],
            "update_gate": block_out["update_gate"],
            "precision": block_out["precision"],
            "support_factors": block_out["support_factors"],
            "conflict_factors": block_out["conflict_factors"],
            "support_summary": block_out["support_summary"],
            "conflict_summary": block_out["conflict_summary"],
            "new_precision": block_out["new_precision"],
            "old_precision": block_out["old_precision"],
        }
        if labels is not None:
            positive_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            masked = logits.masked_fill(F.one_hot(labels, num_classes=self.num_classes).bool(), float("-inf"))
            hard_negative_logits, hard_negative_indices = masked.max(dim=-1)
            out["positive_logit"] = positive_logits
            out["hard_negative_logit"] = hard_negative_logits
            out["margin"] = positive_logits - hard_negative_logits
            out["hard_negative_index"] = hard_negative_indices
            out["positive_drive"] = drive.gather(1, labels.unsqueeze(1)).squeeze(1)
            out["positive_resistance"] = resistance.gather(1, labels.unsqueeze(1)).squeeze(1)
            out["hard_negative_drive"] = drive.gather(1, hard_negative_indices.unsqueeze(1)).squeeze(1)
            out["hard_negative_resistance"] = resistance.gather(1, hard_negative_indices.unsqueeze(1)).squeeze(1)
            out["resistance_score"] = out["positive_resistance"]
        if resistance_target is not None and "resistance_score" in out:
            out["resistance_target"] = resistance_target.reshape(-1).to(device=z.device, dtype=z.dtype)
            out["resistance_supervision_error"] = out["resistance_score"] - out["resistance_target"]
        return out


__all__ = [
    "ConflictFactorBank",
    "MERCCell",
    "MERCHead",
    "MERCResidualBlock",
    "SupportFactorBank",
]
