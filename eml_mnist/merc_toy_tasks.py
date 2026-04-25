from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .merc import MERCCell
from .primitives import EMLUnit


@dataclass
class ToyBatch:
    inputs: torch.Tensor
    labels: torch.Tensor
    evidence_target: torch.Tensor
    conflict_target: torch.Tensor


def _randn(generator: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def conjunctive_evidence_batch(batch_size: int, input_dim: int, seed: int) -> ToyBatch:
    generator = torch.Generator().manual_seed(seed)
    x = _randn(generator, batch_size, input_dim)
    evidence_a = x[:, 0]
    evidence_b = x[:, 1]
    conflict = x[:, 2]
    labels = ((evidence_a > 0.5) & (evidence_b > 0.5) & (conflict < 0.0)).long()
    evidence_target = torch.stack([evidence_a, evidence_b], dim=-1).mean(dim=-1)
    return ToyBatch(x, labels, evidence_target, conflict)


def xor_batch(batch_size: int, input_dim: int, seed: int) -> ToyBatch:
    generator = torch.Generator().manual_seed(seed)
    x = _randn(generator, batch_size, input_dim)
    a = x[:, 0] > 0.0
    b = x[:, 1] > 0.0
    c = x[:, 2] > 0.0
    labels = (a ^ b ^ c).long()
    evidence_target = x[:, :3].abs().mean(dim=-1)
    conflict = (x[:, 3] * x[:, 4]).tanh()
    return ToyBatch(x, labels, evidence_target, conflict)


def conflict_suppression_batch(batch_size: int, input_dim: int, seed: int) -> ToyBatch:
    generator = torch.Generator().manual_seed(seed)
    x = _randn(generator, batch_size, input_dim)
    support = x[:, 0] + x[:, 1]
    conflict = x[:, 2].abs() + 0.5 * x[:, 3].relu()
    labels = ((support > 1.0) & (conflict < 0.75)).long()
    evidence_target = support
    return ToyBatch(x, labels, evidence_target, conflict)


TOY_TASKS = {
    "conjunctive": conjunctive_evidence_batch,
    "xor": xor_batch,
    "conflict_suppression": conflict_suppression_batch,
}


class LinearToyModel(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.linear(x)
        probs = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs}


class MLPToyModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs}


class OldEMLGateToyModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.drive = nn.Linear(input_dim, hidden_dim)
        self.resistance = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, 2)
        self.eml = EMLUnit(dim=hidden_dim, clip_value=3.0, init_gamma=0.1, init_lambda=1.0, init_bias=0.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        drive = self.drive(x)
        resistance = self.resistance(x)
        energy = self.eml(drive, resistance, warmup_eta=1.0)
        gate = torch.sigmoid(energy)
        features = gate * self.value(x)
        logits = self.readout(features)
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
            "drive": drive.mean(dim=-1),
            "resistance": resistance.mean(dim=-1),
            "energy": energy.mean(dim=-1),
        }


class MERCToyModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, class_energy: bool = False) -> None:
        super().__init__()
        self.class_energy = class_energy
        self.cell = MERCCell(
            input_dim=input_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_support_factors=4,
            num_conflict_factors=4,
            init_gamma=0.3,
            old_confidence_init=4.0,
            precision_threshold=1.0,
        )
        self.drive = nn.Linear(hidden_dim, 2)
        self.resistance = nn.Linear(hidden_dim, 2)
        self.readout = nn.Linear(hidden_dim, 2)
        self.class_eml = EMLUnit(dim=2, clip_value=3.0, init_gamma=0.3, init_lambda=1.0, init_bias=0.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cell_out = self.cell(x, warmup_eta=1.0)
        features = cell_out["output"]
        drive = self.drive(features)
        resistance = F.softplus(self.resistance(features))
        logits = self.readout(features)
        if self.class_energy:
            logits = self.class_eml(drive, resistance, warmup_eta=1.0)
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
            "drive": drive.mean(dim=-1),
            "resistance": resistance.mean(dim=-1),
            "energy": cell_out["energy"].reshape(-1),
            "support_factors": cell_out["support_factors"],
            "conflict_factors": cell_out["conflict_factors"],
        }


TOY_MODELS = {
    "linear": LinearToyModel,
    "mlp": MLPToyModel,
    "old_eml_gate": OldEMLGateToyModel,
    "merc": MERCToyModel,
    "merc_energy": lambda input_dim, hidden_dim=32: MERCToyModel(input_dim, hidden_dim=hidden_dim, class_energy=True),
}


__all__ = [
    "TOY_MODELS",
    "TOY_TASKS",
    "ToyBatch",
    "conjunctive_evidence_batch",
    "conflict_suppression_batch",
    "xor_batch",
]
