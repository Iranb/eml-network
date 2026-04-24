from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import EMLUnit, _reset_linear
from .representation import (
    EMLAttractorMemory,
    EMLComposition,
    EMLLocalEvidenceEncoder,
    EMLRepresentationReadout,
    EMLResponsibilityPropagation,
)


def _stats(prefix: str, value: torch.Tensor) -> Dict[str, torch.Tensor]:
    value = value.detach().float()
    return {
        f"{prefix}_mean": value.mean(),
        f"{prefix}_std": value.std(unbiased=False),
    }


class _CausalConv1d(nn.Conv1d):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int) -> None:
        super().__init__(input_dim, output_dim, kernel_size=kernel_size, padding=0)
        self.left_padding = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


class _TextSensor(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, state_dim: int, pad_id: int) -> None:
        super().__init__()
        if vocab_size <= 0 or embed_dim <= 0 or state_dim <= 0:
            raise ValueError("vocab_size, embed_dim, and state_dim must be positive")
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.local = nn.Sequential(
            _CausalConv1d(embed_dim, state_dim, kernel_size=3),
            nn.GELU(),
            _CausalConv1d(state_dim, state_dim, kernel_size=3),
            nn.GELU(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if padding_mask is None:
            padding_mask = input_ids != self.pad_id
        token_features = self.embedding(input_ids)
        local_features = self.local(token_features.transpose(1, 2)).transpose(1, 2)
        local_features = torch.where(padding_mask.unsqueeze(-1), local_features, torch.zeros_like(local_features))
        return {
            "token_features": token_features,
            "local_features": local_features,
            "padding_mask": padding_mask.bool(),
        }


class EfficientEMLTextEncoder(nn.Module):
    """Efficient causal text representation trunk built from EML responsibility propagation."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        state_dim: int = 128,
        hidden_dim: int = 256,
        num_hypotheses: int = 16,
        num_attractors: int = 4,
        representation_dim: int | None = None,
        causal_window_size: int = 16,
        chunk_size: int = 8,
        pad_id: int = 0,
        clip_value: float = 3.0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or embed_dim <= 0 or state_dim <= 0 or hidden_dim <= 0:
            raise ValueError("invalid text encoder dimensions")
        self.state_dim = state_dim
        self.representation_dim = representation_dim or state_dim
        self.pad_id = pad_id
        self.sensor = _TextSensor(vocab_size, embed_dim, state_dim, pad_id)
        self.evidence = EMLLocalEvidenceEncoder(
            input_dim=state_dim,
            state_dim=state_dim,
            num_hypotheses=num_hypotheses,
            hidden_dim=hidden_dim,
        )
        self.propagation = EMLResponsibilityPropagation(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_hypotheses=num_hypotheses,
            mode="text",
            window_size=causal_window_size,
            clip_value=clip_value,
        )
        self.composition = EMLComposition(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            mode="text",
            region_size=chunk_size,
            clip_value=clip_value,
        )
        self.attractor = EMLAttractorMemory(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_attractors=num_attractors,
            clip_value=clip_value,
        )
        self.readout = EMLRepresentationReadout(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            representation_dim=self.representation_dim,
            clip_value=clip_value,
        )

    def _merge_diagnostics(self, *containers: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        diagnostics: Dict[str, torch.Tensor] = {}
        for container in containers:
            for key, value in container.items():
                if torch.is_tensor(value):
                    diagnostics[key] = value
        return diagnostics

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, Any]:
        sensor_out = self.sensor(input_ids, padding_mask=padding_mask)
        padding_mask = sensor_out["padding_mask"]
        evidence = self.evidence(sensor_out["local_features"])
        propagation = self.propagation(
            evidence["measurement"],
            drive_seed=evidence["drive_seed"],
            resistance_seed=evidence["resistance_seed"],
            padding_mask=padding_mask,
            warmup_eta=warmup_eta,
        )
        sequence_states = propagation["state"]
        composition = self.composition(sequence_states, padding_mask=padding_mask, warmup_eta=warmup_eta)
        chunk_states = composition["parent_state"]
        chunk_mask = composition["parent_activation"] > 0.0
        attractor = self.attractor(chunk_states, padding_mask=chunk_mask, warmup_eta=warmup_eta)
        readout = self.readout(attractor["attractor_states"], warmup_eta=warmup_eta)
        diagnostics = self._merge_diagnostics(
            propagation["diagnostics"],
            composition["diagnostics"],  # type: ignore[arg-type]
            attractor["diagnostics"],  # type: ignore[arg-type]
        )
        diagnostics.update(_stats("readout_weight", readout["weights"]))
        diagnostics["sequence_length"] = torch.tensor(float(input_ids.size(1)), device=input_ids.device)
        diagnostics["chunk_count"] = torch.tensor(float(chunk_states.size(1)), device=input_ids.device)
        diagnostics["num_attractors"] = torch.tensor(float(attractor["attractor_states"].size(1)), device=input_ids.device)
        diagnostics["valid_token_rate"] = padding_mask.float().mean().detach()
        return {
            "sequence_states": sequence_states,
            "sequence_features": sequence_states,
            "local_queries": sequence_states,
            "chunk_states": chunk_states,
            "chunk_hypotheses": chunk_states,
            "attractor_states": attractor["attractor_states"],
            "global_slot_features": attractor["attractor_states"],
            "attractor_weights": attractor["attractor_weights"],
            "attractor_activation": attractor["update_strength"],
            "representation": readout["representation"],
            "readout_weights": readout["weights"],
            "drive": readout["drive"],
            "resistance": readout["resistance"],
            "energy": readout["energy"],
            "padding_mask": padding_mask,
            "diagnostics": diagnostics,
            "propagation": propagation,
            "composition": composition,
            "attractor": attractor,
            "readout": readout,
        }


class EfficientEMLTextGenerationHead(nn.Module):
    """EML scorer for causal token prediction from sequence states."""

    def __init__(
        self,
        state_dim: int,
        vocab_size: int,
        hidden_dim: int = 256,
        clip_value: float = 3.0,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or vocab_size <= 0 or hidden_dim <= 0:
            raise ValueError("state_dim, vocab_size, and hidden_dim must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.state_dim = state_dim
        self.vocab_size = vocab_size
        self.temperature = float(temperature)
        self.state_norm = nn.LayerNorm(state_dim)
        self.vocab_prototypes = nn.Parameter(torch.empty(vocab_size, state_dim))
        self.drive_residual = nn.Linear(state_dim, vocab_size)
        self.uncertainty = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.energy = EMLUnit(dim=vocab_size, clip_value=clip_value, init_bias=0.0)
        nn.init.normal_(self.vocab_prototypes, mean=0.0, std=0.05)
        _reset_linear(self.drive_residual)
        for module in self.uncertainty.modules():
            if isinstance(module, nn.Linear):
                _reset_linear(module)

    def forward(
        self,
        sequence_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if sequence_states.ndim != 3 or sequence_states.size(-1) != self.state_dim:
            raise ValueError("sequence_states must have shape [batch, seq_len, state_dim]")
        normalized_states = F.normalize(self.state_norm(sequence_states), dim=-1)
        normalized_vocab = F.normalize(self.vocab_prototypes, dim=-1)
        similarity = normalized_states @ normalized_vocab.t()
        drive = similarity / self.temperature + self.drive_residual(sequence_states)
        uncertainty = F.softplus(self.uncertainty(sequence_states))
        conflict = F.relu(1.0 - similarity)
        resistance = conflict + uncertainty
        logits = self.energy(drive, resistance, warmup_eta=warmup_eta)
        if padding_mask is not None:
            logits = torch.where(padding_mask.unsqueeze(-1), logits, torch.zeros_like(logits))
        probs = torch.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "probs": probs,
            "drive": drive,
            "resistance": resistance,
            "energy": logits,
            "uncertainty": uncertainty,
        }


__all__ = ["EfficientEMLTextEncoder", "EfficientEMLTextGenerationHead"]
