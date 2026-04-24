from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .field import (
    EMLAttractorMemory,
    EMLCompositionField,
    EMLConsensusField,
    EMLFieldReadout,
    EMLHypothesisCompetition,
    EMLHypothesisField,
    EMLSensor,
)
from .primitives import EMLUnit, _reset_linear


def _text_position_features(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    denom = torch.tensor(max(seq_len - 1, 1), device=device, dtype=dtype)
    normalized = positions / denom
    log_scaled = torch.log1p(positions) / torch.log1p(denom)
    phase = torch.sin(positions / 4.0)
    return torch.stack([normalized, log_scaled, phase], dim=-1)


def _stats(prefix: str, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    tensor_fp32 = tensor.detach().to(dtype=torch.float32)
    return {
        f"{prefix}_mean": tensor_fp32.mean(),
        f"{prefix}_std": tensor_fp32.std(unbiased=False),
    }


def _masked_hypotheses(value: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    return value * padding_mask.unsqueeze(-1).to(dtype=value.dtype)


def _chunk_padding_mask(padding_mask: torch.Tensor, chunk_size: int) -> torch.Tensor:
    batch_size, seq_len = padding_mask.shape
    pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
    padded = F.pad(padding_mask.bool(), (0, pad_len), value=False)
    return padded.view(batch_size, -1, chunk_size).any(dim=-1)


class _CausalConv1d(nn.Conv1d):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int) -> None:
        super().__init__(input_dim, output_dim, kernel_size=kernel_size, padding=0)
        self.left_padding = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


class _ThinTextSensor(nn.Module):
    """Embedding plus local causal filters for EML measurements."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        sensor_dim: int,
        pad_id: int = 0,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or embed_dim <= 0 or sensor_dim <= 0:
            raise ValueError("vocab_size, embed_dim, and sensor_dim must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.sensor_dim = sensor_dim
        self.pad_id = pad_id
        hidden_dim = max(embed_dim, sensor_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.local = nn.Sequential(
            _CausalConv1d(embed_dim, hidden_dim, kernel_size=kernel_size),
            nn.GELU(),
            nn.Dropout(dropout),
            _CausalConv1d(hidden_dim, sensor_dim, kernel_size=kernel_size),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(sensor_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].zero_()
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if padding_mask is None:
            padding_mask = input_ids != self.pad_id
        if padding_mask.shape != input_ids.shape:
            raise ValueError("padding_mask must match input_ids")

        token_features = self.embedding(input_ids)
        local_features = self.local(token_features.transpose(1, 2)).transpose(1, 2)
        local_features = self.norm(local_features)
        local_features = local_features * padding_mask.unsqueeze(-1).to(dtype=local_features.dtype)
        return {
            "sensor_features": local_features,
            "token_features": token_features,
            "padding_mask": padding_mask,
        }


class EMLTextFieldEncoder(nn.Module):
    """Causal EML text encoder built from measurement, hypothesis, consensus, composition, and attractor fields."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        sensor_dim: int = 32,
        measurement_dim: int = 32,
        field_dim: int = 32,
        hidden_dim: int = 64,
        num_hypotheses: int = 4,
        num_chunk_hypotheses: int = 4,
        num_attractors: int = 4,
        representation_dim: int | None = None,
        pad_id: int = 0,
        sensor_kernel_size: int = 3,
        causal_window_size: int = 5,
        chunk_size: int = 4,
        chunk_window_size: int = 3,
        clip_value: float = 3.0,
        enable_chunk_consensus: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if sensor_dim <= 0 or measurement_dim <= 0 or field_dim <= 0 or hidden_dim <= 0:
            raise ValueError("sensor_dim, measurement_dim, field_dim, and hidden_dim must be positive")
        if num_hypotheses <= 0 or num_chunk_hypotheses <= 0 or num_attractors <= 0:
            raise ValueError("hypothesis and attractor counts must be positive")
        if causal_window_size <= 0 or chunk_size <= 0 or chunk_window_size <= 0:
            raise ValueError("window and chunk sizes must be positive")

        self.vocab_size = vocab_size
        self.field_dim = field_dim
        self.num_hypotheses = num_hypotheses
        self.num_chunk_hypotheses = num_chunk_hypotheses
        self.pad_id = pad_id
        self.chunk_size = chunk_size
        self.enable_chunk_consensus = enable_chunk_consensus

        self.text_sensor = _ThinTextSensor(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            sensor_dim=sensor_dim,
            pad_id=pad_id,
            kernel_size=sensor_kernel_size,
            dropout=dropout,
        )
        self.sensor = EMLSensor(
            input_dim=sensor_dim,
            measurement_dim=measurement_dim,
            seed_dim=field_dim,
            hidden_dim=hidden_dim,
            position_dim=3,
            dropout=dropout,
        )
        self.local_field = EMLHypothesisField(
            measurement_dim=measurement_dim,
            field_dim=field_dim,
            num_hypotheses=num_hypotheses,
            hidden_dim=hidden_dim,
            clip_value=clip_value,
            dropout=dropout,
        )
        self.local_drive_seed_proj = nn.Linear(field_dim, num_hypotheses)
        self.local_resistance_seed_proj = nn.Linear(field_dim, num_hypotheses)
        self.local_energy = EMLUnit(dim=num_hypotheses, clip_value=clip_value, init_bias=0.0)
        self.local_competition = EMLHypothesisCompetition(top_k=num_hypotheses)
        self.local_consensus = EMLConsensusField(
            field_dim=field_dim,
            hidden_dim=hidden_dim,
            num_hypotheses=num_hypotheses,
            mode="text",
            window_size=causal_window_size,
            clip_value=clip_value,
        )
        self.composition = EMLCompositionField(
            field_dim=field_dim,
            hidden_dim=hidden_dim,
            mode="text",
            chunk_size=chunk_size,
            num_parent_hypotheses=num_chunk_hypotheses,
            clip_value=clip_value,
        )
        self.chunk_consensus = (
            EMLConsensusField(
                field_dim=field_dim,
                hidden_dim=hidden_dim,
                num_hypotheses=num_chunk_hypotheses,
                mode="text",
                window_size=chunk_window_size,
                clip_value=clip_value,
            )
            if enable_chunk_consensus
            else None
        )
        self.attractor = EMLAttractorMemory(
            field_dim=field_dim,
            num_attractors=num_attractors,
            hidden_dim=hidden_dim,
            clip_value=clip_value,
        )
        self.readout = EMLFieldReadout(
            field_dim=field_dim,
            hidden_dim=hidden_dim,
            representation_dim=representation_dim or field_dim,
            clip_value=clip_value,
        )
        self.sequence_norm = nn.LayerNorm(field_dim)
        _reset_linear(self.local_drive_seed_proj)
        _reset_linear(self.local_resistance_seed_proj)

    def _summarize_stage(self, prefix: str, stage: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        stats: Dict[str, torch.Tensor] = {}
        for key in ("drive", "resistance", "energy", "activation", "support", "conflict", "gate_mass"):
            value = stage.get(key)
            if torch.is_tensor(value):
                stats.update(_stats(f"{prefix}_{key}", value))
        return stats

    def _sequence_from_hypotheses(self, state: torch.Tensor, activation: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        mass = activation.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        sequence_states = (state * activation.unsqueeze(-1)).sum(dim=2) / mass
        sequence_states = self.sequence_norm(sequence_states)
        return sequence_states * padding_mask.unsqueeze(-1).to(dtype=sequence_states.dtype)

    def _normalize_chunk_stage(
        self,
        composition_out: Dict[str, torch.Tensor | Dict[str, torch.Tensor]],
        chunk_consensus_out: Dict[str, torch.Tensor] | None,
        chunk_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        chunk_state = composition_out["parent_state"]  # type: ignore[index]
        mask = chunk_mask.unsqueeze(-1).to(dtype=chunk_state.dtype)
        if chunk_consensus_out is not None:
            return {
                "state": chunk_state,
                "drive": _masked_hypotheses(chunk_consensus_out["drive"], chunk_mask),
                "resistance": _masked_hypotheses(chunk_consensus_out["resistance"], chunk_mask),
                "energy": _masked_hypotheses(chunk_consensus_out["energy"], chunk_mask),
                "activation": _masked_hypotheses(chunk_consensus_out["activation"], chunk_mask),
                "support": _masked_hypotheses(chunk_consensus_out["support"], chunk_mask),
                "conflict": _masked_hypotheses(chunk_consensus_out["conflict"], chunk_mask),
                "gate_mass": _masked_hypotheses(chunk_consensus_out["gate_mass"], chunk_mask),
            }

        parent_drive = composition_out["parent_drive"]  # type: ignore[index]
        parent_resistance = composition_out["parent_resistance"]  # type: ignore[index]
        parent_energy = composition_out["parent_energy"]  # type: ignore[index]
        parent_activation = composition_out["parent_activation"]  # type: ignore[index]
        zero = parent_drive.new_zeros(parent_drive.shape)
        return {
            "state": chunk_state * mask.unsqueeze(-1),
            "drive": _masked_hypotheses(parent_drive, chunk_mask),
            "resistance": _masked_hypotheses(parent_resistance, chunk_mask),
            "energy": _masked_hypotheses(parent_energy, chunk_mask),
            "activation": _masked_hypotheses(parent_activation, chunk_mask),
            "support": zero,
            "conflict": zero,
            "gate_mass": zero,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if padding_mask is None:
            padding_mask = input_ids != self.pad_id
        if padding_mask.shape != input_ids.shape:
            raise ValueError("padding_mask must match input_ids")

        batch_size, seq_len = input_ids.shape
        sensor2d_out = self.text_sensor(input_ids, padding_mask=padding_mask)
        position_features = _text_position_features(seq_len, input_ids.device, sensor2d_out["sensor_features"].dtype)
        sensor_out = self.sensor(sensor2d_out["sensor_features"], position_features=position_features)

        local_field_out = self.local_field(sensor_out["measurement"], warmup_eta=warmup_eta)
        hypothesis_mask = padding_mask.unsqueeze(-1).expand(-1, -1, self.num_hypotheses)
        local_drive = local_field_out["drive"] + self.local_drive_seed_proj(sensor_out["drive_seed"])
        local_resistance = local_field_out["resistance"] + F.softplus(
            self.local_resistance_seed_proj(sensor_out["resistance_seed"])
        )
        local_drive = local_drive * hypothesis_mask.to(dtype=local_drive.dtype)
        local_resistance = local_resistance * hypothesis_mask.to(dtype=local_resistance.dtype)
        local_energy = self.local_energy(local_drive, local_resistance, warmup_eta=warmup_eta)
        local_energy = local_energy * hypothesis_mask.to(dtype=local_energy.dtype)
        local_competition_out = self.local_competition(energy=local_energy, resistance=local_resistance)
        local_activation = local_competition_out["activation"] * hypothesis_mask.to(dtype=local_energy.dtype)

        local_consensus_out = self.local_consensus(
            hypothesis_state=local_field_out["hypothesis_state"],
            activation=local_activation,
            drive=local_drive,
            resistance=local_resistance,
            padding_mask=padding_mask,
            warmup_eta=warmup_eta,
        )
        consensus_activation = local_consensus_out["activation"] * hypothesis_mask.to(dtype=local_energy.dtype)
        local_hypotheses = {
            "state": local_field_out["hypothesis_state"] * hypothesis_mask.unsqueeze(-1).to(dtype=local_energy.dtype),
            "drive": _masked_hypotheses(local_consensus_out["drive"], padding_mask),
            "resistance": _masked_hypotheses(local_consensus_out["resistance"], padding_mask),
            "energy": _masked_hypotheses(local_consensus_out["energy"], padding_mask),
            "activation": consensus_activation,
            "support": _masked_hypotheses(local_consensus_out["support"], padding_mask),
            "conflict": _masked_hypotheses(local_consensus_out["conflict"], padding_mask),
            "gate_mass": _masked_hypotheses(local_consensus_out["gate_mass"], padding_mask),
        }
        sequence_states = self._sequence_from_hypotheses(local_hypotheses["state"], local_hypotheses["activation"], padding_mask)

        composition_out = self.composition(
            hypothesis_state=local_field_out["hypothesis_state"],
            activation=consensus_activation,
            padding_mask=padding_mask,
            warmup_eta=warmup_eta,
        )
        chunk_mask = _chunk_padding_mask(padding_mask, self.chunk_size)
        chunk_consensus_out = None
        if self.chunk_consensus is not None:
            chunk_consensus_out = self.chunk_consensus(
                hypothesis_state=composition_out["parent_state"],  # type: ignore[arg-type]
                activation=composition_out["parent_activation"],  # type: ignore[arg-type]
                drive=composition_out["parent_drive"],  # type: ignore[arg-type]
                resistance=composition_out["parent_resistance"],  # type: ignore[arg-type]
                padding_mask=chunk_mask,
                warmup_eta=warmup_eta,
            )
        chunk_hypotheses = self._normalize_chunk_stage(composition_out, chunk_consensus_out, chunk_mask)

        attractor_out = self.attractor(
            hypothesis_state=chunk_hypotheses["state"],
            activation=chunk_hypotheses["activation"],
            warmup_eta=warmup_eta,
        )
        readout_out = self.readout(
            attractor_states=attractor_out["attractor_states"],
            attractor_activation=attractor_out["attractor_activation"],
            warmup_eta=warmup_eta,
        )

        budget_loss = (
            local_field_out["budget_loss"]
            + local_competition_out["budget_loss"]
            + local_consensus_out["budget_loss"]
            + composition_out["diagnostics"]["budget_loss"]  # type: ignore[index]
            + attractor_out["budget_loss"]
        )
        if chunk_consensus_out is not None:
            budget_loss = budget_loss + chunk_consensus_out["budget_loss"]

        diagnostics = {
            "sensor": {
                "measurement": sensor_out["measurement"],
                "drive_seed": sensor_out["drive_seed"],
                "resistance_seed": sensor_out["resistance_seed"],
            },
            "local": local_hypotheses,
            "chunk": chunk_hypotheses,
            "attractor": {
                "drive": attractor_out["attractor_drive"],
                "resistance": attractor_out["attractor_resistance"],
                "energy": attractor_out["attractor_energy"],
                "activation": attractor_out["attractor_activation"],
                "update_gate": attractor_out["update_gate"],
            },
            "readout": readout_out,
            "budget_loss": budget_loss,
            "stats": {
                **self._summarize_stage("local", local_hypotheses),
                **self._summarize_stage("chunk", chunk_hypotheses),
                **self._summarize_stage(
                    "attractor",
                    {
                        "drive": attractor_out["attractor_drive"],
                        "resistance": attractor_out["attractor_resistance"],
                        "energy": attractor_out["attractor_energy"],
                        "activation": attractor_out["attractor_activation"],
                    },
                ),
                **_stats("readout_weight", readout_out["weights"]),
                "valid_token_rate": padding_mask.to(dtype=torch.float32).mean().detach(),
                "budget_loss": budget_loss.detach(),
            },
        }

        return {
            "sequence_states": sequence_states,
            "sequence_features": sequence_states,
            "representation": readout_out["representation"],
            "event": readout_out["representation"],
            "pooled_representation": readout_out["representation"],
            "token_features": sequence_states,
            "global_slot_features": attractor_out["attractor_states"],
            "local_queries": local_field_out["hypothesis_state"].reshape(batch_size, -1, self.field_dim),
            "local_hypotheses": local_hypotheses,
            "chunk_hypotheses": chunk_hypotheses,
            "attractor_states": attractor_out["attractor_states"],
            "attractor_activation": attractor_out["attractor_activation"],
            "readout_weights": readout_out["weights"],
            "pool_weights": readout_out["weights"],
            "pool_energy": attractor_out["attractor_energy"],
            "pool_drive": attractor_out["attractor_drive"],
            "pool_resistance": attractor_out["attractor_resistance"],
            "padding_mask": padding_mask,
            "block_stats": [local_hypotheses, chunk_hypotheses],
            "diagnostics": diagnostics,
        }


class EMLTextFieldGenerationHead(nn.Module):
    """Prototype EML scorer for causal token prediction."""

    def __init__(
        self,
        state_dim: int,
        vocab_size: int,
        hidden_dim: int = 64,
        clip_value: float = 3.0,
        prototype_temperature: float = 0.5,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or vocab_size <= 0 or hidden_dim <= 0:
            raise ValueError("state_dim, vocab_size, and hidden_dim must be positive")
        if prototype_temperature <= 0.0:
            raise ValueError("prototype_temperature must be positive")

        self.state_dim = state_dim
        self.vocab_size = vocab_size
        self.prototype_temperature = float(prototype_temperature)
        self.state_norm = nn.LayerNorm(state_dim)
        self.state_proj = nn.Linear(state_dim, state_dim)
        self.vocab_prototypes = nn.Parameter(torch.empty(vocab_size, state_dim))
        self.prototype_proj = nn.Linear(state_dim, state_dim)
        self.drive_residual = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )
        self.conflict_residual = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )
        self.uncertainty_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.eml = EMLUnit(dim=vocab_size, clip_value=clip_value, init_gamma=1.0, init_bias=0.0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_linear(self.state_proj)
        _reset_linear(self.prototype_proj)
        for module in self.drive_residual.modules():
            if isinstance(module, nn.Linear):
                _reset_linear(module)
        for module in self.conflict_residual.modules():
            if isinstance(module, nn.Linear):
                _reset_linear(module)
        for module in self.uncertainty_net.modules():
            if isinstance(module, nn.Linear):
                _reset_linear(module)
        nn.init.normal_(self.vocab_prototypes, mean=0.0, std=0.02)

    def forward(
        self,
        sequence_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if sequence_states.ndim != 3 or sequence_states.size(-1) != self.state_dim:
            raise ValueError("sequence_states must have shape [batch, seq_len, state_dim]")
        if padding_mask is not None and padding_mask.shape != sequence_states.shape[:2]:
            raise ValueError("padding_mask must have shape [batch, seq_len]")

        normalized = self.state_norm(sequence_states)
        state_features = F.normalize(self.state_proj(normalized), dim=-1)
        prototype_features = F.normalize(self.prototype_proj(self.vocab_prototypes), dim=-1)
        similarity = torch.matmul(state_features, prototype_features.t())
        drive = similarity / self.prototype_temperature + self.drive_residual(normalized)
        ambiguity = torch.logsumexp(similarity, dim=-1, keepdim=True) - similarity
        uncertainty = F.softplus(self.uncertainty_net(normalized))
        resistance = F.softplus(self.conflict_residual(normalized)) + F.softplus(ambiguity) + uncertainty
        energy = self.eml(drive, resistance, warmup_eta=warmup_eta)
        logits = energy

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1)
            logits = logits.masked_fill(~mask, 0.0)
            drive = drive.masked_fill(~mask, 0.0)
            resistance = resistance.masked_fill(~mask, 0.0)
            energy = energy.masked_fill(~mask, 0.0)
            similarity = similarity.masked_fill(~mask, 0.0)

        return {
            "logits": logits,
            "probs": torch.softmax(logits.to(dtype=torch.float32), dim=-1).to(dtype=logits.dtype),
            "drive": drive,
            "resistance": resistance,
            "energy": energy,
            "uncertainty": uncertainty,
            "similarity": similarity,
            "ambiguity": ambiguity,
        }


__all__ = ["EMLTextFieldEncoder", "EMLTextFieldGenerationHead"]
