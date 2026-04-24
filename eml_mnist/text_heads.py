from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .primitives import EMLUpdateGate, _reset_linear


class LocalTextGenerationHead(nn.Module):
    """Local text generation head without attention or transformer decoding."""

    def __init__(
        self,
        context_dim: int,
        token_dim: int,
        hidden_dim: int,
        vocab_size: int,
        clip_value: float = 3.0,
    ) -> None:
        super().__init__()
        if context_dim <= 0 or token_dim <= 0 or hidden_dim <= 0 or vocab_size <= 0:
            raise ValueError("all head dimensions must be positive")

        self.context_dim = context_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.context_norm = nn.LayerNorm(context_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.joint_norm = nn.LayerNorm(hidden_dim * 3)
        self.candidate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.drive_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.resistance_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = EMLUpdateGate(dim=hidden_dim, clip_value=clip_value, init_bias=-0.5)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                _reset_linear(module)

    def forward(
        self,
        representation: torch.Tensor,
        sequence_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if representation.ndim != 2 or representation.size(-1) != self.context_dim:
            raise ValueError("representation must have shape [batch, context_dim]")
        if sequence_features.ndim != 3 or sequence_features.size(-1) != self.token_dim:
            raise ValueError("sequence_features must have shape [batch, seq_len, token_dim]")
        if representation.size(0) != sequence_features.size(0):
            raise ValueError("representation and sequence_features batch sizes must match")
        if padding_mask is not None and padding_mask.shape != sequence_features.shape[:2]:
            raise ValueError("padding_mask must have shape [batch, seq_len]")

        batch_size, seq_len, _ = sequence_features.shape
        context = self.context_proj(self.context_norm(representation)).unsqueeze(1).expand(batch_size, seq_len, -1)
        token_features = self.token_proj(self.token_norm(sequence_features))
        joint = self.joint_norm(torch.cat([context, token_features, context * token_features], dim=-1))

        candidate = self.candidate(joint)
        drive = self.drive_net(joint)
        resistance = self.resistance_net(joint)
        gate_out = self.gate(drive, resistance, warmup_eta=warmup_eta)
        hidden = gate_out["gate"] * candidate
        logits = self.output_proj(hidden)
        if padding_mask is not None:
            logits = logits.masked_fill(~padding_mask.unsqueeze(-1), 0.0)

        return {
            "logits": logits,
            "hidden": hidden,
            "drive": gate_out["drive"],
            "resistance": gate_out["resistance"],
            "energy": gate_out["energy"],
            "gate": gate_out["gate"],
        }


__all__ = ["LocalTextGenerationHead"]
