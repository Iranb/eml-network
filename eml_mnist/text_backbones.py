from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import EMLResidualBankBlock, EMLTokenPool, MLP
from .primitives import EMLUnit
from .text_codecs import LocalTextCodec


class EMLCausalLocalMessageBlock(nn.Module):
    """Causal local sEML message passing over sequence features."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        window_size: int = 5,
        clip_value: float = 3.0,
        dropout: float = 0.1,
        gate_bias: float = -0.3,
        relative_distance_dim: int = 8,
        gate_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if relative_distance_dim <= 0:
            raise ValueError("relative_distance_dim must be positive")
        if gate_eps <= 0.0:
            raise ValueError("gate_eps must be positive")

        self.feature_dim = feature_dim
        self.window_size = window_size
        self.gate_eps = float(gate_eps)
        self.pre_norm = nn.LayerNorm(feature_dim)
        self.post_norm = nn.LayerNorm(feature_dim)
        self.relative_distance_embed = nn.Parameter(torch.empty(window_size, relative_distance_dim))
        edge_input_dim = feature_dim * 3 + relative_distance_dim
        edge_hidden_dim = max(hidden_dim // 2, feature_dim)
        self.drive_net = MLP(edge_input_dim, edge_hidden_dim, 1, dropout=dropout)
        self.resistance_net = MLP(edge_input_dim, edge_hidden_dim, 1, dropout=dropout)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.eml = EMLUnit(
            dim=1,
            clip_value=clip_value,
            init_gamma=0.1,
            init_lambda=1.0,
            init_bias=gate_bias,
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.relative_distance_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.value_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.out_proj.bias)

    def _causal_neighborhoods(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (0, 0, self.window_size - 1, 0))
        return torch.stack(
            [padded[:, offset : offset + x.size(1), :] for offset in range(self.window_size)],
            dim=2,
        )

    def _causal_mask(
        self,
        padding_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        distances = torch.arange(self.window_size - 1, -1, -1, device=device)
        positions = torch.arange(seq_len, device=device).unsqueeze(-1)
        history_mask = positions >= distances.unsqueeze(0)
        edge_mask = history_mask.unsqueeze(0).expand(batch_size, -1, -1)
        if padding_mask is not None:
            padded_mask = F.pad(padding_mask.bool(), (self.window_size - 1, 0), value=False)
            source_mask = torch.stack(
                [padded_mask[:, offset : offset + seq_len] for offset in range(self.window_size)],
                dim=2,
            )
            edge_mask = edge_mask & source_mask & padding_mask.bool().unsqueeze(-1)
        return edge_mask

    def forward(
        self,
        sequence_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if sequence_features.ndim != 3 or sequence_features.size(-1) != self.feature_dim:
            raise ValueError("sequence_features must have shape [batch, seq_len, feature_dim]")
        if padding_mask is not None and padding_mask.shape != sequence_features.shape[:2]:
            raise ValueError("padding_mask must have shape [batch, seq_len]")

        batch_size, seq_len, _ = sequence_features.shape
        normalized = self.pre_norm(sequence_features)
        neighborhoods = self._causal_neighborhoods(normalized)
        center = normalized.unsqueeze(2).expand(-1, -1, self.window_size, -1)
        relative_distance = self.relative_distance_embed.view(1, 1, self.window_size, -1)
        relative_distance = relative_distance.expand(batch_size, seq_len, -1, -1)
        edge_features = torch.cat([center, neighborhoods, center - neighborhoods, relative_distance], dim=-1)

        drive = self.drive_net(edge_features).squeeze(-1)
        resistance = self.resistance_net(edge_features).squeeze(-1)
        energy = self.eml(drive.unsqueeze(-1), resistance.unsqueeze(-1), warmup_eta=warmup_eta).squeeze(-1)
        gate = torch.sigmoid(energy)
        edge_mask = self._causal_mask(padding_mask, batch_size, seq_len, sequence_features.device)
        gate = gate.masked_fill(~edge_mask, 0.0)

        values = self.value_proj(neighborhoods)
        gate_values = gate.unsqueeze(-1)
        gate_mass = gate_values.sum(dim=2).clamp_min(self.gate_eps)
        message = (gate_values * values).sum(dim=2) / gate_mass
        output = self.post_norm(sequence_features + self.dropout(self.out_proj(message)))

        return {
            "output": output,
            "drive": drive,
            "resistance": resistance,
            "energy": energy,
            "gate": gate,
            "gate_mass": gate_mass,
            "edge_mask": edge_mask,
        }


class EMLTextBackbone(nn.Module):
    """Local text codec plus EML-native sequence representation blocks."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        feature_dim: int,
        event_dim: int,
        hidden_dim: int,
        bank_dim: int,
        num_layers: int = 3,
        causal_window_size: int = 5,
        clip_value: float = 3.0,
        dropout: float = 0.1,
        pad_id: int = 0,
        num_global_slots: int = 4,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_global_slots <= 0:
            raise ValueError("num_global_slots must be positive")

        self.feature_dim = feature_dim
        self.event_dim = event_dim
        self.num_global_slots = num_global_slots
        self.codec = LocalTextCodec(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=feature_dim,
            event_dim=event_dim,
            dropout=dropout,
            pad_id=pad_id,
        )
        self.causal_blocks = nn.ModuleList(
            [
                EMLCausalLocalMessageBlock(
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    window_size=causal_window_size,
                    clip_value=clip_value,
                    dropout=dropout,
                    gate_bias=-0.3,
                )
                for _ in range(num_layers)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                EMLResidualBankBlock(
                    input_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    bank_dim=bank_dim,
                    clip_value=clip_value,
                    dropout=dropout,
                    gate_bias=-1.0,
                )
                for _ in range(num_layers)
            ]
        )
        self.pool = EMLTokenPool(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            clip_value=clip_value,
            dropout=dropout,
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, event_dim),
            nn.LayerNorm(event_dim),
        )

        for module in self.readout.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, Any]:
        codec_out = self.codec(input_ids=input_ids, padding_mask=padding_mask)
        sequence_features = codec_out["sequence_features"]
        block_stats: List[Dict[str, torch.Tensor]] = []
        causal_message_stats: List[Dict[str, torch.Tensor]] = []

        for causal_block, block in zip(self.causal_blocks, self.blocks):
            causal_out = causal_block(
                sequence_features,
                padding_mask=codec_out["padding_mask"],
                warmup_eta=warmup_eta,
            )
            sequence_features = causal_out["output"]
            causal_message_stats.append(causal_out)
            block_stats.append(causal_out)
            block_out = block(sequence_features, warmup_eta=warmup_eta)
            sequence_features = block_out["output"]
            block_stats.append(block_out)

        pooled = self.pool(sequence_features, warmup_eta=warmup_eta)
        pooled_representation = self.readout(pooled["pooled"])

        topk_count = min(self.num_global_slots, sequence_features.size(1))
        topk_indices = pooled["weights"].topk(k=topk_count, dim=1).indices
        global_slot_features = sequence_features.gather(
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, sequence_features.size(-1)),
        )

        return {
            "event": pooled_representation,
            "pooled_representation": pooled_representation,
            "sequence_features": sequence_features,
            "global_slot_features": global_slot_features,
            "local_queries": sequence_features,
            "padding_mask": codec_out["padding_mask"],
            "codec": codec_out,
            "block_stats": block_stats,
            "causal_message_stats": causal_message_stats,
            "pool_stats": pooled,
            "pool_weights": pooled["weights"],
            "pool_energy": pooled["energy"],
            "pool_drive": pooled["drive"],
            "pool_resistance": pooled["resistance"],
            "topk_indices": topk_indices,
        }


__all__ = ["EMLCausalLocalMessageBlock", "EMLTextBackbone"]
