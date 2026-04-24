from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import _reset_linear


DEFAULT_VOCAB = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "[]{}()<>"
    "=,:;|+-*/_ "
)


class CharVocabulary:
    def __init__(self, symbols: Iterable[str] | None = None) -> None:
        symbol_list = list(symbols) if symbols is not None else list(dict.fromkeys(DEFAULT_VOCAB))
        if not symbol_list:
            raise ValueError("vocabulary must not be empty")

        special = ["<pad>", "<bos>", "<eos>"]
        merged = []
        for token in special + symbol_list:
            if token not in merged:
                merged.append(token)

        self.tokens = tuple(merged)
        self.stoi = {token: index for index, token in enumerate(self.tokens)}
        self.itos = {index: token for token, index in self.stoi.items()}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, ids: Iterable[int]) -> str:
        chars = []
        for index in ids:
            token = self.itos[int(index)]
            if token.startswith("<"):
                continue
            chars.append(token)
        return "".join(chars)


class _CausalConv1d(nn.Conv1d):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int) -> None:
        super().__init__(input_dim, output_dim, kernel_size=kernel_size, padding=0)
        self.left_padding = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


class LocalTextCodec(nn.Module):
    """Embedding + causal local conv + GRU text codec."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        event_dim: int,
        dropout: float = 0.0,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or embed_dim <= 0 or hidden_dim <= 0 or event_dim <= 0:
            raise ValueError("all codec dimensions must be positive")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.event_dim = event_dim
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.conv = nn.Sequential(
            _CausalConv1d(embed_dim, hidden_dim, kernel_size=3),
            nn.GELU(),
            _CausalConv1d(hidden_dim, hidden_dim, kernel_size=3),
            nn.GELU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.event_proj = nn.Linear(hidden_dim, event_dim)
        self.dropout = nn.Dropout(dropout)
        _reset_linear(self.event_proj)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        text_chunks: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if input_ids is None and text_chunks is None:
            raise ValueError("either input_ids or text_chunks must be provided")

        if input_ids is not None:
            if input_ids.ndim != 2:
                raise ValueError("input_ids must have shape [batch, seq_len]")
            token_features = self.embedding(input_ids)
            if padding_mask is None:
                padding_mask = input_ids != self.pad_id
        else:
            if text_chunks is None or text_chunks.ndim != 3:
                raise ValueError("text_chunks must have shape [batch, seq_len, embed_dim]")
            token_features = text_chunks
            if padding_mask is None:
                padding_mask = torch.ones(token_features.shape[:2], device=token_features.device, dtype=torch.bool)

        conv_features = self.conv(token_features.transpose(1, 2)).transpose(1, 2)
        recurrent_features, _ = self.gru(conv_features)
        sequence_features = self.norm(self.dropout(recurrent_features))

        mask = padding_mask.unsqueeze(-1).float()
        pooled = (sequence_features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        event = self.event_proj(pooled)

        return {
            "event": event,
            "sequence_features": sequence_features,
            "padding_mask": padding_mask,
            "local_queries": sequence_features,
        }


__all__ = ["CharVocabulary", "LocalTextCodec", "DEFAULT_VOCAB"]
