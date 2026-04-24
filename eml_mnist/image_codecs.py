from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import _reset_linear


class LocalImageChunkCodec(nn.Module):
    """Patch-and-convolution image codec without attention."""

    def __init__(
        self,
        input_channels: int,
        image_size: int,
        patch_size: int,
        chunk_dim: int,
        event_dim: int,
        hidden_dim: int,
        patch_stride: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or image_size <= 0 or patch_size <= 0 or chunk_dim <= 0 or event_dim <= 0 or hidden_dim <= 0:
            raise ValueError("all codec dimensions must be positive")

        patch_stride = patch_stride or patch_size
        self.input_channels = input_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        patch_dim = input_channels * patch_size * patch_size

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_stride)
        self.patch_proj = nn.Linear(patch_dim, chunk_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(chunk_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, chunk_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(chunk_dim)
        self.event_proj = nn.Linear(chunk_dim, event_dim)
        self.dropout = nn.Dropout(dropout)

        _reset_linear(self.patch_proj)
        _reset_linear(self.event_proj)

    def forward(
        self,
        images: torch.Tensor | None = None,
        image_chunks: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if images is None and image_chunks is None:
            raise ValueError("either images or image_chunks must be provided")

        if images is not None:
            if images.ndim != 4:
                raise ValueError("images must have shape [batch, channels, height, width]")
            patches = self.unfold(images).transpose(1, 2)
            chunk_features = self.patch_proj(patches)
        else:
            if image_chunks is None or image_chunks.ndim != 3:
                raise ValueError("image_chunks must have shape [batch, num_chunks, chunk_dim]")
            chunk_features = image_chunks

        conv_features = self.conv(chunk_features.transpose(1, 2)).transpose(1, 2)
        chunk_features = self.norm(self.dropout(chunk_features + conv_features))
        event = self.event_proj(chunk_features.mean(dim=1))

        return {
            "event": event,
            "chunk_features": chunk_features,
            "local_queries": chunk_features,
        }


__all__ = ["LocalImageChunkCodec"]
