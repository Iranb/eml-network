from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from .model import EMLLocalStageBlock, EMLPrototypeClassifier, EMLTokenPool, OverlapPatchTokenizer, PatchMerge


class PureEMLImageBackbone(nn.Module):
    """Reusable non-attention image backbone built from local EML stages."""

    def __init__(
        self,
        image_size: int,
        input_channels: int,
        feature_dim: int,
        event_dim: int | None,
        hidden_dim: int,
        bank_dim: int,
        num_layers: int = 6,
        patch_size: int = 7,
        patch_stride: int = 4,
        local_window_size: int = 3,
        merge_every: int = 2,
        clip_value: float = 3.0,
        dropout: float = 0.1,
        num_global_slots: int = 4,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_global_slots <= 0:
            raise ValueError("num_global_slots must be positive")

        self.feature_dim = feature_dim
        self.event_dim = event_dim or feature_dim
        self.num_layers = num_layers
        self.num_global_slots = num_global_slots
        self.tokenizer = OverlapPatchTokenizer(
            image_size=image_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
            input_channels=input_channels,
            embed_dim=feature_dim,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                EMLLocalStageBlock(
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    bank_dim=bank_dim,
                    window_size=local_window_size,
                    clip_value=clip_value,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.merge_every = merge_every
        merge_count = max(0, (num_layers - 1) // max(1, merge_every))
        self.merges = nn.ModuleList([PatchMerge(feature_dim) for _ in range(merge_count)])
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
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.event_proj = nn.Linear(feature_dim, self.event_dim)

        for module in self.readout.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.event_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.event_proj.bias)

    def forward(
        self,
        images: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, Any]:
        tokens, height, width = self.tokenizer(images)
        block_stats: List[Dict[str, torch.Tensor]] = []
        merge_index = 0

        for block_index, block in enumerate(self.blocks, start=1):
            block_out = block(tokens, warmup_eta=warmup_eta)
            tokens = block_out["output"]
            block_stats.append(block_out["message_stats"])
            block_stats.append(block_out["channel_stats"])

            should_merge = (
                self.merge_every > 0
                and block_index % self.merge_every == 0
                and block_index != len(self.blocks)
                and height % 2 == 0
                and width % 2 == 0
                and merge_index < len(self.merges)
            )
            if should_merge:
                tokens = self.merges[merge_index](tokens)
                height //= 2
                width //= 2
                merge_index += 1

        flat_tokens = tokens.view(tokens.size(0), height * width, tokens.size(-1))
        pooled = self.pool(flat_tokens, warmup_eta=warmup_eta)
        pooled_representation = self.readout(pooled["pooled"])
        event = self.event_proj(pooled_representation)

        topk_count = min(self.num_global_slots, flat_tokens.size(1))
        topk_indices = pooled["weights"].topk(k=topk_count, dim=1).indices
        global_slot_features = flat_tokens.gather(
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, flat_tokens.size(-1)),
        )

        return {
            "event": event,
            "pooled_representation": pooled_representation,
            "token_features": flat_tokens,
            "global_slot_features": global_slot_features,
            "local_queries": flat_tokens,
            "block_stats": block_stats,
            "pool_weights": pooled["weights"],
            "pool_energy": pooled["energy"],
            "pool_drive": pooled["drive"],
            "pool_resistance": pooled["resistance"],
            "topk_indices": topk_indices,
        }


class PureEMLImageClassifier(nn.Module):
    """Image backbone + prototype classifier composition."""

    def __init__(
        self,
        num_classes: int,
        image_size: int,
        input_channels: int,
        feature_dim: int,
        event_dim: int | None,
        hidden_dim: int,
        bank_dim: int,
        num_layers: int = 6,
        patch_size: int = 7,
        patch_stride: int = 4,
        local_window_size: int = 3,
        merge_every: int = 2,
        clip_value: float = 3.0,
        dropout: float = 0.1,
        prototype_temperature: float = 0.25,
        num_global_slots: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = PureEMLImageBackbone(
            image_size=image_size,
            input_channels=input_channels,
            feature_dim=feature_dim,
            event_dim=event_dim,
            hidden_dim=hidden_dim,
            bank_dim=bank_dim,
            num_layers=num_layers,
            patch_size=patch_size,
            patch_stride=patch_stride,
            local_window_size=local_window_size,
            merge_every=merge_every,
            clip_value=clip_value,
            dropout=dropout,
            num_global_slots=num_global_slots,
        )
        self.classifier = EMLPrototypeClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            clip_value=clip_value,
            prototype_temperature=prototype_temperature,
        )

    def forward(
        self,
        images: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, Any]:
        backbone_out = self.backbone(images, warmup_eta=warmup_eta)
        head = self.classifier(backbone_out["pooled_representation"], warmup_eta=warmup_eta)
        head["features"] = backbone_out["pooled_representation"]
        head["tokens"] = backbone_out["token_features"]
        head["global_slot_features"] = backbone_out["global_slot_features"]
        head["block_stats"] = backbone_out["block_stats"]
        head["pool_weights"] = backbone_out["pool_weights"]
        head["pool_energy"] = backbone_out["pool_energy"]
        head["pool_drive"] = backbone_out["pool_drive"]
        head["pool_resistance"] = backbone_out["pool_resistance"]
        return head


__all__ = ["PureEMLImageBackbone", "PureEMLImageClassifier"]
