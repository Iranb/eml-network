import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import EMLUnit, inverse_softplus


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        last_bias: bool = True,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim, bias=last_bias),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EMLResidualBankBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bank_dim: int,
        clip_value: float = 3.0,
        dropout: float = 0.1,
        gate_bias: float = -1.0,
    ) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(input_dim)
        self.post_norm = nn.LayerNorm(input_dim)

        self.drive_net = MLP(input_dim, hidden_dim, bank_dim, dropout=dropout)
        self.resistance_net = MLP(input_dim, hidden_dim, bank_dim, dropout=dropout)
        self.value_net = MLP(input_dim, hidden_dim, bank_dim, dropout=dropout)

        self.eml = EMLUnit(
            dim=bank_dim,
            clip_value=clip_value,
            init_gamma=0.1,
            init_lambda=1.0,
            init_bias=gate_bias,
        )
        self.mix = nn.Linear(bank_dim, input_dim)
        nn.init.normal_(self.mix.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mix.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        warmup_eta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        h = self.pre_norm(x)
        drive = self.drive_net(h)
        resistance = self.resistance_net(h)
        value = self.value_net(h)

        energy = self.eml(drive, resistance, warmup_eta=warmup_eta)
        gate = torch.sigmoid(energy)
        update = self.mix(gate * value)
        out = self.post_norm(x + self.dropout(update))

        return {
            "output": out,
            "drive": drive,
            "resistance": resistance,
            "energy": energy,
            "gate": gate,
        }


class ConvBackbone(nn.Module):
    def __init__(self, feature_dim: int, input_channels: int = 1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(x))


class SpatialConvStem(nn.Module):
    def __init__(self, feature_dim: int, input_channels: int = 1) -> None:
        super().__init__()
        stem_dim = max(32, feature_dim // 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, stem_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
            nn.Conv2d(stem_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class PatchTokenizer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        input_channels: int,
        embed_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * input_channels

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.unfold(x).transpose(1, 2)
        tokens = self.proj(patches) + self.pos_embed
        return self.dropout(tokens)


class OverlapPatchTokenizer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        patch_stride: int,
        input_channels: int,
        embed_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.padding = patch_stride // 2
        patch_dim = patch_size * patch_size * input_channels

        self.unfold = nn.Unfold(
            kernel_size=patch_size,
            stride=patch_stride,
            padding=self.padding,
        )
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        grid_size = math.floor((image_size + 2 * self.padding - patch_size) / patch_stride) + 1
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        patches = self.unfold(x).transpose(1, 2)
        tokens = self.dropout(self.proj(patches) + self.pos_embed)
        batch_size = x.size(0)
        return tokens.view(batch_size, self.grid_size, self.grid_size, -1), self.grid_size, self.grid_size


class EMLMixerLayer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_tokens: int,
        hidden_dim: int,
        bank_dim: int,
        clip_value: float = 3.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channel_block = EMLResidualBankBlock(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            bank_dim=bank_dim,
            clip_value=clip_value,
            dropout=dropout,
            gate_bias=-1.0,
        )
        self.token_block = EMLResidualBankBlock(
            input_dim=num_tokens,
            hidden_dim=hidden_dim,
            bank_dim=bank_dim,
            clip_value=clip_value,
            dropout=dropout,
            gate_bias=-1.0,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        warmup_eta: float = 1.0,
    ) -> Dict[str, Any]:
        channel_out = self.channel_block(tokens, warmup_eta=warmup_eta)
        token_in = channel_out["output"].transpose(1, 2)
        token_out = self.token_block(token_in, warmup_eta=warmup_eta)
        mixed_tokens = token_out["output"].transpose(1, 2)

        return {
            "output": mixed_tokens,
            "channel_stats": channel_out,
            "token_stats": token_out,
        }


class EMLLocalMessageBlock(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        window_size: int = 3,
        clip_value: float = 3.0,
        dropout: float = 0.1,
        gate_bias: float = -0.3,
        relative_position_dim: int = 8,
        gate_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        if relative_position_dim <= 0:
            raise ValueError("relative_position_dim must be positive")
        if gate_eps <= 0.0:
            raise ValueError("gate_eps must be positive")

        self.window_size = window_size
        self.neighbor_count = window_size * window_size
        self.gate_eps = float(gate_eps)
        self.pre_norm = nn.LayerNorm(feature_dim)
        self.post_norm = nn.LayerNorm(feature_dim)
        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2)
        self.relative_position_embed = nn.Parameter(torch.empty(self.neighbor_count, relative_position_dim))
        edge_input_dim = feature_dim * 3 + relative_position_dim
        edge_hidden_dim = max(hidden_dim // 2, feature_dim)

        self.drive_net = MLP(edge_input_dim, edge_hidden_dim, 1, dropout=dropout)
        self.resistance_net = MLP(edge_input_dim, edge_hidden_dim, 1, dropout=dropout)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        nn.init.normal_(self.value_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.out_proj.bias)

        self.eml = EMLUnit(
            dim=1,
            clip_value=clip_value,
            init_gamma=0.1,
            init_lambda=1.0,
            init_bias=gate_bias,
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.relative_position_embed, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        warmup_eta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        batch_size, height, width, feature_dim = tokens.shape
        normalized = self.pre_norm(tokens)
        flattened = normalized.view(batch_size, height * width, feature_dim)

        neighborhoods = self.unfold(normalized.permute(0, 3, 1, 2))
        neighborhoods = neighborhoods.transpose(1, 2).reshape(
            batch_size,
            height * width,
            self.neighbor_count,
            feature_dim,
        )
        center = flattened.unsqueeze(2).expand(-1, -1, self.neighbor_count, -1)
        relative_position = self.relative_position_embed.view(1, 1, self.neighbor_count, -1)
        relative_position = relative_position.expand(batch_size, height * width, -1, -1)
        edge_features = torch.cat([center, neighborhoods, center - neighborhoods, relative_position], dim=-1)

        drive = self.drive_net(edge_features).squeeze(-1)
        resistance = self.resistance_net(edge_features).squeeze(-1)
        energy = self.eml(drive.unsqueeze(-1), resistance.unsqueeze(-1), warmup_eta=warmup_eta).squeeze(-1)
        gate = torch.sigmoid(energy)

        values = self.value_proj(neighborhoods)
        gate_values = gate.unsqueeze(-1)
        gate_mass = gate_values.sum(dim=2).clamp_min(self.gate_eps)
        message = (gate_values * values).sum(dim=2) / gate_mass
        update = self.out_proj(message).view(batch_size, height, width, feature_dim)
        output = self.post_norm(tokens + self.dropout(update))

        return {
            "output": output,
            "energy": energy,
            "gate": gate,
            "gate_mass": gate_mass,
            "drive": drive,
            "resistance": resistance,
        }


class PatchMerge(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim * 4)
        self.reduction = nn.Linear(feature_dim * 4, feature_dim)
        nn.init.normal_(self.reduction.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.reduction.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, feature_dim = tokens.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("PatchMerge expects even height and width")

        top_left = tokens[:, 0::2, 0::2, :]
        bottom_left = tokens[:, 1::2, 0::2, :]
        top_right = tokens[:, 0::2, 1::2, :]
        bottom_right = tokens[:, 1::2, 1::2, :]
        merged = torch.cat([top_left, bottom_left, top_right, bottom_right], dim=-1)
        return self.reduction(self.norm(merged))


class ConvSpatialDownsample(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        features = tokens.permute(0, 3, 1, 2).contiguous()
        features = self.downsample(features)
        return features.permute(0, 2, 3, 1).contiguous()


class EMLTokenPool(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        clip_value: float = 3.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.drive_net = MLP(feature_dim, hidden_dim, 1, dropout=dropout)
        self.resistance_net = MLP(feature_dim, hidden_dim, 1, dropout=dropout)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        nn.init.normal_(self.value_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.value_proj.bias)
        self.eml = EMLUnit(
            dim=1,
            clip_value=clip_value,
            init_gamma=0.1,
            init_lambda=1.0,
            init_bias=0.0,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        warmup_eta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        normalized = self.norm(tokens)
        drive = self.drive_net(normalized)
        resistance = self.resistance_net(normalized)
        energy = self.eml(drive, resistance, warmup_eta=warmup_eta).squeeze(-1)
        weights = torch.softmax(energy, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * self.value_proj(normalized), dim=1)
        return {
            "pooled": pooled,
            "weights": weights,
            "energy": energy,
            "drive": drive.squeeze(-1),
            "resistance": resistance.squeeze(-1),
        }


class EMLLocalStageBlock(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        bank_dim: int,
        window_size: int = 3,
        clip_value: float = 3.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.message_block = EMLLocalMessageBlock(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            window_size=window_size,
            clip_value=clip_value,
            dropout=dropout,
            gate_bias=-0.3,
        )
        self.channel_block = EMLResidualBankBlock(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            bank_dim=bank_dim,
            clip_value=clip_value,
            dropout=dropout,
            gate_bias=-0.1,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        warmup_eta: float = 1.0,
    ) -> Dict[str, Any]:
        message_out = self.message_block(tokens, warmup_eta=warmup_eta)
        batch_size, height, width, feature_dim = message_out["output"].shape
        flat_tokens = message_out["output"].view(batch_size, height * width, feature_dim)
        channel_out = self.channel_block(flat_tokens, warmup_eta=warmup_eta)
        output = channel_out["output"].view(batch_size, height, width, feature_dim)

        return {
            "output": output,
            "message_stats": message_out,
            "channel_stats": channel_out,
        }


class EMLPrototypeClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int,
        clip_value: float = 3.0,
        prototype_temperature: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.prototype_temperature = prototype_temperature

        self.prototypes = nn.Parameter(torch.empty(num_classes, feature_dim))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.05)

        self.drive_residual = MLP(feature_dim, hidden_dim, num_classes, dropout=0.1)
        self.uncertainty_head = MLP(feature_dim, hidden_dim, 1, dropout=0.1)
        self.raw_class_resistance = nn.Parameter(
            torch.full((num_classes,), inverse_softplus(0.2), dtype=torch.float32)
        )

        self.eml_score = EMLUnit(
            dim=num_classes,
            clip_value=clip_value,
            init_gamma=0.1,
            init_lambda=1.0,
            init_bias=0.0,
        )

    def compute_ambiguity(self, similarity: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = similarity.shape
        eye_mask = torch.eye(num_classes, device=similarity.device, dtype=torch.bool).unsqueeze(0)
        expanded = similarity.unsqueeze(1).expand(batch_size, num_classes, num_classes)
        masked = expanded.masked_fill(eye_mask, float("-inf"))
        return torch.logsumexp(masked, dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        warmup_eta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        features = F.normalize(features, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)

        similarity = features @ prototypes.t()
        drive = similarity / self.prototype_temperature + self.drive_residual(features)

        ambiguity = self.compute_ambiguity(similarity / self.prototype_temperature)
        class_radius = F.softplus(self.raw_class_resistance).unsqueeze(0).expand_as(drive)
        sample_uncertainty = F.softplus(self.uncertainty_head(features)).expand_as(drive)
        resistance = ambiguity + class_radius + sample_uncertainty

        score_diagnostics = self.eml_score.compute(drive, resistance, warmup_eta=warmup_eta)
        logits = score_diagnostics["energy"]
        probs = torch.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "energy": logits,
            "probs": probs,
            "drive": drive,
            "resistance": resistance,
            "similarity": similarity,
            "class_radius": class_radius,
            "sample_uncertainty": sample_uncertainty[:, :1],
            "prototypes": self.prototypes,
            "eml_gamma": score_diagnostics["gamma_fp32"],
            "eml_lambda": score_diagnostics["lambda_fp32"],
        }


class MNISTEMLNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        bank_dim: int = 128,
        bank_blocks: int = 2,
        clip_value: float = 3.0,
        dropout: float = 0.1,
        prototype_temperature: float = 0.25,
    ) -> None:
        super().__init__()
        self.backbone = ConvBackbone(feature_dim=feature_dim, input_channels=input_channels)
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
                for _ in range(bank_blocks)
            ]
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
        x: torch.Tensor,
        warmup_eta: float = 1.0,
    ) -> Dict[str, Any]:
        features = self.backbone(x)
        block_stats: List[Dict[str, torch.Tensor]] = []

        for block in self.blocks:
            block_out = block(features, warmup_eta=warmup_eta)
            features = block_out["output"]
            block_stats.append(block_out)

        head = self.classifier(features, warmup_eta=warmup_eta)
        head["features"] = features
        head["block_stats"] = block_stats
        return head


class PureEMLMNISTNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 28,
        patch_size: int = 7,
        input_channels: int = 1,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        bank_dim: int = 128,
        bank_blocks: int = 4,
        clip_value: float = 3.0,
        dropout: float = 0.1,
        prototype_temperature: float = 0.25,
    ) -> None:
        super().__init__()
        self.tokenizer = PatchTokenizer(
            image_size=image_size,
            patch_size=patch_size,
            input_channels=input_channels,
            embed_dim=feature_dim,
            dropout=dropout,
        )
        self.layers = nn.ModuleList(
            [
                EMLMixerLayer(
                    feature_dim=feature_dim,
                    num_tokens=self.tokenizer.num_patches,
                    hidden_dim=hidden_dim,
                    bank_dim=bank_dim,
                    clip_value=clip_value,
                    dropout=dropout,
                )
                for _ in range(bank_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(feature_dim)
        self.readout = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.classifier = EMLPrototypeClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            clip_value=clip_value,
            prototype_temperature=prototype_temperature,
        )

        for module in self.readout.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        warmup_eta: float = 1.0,
    ) -> Dict[str, Any]:
        tokens = self.tokenizer(x)
        block_stats: List[Dict[str, torch.Tensor]] = []

        for layer in self.layers:
            layer_out = layer(tokens, warmup_eta=warmup_eta)
            tokens = layer_out["output"]
            block_stats.append(layer_out["channel_stats"])
            block_stats.append(layer_out["token_stats"])

        pooled = self.final_norm(tokens).mean(dim=1)
        features = self.readout(pooled)

        head = self.classifier(features, warmup_eta=warmup_eta)
        head["features"] = features
        head["tokens"] = tokens
        head["block_stats"] = block_stats
        return head


class PureEMLV2MNISTNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 28,
        patch_size: int = 7,
        patch_stride: int = 4,
        input_channels: int = 1,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        bank_dim: int = 128,
        bank_blocks: int = 6,
        clip_value: float = 3.0,
        dropout: float = 0.1,
        prototype_temperature: float = 0.25,
        local_window_size: int = 3,
        merge_every: int = 2,
    ) -> None:
        super().__init__()
        from .image_backbones import PureEMLImageClassifier

        self.model = PureEMLImageClassifier(
            num_classes=num_classes,
            image_size=image_size,
            input_channels=input_channels,
            feature_dim=feature_dim,
            event_dim=feature_dim,
            hidden_dim=hidden_dim,
            bank_dim=bank_dim,
            num_layers=bank_blocks,
            patch_size=patch_size,
            patch_stride=patch_stride,
            local_window_size=local_window_size,
            merge_every=merge_every,
            clip_value=clip_value,
            dropout=dropout,
            prototype_temperature=prototype_temperature,
        )

    def forward(
        self,
        x: torch.Tensor,
        warmup_eta: float = 1.0,
    ) -> Dict[str, Any]:
        return self.model(x, warmup_eta=warmup_eta)


class CNNEMLStageNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        bank_dim: int = 128,
        bank_blocks: int = 2,
        clip_value: float = 3.0,
        dropout: float = 0.1,
        prototype_temperature: float = 0.25,
        local_window_size: int = 3,
    ) -> None:
        super().__init__()
        num_stages = max(2, bank_blocks)
        self.stem = SpatialConvStem(feature_dim=feature_dim, input_channels=input_channels)
        self.stages = nn.ModuleList(
            [
                EMLLocalStageBlock(
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    bank_dim=bank_dim,
                    window_size=local_window_size,
                    clip_value=clip_value,
                    dropout=dropout,
                )
                for _ in range(num_stages)
            ]
        )
        self.downsamples = nn.ModuleList([ConvSpatialDownsample(feature_dim) for _ in range(num_stages - 1)])
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
        self.classifier = EMLPrototypeClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            clip_value=clip_value,
            prototype_temperature=prototype_temperature,
        )

        for module in self.readout.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        warmup_eta: float = 1.0,
    ) -> Dict[str, Any]:
        features_2d = self.stem(x)
        tokens = features_2d.permute(0, 2, 3, 1).contiguous()
        block_stats: List[Dict[str, torch.Tensor]] = []

        for stage_index, stage in enumerate(self.stages):
            stage_out = stage(tokens, warmup_eta=warmup_eta)
            tokens = stage_out["output"]
            block_stats.append(stage_out["message_stats"])
            block_stats.append(stage_out["channel_stats"])
            if stage_index < len(self.downsamples):
                tokens = self.downsamples[stage_index](tokens)

        flat_tokens = tokens.view(tokens.size(0), tokens.size(1) * tokens.size(2), tokens.size(-1))
        pooled = self.pool(flat_tokens, warmup_eta=warmup_eta)
        features = self.readout(pooled["pooled"])

        head = self.classifier(features, warmup_eta=warmup_eta)
        head["features"] = features
        head["tokens"] = flat_tokens
        head["block_stats"] = block_stats
        head["pool_weights"] = pooled["weights"]
        head["pool_energy"] = pooled["energy"]
        return head


def build_mnist_eml_model(config: Optional[Dict[str, Any]] = None, **overrides: Any) -> nn.Module:
    merged: Dict[str, Any] = {
        "model_name": "cnn_eml",
        "num_classes": 10,
        "image_size": 28,
        "patch_size": 7,
        "patch_stride": 4,
        "input_channels": 1,
        "feature_dim": 128,
        "hidden_dim": 256,
        "bank_dim": 128,
        "bank_blocks": 2,
        "clip_value": 3.0,
        "dropout": 0.1,
        "prototype_temperature": 0.25,
        "local_window_size": 3,
        "merge_every": 2,
    }
    if config is not None:
        for key in list(merged.keys()):
            if key in config:
                merged[key] = config[key]
    merged.update(overrides)
    model_name = merged.pop("model_name")

    if model_name == "cnn_eml":
        merged.pop("image_size", None)
        merged.pop("patch_size", None)
        merged.pop("patch_stride", None)
        merged.pop("local_window_size", None)
        merged.pop("merge_every", None)
        return MNISTEMLNet(**merged)
    if model_name == "cnn_eml_stage":
        merged.pop("image_size", None)
        merged.pop("patch_size", None)
        merged.pop("patch_stride", None)
        merged.pop("merge_every", None)
        return CNNEMLStageNet(**merged)
    if model_name == "pure_eml":
        merged.pop("patch_stride", None)
        merged.pop("local_window_size", None)
        merged.pop("merge_every", None)
        return PureEMLMNISTNet(**merged)
    if model_name == "pure_eml_v2":
        return PureEMLV2MNISTNet(**merged)
    raise ValueError(f"Unsupported model_name: {model_name}")
