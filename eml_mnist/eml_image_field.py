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
from .model import EMLPrototypeClassifier
from .primitives import EMLUnit, _reset_linear


def _position_features(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    radius = torch.sqrt(xx.square() + yy.square()).clamp_max(1.5)
    return torch.stack([yy, xx, radius], dim=-1).view(height * width, 3)


def _downsampled_shape(height: int, width: int, region_size: int) -> tuple[int, int]:
    return (
        (height + region_size - 1) // region_size,
        (width + region_size - 1) // region_size,
    )


def _stats(prefix: str, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    tensor_fp32 = tensor.detach().to(dtype=torch.float32)
    return {
        f"{prefix}_mean": tensor_fp32.mean(),
        f"{prefix}_std": tensor_fp32.std(unbiased=False),
    }


def _safe_zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(reference.shape)


class _ThinImageSensor2d(nn.Module):
    """Small local convolutional image sensor feeding the EML field stack."""

    def __init__(
        self,
        input_channels: int,
        sensor_dim: int,
        patch_size: int = 5,
        patch_stride: int = 4,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or sensor_dim <= 0:
            raise ValueError("input_channels and sensor_dim must be positive")
        if patch_size <= 0 or patch_stride <= 0:
            raise ValueError("patch_size and patch_stride must be positive")

        hidden_dim = max(16, sensor_dim // 2)
        padding = patch_size // 2
        self.sensor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, sensor_dim, kernel_size=patch_size, stride=patch_stride, padding=padding),
            nn.GroupNorm(1, sensor_dim),
            nn.GELU(),
        )
        self.sensor_dim = sensor_dim
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width]")
        features = self.sensor(images)
        batch_size, sensor_dim, height, width = features.shape
        flattened = features.permute(0, 2, 3, 1).reshape(batch_size, height * width, sensor_dim)
        return {
            "grid_features": features,
            "flattened_features": flattened,
            "image_shape": torch.tensor([height, width], device=images.device),
        }


class EMLImageFieldEncoder(nn.Module):
    """EML-native image encoder built from local energy-field primitives."""

    def __init__(
        self,
        input_channels: int = 3,
        sensor_dim: int = 32,
        measurement_dim: int = 32,
        field_dim: int = 32,
        hidden_dim: int = 64,
        num_hypotheses: int = 4,
        num_parent_hypotheses: int = 4,
        num_attractors: int = 4,
        representation_dim: int | None = None,
        patch_size: int = 5,
        patch_stride: int = 4,
        local_window_size: int = 3,
        parent_window_size: int = 3,
        composition_region_size: int = 2,
        clip_value: float = 3.0,
        enable_parent_consensus: bool = True,
    ) -> None:
        super().__init__()
        if sensor_dim <= 0 or measurement_dim <= 0 or field_dim <= 0 or hidden_dim <= 0:
            raise ValueError("sensor_dim, measurement_dim, field_dim, and hidden_dim must be positive")
        if num_hypotheses <= 0 or num_parent_hypotheses <= 0 or num_attractors <= 0:
            raise ValueError("num_hypotheses, num_parent_hypotheses, and num_attractors must be positive")

        self.field_dim = field_dim
        self.num_hypotheses = num_hypotheses
        self.num_parent_hypotheses = num_parent_hypotheses
        self.composition_region_size = composition_region_size
        self.enable_parent_consensus = enable_parent_consensus

        self.image_sensor = _ThinImageSensor2d(
            input_channels=input_channels,
            sensor_dim=sensor_dim,
            patch_size=patch_size,
            patch_stride=patch_stride,
        )
        self.sensor = EMLSensor(
            input_dim=sensor_dim,
            measurement_dim=measurement_dim,
            seed_dim=field_dim,
            hidden_dim=hidden_dim,
            position_dim=3,
        )
        self.local_field = EMLHypothesisField(
            measurement_dim=measurement_dim,
            field_dim=field_dim,
            num_hypotheses=num_hypotheses,
            hidden_dim=hidden_dim,
            clip_value=clip_value,
        )
        self.local_drive_seed_proj = nn.Linear(field_dim, num_hypotheses)
        self.local_resistance_seed_proj = nn.Linear(field_dim, num_hypotheses)
        self.local_energy = EMLUnit(dim=num_hypotheses, clip_value=clip_value, init_bias=0.0)
        self.local_competition = EMLHypothesisCompetition(top_k=num_hypotheses)
        self.local_consensus = EMLConsensusField(
            field_dim=field_dim,
            hidden_dim=hidden_dim,
            num_hypotheses=num_hypotheses,
            mode="image",
            window_size=local_window_size,
            clip_value=clip_value,
        )
        self.composition = EMLCompositionField(
            field_dim=field_dim,
            hidden_dim=hidden_dim,
            mode="image",
            region_size=composition_region_size,
            num_parent_hypotheses=num_parent_hypotheses,
            clip_value=clip_value,
        )
        self.parent_consensus = (
            EMLConsensusField(
                field_dim=field_dim,
                hidden_dim=hidden_dim,
                num_hypotheses=num_parent_hypotheses,
                mode="image",
                window_size=parent_window_size,
                clip_value=clip_value,
            )
            if enable_parent_consensus
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
        _reset_linear(self.local_drive_seed_proj)
        _reset_linear(self.local_resistance_seed_proj)

    def _summarize_stage(self, prefix: str, stage: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        stats: Dict[str, torch.Tensor] = {}
        for key in ("drive", "resistance", "energy", "activation", "support", "conflict", "gate_mass"):
            value = stage.get(key)
            if torch.is_tensor(value):
                stats.update(_stats(f"{prefix}_{key}", value))
        return stats

    def _normalize_parent_stage(
        self,
        composition_out: Dict[str, torch.Tensor | Dict[str, torch.Tensor]],
        parent_consensus_out: Dict[str, torch.Tensor] | None,
    ) -> Dict[str, torch.Tensor]:
        parent_state = composition_out["parent_state"]  # type: ignore[index]
        if parent_consensus_out is not None:
            return {
                "state": parent_state,
                "drive": parent_consensus_out["drive"],
                "resistance": parent_consensus_out["resistance"],
                "energy": parent_consensus_out["energy"],
                "activation": parent_consensus_out["activation"],
                "support": parent_consensus_out["support"],
                "conflict": parent_consensus_out["conflict"],
                "gate_mass": parent_consensus_out["gate_mass"],
            }

        parent_drive = composition_out["parent_drive"]  # type: ignore[index]
        parent_resistance = composition_out["parent_resistance"]  # type: ignore[index]
        parent_energy = composition_out["parent_energy"]  # type: ignore[index]
        parent_activation = composition_out["parent_activation"]  # type: ignore[index]
        return {
            "state": parent_state,
            "drive": parent_drive,
            "resistance": parent_resistance,
            "energy": parent_energy,
            "activation": parent_activation,
            "support": _safe_zero(parent_drive),
            "conflict": _safe_zero(parent_drive),
            "gate_mass": _safe_zero(parent_drive),
        }

    def forward(
        self,
        images: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, Any]:
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width]")

        image_sensor_out = self.image_sensor(images)
        sensor_features = image_sensor_out["flattened_features"]
        grid_height, grid_width = image_sensor_out["grid_features"].shape[-2:]
        position_features = _position_features(grid_height, grid_width, images.device, images.dtype)

        sensor_out = self.sensor(sensor_features, position_features=position_features)
        local_field_out = self.local_field(sensor_out["measurement"], warmup_eta=warmup_eta)
        local_drive = local_field_out["drive"] + self.local_drive_seed_proj(sensor_out["drive_seed"])
        local_resistance = local_field_out["resistance"] + F.softplus(
            self.local_resistance_seed_proj(sensor_out["resistance_seed"])
        )
        local_energy = self.local_energy(local_drive, local_resistance, warmup_eta=warmup_eta)
        local_competition_out = self.local_competition(
            energy=local_energy,
            resistance=local_resistance,
        )
        local_consensus_out = self.local_consensus(
            hypothesis_state=local_field_out["hypothesis_state"],
            activation=local_competition_out["activation"],
            drive=local_drive,
            resistance=local_resistance,
            image_shape=(grid_height, grid_width),
            warmup_eta=warmup_eta,
        )

        composition_out = self.composition(
            hypothesis_state=local_field_out["hypothesis_state"],
            activation=local_consensus_out["activation"],
            image_shape=(grid_height, grid_width),
            warmup_eta=warmup_eta,
        )
        parent_shape = _downsampled_shape(grid_height, grid_width, self.composition_region_size)
        parent_consensus_out = None
        if self.parent_consensus is not None:
            parent_consensus_out = self.parent_consensus(
                hypothesis_state=composition_out["parent_state"],  # type: ignore[arg-type]
                activation=composition_out["parent_activation"],  # type: ignore[arg-type]
                drive=composition_out["parent_drive"],  # type: ignore[arg-type]
                resistance=composition_out["parent_resistance"],  # type: ignore[arg-type]
                image_shape=parent_shape,
                warmup_eta=warmup_eta,
            )
        parent_stage = self._normalize_parent_stage(composition_out, parent_consensus_out)

        attractor_out = self.attractor(
            hypothesis_state=parent_stage["state"],
            activation=parent_stage["activation"],
            warmup_eta=warmup_eta,
        )
        readout_out = self.readout(
            attractor_states=attractor_out["attractor_states"],
            attractor_activation=attractor_out["attractor_activation"],
            warmup_eta=warmup_eta,
        )

        local_hypotheses = {
            "state": local_field_out["hypothesis_state"],
            "drive": local_consensus_out["drive"],
            "resistance": local_consensus_out["resistance"],
            "energy": local_consensus_out["energy"],
            "activation": local_consensus_out["activation"],
            "support": local_consensus_out["support"],
            "conflict": local_consensus_out["conflict"],
            "gate_mass": local_consensus_out["gate_mass"],
        }
        parent_hypotheses = {
            "state": parent_stage["state"],
            "drive": parent_stage["drive"],
            "resistance": parent_stage["resistance"],
            "energy": parent_stage["energy"],
            "activation": parent_stage["activation"],
            "support": parent_stage["support"],
            "conflict": parent_stage["conflict"],
            "gate_mass": parent_stage["gate_mass"],
        }

        budget_loss = (
            local_field_out["budget_loss"]
            + local_competition_out["budget_loss"]
            + local_consensus_out["budget_loss"]
            + composition_out["diagnostics"]["budget_loss"]  # type: ignore[index]
            + attractor_out["budget_loss"]
        )
        if parent_consensus_out is not None:
            budget_loss = budget_loss + parent_consensus_out["budget_loss"]

        diagnostics = {
            "sensor": {
                "measurement": sensor_out["measurement"],
                "drive_seed": sensor_out["drive_seed"],
                "resistance_seed": sensor_out["resistance_seed"],
            },
            "local": local_hypotheses,
            "parent": parent_hypotheses,
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
                **self._summarize_stage("parent", parent_hypotheses),
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
                "sensor_grid_height": torch.tensor(grid_height, device=images.device),
                "sensor_grid_width": torch.tensor(grid_width, device=images.device),
                "budget_loss": budget_loss.detach(),
            },
        }

        return {
            "representation": readout_out["representation"],
            "event": readout_out["representation"],
            "pooled_representation": readout_out["representation"],
            "token_features": sensor_out["measurement"],
            "global_slot_features": attractor_out["attractor_states"],
            "local_queries": local_field_out["hypothesis_state"].reshape(images.size(0), -1, self.field_dim),
            "local_hypotheses": local_hypotheses,
            "parent_hypotheses": parent_hypotheses,
            "attractor_states": attractor_out["attractor_states"],
            "attractor_activation": attractor_out["attractor_activation"],
            "readout_weights": readout_out["weights"],
            "pool_weights": readout_out["weights"],
            "pool_energy": attractor_out["attractor_energy"],
            "pool_drive": attractor_out["attractor_drive"],
            "pool_resistance": attractor_out["attractor_resistance"],
            "block_stats": [local_hypotheses, parent_hypotheses],
            "diagnostics": diagnostics,
        }


class EMLImageFieldClassifier(nn.Module):
    """EML image-field encoder plus ambiguity-aware prototype classification."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        sensor_dim: int = 32,
        measurement_dim: int = 32,
        field_dim: int = 32,
        hidden_dim: int = 64,
        num_hypotheses: int = 4,
        num_parent_hypotheses: int = 4,
        num_attractors: int = 4,
        representation_dim: int | None = None,
        patch_size: int = 5,
        patch_stride: int = 4,
        local_window_size: int = 3,
        parent_window_size: int = 3,
        composition_region_size: int = 2,
        clip_value: float = 3.0,
        prototype_temperature: float = 0.25,
        enable_parent_consensus: bool = True,
    ) -> None:
        super().__init__()
        representation_dim = representation_dim or field_dim
        self.encoder = EMLImageFieldEncoder(
            input_channels=input_channels,
            sensor_dim=sensor_dim,
            measurement_dim=measurement_dim,
            field_dim=field_dim,
            hidden_dim=hidden_dim,
            num_hypotheses=num_hypotheses,
            num_parent_hypotheses=num_parent_hypotheses,
            num_attractors=num_attractors,
            representation_dim=representation_dim,
            patch_size=patch_size,
            patch_stride=patch_stride,
            local_window_size=local_window_size,
            parent_window_size=parent_window_size,
            composition_region_size=composition_region_size,
            clip_value=clip_value,
            enable_parent_consensus=enable_parent_consensus,
        )
        self.classifier = EMLPrototypeClassifier(
            feature_dim=representation_dim,
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
        encoder_out = self.encoder(images, warmup_eta=warmup_eta)
        classifier_out = self.classifier(encoder_out["representation"], warmup_eta=warmup_eta)

        return {
            "logits": classifier_out["logits"],
            "probs": classifier_out["probs"],
            "representation": encoder_out["representation"],
            "drive": classifier_out["drive"],
            "resistance": classifier_out["resistance"],
            "energy": classifier_out["energy"],
            "ambiguity": classifier_out["ambiguity"],
            "weighted_ambiguity": classifier_out["weighted_ambiguity"],
            "ambiguity_weight": classifier_out["ambiguity_weight"],
            "class_radius": classifier_out["class_radius"],
            "sample_uncertainty": classifier_out["sample_uncertainty"],
            "similarity": classifier_out["similarity"],
            "prototypes": classifier_out["prototypes"],
            "local_hypotheses": encoder_out["local_hypotheses"],
            "parent_hypotheses": encoder_out["parent_hypotheses"],
            "attractor_states": encoder_out["attractor_states"],
            "attractor_activation": encoder_out["attractor_activation"],
            "diagnostics": encoder_out["diagnostics"],
            "encoder": encoder_out,
        }


__all__ = ["EMLImageFieldClassifier", "EMLImageFieldEncoder"]
