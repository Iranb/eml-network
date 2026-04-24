from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .heads import ClassificationHead
from .primitives import _reset_linear
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


class _ImageStem(nn.Module):
    def __init__(self, input_channels: int, state_dim: int, patch_stride: int) -> None:
        super().__init__()
        if input_channels <= 0 or state_dim <= 0 or patch_stride <= 0:
            raise ValueError("input_channels, state_dim, and patch_stride must be positive")
        hidden_dim = max(16, state_dim // 2)
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, state_dim, kernel_size=3, stride=patch_stride, padding=1),
            nn.GroupNorm(1, state_dim),
            nn.GELU(),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width]")
        features = self.net(images)
        batch_size, channels, height, width = features.shape
        states = features.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        return states, (height, width)


class EfficientEMLImageEncoder(nn.Module):
    """Efficient image representation trunk built from EML responsibility propagation."""

    def __init__(
        self,
        input_channels: int = 3,
        state_dim: int = 128,
        hidden_dim: int = 256,
        num_hypotheses: int = 16,
        num_attractors: int = 4,
        representation_dim: int | None = None,
        patch_stride: int = 4,
        local_window_size: int = 3,
        composition_region_size: int = 2,
        enable_second_stage: bool = True,
        clip_value: float = 3.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or hidden_dim <= 0 or num_hypotheses <= 0 or num_attractors <= 0:
            raise ValueError("invalid image encoder dimensions")
        self.state_dim = state_dim
        self.representation_dim = representation_dim or state_dim
        self.enable_second_stage = enable_second_stage
        self.stem = _ImageStem(input_channels, state_dim, patch_stride)
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
            mode="image",
            window_size=local_window_size,
            clip_value=clip_value,
        )
        self.composition = EMLComposition(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            mode="image",
            region_size=composition_region_size,
            clip_value=clip_value,
        )
        self.parent_propagation = (
            EMLResponsibilityPropagation(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                num_hypotheses=num_hypotheses,
                mode="image",
                window_size=local_window_size,
                clip_value=clip_value,
            )
            if enable_second_stage
            else None
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
        images: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, Any]:
        local_input, image_shape = self.stem(images)
        evidence = self.evidence(local_input)
        local_states = evidence["measurement"]
        propagation = self.propagation(
            local_states,
            drive_seed=evidence["drive_seed"],
            resistance_seed=evidence["resistance_seed"],
            image_shape=image_shape,
            warmup_eta=warmup_eta,
        )
        local_states = propagation["state"]
        composition = self.composition(local_states, image_shape=image_shape, warmup_eta=warmup_eta)
        parent_states = composition["parent_state"]
        parent_shape = composition["parent_shape"]
        parent_stage = None
        if self.parent_propagation is not None:
            parent_stage = self.parent_propagation(
                parent_states,
                image_shape=parent_shape,  # type: ignore[arg-type]
                warmup_eta=warmup_eta,
            )
            parent_states = parent_stage["state"]

        attractor = self.attractor(parent_states, warmup_eta=warmup_eta)
        readout = self.readout(attractor["attractor_states"], warmup_eta=warmup_eta)
        diagnostics = self._merge_diagnostics(
            propagation["diagnostics"],
            composition["diagnostics"],  # type: ignore[arg-type]
            attractor["diagnostics"],  # type: ignore[arg-type]
        )
        if parent_stage is not None:
            diagnostics.update(parent_stage["diagnostics"])
        diagnostics.update(_stats("readout_weight", readout["weights"]))
        diagnostics["local_window_size"] = torch.tensor(float(self.propagation.window_size), device=images.device)
        diagnostics["local_positions"] = torch.tensor(float(local_states.size(1)), device=images.device)
        diagnostics["parent_positions"] = torch.tensor(float(parent_states.size(1)), device=images.device)
        diagnostics["num_attractors"] = torch.tensor(float(attractor["attractor_states"].size(1)), device=images.device)

        return {
            "representation": readout["representation"],
            "local_states": local_states,
            "local_queries": local_states,
            "parent_states": parent_states,
            "parent_hypotheses": parent_states,
            "attractor_states": attractor["attractor_states"],
            "global_slot_features": attractor["attractor_states"],
            "attractor_weights": attractor["attractor_weights"],
            "attractor_activation": attractor["update_strength"],
            "readout_weights": readout["weights"],
            "drive": readout["drive"],
            "resistance": readout["resistance"],
            "energy": readout["energy"],
            "image_shape": image_shape,
            "parent_shape": parent_shape,
            "diagnostics": diagnostics,
            "propagation": propagation,
            "composition": composition,
            "attractor": attractor,
            "readout": readout,
        }


class EfficientEMLImageClassifier(nn.Module):
    """Efficient EML image encoder plus ambiguity-aware prototype classification."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        state_dim: int = 128,
        hidden_dim: int = 256,
        num_hypotheses: int = 16,
        num_attractors: int = 4,
        representation_dim: int | None = None,
        patch_stride: int = 4,
        local_window_size: int = 3,
        composition_region_size: int = 2,
        clip_value: float = 3.0,
    ) -> None:
        super().__init__()
        representation_dim = representation_dim or state_dim
        self.encoder = EfficientEMLImageEncoder(
            input_channels=input_channels,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_hypotheses=num_hypotheses,
            num_attractors=num_attractors,
            representation_dim=representation_dim,
            patch_stride=patch_stride,
            local_window_size=local_window_size,
            composition_region_size=composition_region_size,
            clip_value=clip_value,
        )
        self.classifier = ClassificationHead(
            input_dim=representation_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            clip_value=clip_value,
            temperature=0.25,
        )

    def forward(
        self,
        images: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, Any]:
        encoder_out = self.encoder(images, warmup_eta=warmup_eta)
        head = self.classifier(encoder_out["representation"], warmup_eta=warmup_eta)
        return {
            **head,
            "representation": encoder_out["representation"],
            "local_states": encoder_out["local_states"],
            "parent_states": encoder_out["parent_states"],
            "attractor_states": encoder_out["attractor_states"],
            "attractor_activation": encoder_out["attractor_activation"],
            "diagnostics": encoder_out["diagnostics"],
            "encoder": encoder_out,
        }


__all__ = ["EfficientEMLImageClassifier", "EfficientEMLImageEncoder"]
