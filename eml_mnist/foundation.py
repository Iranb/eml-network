from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict

import torch
import torch.nn as nn

from .eml_image_field import EMLImageFieldEncoder
from .eml_text_field import EMLTextFieldEncoder, EMLTextFieldGenerationHead
from .graph import EMLSlotGraphLayer, SlotBank
from .heads import (
    ActionHead,
    ClassificationHead,
    LocalReconstructionHead,
    PatchRankHead,
    PrototypeNoveltyHead,
    RepresentationHead,
    RiskResistanceHead,
)
from .image_backbones import PureEMLImageBackbone
from .text_backbones import EMLTextBackbone
from .text_heads import LocalTextGenerationHead


def _collect_nested_stats(
    container: Any,
    names: Sequence[str] = ("drive", "resistance", "energy", "gate", "active_route_strength", "gate_mass"),
) -> Dict[str, float]:
    collected = {name: [] for name in names}

    def _visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in collected and torch.is_tensor(item):
                    collected[key].append(item.detach().float().reshape(-1))
                _visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _visit(item)

    _visit(container)

    stats: Dict[str, float] = {}
    for name, tensors in collected.items():
        if not tensors:
            continue
        flat = torch.cat(tensors, dim=0)
        stats[f"{name}_mean"] = float(flat.mean().item())
        stats[f"{name}_std"] = float(flat.std(unbiased=False).item())
    if "gate_mean" in stats:
        stats["gate_activation_rate"] = stats["gate_mean"]
    return stats


class EMLFoundationCore(nn.Module):
    """General EML-native foundation core with optional image and text paths."""

    def __init__(
        self,
        slot_dim: int,
        event_dim: int,
        hidden_dim: int,
        slot_layout: Mapping[str, int] | Sequence[str],
        num_layers: int = 2,
        top_k: int = 4,
        representation_dim: int | None = None,
        action_dim: int | None = None,
        patch_dim: int | None = None,
        local_query_dim: int | None = None,
        reconstruction_dim: int | None = None,
        num_risk_outputs: int = 1,
        local_num_queries: int = 4,
        clip_value: float = 3.0,
        enable_action_head: bool = True,
        enable_patch_rank_head: bool = True,
        enable_prototype_novelty: bool = False,
        num_novelty_prototypes: int = 16,
        image_input_channels: int | None = None,
        image_size: int = 32,
        image_patch_size: int = 4,
        image_patch_stride: int | None = None,
        image_feature_dim: int | None = None,
        image_bank_dim: int | None = None,
        image_num_layers: int = 3,
        image_local_window_size: int = 3,
        image_merge_every: int = 2,
        image_num_global_slots: int = 4,
        image_head_specs: Mapping[str, int] | None = None,
        text_vocab_size: int | None = None,
        text_embed_dim: int | None = None,
        text_feature_dim: int | None = None,
        text_hidden_dim: int | None = None,
        text_bank_dim: int | None = None,
        text_num_layers: int = 3,
        text_num_global_slots: int = 4,
        text_pad_id: int = 0,
        enable_text_generation_head: bool = False,
        enable_modality_slot_injection: bool = True,
        modality_slot_injection_mode: str = "residual",
        enable_image_field_encoder: bool = False,
        enable_text_field_encoder: bool = False,
        image_field_config: Mapping[str, Any] | None = None,
        text_field_config: Mapping[str, Any] | None = None,
        inject_attractors: bool = True,
        attractor_injection_mode: str = "residual",
    ) -> None:
        super().__init__()
        if slot_dim <= 0 or event_dim <= 0 or hidden_dim <= 0:
            raise ValueError("slot_dim, event_dim, and hidden_dim must be positive")
        if num_layers <= 0 or top_k <= 0:
            raise ValueError("num_layers and top_k must be positive")
        if num_risk_outputs <= 0:
            raise ValueError("num_risk_outputs must be positive")

        representation_dim = representation_dim or slot_dim
        action_dim = action_dim or slot_dim
        patch_dim = patch_dim or slot_dim
        local_query_dim = local_query_dim or representation_dim
        reconstruction_dim = reconstruction_dim or event_dim
        image_feature_dim = image_feature_dim or local_query_dim
        image_bank_dim = image_bank_dim or hidden_dim
        text_embed_dim = text_embed_dim or local_query_dim
        text_feature_dim = text_feature_dim or local_query_dim
        text_hidden_dim = text_hidden_dim or hidden_dim
        text_bank_dim = text_bank_dim or hidden_dim
        if modality_slot_injection_mode not in {"residual", "overwrite"}:
            raise ValueError("modality_slot_injection_mode must be 'residual' or 'overwrite'")
        if attractor_injection_mode not in {"residual", "overwrite"}:
            raise ValueError("attractor_injection_mode must be 'residual' or 'overwrite'")

        image_field_config_dict = dict(image_field_config or {})
        text_field_config_dict = dict(text_field_config or {})
        image_field_dim = int(image_field_config_dict.get("field_dim", image_feature_dim))
        image_field_representation_dim = int(image_field_config_dict.get("representation_dim", image_field_dim))
        text_field_dim = int(text_field_config_dict.get("field_dim", text_feature_dim))
        text_field_representation_dim = int(text_field_config_dict.get("representation_dim", text_field_dim))

        self.slot_dim = slot_dim
        self.event_dim = event_dim
        self.representation_dim = representation_dim
        self.local_query_dim = local_query_dim
        self.enable_modality_slot_injection = enable_modality_slot_injection
        self.modality_slot_injection_mode = modality_slot_injection_mode
        self.inject_attractors = inject_attractors
        self.attractor_injection_mode = attractor_injection_mode
        self.image_field_dim = image_field_dim
        self.text_field_dim = text_field_dim
        self.slot_bank = SlotBank(slot_dim=slot_dim, slot_layout=slot_layout)
        self.image_slot_proj = nn.Linear(image_feature_dim, slot_dim)
        self.text_slot_proj = nn.Linear(text_feature_dim, slot_dim)
        self.image_query_proj = nn.Linear(image_feature_dim, local_query_dim)
        self.text_query_proj = nn.Linear(text_feature_dim, local_query_dim)
        self.image_field_event_proj = nn.Linear(image_field_representation_dim, event_dim)
        self.text_field_event_proj = nn.Linear(text_field_representation_dim, event_dim)
        self.image_attractor_slot_proj = nn.Linear(image_field_dim, slot_dim)
        self.text_attractor_slot_proj = nn.Linear(text_field_dim, slot_dim)
        self.image_field_query_proj = nn.Linear(image_field_dim, local_query_dim)
        self.text_field_query_proj = nn.Linear(text_field_dim, local_query_dim)
        self.text_field_token_proj = nn.Linear(text_field_dim, text_feature_dim)
        nn.init.normal_(self.image_slot_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.image_slot_proj.bias)
        nn.init.normal_(self.text_slot_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.text_slot_proj.bias)
        nn.init.normal_(self.image_query_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.image_query_proj.bias)
        nn.init.normal_(self.text_query_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.text_query_proj.bias)
        for projector in (
            self.image_field_event_proj,
            self.text_field_event_proj,
            self.image_attractor_slot_proj,
            self.text_attractor_slot_proj,
            self.image_field_query_proj,
            self.text_field_query_proj,
            self.text_field_token_proj,
        ):
            nn.init.normal_(projector.weight, mean=0.0, std=0.01)
            nn.init.zeros_(projector.bias)
        self.graph_layers = nn.ModuleList(
            [
                EMLSlotGraphLayer(
                    slot_dim=slot_dim,
                    event_dim=event_dim,
                    hidden_dim=hidden_dim,
                    top_k=top_k,
                    clip_value=clip_value,
                )
                for _ in range(num_layers)
            ]
        )
        self.representation_head = RepresentationHead(
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
            representation_dim=representation_dim,
            clip_value=clip_value,
        )
        self.risk_resistance_head = RiskResistanceHead(
            input_dim=representation_dim,
            hidden_dim=hidden_dim,
            output_dim=num_risk_outputs,
            clip_value=clip_value,
        )
        self.local_reconstruction_head = LocalReconstructionHead(
            context_dim=representation_dim,
            query_dim=local_query_dim,
            output_dim=reconstruction_dim,
            hidden_dim=hidden_dim,
            num_queries=local_num_queries,
            clip_value=clip_value,
        )

        self.action_head = (
            ActionHead(
                context_dim=representation_dim,
                candidate_dim=action_dim,
                hidden_dim=hidden_dim,
                clip_value=clip_value,
            )
            if enable_action_head
            else None
        )
        self.patch_rank_head = (
            PatchRankHead(
                context_dim=representation_dim,
                candidate_dim=patch_dim,
                hidden_dim=hidden_dim,
                clip_value=clip_value,
            )
            if enable_patch_rank_head
            else None
        )
        self.prototype_novelty_head = (
            PrototypeNoveltyHead(
                input_dim=representation_dim,
                hidden_dim=hidden_dim,
                num_prototypes=num_novelty_prototypes,
                clip_value=clip_value,
            )
            if enable_prototype_novelty
            else None
        )

        self.image_backbone = (
            PureEMLImageBackbone(
                image_size=image_size,
                input_channels=image_input_channels,
                feature_dim=image_feature_dim,
                event_dim=event_dim,
                hidden_dim=hidden_dim,
                bank_dim=image_bank_dim,
                num_layers=image_num_layers,
                patch_size=image_patch_size,
                patch_stride=image_patch_stride or image_patch_size,
                local_window_size=image_local_window_size,
                merge_every=image_merge_every,
                clip_value=clip_value,
                dropout=0.0,
                num_global_slots=image_num_global_slots,
            )
            if image_input_channels is not None
            else None
        )
        if enable_image_field_encoder:
            image_field_config_dict.setdefault("input_channels", image_input_channels or 3)
            image_field_config_dict.setdefault("sensor_dim", image_field_dim)
            image_field_config_dict.setdefault("measurement_dim", image_field_dim)
            image_field_config_dict.setdefault("field_dim", image_field_dim)
            image_field_config_dict.setdefault("hidden_dim", hidden_dim)
            image_field_config_dict.setdefault("representation_dim", image_field_representation_dim)
            self.image_field_encoder = EMLImageFieldEncoder(**image_field_config_dict)
        else:
            self.image_field_encoder = None
        self.image_heads = nn.ModuleDict()
        if image_head_specs is not None:
            for name, num_classes in image_head_specs.items():
                self.image_heads[name] = ClassificationHead(
                    input_dim=representation_dim,
                    num_classes=int(num_classes),
                    hidden_dim=hidden_dim,
                    clip_value=clip_value,
                )

        self.text_backbone = (
            EMLTextBackbone(
                vocab_size=text_vocab_size,
                embed_dim=text_embed_dim,
                feature_dim=text_feature_dim,
                event_dim=event_dim,
                hidden_dim=text_hidden_dim,
                bank_dim=text_bank_dim,
                num_layers=text_num_layers,
                clip_value=clip_value,
                dropout=0.0,
                pad_id=text_pad_id,
                num_global_slots=text_num_global_slots,
            )
            if text_vocab_size is not None
            else None
        )
        if enable_text_field_encoder:
            if text_vocab_size is None:
                raise ValueError("text_vocab_size is required when enable_text_field_encoder is true")
            text_field_config_dict.setdefault("vocab_size", text_vocab_size)
            text_field_config_dict.setdefault("embed_dim", text_embed_dim)
            text_field_config_dict.setdefault("sensor_dim", text_field_dim)
            text_field_config_dict.setdefault("measurement_dim", text_field_dim)
            text_field_config_dict.setdefault("field_dim", text_field_dim)
            text_field_config_dict.setdefault("hidden_dim", text_hidden_dim)
            text_field_config_dict.setdefault("representation_dim", text_field_representation_dim)
            text_field_config_dict.setdefault("pad_id", text_pad_id)
            self.text_field_encoder = EMLTextFieldEncoder(**text_field_config_dict)
        else:
            self.text_field_encoder = None
        self.text_generation_head = (
            LocalTextGenerationHead(
                context_dim=representation_dim,
                token_dim=text_feature_dim,
                hidden_dim=hidden_dim,
                vocab_size=text_vocab_size,
                clip_value=clip_value,
            )
            if enable_text_generation_head and text_vocab_size is not None
            else None
        )
        self.text_field_generation_head = (
            EMLTextFieldGenerationHead(
                state_dim=text_field_dim,
                hidden_dim=text_hidden_dim,
                vocab_size=text_vocab_size,
                clip_value=clip_value,
            )
            if enable_text_field_encoder and text_vocab_size is not None
            else None
        )

    def _project_local_queries(self, local_queries: torch.Tensor, modality: str) -> torch.Tensor:
        if local_queries.size(-1) == self.local_query_dim:
            return local_queries
        projector = self.image_query_proj if modality == "image" else self.text_query_proj
        if local_queries.size(-1) != projector.in_features:
            raise ValueError(f"{modality} local_queries last dimension does not match configured feature_dim")
        return projector(local_queries)

    def _resolve_event_inputs(
        self,
        event: torch.Tensor | None,
        image_inputs: torch.Tensor | None,
        image_backbone_outputs: Dict[str, Any] | None,
        image_field_outputs: Dict[str, Any] | None,
        text_inputs: torch.Tensor | None,
        text_backbone_outputs: Dict[str, Any] | None,
        text_field_outputs: Dict[str, Any] | None,
        text_padding_mask: torch.Tensor | None,
        local_queries: torch.Tensor | None,
        warmup_eta: float | torch.Tensor,
        use_field_path: bool,
    ) -> tuple[torch.Tensor, Dict[str, Any], torch.Tensor | None]:
        outputs: Dict[str, Any] = {}
        resolved_local_queries = local_queries

        if event is not None:
            if event.ndim != 2 or event.size(-1) != self.event_dim:
                raise ValueError("event must have shape [batch, event_dim]")
            outputs["event_input"] = {"event": event}
            return event, outputs, resolved_local_queries

        if image_field_outputs is not None:
            image_out = image_field_outputs
            outputs["image_field"] = image_out
            if resolved_local_queries is None:
                resolved_local_queries = self.image_field_query_proj(image_out["local_queries"])
            return self.image_field_event_proj(image_out["representation"]), outputs, resolved_local_queries

        if image_inputs is not None and use_field_path and self.image_field_encoder is not None:
            image_out = self.image_field_encoder(image_inputs, warmup_eta=warmup_eta)
            outputs["image_field"] = image_out
            if resolved_local_queries is None:
                resolved_local_queries = self.image_field_query_proj(image_out["local_queries"])
            return self.image_field_event_proj(image_out["representation"]), outputs, resolved_local_queries

        if image_backbone_outputs is not None:
            image_out = image_backbone_outputs
            outputs["image_backbone"] = image_out
            if resolved_local_queries is None:
                resolved_local_queries = self._project_local_queries(image_out["local_queries"], "image")
            return image_out["event"], outputs, resolved_local_queries

        if image_inputs is not None:
            if self.image_backbone is None:
                raise ValueError("this EMLFoundationCore was created without an image backbone")
            image_out = self.image_backbone(image_inputs, warmup_eta=warmup_eta)
            outputs["image_backbone"] = image_out
            if resolved_local_queries is None:
                resolved_local_queries = self._project_local_queries(image_out["local_queries"], "image")
            return image_out["event"], outputs, resolved_local_queries

        if text_field_outputs is not None:
            text_out = text_field_outputs
            outputs["text_field"] = text_out
            if resolved_local_queries is None:
                resolved_local_queries = self.text_field_query_proj(text_out["sequence_states"])
            return self.text_field_event_proj(text_out["representation"]), outputs, resolved_local_queries

        if text_inputs is not None and use_field_path and self.text_field_encoder is not None:
            text_out = self.text_field_encoder(text_inputs, padding_mask=text_padding_mask, warmup_eta=warmup_eta)
            outputs["text_field"] = text_out
            if resolved_local_queries is None:
                resolved_local_queries = self.text_field_query_proj(text_out["sequence_states"])
            return self.text_field_event_proj(text_out["representation"]), outputs, resolved_local_queries

        if text_backbone_outputs is not None:
            text_out = text_backbone_outputs
            outputs["text_backbone"] = text_out
            if resolved_local_queries is None:
                resolved_local_queries = self._project_local_queries(text_out["local_queries"], "text")
            return text_out["event"], outputs, resolved_local_queries

        if text_inputs is not None:
            if self.text_backbone is None:
                raise ValueError("this EMLFoundationCore was created without a text backbone")
            text_out = self.text_backbone(input_ids=text_inputs, padding_mask=text_padding_mask, warmup_eta=warmup_eta)
            outputs["text_backbone"] = text_out
            if resolved_local_queries is None:
                resolved_local_queries = self._project_local_queries(text_out["local_queries"], "text")
            return text_out["event"], outputs, resolved_local_queries

        raise ValueError("one of event, image backbone/input, or text backbone/input must be provided")

    def _slot_indices_for_modality(self, modality: str, device: torch.device) -> torch.Tensor:
        matching_type_ids = [
            type_id
            for type_id, type_name in enumerate(self.slot_bank.slot_type_names)
            if modality in type_name.lower()
        ]
        if not matching_type_ids:
            return torch.empty(0, device=device, dtype=torch.long)
        type_ids = self.slot_bank.slot_type_ids.to(device=device)
        mask = torch.zeros_like(type_ids, dtype=torch.bool)
        for type_id in matching_type_ids:
            mask |= type_ids == type_id
        return torch.nonzero(mask, as_tuple=False).flatten()

    def _slot_indices_for_attractors(self, modality: str, device: torch.device) -> torch.Tensor:
        modality_type_ids = []
        fallback_type_ids = []
        for type_id, type_name in enumerate(self.slot_bank.slot_type_names):
            lowered = type_name.lower()
            if modality in lowered:
                modality_type_ids.append(type_id)
            elif "attractor" in lowered:
                fallback_type_ids.append(type_id)

        matching_type_ids = modality_type_ids or fallback_type_ids
        if not matching_type_ids:
            return torch.empty(0, device=device, dtype=torch.long)
        type_ids = self.slot_bank.slot_type_ids.to(device=device)
        mask = torch.zeros_like(type_ids, dtype=torch.bool)
        for type_id in matching_type_ids:
            mask |= type_ids == type_id
        return torch.nonzero(mask, as_tuple=False).flatten()

    def _inject_modality_slots(
        self,
        bank: Dict[str, torch.Tensor],
        modality_slots: torch.Tensor | None,
        modality: str,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {
            "injected_slots": 0,
            "slot_indices": [],
            "skipped": True,
        }
        if modality_slots is None:
            return bank, diagnostics
        if modality_slots.ndim == 2:
            modality_slots = modality_slots.unsqueeze(1)
        if modality_slots.ndim != 3:
            raise ValueError("modality_slots must have shape [batch, slots, feature_dim]")
        if modality_slots.size(0) != bank["slot_states"].size(0):
            raise ValueError("modality_slots batch size must match the slot bank")

        slot_indices = self._slot_indices_for_modality(modality, bank["slot_states"].device)
        if slot_indices.numel() == 0:
            diagnostics["skipped"] = True
            return bank, diagnostics

        projector = self.image_slot_proj if modality == "image" else self.text_slot_proj
        projected_slots = projector(modality_slots)
        injected_count = min(projected_slots.size(1), slot_indices.numel())
        if injected_count <= 0:
            return bank, diagnostics

        selected_indices = slot_indices[:injected_count]
        selected_updates = projected_slots[:, :injected_count, :]
        updated_slot_states = bank["slot_states"].clone()
        if self.modality_slot_injection_mode == "overwrite":
            updated_slot_states[:, selected_indices, :] = selected_updates
        else:
            updated_slot_states[:, selected_indices, :] = updated_slot_states[:, selected_indices, :] + selected_updates

        updated_bank = self.slot_bank.compose(updated_slot_states)
        updated_bank["slot_mask"] = bank["slot_mask"]
        diagnostics = {
            "injected_slots": int(injected_count),
            "slot_indices": [int(index) for index in selected_indices.detach().cpu().tolist()],
            "skipped": False,
        }
        return updated_bank, diagnostics

    def _inject_attractor_slots(
        self,
        bank: Dict[str, torch.Tensor],
        attractor_states: torch.Tensor | None,
        modality: str,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {
            "injected_slots": 0,
            "injection_norm": 0.0,
            "slot_indices": [],
            "skipped_reason": None,
        }
        if attractor_states is None:
            diagnostics["skipped_reason"] = "no_attractor_states"
            return bank, diagnostics
        if attractor_states.ndim != 3:
            raise ValueError("attractor_states must have shape [batch, attractors, field_dim]")
        if attractor_states.size(0) != bank["slot_states"].size(0):
            raise ValueError("attractor_states batch size must match the slot bank")

        slot_indices = self._slot_indices_for_attractors(modality, bank["slot_states"].device)
        if slot_indices.numel() == 0:
            diagnostics["skipped_reason"] = "no_matching_slots"
            return bank, diagnostics

        projector = self.image_attractor_slot_proj if modality == "image" else self.text_attractor_slot_proj
        if attractor_states.size(-1) != projector.in_features:
            raise ValueError(f"{modality} attractor_states last dimension does not match configured field_dim")
        projected_slots = projector(attractor_states)
        injected_count = min(projected_slots.size(1), slot_indices.numel())
        if injected_count <= 0:
            diagnostics["skipped_reason"] = "no_available_slots"
            return bank, diagnostics

        selected_indices = slot_indices[:injected_count]
        selected_updates = projected_slots[:, :injected_count, :]
        updated_slot_states = bank["slot_states"].clone()
        if self.attractor_injection_mode == "overwrite":
            updated_slot_states[:, selected_indices, :] = selected_updates
        else:
            updated_slot_states[:, selected_indices, :] = updated_slot_states[:, selected_indices, :] + selected_updates

        updated_bank = self.slot_bank.compose(updated_slot_states)
        updated_bank["slot_mask"] = bank["slot_mask"]
        diagnostics = {
            "injected_slots": int(injected_count),
            "injection_norm": float(selected_updates.detach().norm(dim=-1).mean().item()),
            "slot_indices": [int(index) for index in selected_indices.detach().cpu().tolist()],
            "skipped_reason": None,
        }
        return updated_bank, diagnostics

    def _empty_modality_injection_diagnostics(self, enabled: bool) -> Dict[str, Any]:
        return {
            "enabled": enabled,
            "injected_image_slots": 0,
            "injected_text_slots": 0,
            "image_slot_injection_skipped": True,
            "text_slot_injection_skipped": True,
            "modality_slot_indices": {"image": [], "text": []},
        }

    def _empty_attractor_injection_diagnostics(self, enabled: bool) -> Dict[str, Any]:
        return {
            "enabled": enabled,
            "image": {
                "injected_slots": 0,
                "injection_norm": 0.0,
                "slot_indices": [],
                "skipped_reason": "disabled" if not enabled else "not_requested",
            },
            "text": {
                "injected_slots": 0,
                "injection_norm": 0.0,
                "slot_indices": [],
                "skipped_reason": "disabled" if not enabled else "not_requested",
            },
        }

    def _representation_modality_diagnostics(
        self,
        weights: torch.Tensor,
        modality_slot_indices: Dict[str, list[int]],
    ) -> Dict[str, torch.Tensor]:
        diagnostics: Dict[str, torch.Tensor] = {}
        for modality, indices in modality_slot_indices.items():
            if indices:
                index_tensor = torch.tensor(indices, device=weights.device, dtype=torch.long)
                slot_weight = weights.index_select(1, index_tensor).sum(dim=1)
            else:
                slot_weight = weights.new_zeros(weights.size(0))
            diagnostics[f"{modality}_slot_readout_weight"] = slot_weight
            diagnostics[f"{modality}_slot_readout_weight_mean"] = slot_weight.mean()
        return diagnostics

    def _run_core(
        self,
        event: torch.Tensor,
        slot_states: torch.Tensor | None,
        slot_mask: torch.Tensor | None,
        warmup_eta: float | torch.Tensor,
        top_k: int | None,
        modality_inputs: Dict[str, Any] | None = None,
        inject_modality_slots: bool | None = None,
        inject_attractors: bool | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        bank = self.slot_bank(batch_size=event.size(0), slot_states=slot_states)
        effective_slot_mask = bank["slot_mask"] if slot_mask is None else slot_mask
        if effective_slot_mask.shape != bank["slot_mask"].shape:
            raise ValueError("slot_mask must match the slot bank shape [batch, num_slots]")
        bank["slot_mask"] = effective_slot_mask

        should_inject = self.enable_modality_slot_injection if inject_modality_slots is None else inject_modality_slots
        modality_injection = self._empty_modality_injection_diagnostics(enabled=should_inject)
        if should_inject and modality_inputs is not None:
            image_out = modality_inputs.get("image_backbone")
            if isinstance(image_out, dict) and torch.is_tensor(image_out.get("global_slot_features")):
                bank, image_diag = self._inject_modality_slots(bank, image_out["global_slot_features"], "image")
                modality_injection["injected_image_slots"] = image_diag["injected_slots"]
                modality_injection["image_slot_injection_skipped"] = image_diag["skipped"]
                modality_injection["modality_slot_indices"]["image"] = image_diag["slot_indices"]

            text_out = modality_inputs.get("text_backbone")
            if isinstance(text_out, dict) and torch.is_tensor(text_out.get("global_slot_features")):
                bank, text_diag = self._inject_modality_slots(bank, text_out["global_slot_features"], "text")
                modality_injection["injected_text_slots"] = text_diag["injected_slots"]
                modality_injection["text_slot_injection_skipped"] = text_diag["skipped"]
                modality_injection["modality_slot_indices"]["text"] = text_diag["slot_indices"]
            bank["slot_mask"] = effective_slot_mask

        should_inject_attractors = self.inject_attractors if inject_attractors is None else inject_attractors
        attractor_injection = self._empty_attractor_injection_diagnostics(enabled=should_inject_attractors)
        if should_inject_attractors and modality_inputs is not None:
            image_field_out = modality_inputs.get("image_field")
            if isinstance(image_field_out, dict) and torch.is_tensor(image_field_out.get("attractor_states")):
                bank, image_diag = self._inject_attractor_slots(bank, image_field_out["attractor_states"], "image")
                attractor_injection["image"] = image_diag

            text_field_out = modality_inputs.get("text_field")
            if isinstance(text_field_out, dict) and torch.is_tensor(text_field_out.get("attractor_states")):
                bank, text_diag = self._inject_attractor_slots(bank, text_field_out["attractor_states"], "text")
                attractor_injection["text"] = text_diag
            bank["slot_mask"] = effective_slot_mask

        current_slot_states = bank["slot_states"]
        type_features = bank["type_features"]
        graph_layer_outputs = []
        for layer in self.graph_layers:
            layer_out = layer(
                event=event,
                slot_states=current_slot_states,
                type_features=type_features,
                slot_mask=effective_slot_mask,
                warmup_eta=warmup_eta,
                top_k=top_k,
            )
            current_slot_states = layer_out["slot_states"]
            graph_layer_outputs.append(layer_out)

        bank = self.slot_bank.compose(current_slot_states)
        bank["slot_mask"] = effective_slot_mask
        representation_out = self.representation_head(
            slot_states=bank["slot_states"],
            type_features=bank["type_features"],
            slot_mask=bank["slot_mask"],
            warmup_eta=warmup_eta,
        )
        readout_slot_indices = {
            "image": list(modality_injection["modality_slot_indices"]["image"]),
            "text": list(modality_injection["modality_slot_indices"]["text"]),
        }
        for modality in ("image", "text"):
            for index in attractor_injection[modality]["slot_indices"]:
                if index not in readout_slot_indices[modality]:
                    readout_slot_indices[modality].append(index)
        readout_modality = self._representation_modality_diagnostics(
            representation_out["weights"],
            readout_slot_indices,
        )
        representation_out["modality_slot_readout"] = readout_modality
        return bank, {
            "representation": representation_out["representation"],
            "representation_head": representation_out,
            "graph_layers": graph_layer_outputs,
            "active_slot_indices": graph_layer_outputs[-1]["active_indices"] if graph_layer_outputs else None,
            "modality_injection": modality_injection,
            "attractor_injection": attractor_injection,
            "representation_modality": readout_modality,
        }

    def forward(
        self,
        event: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        image_inputs: torch.Tensor | None = None,
        image_features: Dict[str, Any] | None = None,
        image_backbone_outputs: Dict[str, Any] | None = None,
        image_field_outputs: Dict[str, Any] | None = None,
        text_tokens: torch.Tensor | None = None,
        text_inputs: torch.Tensor | None = None,
        text_features: Dict[str, Any] | None = None,
        text_backbone_outputs: Dict[str, Any] | None = None,
        text_field_outputs: Dict[str, Any] | None = None,
        text_padding_mask: torch.Tensor | None = None,
        slot_states: torch.Tensor | None = None,
        slot_mask: torch.Tensor | None = None,
        candidate_actions: torch.Tensor | None = None,
        candidate_patches: torch.Tensor | None = None,
        local_queries: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
        top_k: int | None = None,
        inject_modality_slots: bool | None = None,
        inject_attractors: bool | None = None,
        use_field_path: bool = True,
    ) -> Dict[str, Any]:
        resolved_event, modality_inputs, resolved_local_queries = self._resolve_event_inputs(
            event=event,
            image_inputs=images if image_inputs is None else image_inputs,
            image_backbone_outputs=image_backbone_outputs if image_features is None else image_features,
            image_field_outputs=image_field_outputs,
            text_inputs=text_tokens if text_inputs is None else text_inputs,
            text_backbone_outputs=text_backbone_outputs if text_features is None else text_features,
            text_field_outputs=text_field_outputs,
            text_padding_mask=text_padding_mask,
            local_queries=local_queries,
            warmup_eta=warmup_eta,
            use_field_path=use_field_path,
        )
        bank, core_out = self._run_core(
            event=resolved_event,
            slot_states=slot_states,
            slot_mask=slot_mask,
            warmup_eta=warmup_eta,
            top_k=top_k,
            modality_inputs=modality_inputs,
            inject_modality_slots=inject_modality_slots,
            inject_attractors=inject_attractors,
        )

        outputs: Dict[str, Any] = {
            **modality_inputs,
            "slot_states": bank["slot_states"],
            "typed_states": bank["typed_states"],
            "slot_mask": bank["slot_mask"],
            "type_ids": bank["type_ids"],
            "type_features": bank["type_features"],
            "representation": core_out["representation"],
            "representation_head": core_out["representation_head"],
            "graph_layers": core_out["graph_layers"],
            "active_slot_indices": core_out["active_slot_indices"],
            "modality_injection": core_out["modality_injection"],
            "attractor_injection": core_out["attractor_injection"],
            "representation_modality": core_out["representation_modality"],
            "risk_resistance": self.risk_resistance_head(core_out["representation"], warmup_eta=warmup_eta),
            "local_reconstruction": self.local_reconstruction_head(
                core_out["representation"],
                local_queries=resolved_local_queries,
                warmup_eta=warmup_eta,
            ),
        }

        if self.action_head is not None and candidate_actions is not None:
            outputs["action"] = self.action_head(core_out["representation"], candidate_actions, warmup_eta=warmup_eta)
        if self.patch_rank_head is not None and candidate_patches is not None:
            outputs["patch_rank"] = self.patch_rank_head(core_out["representation"], candidate_patches, warmup_eta=warmup_eta)
        if self.prototype_novelty_head is not None:
            outputs["prototype_novelty"] = self.prototype_novelty_head(core_out["representation"], warmup_eta=warmup_eta)
        if self.image_heads and any(key in outputs for key in ("image_backbone", "image_field")):
            outputs["image_heads"] = {
                name: head(core_out["representation"], warmup_eta=warmup_eta)
                for name, head in self.image_heads.items()
            }
        if self.text_generation_head is not None and any(key in outputs for key in ("text_backbone", "text_field")):
            text_codec_out = outputs["text_backbone"] if "text_backbone" in outputs else outputs["text_field"]
            sequence_features = text_codec_out["sequence_features"]
            if "text_field" in outputs:
                sequence_features = self.text_field_token_proj(sequence_features)
            outputs["text_generation"] = self.text_generation_head(
                representation=core_out["representation"],
                sequence_features=sequence_features,
                padding_mask=text_codec_out["padding_mask"],
                warmup_eta=warmup_eta,
            )
        if self.text_field_generation_head is not None and "text_field" in outputs:
            text_field_out = outputs["text_field"]
            outputs["text_field_generation"] = self.text_field_generation_head(
                sequence_states=text_field_out["sequence_states"],
                padding_mask=text_field_out["padding_mask"],
                warmup_eta=warmup_eta,
            )

        outputs["diagnostics"] = {
            "router": [layer_out["router"] for layer_out in core_out["graph_layers"]],
            "graph_update": [layer_out["state_update"] for layer_out in core_out["graph_layers"]],
            "graph": core_out["graph_layers"],
            "representation": outputs["representation_head"],
            "modality_injection": outputs["modality_injection"],
            "attractor_injection": outputs["attractor_injection"],
            "representation_modality": outputs["representation_modality"],
            "risk_resistance": outputs["risk_resistance"],
            "local_reconstruction": outputs["local_reconstruction"],
            "active_slot_indices": outputs["active_slot_indices"],
            "stats": _collect_nested_stats(outputs),
        }
        if "action" in outputs:
            outputs["diagnostics"]["action"] = outputs["action"]
        if "patch_rank" in outputs:
            outputs["diagnostics"]["patch_rank"] = outputs["patch_rank"]
        if "prototype_novelty" in outputs:
            outputs["diagnostics"]["prototype_novelty"] = outputs["prototype_novelty"]
        if "image_heads" in outputs:
            outputs["diagnostics"]["image_heads"] = outputs["image_heads"]
        if "image_field" in outputs:
            outputs["diagnostics"]["field_image"] = outputs["image_field"]["diagnostics"]
            outputs["diagnostics"]["image_field"] = outputs["image_field"]["diagnostics"]
        if "text_generation" in outputs:
            outputs["diagnostics"]["text_generation"] = outputs["text_generation"]
        if "text_field" in outputs:
            outputs["diagnostics"]["field_text"] = outputs["text_field"]["diagnostics"]
            outputs["diagnostics"]["text_field"] = outputs["text_field"]["diagnostics"]
        if "text_field_generation" in outputs:
            outputs["diagnostics"]["text_field_generation"] = outputs["text_field_generation"]

        return outputs


__all__ = ["EMLFoundationCore"]
