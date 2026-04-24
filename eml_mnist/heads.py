from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import EMLScore, EMLUpdateGate, _reset_linear, inverse_softplus


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None, dim: int = -1) -> torch.Tensor:
    if mask is None:
        return torch.softmax(logits, dim=dim)
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(masked_logits, dim=dim)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    normalizer = weights.sum(dim=dim, keepdim=True)
    return torch.where(normalizer > 0.0, weights / normalizer.clamp_min(1.0e-12), torch.zeros_like(weights))


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, final_tanh: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        ]
        if final_tanh:
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _reset_linear(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RepresentationHead(nn.Module):
    """Pool typed slot states into a global representation with sEML scoring."""

    def __init__(
        self,
        slot_dim: int,
        hidden_dim: int,
        representation_dim: int | None = None,
        clip_value: float = 3.0,
    ) -> None:
        super().__init__()
        if slot_dim <= 0 or hidden_dim <= 0:
            raise ValueError("slot_dim and hidden_dim must be positive")

        self.slot_dim = slot_dim
        self.representation_dim = representation_dim or slot_dim
        self.norm = nn.LayerNorm(slot_dim)
        self.drive_net = _MLP(slot_dim, hidden_dim, 1)
        self.resistance_net = _MLP(slot_dim, hidden_dim, 1)
        self.value_proj = nn.Linear(slot_dim, self.representation_dim)
        self.score = EMLScore(dim=1, clip_value=clip_value, init_bias=0.0)
        _reset_linear(self.value_proj)

    def forward(
        self,
        slot_states: torch.Tensor,
        type_features: torch.Tensor | None = None,
        slot_mask: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if slot_states.ndim != 3 or slot_states.size(-1) != self.slot_dim:
            raise ValueError("slot_states must have shape [batch, num_slots, slot_dim]")
        if type_features is not None and type_features.shape != slot_states.shape:
            raise ValueError("type_features must match slot_states shape")
        if slot_mask is not None and slot_mask.shape != slot_states.shape[:2]:
            raise ValueError("slot_mask must have shape [batch, num_slots]")

        typed_states = slot_states if type_features is None else slot_states + type_features
        normalized = self.norm(typed_states)
        drive = self.drive_net(normalized)
        resistance = self.resistance_net(normalized)
        score_out = self.score(drive, resistance, warmup_eta=warmup_eta)
        logits = score_out["score"].squeeze(-1)
        weights = _masked_softmax(logits, slot_mask, dim=1)
        representation = torch.sum(weights.unsqueeze(-1) * self.value_proj(normalized), dim=1)
        top_count = min(3, weights.size(1))
        top_weights, top_indices = weights.topk(k=top_count, dim=1)

        return {
            "representation": representation,
            "weights": weights,
            "top_indices": top_indices,
            "top_weights": top_weights,
            "drive": score_out["drive"].squeeze(-1),
            "resistance": score_out["resistance"].squeeze(-1),
            "energy": logits,
            "score": logits,
            "drive_mean": score_out["drive"].mean(),
            "resistance_mean": score_out["resistance"].mean(),
            "energy_mean": logits.mean(),
        }


class _CandidateRankHead(nn.Module):
    def __init__(
        self,
        context_dim: int,
        candidate_dim: int,
        hidden_dim: int,
        clip_value: float = 3.0,
        temperature: float = 1.0,
        init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if context_dim <= 0 or candidate_dim <= 0 or hidden_dim <= 0:
            raise ValueError("context_dim, candidate_dim, and hidden_dim must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self.context_dim = context_dim
        self.candidate_dim = candidate_dim
        self.context_norm = nn.LayerNorm(context_dim)
        self.candidate_norm = nn.LayerNorm(candidate_dim)
        self.context_proj = nn.Linear(context_dim, context_dim)
        self.candidate_proj = nn.Linear(candidate_dim, context_dim)
        self.joint_norm = nn.LayerNorm(context_dim * 3)
        self.drive_net = _MLP(context_dim * 3, hidden_dim, 1)
        self.resistance_net = _MLP(context_dim * 3, hidden_dim, 1)
        self.score = EMLScore(
            dim=1,
            clip_value=clip_value,
            temperature=temperature,
            init_bias=init_bias,
        )
        _reset_linear(self.context_proj)
        _reset_linear(self.candidate_proj)

    def _forward_candidates(
        self,
        representation: torch.Tensor,
        candidates: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if representation.ndim != 2 or representation.size(-1) != self.context_dim:
            raise ValueError("representation must have shape [batch, context_dim]")
        if candidates.ndim != 3 or candidates.size(-1) != self.candidate_dim:
            raise ValueError("candidates must have shape [batch, num_candidates, candidate_dim]")
        if representation.size(0) != candidates.size(0):
            raise ValueError("representation and candidates batch sizes must match")

        batch_size, num_candidates, _ = candidates.shape
        context = self.context_proj(self.context_norm(representation)).unsqueeze(1).expand(batch_size, num_candidates, -1)
        candidate_features = self.candidate_proj(self.candidate_norm(candidates))
        joint = self.joint_norm(torch.cat([context, candidate_features, context * candidate_features], dim=-1))
        drive = self.drive_net(joint)
        resistance = self.resistance_net(joint)
        score_out = self.score(drive, resistance, warmup_eta=warmup_eta)
        logits = score_out["score"].squeeze(-1)
        probs = score_out["probs"].squeeze(-1)
        best_indices = logits.argmax(dim=-1)

        return {
            "score": logits,
            "probs": probs,
            "best_indices": best_indices,
            "drive": score_out["drive"].squeeze(-1),
            "resistance": score_out["resistance"].squeeze(-1),
            "energy": logits,
        }


class ActionHead(_CandidateRankHead):
    def forward(
        self,
        representation: torch.Tensor,
        candidate_actions: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        out = self._forward_candidates(representation, candidate_actions, warmup_eta=warmup_eta)
        return {
            "action_score": out["score"],
            "action_probs": out["probs"],
            "best_action_indices": out["best_indices"],
            "drive": out["drive"],
            "resistance": out["resistance"],
            "energy": out["energy"],
        }


class PatchRankHead(_CandidateRankHead):
    def forward(
        self,
        representation: torch.Tensor,
        candidate_patches: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        out = self._forward_candidates(representation, candidate_patches, warmup_eta=warmup_eta)
        return {
            "patch_score": out["score"],
            "patch_probs": out["probs"],
            "best_patch_indices": out["best_indices"],
            "drive": out["drive"],
            "resistance": out["resistance"],
            "energy": out["energy"],
        }


class ClassificationHead(nn.Module):
    """Prototype-style general classification head using sEML scores."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        clip_value: float = 3.0,
        temperature: float = 0.5,
        center_ambiguity: bool = True,
        ambiguity_weight: float = 1.0,
        schedule_ambiguity_weight: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_classes <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim, num_classes, and hidden_dim must be positive")
        if ambiguity_weight < 0.0:
            raise ValueError("ambiguity_weight must be non-negative")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.temperature = temperature
        self.center_ambiguity = bool(center_ambiguity)
        self.ambiguity_weight = float(ambiguity_weight)
        self.schedule_ambiguity_weight = bool(schedule_ambiguity_weight)

        self.prototypes = nn.Parameter(torch.empty(num_classes, input_dim))
        self.drive_residual = _MLP(input_dim, hidden_dim, num_classes)
        self.uncertainty_head = _MLP(input_dim, hidden_dim, 1)
        self.raw_class_resistance = nn.Parameter(
            torch.full((num_classes,), inverse_softplus(0.2), dtype=torch.float32)
        )
        self.score = EMLScore(dim=num_classes, clip_value=clip_value, init_bias=0.0)
        nn.init.normal_(self.prototypes, mean=0.0, std=0.05)

    def forward(
        self,
        representation: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if representation.ndim != 2 or representation.size(-1) != self.input_dim:
            raise ValueError("representation must have shape [batch, input_dim]")

        normalized_representation = F.normalize(representation, dim=-1)
        normalized_prototypes = F.normalize(self.prototypes, dim=-1)
        similarity = normalized_representation @ normalized_prototypes.t()
        scaled_similarity = similarity / self.temperature
        drive = scaled_similarity + self.drive_residual(representation)
        if self.num_classes > 1:
            eye_mask = torch.eye(self.num_classes, device=similarity.device, dtype=torch.bool).unsqueeze(0)
            expanded_similarity = scaled_similarity.unsqueeze(1).expand(-1, self.num_classes, -1)
            ambiguity = torch.logsumexp(expanded_similarity.masked_fill(eye_mask, float("-inf")), dim=-1)
            if self.center_ambiguity:
                ambiguity = ambiguity - torch.log(
                    torch.tensor(float(self.num_classes - 1), device=similarity.device, dtype=similarity.dtype)
                )
        else:
            ambiguity = torch.zeros_like(drive)
        if torch.is_tensor(warmup_eta):
            eta = warmup_eta.to(device=representation.device, dtype=representation.dtype).clamp(0.0, 1.0)
        else:
            eta = torch.tensor(float(warmup_eta), device=representation.device, dtype=representation.dtype).clamp(0.0, 1.0)
        ambiguity_weight = torch.tensor(self.ambiguity_weight, device=representation.device, dtype=representation.dtype)
        if self.schedule_ambiguity_weight:
            ambiguity_weight = ambiguity_weight * eta
        weighted_ambiguity = ambiguity_weight * ambiguity
        class_resistance = F.softplus(self.raw_class_resistance).unsqueeze(0).expand_as(drive)
        sample_uncertainty = F.softplus(self.uncertainty_head(representation)).expand_as(drive)
        resistance = weighted_ambiguity + class_resistance + sample_uncertainty
        score_out = self.score(drive, resistance, warmup_eta=warmup_eta)
        if self.num_classes > 1:
            prototype_cosine = normalized_prototypes @ normalized_prototypes.t()
            off_diagonal = prototype_cosine.masked_select(
                ~torch.eye(self.num_classes, device=prototype_cosine.device, dtype=torch.bool)
            )
            prototype_diversity_penalty = off_diagonal.square().mean()
        else:
            prototype_diversity_penalty = representation.new_tensor(0.0)

        return {
            "logits": score_out["score"],
            "probs": score_out["probs"],
            "drive": score_out["drive"],
            "resistance": score_out["resistance"],
            "energy": score_out["energy"],
            "similarity": similarity,
            "ambiguity": ambiguity,
            "weighted_ambiguity": weighted_ambiguity,
            "ambiguity_weight": ambiguity_weight.expand_as(ambiguity),
            "class_resistance": class_resistance,
            "class_radius": class_resistance,
            "sample_uncertainty": sample_uncertainty[:, :1],
            "prototype_diversity_penalty": prototype_diversity_penalty,
            "prototypes": self.prototypes,
        }


class RiskResistanceHead(nn.Module):
    """Predict general risk/resistance scores from the global representation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        clip_value: float = 3.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim, hidden_dim, and output_dim must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.drive_net = _MLP(input_dim, hidden_dim, output_dim)
        self.resistance_net = _MLP(input_dim, hidden_dim, output_dim)
        self.score = EMLScore(dim=output_dim, clip_value=clip_value, init_bias=0.0)

    def forward(
        self,
        representation: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if representation.ndim != 2 or representation.size(-1) != self.input_dim:
            raise ValueError("representation must have shape [batch, input_dim]")

        normalized = self.norm(representation)
        drive = self.drive_net(normalized)
        resistance = self.resistance_net(normalized)
        score_out = self.score(drive, resistance, warmup_eta=warmup_eta)
        risk_prob = torch.sigmoid(score_out["score"])

        return {
            "risk_score": score_out["score"],
            "risk_prob": risk_prob,
            "drive": score_out["drive"],
            "resistance": score_out["resistance"],
            "energy": score_out["energy"],
        }


class LocalReconstructionHead(nn.Module):
    """Locally reconstruct latent targets from global context without attention."""

    def __init__(
        self,
        context_dim: int,
        query_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_queries: int | None = None,
        clip_value: float = 3.0,
    ) -> None:
        super().__init__()
        if context_dim <= 0 or query_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("context_dim, query_dim, output_dim, and hidden_dim must be positive")
        if num_queries is not None and num_queries <= 0:
            raise ValueError("num_queries must be positive when provided")

        self.context_dim = context_dim
        self.query_dim = query_dim
        self.output_dim = output_dim
        self.num_queries = num_queries

        self.learned_queries = (
            nn.Parameter(torch.empty(num_queries, query_dim)) if num_queries is not None else None
        )
        self.context_norm = nn.LayerNorm(context_dim)
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_proj = nn.Linear(context_dim, context_dim)
        self.query_proj = nn.Linear(query_dim, context_dim)
        self.joint_norm = nn.LayerNorm(context_dim * 3)
        self.candidate = _MLP(context_dim * 3, hidden_dim, output_dim, final_tanh=True)
        self.drive_net = _MLP(context_dim * 3, hidden_dim, output_dim)
        self.resistance_net = _MLP(context_dim * 3, hidden_dim, output_dim)
        self.gate = EMLUpdateGate(dim=output_dim, clip_value=clip_value, init_bias=-0.5)
        _reset_linear(self.context_proj)
        _reset_linear(self.query_proj)
        if self.learned_queries is not None:
            nn.init.normal_(self.learned_queries, mean=0.0, std=0.02)

    def forward(
        self,
        representation: torch.Tensor,
        local_queries: torch.Tensor | None = None,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if representation.ndim != 2 or representation.size(-1) != self.context_dim:
            raise ValueError("representation must have shape [batch, context_dim]")

        batch_size = representation.size(0)
        if local_queries is None:
            if self.learned_queries is None:
                raise ValueError("local_queries are required when no learned queries are configured")
            local_queries = self.learned_queries.unsqueeze(0).expand(batch_size, -1, -1)
        elif local_queries.ndim == 2:
            local_queries = local_queries.unsqueeze(0).expand(batch_size, -1, -1)
        elif local_queries.ndim != 3:
            raise ValueError("local_queries must have shape [batch, num_queries, query_dim]")

        if local_queries.size(0) != batch_size or local_queries.size(-1) != self.query_dim:
            raise ValueError("local_queries shape does not match this LocalReconstructionHead")

        num_queries = local_queries.size(1)
        context = self.context_proj(self.context_norm(representation)).unsqueeze(1).expand(batch_size, num_queries, -1)
        queries = self.query_proj(self.query_norm(local_queries))
        joint = self.joint_norm(torch.cat([context, queries, context * queries], dim=-1))
        candidate = self.candidate(joint)
        drive = self.drive_net(joint)
        resistance = self.resistance_net(joint)
        gate_out = self.gate(drive, resistance, warmup_eta=warmup_eta)
        reconstruction = gate_out["gate"] * candidate

        return {
            "reconstruction": reconstruction,
            "reconstruction_gate": gate_out["gate"],
            "drive": gate_out["drive"],
            "resistance": gate_out["resistance"],
            "energy": gate_out["energy"],
        }


class PrototypeNoveltyHead(nn.Module):
    """Optional prototype-based novelty detector."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_prototypes: int,
        clip_value: float = 3.0,
        prototype_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_prototypes <= 0:
            raise ValueError("input_dim, hidden_dim, and num_prototypes must be positive")
        if prototype_temperature <= 0.0:
            raise ValueError("prototype_temperature must be positive")

        self.input_dim = input_dim
        self.num_prototypes = num_prototypes
        self.prototype_temperature = prototype_temperature

        self.prototypes = nn.Parameter(torch.empty(num_prototypes, input_dim))
        self.drive_residual = _MLP(input_dim, hidden_dim, 1)
        self.uncertainty_head = _MLP(input_dim, hidden_dim, 1)
        self.raw_prototype_resistance = nn.Parameter(
            torch.full((num_prototypes,), inverse_softplus(0.2), dtype=torch.float32)
        )
        self.score = EMLScore(dim=1, clip_value=clip_value, init_bias=0.0)
        nn.init.normal_(self.prototypes, mean=0.0, std=0.05)

    def forward(
        self,
        representation: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if representation.ndim != 2 or representation.size(-1) != self.input_dim:
            raise ValueError("representation must have shape [batch, input_dim]")

        normalized_representation = F.normalize(representation, dim=-1)
        normalized_prototypes = F.normalize(self.prototypes, dim=-1)
        similarities = normalized_representation @ normalized_prototypes.t()
        scaled_similarities = similarities / self.prototype_temperature
        best_similarity, best_prototype_indices = scaled_similarities.max(dim=-1, keepdim=True)

        drive = (1.0 - best_similarity) + self.drive_residual(representation)
        prototype_resistance = F.softplus(self.raw_prototype_resistance)[best_prototype_indices.squeeze(-1)].unsqueeze(-1)
        resistance = prototype_resistance + F.softplus(self.uncertainty_head(representation))
        score_out = self.score(drive, resistance, warmup_eta=warmup_eta)
        novelty_prob = torch.sigmoid(score_out["score"])

        return {
            "novelty_score": score_out["score"],
            "novelty_prob": novelty_prob,
            "similarities": similarities,
            "best_similarity": best_similarity,
            "best_prototype_indices": best_prototype_indices.squeeze(-1),
            "drive": score_out["drive"],
            "resistance": score_out["resistance"],
            "energy": score_out["energy"],
        }


__all__ = [
    "ActionHead",
    "ClassificationHead",
    "LocalReconstructionHead",
    "PatchRankHead",
    "PrototypeNoveltyHead",
    "RepresentationHead",
    "RiskResistanceHead",
]
