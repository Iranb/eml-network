from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _projection(generator: torch.Generator, input_dim: int, output_dim: int) -> torch.Tensor:
    scale = 1.0 / math.sqrt(max(1, input_dim))
    return torch.randn(input_dim, output_dim, generator=generator) * scale


def _shuffle_with_target(candidates: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    size, num_candidates, candidate_dim = candidates.shape
    permutations = torch.stack([torch.randperm(num_candidates, generator=generator) for _ in range(size)], dim=0)
    shuffled = candidates.gather(1, permutations.unsqueeze(-1).expand(-1, -1, candidate_dim))
    targets = (permutations == 0).nonzero(as_tuple=False)[:, 1]
    return shuffled, targets


def _build_rank_candidates(
    generator: torch.Generator,
    context: torch.Tensor,
    num_candidates: int,
    candidate_dim: int,
    positive_scale: float = 0.05,
    negative_scale: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor]:
    size = context.size(0)
    positive = context.unsqueeze(1) + positive_scale * torch.randn(size, 1, candidate_dim, generator=generator)
    negatives = negative_scale * torch.randn(size, num_candidates - 1, candidate_dim, generator=generator)
    candidates = torch.cat([positive, negatives], dim=1)
    return _shuffle_with_target(candidates, generator)


def _scatter_active_updates(
    slot_states: torch.Tensor,
    active_indices: torch.Tensor,
    updated_active_states: torch.Tensor,
) -> torch.Tensor:
    scatter_index = active_indices.unsqueeze(-1).expand_as(updated_active_states)
    updated_slot_states = slot_states.clone()
    updated_slot_states.scatter_(1, scatter_index, updated_active_states)
    return updated_slot_states


class _TensorDictDataset(Dataset):
    def __init__(self, payload: Dict[str, torch.Tensor]) -> None:
        if not payload:
            raise ValueError("payload must not be empty")
        lengths = {value.size(0) for value in payload.values()}
        if len(lengths) != 1:
            raise ValueError("all payload tensors must share the same leading dimension")
        self.payload = payload
        self.size = lengths.pop()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.payload.items()}


class ToyActionDataset(_TensorDictDataset):
    def __init__(
        self,
        size: int,
        event_dim: int = 8,
        action_dim: int = 8,
        num_actions: int = 5,
        latent_dim: int | None = None,
        seed: int = 0,
    ) -> None:
        if size <= 0 or event_dim <= 0 or action_dim <= 0 or num_actions < 2:
            raise ValueError("invalid ToyActionDataset configuration")

        latent_dim = latent_dim or max(event_dim, action_dim)
        generator = torch.Generator().manual_seed(seed)

        latent = torch.randn(size, latent_dim, generator=generator)
        event = torch.tanh(latent @ _projection(generator, latent_dim, event_dim))
        action_context = torch.tanh(latent @ _projection(generator, latent_dim, action_dim))
        candidate_actions, action_target = _build_rank_candidates(generator, action_context, num_actions, action_dim)

        super().__init__(
            {
                "event": event.float(),
                "candidate_actions": candidate_actions.float(),
                "action_target": action_target.long(),
            }
        )


class ToyPatchRankingDataset(_TensorDictDataset):
    def __init__(
        self,
        size: int,
        event_dim: int = 8,
        patch_dim: int = 10,
        num_patches: int = 6,
        latent_dim: int | None = None,
        seed: int = 0,
    ) -> None:
        if size <= 0 or event_dim <= 0 or patch_dim <= 0 or num_patches < 2:
            raise ValueError("invalid ToyPatchRankingDataset configuration")

        latent_dim = latent_dim or max(event_dim, patch_dim)
        generator = torch.Generator().manual_seed(seed)

        latent = torch.randn(size, latent_dim, generator=generator)
        event = torch.tanh(latent @ _projection(generator, latent_dim, event_dim))
        patch_context = torch.tanh(latent @ _projection(generator, latent_dim, patch_dim))
        candidate_patches, patch_target = _build_rank_candidates(generator, patch_context, num_patches, patch_dim)

        super().__init__(
            {
                "event": event.float(),
                "candidate_patches": candidate_patches.float(),
                "patch_target": patch_target.long(),
            }
        )


class ToyFoundationDataset(_TensorDictDataset):
    def __init__(
        self,
        size: int,
        event_dim: int = 8,
        action_dim: int = 8,
        patch_dim: int = 10,
        local_query_dim: int = 6,
        reconstruction_dim: int = 8,
        num_actions: int = 5,
        num_patches: int = 6,
        num_queries: int = 4,
        num_risk_outputs: int = 1,
        num_novelty_prototypes: int = 8,
        latent_dim: int | None = None,
        seed: int = 0,
    ) -> None:
        if (
            size <= 0
            or event_dim <= 0
            or action_dim <= 0
            or patch_dim <= 0
            or local_query_dim <= 0
            or reconstruction_dim <= 0
            or num_actions < 2
            or num_patches < 2
            or num_queries <= 0
            or num_risk_outputs <= 0
            or num_novelty_prototypes <= 0
        ):
            raise ValueError("invalid ToyFoundationDataset configuration")

        latent_dim = latent_dim or max(
            event_dim,
            action_dim,
            patch_dim,
            local_query_dim,
            reconstruction_dim,
            num_risk_outputs,
        )
        generator = torch.Generator().manual_seed(seed)

        latent = torch.randn(size, latent_dim, generator=generator)
        event = torch.tanh(latent @ _projection(generator, latent_dim, event_dim))

        action_context = torch.tanh(latent @ _projection(generator, latent_dim, action_dim))
        candidate_actions, action_target = _build_rank_candidates(generator, action_context, num_actions, action_dim)

        patch_context = torch.tanh(latent @ _projection(generator, latent_dim, patch_dim))
        candidate_patches, patch_target = _build_rank_candidates(generator, patch_context, num_patches, patch_dim)

        local_queries = torch.randn(size, num_queries, local_query_dim, generator=generator)
        reconstruction_context = torch.tanh(latent @ _projection(generator, latent_dim, reconstruction_dim))
        query_projection = _projection(generator, local_query_dim, reconstruction_dim)
        query_features = torch.tanh(local_queries @ query_projection)
        reconstruction_target = torch.tanh(
            reconstruction_context.unsqueeze(1)
            + query_features
            + 0.25 * reconstruction_context.unsqueeze(1) * query_features
        )

        risk_logits = latent @ _projection(generator, latent_dim, num_risk_outputs)
        risk_logits = risk_logits + 0.15 * torch.randn(size, num_risk_outputs, generator=generator)
        risk_target = (risk_logits > 0.0).float()

        prototype_latent = F.normalize(torch.randn(num_novelty_prototypes, latent_dim, generator=generator), dim=-1)
        latent_normalized = F.normalize(latent, dim=-1)
        max_similarity = (latent_normalized @ prototype_latent.t()).max(dim=-1, keepdim=True).values
        novelty_target = (max_similarity < 0.35).float()

        super().__init__(
            {
                "event": event.float(),
                "candidate_actions": candidate_actions.float(),
                "action_target": action_target.long(),
                "candidate_patches": candidate_patches.float(),
                "patch_target": patch_target.long(),
                "local_queries": local_queries.float(),
                "reconstruction_target": reconstruction_target.float(),
                "risk_target": risk_target.float(),
                "novelty_target": novelty_target.float(),
            }
        )


class ToyStateTransitionDataset(_TensorDictDataset):
    def __init__(
        self,
        size: int,
        num_slots: int = 6,
        slot_dim: int = 8,
        event_dim: int = 8,
        top_k: int = 3,
        latent_dim: int | None = None,
        seed: int = 0,
    ) -> None:
        if size <= 0 or num_slots <= 0 or slot_dim <= 0 or event_dim <= 0 or top_k <= 0:
            raise ValueError("invalid ToyStateTransitionDataset configuration")

        latent_dim = latent_dim or max(slot_dim, event_dim)
        generator = torch.Generator().manual_seed(seed)
        top_k = min(top_k, num_slots)

        latent = torch.randn(size, latent_dim, generator=generator)
        slot_states = torch.randn(size, num_slots, slot_dim, generator=generator)
        event = torch.tanh(latent @ _projection(generator, latent_dim, event_dim))
        event_slot = torch.tanh(event @ _projection(generator, event_dim, slot_dim))

        routing_scores = (slot_states * event_slot.unsqueeze(1)).sum(dim=-1) / math.sqrt(slot_dim)
        topk_scores, topk_indices = routing_scores.topk(k=top_k, dim=-1)
        active_states = slot_states.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, slot_dim))
        active_mean = active_states.mean(dim=1, keepdim=True)
        candidate = torch.tanh(0.55 * active_states + 0.35 * active_mean + 0.75 * event_slot.unsqueeze(1))
        update_gate = torch.sigmoid(topk_scores.unsqueeze(-1))
        updated_active_states = active_states + update_gate * (candidate - active_states)
        next_slot_states = _scatter_active_updates(slot_states, topk_indices, updated_active_states)
        slot_mask = torch.ones(size, num_slots, dtype=torch.bool)

        super().__init__(
            {
                "event": event.float(),
                "slot_states": slot_states.float(),
                "next_slot_states": next_slot_states.float(),
                "target_topk_indices": topk_indices.long(),
                "slot_mask": slot_mask,
            }
        )


class ToyPrototypeDataset(_TensorDictDataset):
    def __init__(
        self,
        size: int,
        event_dim: int = 8,
        num_known_prototypes: int = 6,
        novelty_prob: float = 0.3,
        latent_dim: int | None = None,
        seed: int = 0,
    ) -> None:
        if size <= 0 or event_dim <= 0 or num_known_prototypes <= 0:
            raise ValueError("invalid ToyPrototypeDataset configuration")
        if not 0.0 <= novelty_prob <= 1.0:
            raise ValueError("novelty_prob must be in [0, 1]")

        latent_dim = latent_dim or event_dim
        generator = torch.Generator().manual_seed(seed)

        known_centers = torch.randn(num_known_prototypes, latent_dim, generator=generator)
        unknown_center = 3.0 * torch.randn(latent_dim, generator=generator)
        known_indices = torch.randint(0, num_known_prototypes, (size,), generator=generator)
        novelty_target = (torch.rand(size, generator=generator) < novelty_prob).float()

        latent = known_centers[known_indices] + 0.15 * torch.randn(size, latent_dim, generator=generator)
        unknown_latent = unknown_center.unsqueeze(0) + 0.2 * torch.randn(size, latent_dim, generator=generator)
        latent = torch.where(novelty_target.unsqueeze(-1).bool(), unknown_latent, latent)
        prototype_index = torch.where(
            novelty_target.bool(),
            torch.full_like(known_indices, -1),
            known_indices,
        )

        event = torch.tanh(latent @ _projection(generator, latent_dim, event_dim))

        super().__init__(
            {
                "event": event.float(),
                "novelty_target": novelty_target.unsqueeze(-1).float(),
                "prototype_index": prototype_index.long(),
            }
        )


__all__ = [
    "ToyActionDataset",
    "ToyFoundationDataset",
    "ToyPatchRankingDataset",
    "ToyPrototypeDataset",
    "ToyStateTransitionDataset",
]
