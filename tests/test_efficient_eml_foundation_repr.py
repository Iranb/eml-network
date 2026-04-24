from __future__ import annotations

import torch

from eml_mnist import EMLFoundationCore, SyntheticShapeEnergyDataset, SyntheticTextEnergyDataset
from eml_mnist.text_codecs import CharVocabulary


def _core(slot_layout: dict[str, int] | None = None) -> EMLFoundationCore:
    vocab = CharVocabulary()
    return EMLFoundationCore(
        slot_dim=16,
        event_dim=16,
        hidden_dim=32,
        slot_layout=slot_layout or {"goal": 1, "image": 2, "text": 2, "memory": 2},
        num_layers=1,
        top_k=2,
        representation_dim=16,
        local_query_dim=16,
        reconstruction_dim=16,
        image_input_channels=3,
        image_head_specs={"shape": 5},
        text_vocab_size=len(vocab),
        text_embed_dim=16,
        text_feature_dim=16,
        text_hidden_dim=32,
        enable_text_generation_head=True,
        enable_action_head=False,
        enable_patch_rank_head=False,
        enable_efficient_image_encoder=True,
        enable_efficient_text_encoder=True,
        image_repr_config={
            "state_dim": 16,
            "hidden_dim": 32,
            "num_hypotheses": 3,
            "num_attractors": 3,
            "representation_dim": 16,
            "patch_stride": 8,
        },
        text_repr_config={
            "embed_dim": 16,
            "state_dim": 16,
            "hidden_dim": 32,
            "num_hypotheses": 3,
            "num_attractors": 3,
            "representation_dim": 16,
            "causal_window_size": 4,
            "chunk_size": 4,
        },
    )


def test_foundation_efficient_image_path_forward() -> None:
    dataset = SyntheticShapeEnergyDataset(size=2, image_size=32, seed=20)
    images = torch.stack([dataset[index]["image"] for index in range(2)], dim=0)
    model = _core()

    out = model(images=images, use_efficient_repr_path=True, warmup_eta=0.5)

    assert out["representation"].shape == (2, 16)
    assert out["image_heads"]["shape"]["logits"].shape == (2, 5)
    assert out["efficient_image"]["attractor_states"].shape[:2] == (2, 3)
    assert out["attractor_injection"]["efficient_image"]["injected_slots"] == 2
    assert torch.isfinite(out["representation"]).all()


def test_foundation_efficient_text_path_forward_and_injection_changes_representation() -> None:
    dataset = SyntheticTextEnergyDataset(size=2, seq_len=20, seed=21)
    input_ids = torch.stack([dataset[index]["input_ids"] for index in range(2)], dim=0)
    mask = torch.stack([dataset[index]["input_mask"] for index in range(2)], dim=0)
    model = _core()

    with_injection = model(text_tokens=input_ids, text_padding_mask=mask, use_efficient_repr_path=True, inject_attractors=True, warmup_eta=0.5)
    without_injection = model(text_tokens=input_ids, text_padding_mask=mask, use_efficient_repr_path=True, inject_attractors=False, warmup_eta=0.5)

    assert with_injection["representation"].shape == (2, 16)
    assert with_injection["efficient_text_generation"]["logits"].shape[:2] == (2, 20)
    assert with_injection["attractor_injection"]["efficient_text"]["injected_slots"] == 2
    assert without_injection["attractor_injection"]["efficient_text"]["skipped_reason"] == "disabled"
    assert (with_injection["representation"] - without_injection["representation"]).norm() > 1.0e-7


def test_foundation_efficient_injection_skips_without_matching_slots() -> None:
    dataset = SyntheticShapeEnergyDataset(size=2, image_size=32, seed=22)
    images = torch.stack([dataset[index]["image"] for index in range(2)], dim=0)
    model = _core(slot_layout={"goal": 1, "task": 2})

    out = model(images=images, use_efficient_repr_path=True, warmup_eta=0.5)

    assert out["attractor_injection"]["efficient_image"]["injected_slots"] == 0
    assert out["attractor_injection"]["efficient_image"]["skipped_reason"] == "no_matching_slots"
    assert torch.isfinite(out["representation"]).all()
