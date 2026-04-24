from __future__ import annotations

import torch

from eml_mnist import EMLFoundationCore, SyntheticShapeEnergyDataset, SyntheticTextEnergyDataset


def _field_core(slot_layout: dict[str, int] | None = None) -> EMLFoundationCore:
    return EMLFoundationCore(
        slot_dim=16,
        event_dim=16,
        hidden_dim=32,
        slot_layout=slot_layout or {"goal": 1, "image": 2, "text": 2, "memory": 1},
        num_layers=1,
        top_k=2,
        representation_dim=16,
        local_query_dim=16,
        reconstruction_dim=16,
        image_head_specs={"shape": 5},
        text_vocab_size=96,
        text_embed_dim=16,
        text_feature_dim=16,
        text_hidden_dim=32,
        enable_text_generation_head=True,
        enable_action_head=False,
        enable_patch_rank_head=False,
        enable_image_field_encoder=True,
        enable_text_field_encoder=True,
        image_field_config={
            "sensor_dim": 16,
            "measurement_dim": 16,
            "field_dim": 16,
            "hidden_dim": 32,
            "num_hypotheses": 3,
            "num_parent_hypotheses": 3,
            "num_attractors": 3,
            "representation_dim": 16,
            "patch_stride": 8,
        },
        text_field_config={
            "embed_dim": 16,
            "sensor_dim": 16,
            "measurement_dim": 16,
            "field_dim": 16,
            "hidden_dim": 32,
            "num_hypotheses": 3,
            "num_chunk_hypotheses": 3,
            "num_attractors": 3,
            "representation_dim": 16,
            "causal_window_size": 3,
            "chunk_size": 4,
        },
    )


def test_foundation_core_old_event_path_still_works() -> None:
    model = _field_core()
    out = model(event=torch.randn(2, 16), warmup_eta=0.5)

    assert out["representation"].shape == (2, 16)
    assert out["active_slot_indices"].shape == (2, 2)
    assert torch.isfinite(out["representation"]).all()


def test_foundation_core_image_field_path_works() -> None:
    dataset = SyntheticShapeEnergyDataset(size=2, image_size=32, seed=2)
    images = torch.stack([dataset[index]["image"] for index in range(2)], dim=0)
    model = _field_core()

    out = model(images=images, use_field_path=True, warmup_eta=0.5)

    assert out["representation"].shape == (2, 16)
    assert out["image_heads"]["shape"]["logits"].shape == (2, 5)
    assert out["image_field"]["attractor_states"].shape[:2] == (2, 3)
    assert out["attractor_injection"]["image"]["injected_slots"] == 2
    assert "field_image" in out["diagnostics"]
    assert torch.isfinite(out["representation"]).all()


def test_foundation_core_text_field_path_works() -> None:
    dataset = SyntheticTextEnergyDataset(size=2, seq_len=24, seed=3)
    input_ids = torch.stack([dataset[index]["input_ids"] for index in range(2)], dim=0)
    input_mask = torch.stack([dataset[index]["input_mask"] for index in range(2)], dim=0)
    model = _field_core()

    out = model(text_tokens=input_ids, text_padding_mask=input_mask, use_field_path=True, warmup_eta=0.5)

    assert out["representation"].shape == (2, 16)
    assert out["text_generation"]["logits"].shape == (2, 24, 96)
    assert out["text_field_generation"]["logits"].shape == (2, 24, 96)
    assert out["text_field"]["attractor_states"].shape[:2] == (2, 3)
    assert out["attractor_injection"]["text"]["injected_slots"] == 2
    assert "field_text" in out["diagnostics"]
    assert torch.isfinite(out["representation"]).all()


def test_attractor_injection_changes_representation() -> None:
    dataset = SyntheticTextEnergyDataset(size=2, seq_len=20, seed=4)
    input_ids = torch.stack([dataset[index]["input_ids"] for index in range(2)], dim=0)
    input_mask = torch.stack([dataset[index]["input_mask"] for index in range(2)], dim=0)
    model = _field_core()

    with_injection = model(text_tokens=input_ids, text_padding_mask=input_mask, inject_attractors=True, warmup_eta=0.5)
    without_injection = model(text_tokens=input_ids, text_padding_mask=input_mask, inject_attractors=False, warmup_eta=0.5)

    assert with_injection["attractor_injection"]["text"]["injected_slots"] == 2
    assert without_injection["attractor_injection"]["text"]["skipped_reason"] == "disabled"
    assert (with_injection["representation"] - without_injection["representation"]).norm() > 1.0e-7


def test_attractor_injection_skips_gracefully_without_matching_slots() -> None:
    dataset = SyntheticShapeEnergyDataset(size=2, image_size=32, seed=5)
    images = torch.stack([dataset[index]["image"] for index in range(2)], dim=0)
    model = _field_core(slot_layout={"goal": 1, "memory": 3})

    out = model(images=images, use_field_path=True, warmup_eta=0.5)

    assert out["attractor_injection"]["image"]["injected_slots"] == 0
    assert out["attractor_injection"]["image"]["skipped_reason"] == "no_matching_slots"
    assert out["representation"].shape == (2, 16)
    assert torch.isfinite(out["representation"]).all()
