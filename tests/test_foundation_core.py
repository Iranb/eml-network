from __future__ import annotations

import torch

from eml_mnist import (
    CharVocabulary,
    EMLFoundationCore,
    EMLTextBackbone,
    PureEMLImageBackbone,
    SyntheticGrammarDataset,
    SyntheticShapeDataset,
)


def test_foundation_core_supports_event_image_and_text_paths() -> None:
    vocab = CharVocabulary()
    model = EMLFoundationCore(
        slot_dim=24,
        event_dim=24,
        hidden_dim=48,
        slot_layout={"goal": 1, "image": 2, "text": 2, "memory": 2, "risk": 1},
        num_layers=2,
        top_k=3,
        representation_dim=24,
        local_query_dim=24,
        reconstruction_dim=24,
        image_input_channels=3,
        image_size=32,
        image_patch_size=4,
        image_feature_dim=24,
        image_bank_dim=48,
        image_num_layers=2,
        image_head_specs={"shape": 5},
        text_vocab_size=len(vocab),
        text_embed_dim=24,
        text_feature_dim=24,
        text_hidden_dim=48,
        text_bank_dim=48,
        text_num_layers=2,
        enable_text_generation_head=True,
        enable_action_head=False,
        enable_patch_rank_head=False,
    )

    event = torch.randn(2, 24)
    event_out = model(event=event, warmup_eta=0.5)
    assert event_out["representation"].shape == (2, 24)

    image_dataset = SyntheticShapeDataset(size=2, image_size=32, seed=3)
    images = torch.stack([image_dataset[index]["image"] for index in range(2)], dim=0)
    image_out = model(images=images, warmup_eta=0.5)
    assert image_out["image_heads"]["shape"]["logits"].shape == (2, 5)

    text_dataset = SyntheticGrammarDataset(size=2, vocab=vocab, max_length=32, seed=3)
    input_ids = torch.stack([text_dataset[index]["input_ids"] for index in range(2)], dim=0)
    input_mask = torch.stack([text_dataset[index]["input_mask"] for index in range(2)], dim=0)
    text_out = model(text_tokens=input_ids, text_padding_mask=input_mask, warmup_eta=0.5)
    assert text_out["text_generation"]["logits"].shape == (2, 32, len(vocab))

    for outputs in (event_out, image_out, text_out):
        assert outputs["active_slot_indices"] is not None
        assert "stats" in outputs["diagnostics"]
        assert torch.isfinite(outputs["representation"]).all()


def test_foundation_core_accepts_backbone_outputs() -> None:
    vocab = CharVocabulary()
    image_backbone = PureEMLImageBackbone(
        image_size=32,
        input_channels=3,
        feature_dim=24,
        event_dim=24,
        hidden_dim=48,
        bank_dim=48,
        num_layers=2,
        patch_size=4,
        patch_stride=4,
        local_window_size=3,
        merge_every=2,
        dropout=0.0,
    )
    text_backbone = EMLTextBackbone(
        vocab_size=len(vocab),
        embed_dim=24,
        feature_dim=24,
        event_dim=24,
        hidden_dim=48,
        bank_dim=48,
        num_layers=2,
        pad_id=vocab.pad_id,
        dropout=0.0,
    )
    model = EMLFoundationCore(
        slot_dim=24,
        event_dim=24,
        hidden_dim=48,
        slot_layout={"goal": 1, "image": 2, "text": 2, "memory": 2, "risk": 1},
        num_layers=2,
        top_k=3,
        representation_dim=24,
        local_query_dim=24,
        reconstruction_dim=24,
        image_head_specs={"shape": 5},
        text_vocab_size=len(vocab),
        text_feature_dim=24,
        enable_text_generation_head=True,
        enable_action_head=False,
        enable_patch_rank_head=False,
    )

    image_dataset = SyntheticShapeDataset(size=2, image_size=32, seed=4)
    images = torch.stack([image_dataset[index]["image"] for index in range(2)], dim=0)
    image_backbone_out = image_backbone(images, warmup_eta=0.5)
    image_out = model(image_backbone_outputs=image_backbone_out, warmup_eta=0.5)
    assert image_out["image_heads"]["shape"]["logits"].shape == (2, 5)

    text_dataset = SyntheticGrammarDataset(size=2, vocab=vocab, max_length=32, seed=4)
    input_ids = torch.stack([text_dataset[index]["input_ids"] for index in range(2)], dim=0)
    input_mask = torch.stack([text_dataset[index]["input_mask"] for index in range(2)], dim=0)
    text_backbone_out = text_backbone(input_ids=input_ids, padding_mask=input_mask, warmup_eta=0.5)
    text_out = model(text_backbone_outputs=text_backbone_out, warmup_eta=0.5)
    assert text_out["text_generation"]["logits"].shape == (2, 32, len(vocab))


def _backbone_outputs(event: torch.Tensor, global_slots: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "event": event,
        "pooled_representation": event,
        "sequence_features": global_slots,
        "token_features": global_slots,
        "global_slot_features": global_slots,
        "local_queries": global_slots,
        "padding_mask": torch.ones(global_slots.shape[:2], dtype=torch.bool),
    }


def test_modality_slot_injection_image() -> None:
    model = EMLFoundationCore(
        slot_dim=12,
        event_dim=12,
        hidden_dim=24,
        slot_layout={"goal": 1, "image": 2, "memory": 1},
        num_layers=1,
        top_k=2,
        representation_dim=12,
        local_query_dim=12,
        reconstruction_dim=12,
        image_feature_dim=5,
        enable_action_head=False,
        enable_patch_rank_head=False,
    )
    event = torch.zeros(2, 12)
    base_slots = torch.randn(2, 2, 5)
    changed_slots = base_slots + 50.0

    base = model(image_backbone_outputs=_backbone_outputs(event, base_slots), warmup_eta=0.5)
    changed = model(image_backbone_outputs=_backbone_outputs(event, changed_slots), warmup_eta=0.5)

    assert base["modality_injection"]["injected_image_slots"] == 2
    assert base["modality_injection"]["modality_slot_indices"]["image"] == [1, 2]
    assert (base["representation"] - changed["representation"]).norm() > 1.0e-6
    assert torch.isfinite(base["representation"]).all()


def test_modality_slot_injection_text() -> None:
    model = EMLFoundationCore(
        slot_dim=12,
        event_dim=12,
        hidden_dim=24,
        slot_layout={"goal": 1, "text": 2, "memory": 1},
        num_layers=1,
        top_k=2,
        representation_dim=12,
        local_query_dim=12,
        reconstruction_dim=12,
        text_feature_dim=6,
        text_vocab_size=32,
        enable_text_generation_head=True,
        enable_action_head=False,
        enable_patch_rank_head=False,
    )
    event = torch.zeros(2, 12)
    base_slots = torch.randn(2, 2, 6)
    changed_slots = base_slots - 50.0

    base = model(text_backbone_outputs=_backbone_outputs(event, base_slots), warmup_eta=0.5)
    changed = model(text_backbone_outputs=_backbone_outputs(event, changed_slots), warmup_eta=0.5)

    assert base["modality_injection"]["injected_text_slots"] == 2
    assert base["modality_injection"]["modality_slot_indices"]["text"] == [1, 2]
    assert (base["representation"] - changed["representation"]).norm() > 1.0e-6
    assert base["text_generation"]["logits"].shape[:2] == (2, 2)


def test_modality_slot_injection_skips_missing_layout_slots() -> None:
    model = EMLFoundationCore(
        slot_dim=10,
        event_dim=10,
        hidden_dim=20,
        slot_layout={"goal": 1, "memory": 2},
        num_layers=1,
        top_k=2,
        representation_dim=10,
        local_query_dim=10,
        reconstruction_dim=10,
        image_feature_dim=5,
        enable_action_head=False,
        enable_patch_rank_head=False,
    )
    event = torch.randn(2, 10)
    slots = torch.randn(2, 4, 5)

    out = model(image_backbone_outputs=_backbone_outputs(event, slots), warmup_eta=0.5)

    assert out["modality_injection"]["injected_image_slots"] == 0
    assert out["modality_injection"]["image_slot_injection_skipped"] is True
    assert out["representation"].shape == (2, 10)


def test_injected_modality_slots_can_receive_nonzero_readout_weight() -> None:
    model = EMLFoundationCore(
        slot_dim=12,
        event_dim=12,
        hidden_dim=24,
        slot_layout={"image": 2, "memory": 2},
        num_layers=1,
        top_k=2,
        representation_dim=12,
        local_query_dim=12,
        reconstruction_dim=12,
        image_feature_dim=5,
        enable_action_head=False,
        enable_patch_rank_head=False,
    )
    event = torch.randn(2, 12)
    slots = torch.randn(2, 2, 5)

    out = model(image_backbone_outputs=_backbone_outputs(event, slots), warmup_eta=0.5)
    readout_weight = out["representation_modality"]["image_slot_readout_weight"]

    assert readout_weight.shape == (2,)
    assert torch.all(readout_weight > 0.0)


def test_foundation_image_text_forward() -> None:
    vocab = CharVocabulary()
    model = EMLFoundationCore(
        slot_dim=16,
        event_dim=16,
        hidden_dim=32,
        slot_layout={"image": 2, "text": 2, "memory": 1},
        num_layers=1,
        top_k=2,
        representation_dim=16,
        local_query_dim=16,
        reconstruction_dim=16,
        image_feature_dim=8,
        text_feature_dim=8,
        text_vocab_size=len(vocab),
        enable_text_generation_head=True,
        image_head_specs={"shape": 5},
        enable_action_head=False,
        enable_patch_rank_head=False,
    )
    image_event = torch.randn(2, 16)
    text_event = torch.randn(2, 16)
    image_slots = torch.randn(2, 2, 8)
    text_slots = torch.randn(2, 2, 8)

    image_out = model(image_backbone_outputs=_backbone_outputs(image_event, image_slots), warmup_eta=0.5)
    text_out = model(text_backbone_outputs=_backbone_outputs(text_event, text_slots), warmup_eta=0.5)

    assert image_out["image_heads"]["shape"]["logits"].shape == (2, 5)
    assert text_out["text_generation"]["logits"].shape == (2, 2, len(vocab))
    assert image_out["modality_injection"]["injected_image_slots"] == 2
    assert text_out["modality_injection"]["injected_text_slots"] == 2
