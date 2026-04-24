from __future__ import annotations

import torch

from eml_mnist import CharVocabulary, EMLTextBackbone, SyntheticGrammarDataset


def test_synthetic_grammar_dataset_works_offline() -> None:
    vocab = CharVocabulary()
    dataset = SyntheticGrammarDataset(size=8, vocab=vocab, max_length=32, seed=0)
    sample = dataset[0]
    assert sample["input_ids"].shape == (32,)
    assert sample["target_ids"].shape == (32,)
    assert sample["input_mask"].shape == (32,)
    assert sample["validity_label"].shape == ()


def test_text_backbone_forward_returns_finite_outputs() -> None:
    vocab = CharVocabulary()
    dataset = SyntheticGrammarDataset(size=4, vocab=vocab, max_length=32, seed=1)
    input_ids = torch.stack([dataset[index]["input_ids"] for index in range(4)], dim=0)
    input_mask = torch.stack([dataset[index]["input_mask"] for index in range(4)], dim=0)

    backbone = EMLTextBackbone(
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
    outputs = backbone(input_ids=input_ids, padding_mask=input_mask, warmup_eta=0.5)

    assert outputs["event"].shape == (4, 24)
    assert outputs["pooled_representation"].shape == (4, 24)
    assert outputs["sequence_features"].shape == (4, 32, 24)
    assert outputs["global_slot_features"].shape[:2] == (4, 4)
    assert outputs["causal_message_stats"][0]["gate_mass"].shape == (4, 32, 1)
    assert torch.isfinite(outputs["event"]).all()
    assert torch.isfinite(outputs["sequence_features"]).all()
    assert torch.isfinite(outputs["causal_message_stats"][0]["gate_mass"]).all()


def test_text_causal_no_future_leakage() -> None:
    torch.manual_seed(7)
    vocab = CharVocabulary()
    backbone = EMLTextBackbone(
        vocab_size=len(vocab),
        embed_dim=16,
        feature_dim=16,
        event_dim=16,
        hidden_dim=32,
        bank_dim=32,
        num_layers=2,
        pad_id=vocab.pad_id,
        dropout=0.0,
    )
    backbone.eval()
    input_ids = torch.randint(3, len(vocab), (2, 14))
    padding_mask = torch.ones_like(input_ids, dtype=torch.bool)

    base = backbone(input_ids=input_ids, padding_mask=padding_mask, warmup_eta=0.5)["sequence_features"]
    perturbed = input_ids.clone()
    perturbed[:, 8:] = (perturbed[:, 8:] + 17) % len(vocab)
    changed = backbone(input_ids=perturbed, padding_mask=padding_mask, warmup_eta=0.5)["sequence_features"]

    assert torch.allclose(base[:, :8], changed[:, :8], atol=1.0e-5)


def test_text_backbone_has_no_forbidden_sequence_modules() -> None:
    vocab = CharVocabulary()
    backbone = EMLTextBackbone(
        vocab_size=len(vocab),
        embed_dim=16,
        feature_dim=16,
        event_dim=16,
        hidden_dim=32,
        bank_dim=32,
        num_layers=1,
        pad_id=vocab.pad_id,
        dropout=0.0,
    )

    forbidden_names = ("Transformer", "MultiheadAttention", "Mamba", "SSM")
    module_names = [module.__class__.__name__ for module in backbone.modules()]
    assert not any(any(forbidden in name for forbidden in forbidden_names) for name in module_names)
