from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from eml_mnist import CharVocabulary, EMLTextFieldEncoder, EMLTextFieldGenerationHead, SyntheticTextEnergyDataset


def _assert_finite(value: Any) -> None:
    if torch.is_tensor(value):
        if value.dtype.is_floating_point or value.dtype.is_complex:
            assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for child in value.values():
            _assert_finite(child)


def _sample_batch(batch_size: int = 3, seq_len: int = 32) -> dict[str, torch.Tensor]:
    dataset = SyntheticTextEnergyDataset(size=batch_size, seq_len=seq_len, seed=11)
    batch = [dataset[index] for index in range(batch_size)]
    return {
        key: torch.stack([sample[key] for sample in batch], dim=0)
        for key in batch[0]
    }


def _encoder(vocab_size: int, pad_id: int = 0, field_dim: int = 16) -> EMLTextFieldEncoder:
    return EMLTextFieldEncoder(
        vocab_size=vocab_size,
        embed_dim=16,
        sensor_dim=16,
        measurement_dim=16,
        field_dim=field_dim,
        hidden_dim=32,
        num_hypotheses=3,
        num_chunk_hypotheses=3,
        num_attractors=3,
        representation_dim=field_dim,
        pad_id=pad_id,
        causal_window_size=3,
        chunk_size=4,
        chunk_window_size=3,
    )


def test_synthetic_text_energy_dataset_works_offline() -> None:
    dataset = SyntheticTextEnergyDataset(size=8, seq_len=32, seed=0)
    sample = dataset[0]

    assert sample["input_ids"].shape == (32,)
    assert sample["target_ids"].shape == (32,)
    assert sample["corruption_mask"].shape == (32,)
    assert sample["boundary_labels"].shape == (32,)
    assert sample["ambiguity_labels"].shape == (32,)
    assert sample["resistance_target"].shape == (32,)
    assert int(sample["input_ids"][0]) == dataset.vocab.bos_id
    _assert_finite(sample)


def test_eml_text_field_encoder_forward_pass_works() -> None:
    batch = _sample_batch(batch_size=3, seq_len=28)
    vocab = CharVocabulary()
    encoder = _encoder(vocab_size=len(vocab), pad_id=vocab.pad_id)

    out = encoder(batch["input_ids"], padding_mask=batch["input_mask"], warmup_eta=0.5)

    assert out["sequence_states"].shape == (3, 28, 16)
    assert out["representation"].shape == (3, 16)
    assert out["attractor_states"].shape == (3, 3, 16)
    assert out["local_hypotheses"]["state"].shape[:3] == (3, 28, 3)
    assert out["chunk_hypotheses"]["state"].ndim == 4
    _assert_finite(out)


def test_eml_text_field_generation_head_logits_shape_is_correct() -> None:
    vocab = CharVocabulary()
    head = EMLTextFieldGenerationHead(state_dim=16, vocab_size=len(vocab), hidden_dim=32)
    sequence_states = torch.randn(2, 21, 16)
    padding_mask = torch.ones(2, 21, dtype=torch.bool)

    out = head(sequence_states, padding_mask=padding_mask, warmup_eta=0.5)

    assert out["logits"].shape == (2, 21, len(vocab))
    assert out["drive"].shape == (2, 21, len(vocab))
    assert out["resistance"].shape == (2, 21, len(vocab))
    assert out["energy"].shape == (2, 21, len(vocab))
    assert out["uncertainty"].shape == (2, 21, 1)
    _assert_finite(out)


def test_text_field_outputs_are_finite() -> None:
    batch = _sample_batch(batch_size=2, seq_len=24)
    vocab = CharVocabulary()
    encoder = _encoder(vocab_size=len(vocab), pad_id=vocab.pad_id)
    head = EMLTextFieldGenerationHead(state_dim=16, vocab_size=len(vocab), hidden_dim=32)

    encoder_out = encoder(batch["input_ids"], padding_mask=batch["input_mask"], warmup_eta=0.5)
    head_out = head(encoder_out["sequence_states"], padding_mask=batch["input_mask"], warmup_eta=0.5)

    _assert_finite(encoder_out)
    _assert_finite(head_out)


def test_diagnostics_contain_drive_resistance_energy_activation() -> None:
    batch = _sample_batch(batch_size=2, seq_len=24)
    vocab = CharVocabulary()
    encoder = _encoder(vocab_size=len(vocab), pad_id=vocab.pad_id)
    out = encoder(batch["input_ids"], padding_mask=batch["input_mask"], warmup_eta=0.5)

    diagnostics = out["diagnostics"]
    for section in ("local", "chunk", "attractor"):
        assert section in diagnostics
    for key in ("drive", "resistance", "energy", "activation"):
        assert key in diagnostics["local"]
        assert key in diagnostics["chunk"]
        assert key in diagnostics["attractor"]


def test_no_future_leakage_in_sequence_states() -> None:
    torch.manual_seed(3)
    vocab = CharVocabulary()
    seq_len = 30
    cutoff = 12
    input_ids = torch.randint(3, len(vocab), (2, seq_len), dtype=torch.long)
    input_ids[:, 0] = vocab.bos_id
    padding_mask = torch.ones_like(input_ids, dtype=torch.bool)
    changed = input_ids.clone()
    changed[:, cutoff + 1 :] = ((changed[:, cutoff + 1 :] - 3 + 7) % (len(vocab) - 3)) + 3

    encoder = _encoder(vocab_size=len(vocab), pad_id=vocab.pad_id)
    encoder.eval()
    with torch.no_grad():
        baseline = encoder(input_ids, padding_mask=padding_mask, warmup_eta=0.5)["sequence_states"]
        perturbed = encoder(changed, padding_mask=padding_mask, warmup_eta=0.5)["sequence_states"]

    assert torch.allclose(baseline[:, : cutoff + 1], perturbed[:, : cutoff + 1], atol=1.0e-5, rtol=1.0e-5)


def test_tiny_training_step_produces_finite_loss() -> None:
    batch = _sample_batch(batch_size=4, seq_len=28)
    vocab = CharVocabulary()
    encoder = _encoder(vocab_size=len(vocab), pad_id=vocab.pad_id)
    head = EMLTextFieldGenerationHead(state_dim=16, vocab_size=len(vocab), hidden_dim=32)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=1.0e-3)

    encoder_out = encoder(batch["input_ids"], padding_mask=batch["input_mask"], warmup_eta=0.5)
    head_out = head(encoder_out["sequence_states"], padding_mask=batch["input_mask"], warmup_eta=0.5)
    loss = F.cross_entropy(
        head_out["logits"].reshape(-1, len(vocab)),
        batch["target_ids"].reshape(-1),
        ignore_index=vocab.pad_id,
    )
    loss = loss + 0.02 * encoder_out["diagnostics"]["budget_loss"]

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_no_forbidden_modules_in_text_field_models() -> None:
    vocab = CharVocabulary()
    modules = nn.ModuleList(
        [
            _encoder(vocab_size=len(vocab), pad_id=vocab.pad_id),
            EMLTextFieldGenerationHead(state_dim=16, vocab_size=len(vocab), hidden_dim=32),
        ]
    )

    assert not any(isinstance(module, nn.MultiheadAttention) for module in modules.modules())
    assert not any("transformer" in module.__class__.__name__.lower() for module in modules.modules())
