from __future__ import annotations

import torch
import torch.nn.functional as F

from eml_mnist import CharVocabulary, EfficientEMLTextEncoder, EfficientEMLTextGenerationHead, SyntheticTextEnergyDataset


def _batch(batch_size: int = 2, seq_len: int = 24) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    vocab = CharVocabulary()
    dataset = SyntheticTextEnergyDataset(size=batch_size, seq_len=seq_len, vocab=vocab, seed=13)
    input_ids = torch.stack([dataset[index]["input_ids"] for index in range(batch_size)], dim=0)
    target_ids = torch.stack([dataset[index]["target_ids"] for index in range(batch_size)], dim=0)
    mask = torch.stack([dataset[index]["input_mask"] for index in range(batch_size)], dim=0)
    return input_ids, target_ids, mask, len(vocab)


def test_efficient_eml_text_encoder_forward() -> None:
    input_ids, _, mask, vocab_size = _batch()
    model = EfficientEMLTextEncoder(
        vocab_size=vocab_size,
        embed_dim=16,
        state_dim=24,
        hidden_dim=48,
        num_hypotheses=4,
        num_attractors=3,
        representation_dim=24,
        causal_window_size=5,
        chunk_size=4,
    )

    out = model(input_ids, padding_mask=mask, warmup_eta=0.5)

    assert out["sequence_states"].shape == (2, 24, 24)
    assert out["chunk_states"].ndim == 3
    assert out["attractor_states"].shape[:2] == (2, 3)
    assert out["representation"].shape == (2, 24)
    assert torch.isfinite(out["sequence_states"]).all()


def test_efficient_eml_text_generation_head_logits_and_tiny_step() -> None:
    input_ids, target_ids, mask, vocab_size = _batch()
    encoder = EfficientEMLTextEncoder(
        vocab_size=vocab_size,
        embed_dim=16,
        state_dim=24,
        hidden_dim=48,
        num_hypotheses=4,
        num_attractors=3,
        representation_dim=24,
        causal_window_size=5,
        chunk_size=4,
    )
    head = EfficientEMLTextGenerationHead(state_dim=24, vocab_size=vocab_size, hidden_dim=48)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=1.0e-3)

    encoded = encoder(input_ids, padding_mask=mask, warmup_eta=0.5)
    out = head(encoded["sequence_states"], padding_mask=mask, warmup_eta=0.5)
    loss = F.cross_entropy(out["logits"].reshape(-1, vocab_size), target_ids.reshape(-1), ignore_index=0)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert out["logits"].shape == (2, 24, vocab_size)
    assert torch.isfinite(loss)


def test_efficient_eml_text_no_future_leakage() -> None:
    input_ids, _, mask, vocab_size = _batch(batch_size=1, seq_len=28)
    model = EfficientEMLTextEncoder(
        vocab_size=vocab_size,
        embed_dim=16,
        state_dim=20,
        hidden_dim=40,
        num_hypotheses=4,
        num_attractors=3,
        representation_dim=20,
        causal_window_size=5,
        chunk_size=4,
    )
    model.eval()
    cutoff = 10
    changed = input_ids.clone()
    changed[:, cutoff + 1 :] = (changed[:, cutoff + 1 :] + 7) % vocab_size

    with torch.no_grad():
        original = model(input_ids, padding_mask=mask, warmup_eta=0.5)["sequence_states"]
        perturbed = model(changed, padding_mask=mask, warmup_eta=0.5)["sequence_states"]

    assert torch.allclose(original[:, : cutoff + 1], perturbed[:, : cutoff + 1], atol=1.0e-5, rtol=1.0e-5)


def test_efficient_eml_text_thresholded_and_no_attractor_switches() -> None:
    input_ids, _, mask, vocab_size = _batch(batch_size=2, seq_len=24)
    model = EfficientEMLTextEncoder(
        vocab_size=vocab_size,
        embed_dim=16,
        state_dim=20,
        hidden_dim=40,
        num_hypotheses=4,
        num_attractors=1,
        representation_dim=20,
        causal_window_size=5,
        chunk_size=4,
        responsibility_mode="thresholded_null",
        enable_composition=False,
        enable_attractor=False,
    )

    out = model(input_ids, padding_mask=mask, warmup_eta=0.5)

    assert out["representation"].shape == (2, 20)
    assert out["diagnostics"]["num_attractors"].item() == 0
    assert torch.isfinite(out["sequence_states"]).all()
