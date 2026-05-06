from __future__ import annotations

import torch
import torch.nn.functional as F

from eml_mnist import (
    CharVocabulary,
    EMLEdgeFunctionLayer,
    EMLEdgeImageClassifier,
    EMLEdgeTextLM,
    SyntheticShapeEnergyDataset,
    SyntheticTextEnergyDataset,
)


def test_eml_edge_function_layer_forward_is_finite() -> None:
    layer = EMLEdgeFunctionLayer(in_features=4, out_features=3, dropout=0.0)
    x = torch.randn(2, 4)

    out = layer(x, warmup_eta=0.5)

    assert out["output"].shape == (2, 3)
    assert out["drive"].shape == (2, 4, 3)
    assert out["resistance"].shape == (2, 4, 3)
    assert torch.isfinite(out["output"]).all()
    assert torch.isfinite(out["energy"]).all()


def test_eml_edge_image_classifier_logits_and_tiny_step() -> None:
    dataset = SyntheticShapeEnergyDataset(size=2, image_size=32, seed=23)
    images = torch.stack([dataset[index]["image"] for index in range(2)], dim=0)
    labels = torch.tensor([int(dataset[index]["label"]) for index in range(2)], dtype=torch.long)
    model = EMLEdgeImageClassifier(num_classes=5, state_dim=20, edge_width=24)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    out = model(images, warmup_eta=0.5)
    loss = F.cross_entropy(out["logits"], labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert out["logits"].shape == (2, 5)
    assert torch.isfinite(loss)
    assert "edge_energy_mean" in out["diagnostics"]


def test_eml_edge_text_lm_is_causal_and_trainable() -> None:
    vocab = CharVocabulary()
    dataset = SyntheticTextEnergyDataset(size=1, seq_len=24, vocab=vocab, seed=29)
    input_ids = dataset[0]["input_ids"].unsqueeze(0)
    targets = dataset[0]["target_ids"].unsqueeze(0)
    mask = dataset[0]["input_mask"].unsqueeze(0).bool()
    model = EMLEdgeTextLM(vocab_size=len(vocab), pad_id=vocab.pad_id, state_dim=18, edge_width=24)

    cutoff = 9
    changed = input_ids.clone()
    changed[:, cutoff + 1 :] = (changed[:, cutoff + 1 :] + 5) % len(vocab)
    model.eval()
    with torch.no_grad():
        original = model(input_ids, padding_mask=mask, warmup_eta=0.5)["sequence_states"]
        perturbed = model(changed, padding_mask=mask, warmup_eta=0.5)["sequence_states"]
    assert torch.allclose(original[:, : cutoff + 1], perturbed[:, : cutoff + 1], atol=1.0e-5, rtol=1.0e-5)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    out = model(input_ids, padding_mask=mask, warmup_eta=0.5)
    loss = F.cross_entropy(out["logits"].reshape(-1, len(vocab)), targets.reshape(-1), ignore_index=vocab.pad_id)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert out["logits"].shape == (1, 24, len(vocab))
    assert torch.isfinite(loss)
