from __future__ import annotations

import torch

from eml_mnist import build_mnist_eml_model
from eml_mnist.model import EMLLocalMessageBlock


def test_cnn_eml_stage_forward_refines_spatial_tokens() -> None:
    model = build_mnist_eml_model(
        model_name="cnn_eml_stage",
        num_classes=10,
        input_channels=1,
        feature_dim=16,
        hidden_dim=32,
        bank_dim=16,
        bank_blocks=2,
        dropout=0.0,
    )
    images = torch.randn(2, 1, 28, 28)

    outputs = model(images, warmup_eta=0.5)

    assert outputs["logits"].shape == (2, 10)
    assert outputs["tokens"].shape == (2, 14 * 14, 16)
    assert len(outputs["block_stats"]) == 4
    assert outputs["pool_weights"].shape == (2, 14 * 14)
    assert torch.allclose(outputs["pool_weights"].sum(dim=1), torch.ones(2), atol=1e-5)
    assert torch.isfinite(outputs["logits"]).all()
    assert torch.isfinite(outputs["pool_energy"]).all()


def test_pure_eml_v2_api_and_local_message_block_stay_finite() -> None:
    model = build_mnist_eml_model(
        model_name="pure_eml_v2",
        num_classes=10,
        image_size=28,
        patch_size=7,
        patch_stride=4,
        input_channels=1,
        feature_dim=16,
        hidden_dim=32,
        bank_dim=16,
        bank_blocks=2,
        dropout=0.0,
    )
    images = torch.randn(2, 1, 28, 28)

    outputs = model(images, warmup_eta=0.5)

    assert outputs["logits"].shape == (2, 10)
    assert outputs["tokens"].shape[-1] == 16
    assert outputs["pool_weights"].ndim == 2
    assert torch.isfinite(outputs["logits"]).all()


def test_local_message_block_has_relative_position_embedding_and_normalized_aggregation() -> None:
    block = EMLLocalMessageBlock(feature_dim=8, hidden_dim=16, window_size=3, dropout=0.0)
    tokens = torch.randn(2, 4, 4, 8)

    outputs = block(tokens, warmup_eta=0.5)

    assert block.relative_position_embed.shape == (9, 8)
    assert outputs["output"].shape == tokens.shape
    assert outputs["gate"].shape == (2, 16, 9)
    assert outputs["gate_mass"].shape == (2, 16, 1)
    assert torch.isfinite(outputs["output"]).all()
    assert torch.isfinite(outputs["gate"]).all()
    assert torch.isfinite(outputs["gate_mass"]).all()


def test_image_local_message_relpos() -> None:
    block = EMLLocalMessageBlock(feature_dim=8, hidden_dim=16, window_size=3, dropout=0.0)
    tokens = torch.randn(2, 4, 4, 8)

    outputs = block(tokens, warmup_eta=0.5)

    assert hasattr(block, "relative_position_embed")
    assert block.relative_position_embed.shape[0] == 9
    assert outputs["output"].shape == tokens.shape
    assert torch.isfinite(outputs["output"]).all()
    assert torch.isfinite(outputs["gate_mass"]).all()
