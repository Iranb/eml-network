from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from eml_mnist import EMLFoundationCore
from eml_mnist.eml_image_field import EMLImageFieldClassifier, EMLImageFieldEncoder
from eml_mnist.eml_repr_image import EfficientEMLImageClassifier, EfficientEMLImageEncoder
from eml_mnist.eml_repr_text import EfficientEMLTextEncoder, EfficientEMLTextGenerationHead
from eml_mnist.eml_text_field import EMLTextFieldEncoder, EMLTextFieldGenerationHead
from eml_mnist.field import (
    EMLAttractorMemory,
    EMLCompositionField,
    EMLConsensusField,
    EMLFieldReadout,
    EMLHypothesisCompetition,
    EMLHypothesisField,
    EMLSensor,
)


FORBIDDEN_PATTERNS = [
    "Transformer",
    "MultiheadAttention",
    "self_attention",
    "SelfAttention",
    "Mamba",
    "SSM",
    "selective_scan",
]


def test_no_forbidden_modules_in_foundation_core() -> None:
    model = EMLFoundationCore(
        slot_dim=16,
        event_dim=16,
        hidden_dim=32,
        slot_layout={"goal": 1, "memory": 2, "text": 2},
        num_layers=2,
        top_k=3,
        text_vocab_size=32,
        text_embed_dim=16,
        text_hidden_dim=16,
        enable_text_generation_head=True,
        enable_action_head=False,
        enable_patch_rank_head=False,
    )

    assert not any(isinstance(module, nn.MultiheadAttention) for module in model.modules())
    assert not any("transformer" in module.__class__.__name__.lower() for module in model.modules())


def test_no_forbidden_modules_in_field_core() -> None:
    modules = nn.ModuleList(
        [
            EMLSensor(input_dim=4, measurement_dim=6),
            EMLHypothesisField(measurement_dim=6, field_dim=8, num_hypotheses=3, hidden_dim=16),
            EMLHypothesisCompetition(),
            EMLConsensusField(field_dim=8, hidden_dim=16, num_hypotheses=3, mode="image"),
            EMLCompositionField(field_dim=8, hidden_dim=16, mode="image"),
            EMLAttractorMemory(field_dim=8, num_attractors=3, hidden_dim=16),
            EMLFieldReadout(field_dim=8, hidden_dim=16),
        ]
    )

    assert not any(isinstance(module, nn.MultiheadAttention) for module in modules.modules())
    assert not any("transformer" in module.__class__.__name__.lower() for module in modules.modules())


def test_no_forbidden_modules_in_image_field_core() -> None:
    modules = nn.ModuleList(
        [
            EMLImageFieldEncoder(representation_dim=24, field_dim=24, measurement_dim=24, sensor_dim=24),
            EMLImageFieldClassifier(num_classes=5, representation_dim=24, field_dim=24, measurement_dim=24, sensor_dim=24),
        ]
    )

    assert not any(isinstance(module, nn.MultiheadAttention) for module in modules.modules())
    assert not any("transformer" in module.__class__.__name__.lower() for module in modules.modules())


def test_no_forbidden_modules_in_text_field_core() -> None:
    modules = nn.ModuleList(
        [
            EMLTextFieldEncoder(
                vocab_size=32,
                embed_dim=16,
                sensor_dim=16,
                measurement_dim=16,
                field_dim=16,
                hidden_dim=32,
                num_hypotheses=3,
                num_chunk_hypotheses=3,
                num_attractors=3,
                representation_dim=16,
            ),
            EMLTextFieldGenerationHead(state_dim=16, vocab_size=32, hidden_dim=32),
        ]
    )

    assert not any(isinstance(module, nn.MultiheadAttention) for module in modules.modules())
    assert not any("transformer" in module.__class__.__name__.lower() for module in modules.modules())


def test_no_forbidden_modules_in_efficient_representation_core() -> None:
    modules = nn.ModuleList(
        [
            EfficientEMLImageEncoder(state_dim=16, hidden_dim=32, num_hypotheses=3, num_attractors=3),
            EfficientEMLImageClassifier(num_classes=5, state_dim=16, hidden_dim=32, num_hypotheses=3, num_attractors=3),
            EfficientEMLTextEncoder(
                vocab_size=32,
                embed_dim=16,
                state_dim=16,
                hidden_dim=32,
                num_hypotheses=3,
                num_attractors=3,
                causal_window_size=4,
            ),
            EfficientEMLTextGenerationHead(state_dim=16, vocab_size=32, hidden_dim=32),
        ]
    )

    assert not any(isinstance(module, nn.MultiheadAttention) for module in modules.modules())
    assert not any("transformer" in module.__class__.__name__.lower() for module in modules.modules())


def test_no_forbidden_patterns_in_source_tree() -> None:
    source_roots = [Path("eml_mnist"), Path("scripts")]
    for root in source_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for pattern in FORBIDDEN_PATTERNS:
                assert pattern not in text, f"forbidden pattern {pattern!r} found in {path}"
