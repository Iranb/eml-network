# General EML-Native Energy-Field Core v0

This repository is no longer just EML MNIST. It is an EML-native energy-field research prototype for validating image classification and text generation with drive/resistance primitives.

It is:
- not MNIST-only
- not GCD-specific
- not a Transformer
- not a Mamba or SSM model
- not a backend or OpenClaw integration

EML is not used only as a head. The preferred architecture uses EML as the core field primitive:

```text
EMLSensor
-> EMLHypothesisField
-> EMLHypothesisCompetition
-> EMLConsensusField
-> EMLCompositionField
-> EMLAttractorMemory
-> EMLFieldReadout
```

This branch also explores EML as the representation mechanism itself:

```text
local evidence
-> support/conflict decomposition
-> EML responsibility propagation
-> precision update
-> composition
-> attractor memory
-> representation readout
```

Image field encoder and text field encoder are now the primary validation paths. Old MNIST, CNN, and PureEML models remain compatibility baselines. Prototype and novelty behavior remains auxiliary only.

## Core Primitive

Conceptual EML:

```text
eml(x, y) = exp(x) - log(y)
```

Production code uses stable sEML:

```text
D = c * tanh(d / c)
R = log(1 + softplus(r)) - rho0

sEML(d, r) =
    gamma * [
        (1 - warmup_eta) * D
        + warmup_eta * expm1(D)
        - lambda * R
    ] + bias
```

Design rules:
- preserve separate `drive` and `resistance` branches
- run `expm1`, `log`, and `softplus` in fp32 islands
- expose diagnostics for drive, resistance, energy, activation, gates, consensus support/conflict, and attractors

## Primary Field Paths

Image field path:

```text
image
-> thin image sensor
-> EMLHypothesisField
-> EMLHypothesisCompetition
-> image EMLConsensusField
-> EMLCompositionField
-> EMLAttractorMemory
-> EMLFieldReadout
-> EMLFoundationCore
```

Text field path:

```text
text ids
-> thin text sensor
-> EMLHypothesisField
-> EMLHypothesisCompetition
-> causal EMLConsensusField
-> chunk EMLCompositionField
-> EMLAttractorMemory
-> EMLFieldReadout
-> EMLFoundationCore
```

The foundation core can inject field attractor states into typed slots instead of reducing field outputs to one event too early.

## Efficient Representation Path

The efficient representation path avoids global pairwise token work:
- images use local 2D windows
- text uses causal local windows
- attractor work is bounded by a small fixed attractor count
- responsibility weights include an optional null route for weak evidence
- precision updates balance old confidence against new evidence

New validation models:
- `EfficientEMLImageEncoder`
- `EfficientEMLImageClassifier`
- `EfficientEMLTextEncoder`
- `EfficientEMLTextGenerationHead`

## Repository Layout

Core modules:
- [eml_mnist/primitives.py](./eml_mnist/primitives.py): stable sEML primitives
- [eml_mnist/representation.py](./eml_mnist/representation.py): efficient local EML representation modules
- [eml_mnist/eml_repr_image.py](./eml_mnist/eml_repr_image.py): efficient image representation encoder/classifier
- [eml_mnist/eml_repr_text.py](./eml_mnist/eml_repr_text.py): efficient causal text representation encoder/generation head
- [eml_mnist/field.py](./eml_mnist/field.py): EML energy-field modules
- [eml_mnist/eml_image_field.py](./eml_mnist/eml_image_field.py): image field encoder/classifier
- [eml_mnist/eml_text_field.py](./eml_mnist/eml_text_field.py): text field encoder/generation head
- [eml_mnist/graph.py](./eml_mnist/graph.py): typed slots, sparse routing, gate-normalized message passing, slot updates
- [eml_mnist/heads.py](./eml_mnist/heads.py): representation, classification, action, patch, risk, reconstruction, novelty heads
- [eml_mnist/foundation.py](./eml_mnist/foundation.py): `EMLFoundationCore` with old event/backbone paths and new field paths

Datasets:
- [eml_mnist/image_datasets.py](./eml_mnist/image_datasets.py): `SyntheticShapeDataset`, `SyntheticShapeEnergyDataset`
- [eml_mnist/text_datasets.py](./eml_mnist/text_datasets.py): `SyntheticGrammarDataset`, `SyntheticTextEnergyDataset`

Compatibility baselines:
- [eml_mnist/image_backbones.py](./eml_mnist/image_backbones.py): `PureEMLImageBackbone`, `PureEMLImageClassifier`
- [eml_mnist/text_backbones.py](./eml_mnist/text_backbones.py): `EMLTextBackbone`
- [eml_mnist/model.py](./eml_mnist/model.py): MNIST/CNN/PureEML variants

## Run Commands

Core test suite:

```bash
pytest
```

Primary field smoke runs:

```bash
python scripts/train_eml_image_field.py --steps 50 --device cpu
python scripts/train_eml_text_field.py --steps 50 --device cpu
python scripts/train_eml_field_foundation.py --steps 50 --device cpu
```

Efficient representation smoke runs:

```bash
python scripts/train_efficient_eml_image_repr.py --steps 50 --device cpu
python scripts/train_efficient_eml_text_repr.py --steps 50 --device cpu
python scripts/train_efficient_eml_foundation_repr.py --steps 50 --device cpu
```

Compatibility smoke runs:

```bash
python scripts/train_image_shapes.py --steps 50 --device cpu
python scripts/train_text_grammar.py --steps 50 --device cpu
python scripts/train_foundation_core.py --steps 50 --device cpu
```

Optional MNIST baseline:

```bash
python train_mnist.py --epochs 1
```

## Diagnostics

Training scripts and the foundation core expose:
- drive / resistance / energy statistics
- local activation rate
- support and conflict means
- null route weight
- responsibility entropy
- precision update strength
- graph gate mass
- active route strength
- attractor activation and injection norm
- representation readout weights

The field foundation smoke run proves:
- image field path runs
- text field path runs
- attractor states are consumed by the foundation slot graph
- losses stay finite
- diagnostics remain available

## Non-Goals

This phase does not build:
- backend integration
- OpenClaw integration
- web UI
- production serving
- GCD-centered behavior
- Transformer/Mamba/SSM stacks
