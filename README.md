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

Image field encoder and text field encoder are primary validation paths for the research direction. Old MNIST, CNN, PureEML, and text backbones remain compatibility baselines and must not be deleted.

Current evidence is deliberately conservative:
- `cnn_eml` is the strongest stable image baseline observed so far.
- EML as a CNN head has not yet been proven better than ordinary linear, MLP, or cosine prototype heads on the same frozen features.
- Efficient EML representation trunks are still under validation and should not be described as proven replacements for CNN/local text baselines.
- Prototype, novelty, and GCD-style behavior remain auxiliary, not the central claim.

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

## Validation and Ablation Reports

The validation suite is the preferred way to compare EML baselines, field models, and efficient representation models without fabricating missing results.

Smoke validation:

```bash
python scripts/run_eml_validation_suite.py --mode smoke --device cpu
python scripts/generate_eml_report.py
```

Longer modes:

```bash
python scripts/run_eml_validation_suite.py --mode ablation --device cuda
python scripts/run_eml_validation_suite.py --mode cifar-medium --device cuda
python scripts/run_eml_validation_suite.py --mode text-medium --device cuda
python scripts/generate_eml_report.py
```

Artifacts are written under:

```text
reports/runs/<timestamp>_<run_id>/
  config.json
  history.json
  metrics.csv
  diagnostics.csv
  summary.json
  model_info.json
  artifacts.json
reports/runs/summary.csv
reports/EML_VALIDATION_REPORT.md
```

Planning documents:
- [reports/EXPERIMENT_PLAN.md](./reports/EXPERIMENT_PLAN.md)
- [reports/ABLATION_MATRIX.md](./reports/ABLATION_MATRIX.md)

Focused stabilization reports:
- [reports/HEAD_ABLATION_REPORT.md](./reports/HEAD_ABLATION_REPORT.md): frozen CNN feature head isolation.
- [reports/CNN_HEAD_END_TO_END_REPORT.md](./reports/CNN_HEAD_END_TO_END_REPORT.md): end-to-end CNN plus head ablation.
- [reports/MECHANISM_PROBE_REPORT.md](./reports/MECHANISM_PROBE_REPORT.md): nontrivial mechanism probes for responsibility, null routing, precision updates, composition, and attractors.
- [reports/IMAGE_REPRESENTATION_ABLATION_REPORT.md](./reports/IMAGE_REPRESENTATION_ABLATION_REPORT.md): synthetic image representation ablations, including no-composition/no-attractor/head-without-ambiguity variants.
- [reports/TEXT_REPRESENTATION_ABLATION_REPORT.md](./reports/TEXT_REPRESENTATION_ABLATION_REPORT.md): synthetic text representation ablations with window-8 as the default efficient path.
- [reports/CIFAR_MEDIUM_REPORT.md](./reports/CIFAR_MEDIUM_REPORT.md): CIFAR medium status, gated by synthetic image success.
- [reports/EML_MASTER_NEXT_STEP_REPORT.md](./reports/EML_MASTER_NEXT_STEP_REPORT.md): master stop/go report.

Stabilization smoke commands:

```bash
python scripts/run_head_ablation.py --dataset synthetic_shape --mode smoke --seeds 0 1 --num-workers 0
python scripts/run_cnn_head_end_to_end_ablation.py --dataset synthetic_shape --mode smoke --seeds 0 1 --num-workers 0
python scripts/run_mechanism_probes.py --mode smoke --seeds 0 1
python scripts/run_image_representation_ablation.py --mode smoke --seeds 0 1 --num-workers 0
python scripts/run_text_representation_ablation.py --mode smoke --seeds 0 1 --num-workers 0
python scripts/generate_master_eml_report.py
```

Current verified result status: only completed report artifacts should be treated as verified. Medium/full CIFAR and multi-seed ablations are not claimed until they appear in the relevant `summary.csv`.

Known validation limitations:
- smoke runs are too short for research conclusions
- torchvision may be unavailable in some environments
- real-data runs are optional and must be marked `NOT RUN` when unavailable
- efficiency metrics are approximate unless the runner records device memory and stable timing

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
