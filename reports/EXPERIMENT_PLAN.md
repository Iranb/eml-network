# EML Validation Experiment Plan

## Purpose

This plan defines a rigorous, artifact-first validation path for the EML repository. The goal is to test whether EML is useful beyond gates and classifier heads by measuring representation learning behavior, mechanism ablations, diagnostics, and cost.

No result should be claimed unless it appears in a raw run artifact. Experiments that are planned but not executed must be recorded as `NOT RUN` with a reason.

## Existing Models Discovered

### Image Baselines

| model | status | notes |
| --- | --- | --- |
| `cnn_eml` | AVAILABLE | CNN visual stem with EML residual bank and prototype classifier. Current strongest practical CIFAR-10 short-run baseline. |
| `pure_eml` | AVAILABLE | Patch/token style EML baseline. |
| `pure_eml_v2` | AVAILABLE | Improved local image EML baseline through `PureEMLImageClassifier`. |
| `cnn_eml_stage` | AVAILABLE | CNN/local-stage EML variant if requested by model builder. |
| `PureEMLImageBackbone` | AVAILABLE | Compatibility image backbone returning token and global-slot features. |
| `PureEMLImageClassifier` | AVAILABLE | Compatibility image classifier using EML prototype head. |
| `EMLImageFieldClassifier` | AVAILABLE | EML energy-field image path. |
| `EfficientEMLImageClassifier` | AVAILABLE | Efficient local-window EML representation path. |
| `LocalConvBaseline` | NOT AVAILABLE | Useful future sanity baseline, not yet implemented. |

### Text Models

| model | status | notes |
| --- | --- | --- |
| `EMLTextBackbone` | AVAILABLE | Compatibility text path using local codec, causal local messages, residual EML bank, and EML pooling. |
| `LocalTextGenerationHead` | AVAILABLE | Existing local generation head. |
| `EMLTextFieldEncoder` | AVAILABLE | EML field text encoder. |
| `EMLTextFieldGenerationHead` | AVAILABLE | EML field next-token head. |
| `EfficientEMLTextEncoder` | AVAILABLE | Efficient causal local-window EML representation path. |
| `EfficientEMLTextGenerationHead` | AVAILABLE | EML next-token scoring head over vocabulary prototypes. |
| `LocalTextCodec + linear next-token head` | NOT AVAILABLE AS SCRIPT | Components exist, but a standardized comparison wrapper is not yet implemented. |
| `GRU/Conv text sanity baseline` | NOT AVAILABLE | Useful future sanity baseline. |

### Foundation and Graph

| component | status | notes |
| --- | --- | --- |
| `EMLFoundationCore` | AVAILABLE | Old event path, old backbone path, field path, and efficient representation path are supported. |
| `SlotBank` | AVAILABLE | Typed slot memory. |
| `EMLSlotGraphLayer` | AVAILABLE | Sparse route/message/update layer. |
| `EMLMessagePassing` | AVAILABLE | Supports responsibility mode and compatibility gate-normalized mode. |
| `EMLStateUpdateCell` | AVAILABLE | Supports sigmoid update and precision update modes. |

### EML Primitives

| primitive | status | notes |
| --- | --- | --- |
| `EMLUnit` | AVAILABLE | Stable fp32 island implementation. |
| `EMLGate` / `EMLUpdateGate` / `EMLMessageGate` | AVAILABLE | Existing gate primitives. |
| `EMLScore` | AVAILABLE | Returns `score` and `energy`. |
| `EMLBank` | AVAILABLE | Existing EML bank primitive. |
| `EMLResponsibility` | AVAILABLE | Converts EML energies into responsibility weights with optional null route. |
| `EMLPrecisionUpdate` | AVAILABLE | Precision-style state update with sigmoid compatibility mode. |

## Available Datasets Discovered

| dataset | status | dependency | notes |
| --- | --- | --- | --- |
| `SyntheticShapeEnergyDataset` | AVAILABLE | none | Offline image dataset with noise, occlusion, clutter, resistance target. |
| `SyntheticShapeDataset` | AVAILABLE | none | Older synthetic shape baseline dataset. |
| `SyntheticTextEnergyDataset` | AVAILABLE | none | Offline character-level sequence data with corruption and resistance labels. |
| `SyntheticGrammarDataset` | AVAILABLE | none | Older grammar dataset for next-token smoke training. |
| `MNIST` | OPTIONAL | torchvision/data | Do not require for tests. |
| `CIFAR-10` | OPTIONAL | torchvision/data | Use if torchvision works and data is local or download allowed. |
| `Fashion-MNIST` | NOT CONFIGURED | torchvision/data | Listed by task but no loader currently implemented. |
| local char corpus | NOT DISCOVERED | local files | No local corpus path standardized yet. |

## Missing but Required Experiment Utilities

1. Common run logger writing `config.json`, `history.json`, `metrics.csv`, `diagnostics.csv`, `summary.json`, `model_info.json`, and `artifacts.json`.
2. `reports/runs/summary.csv` registry with run and not-run rows.
3. Diagnostic collector that can flatten nested EML outputs.
4. Standard metric helpers for image, text, correlations, efficiency, and parameter counts.
5. Validation runner with smoke, ablation, CIFAR medium, and text medium modes.
6. Report generator reading raw artifacts and producing `reports/EML_VALIDATION_REPORT.md`.
7. Tests for logger file creation and diagnostic flattening.

## Proposed Ablation Matrix

The full matrix is too large for one pass. Use a minimal one-factor-at-a-time plan:

1. Baselines:
   - `cnn_eml`
   - `pure_eml_v2`
   - `EfficientEMLImageClassifier`
   - `EMLTextBackbone + LocalTextGenerationHead`
   - `EfficientEMLTextEncoder + EfficientEMLTextGenerationHead`

2. Mechanism probes:
   - gate-normalized message aggregation
   - responsibility aggregation without null
   - responsibility aggregation with null
   - sigmoid update
   - precision update

3. Efficient representation ablations:
   - no composition if supported
   - EML composition
   - attractor 4
   - attractor 8
   - warmup disabled
   - warmup enabled

4. Classification head:
   - ambiguity-aware EML classification head
   - simpler linear or cosine head when implemented

## Proposed Commands

Smoke:

```bash
pytest
python scripts/run_eml_validation_suite.py --mode smoke --device cpu
python scripts/generate_eml_report.py
```

Ablation:

```bash
python scripts/run_eml_validation_suite.py --mode ablation --device cuda
python scripts/generate_eml_report.py
```

CIFAR medium:

```bash
python scripts/run_eml_validation_suite.py --mode cifar-medium --device cuda --data-dir ~/dataset/data
python scripts/generate_eml_report.py
```

Text medium:

```bash
python scripts/run_eml_validation_suite.py --mode text-medium --device cuda
python scripts/generate_eml_report.py
```

## Expected Output Artifacts

Each run writes:

```text
reports/runs/<timestamp>_<run_id>/
  config.json
  history.json
  metrics.csv
  diagnostics.csv
  summary.json
  stdout.log
  model_info.json
  artifacts.json
```

The suite also writes:

```text
reports/runs/summary.csv
reports/EML_VALIDATION_REPORT.md
```

## Risks and Missing Dependencies

| risk | handling |
| --- | --- |
| torchvision unavailable | record CIFAR/MNIST experiments as `NOT RUN`. |
| CUDA unavailable | run smoke on CPU and record GPU modes as `NOT RUN`. |
| CIFAR-10 data missing | do not download unless explicitly enabled; record as `NOT RUN`. |
| Long-running field models | keep smoke small; record medium/full as `NOT RUN` unless explicitly launched. |
| Incomplete ablation support | record unsupported factors as `NOT RUN` rather than inventing results. |
| Existing branches overlap | report model configs and raw artifact paths so comparisons remain auditable. |
