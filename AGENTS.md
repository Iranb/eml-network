# AGENTS.md

## Mission

This repository is a general EML-native foundation model core v0.

Primary validation tasks in this phase:
1. image classification
2. text generation / language modeling

MNIST remains an optional baseline only.
Prototype / novelty / GCD behavior is auxiliary only.

Think from EML drive/resistance primitives, not Transformer blocks. Field, efficient representation, and MERC modules are experimental branches, not validated preferred architectures.

Experimental field path:

```text
EMLSensor
-> EMLHypothesisField
-> EMLHypothesisCompetition
-> EMLConsensusField
-> EMLCompositionField
-> EMLAttractorMemory
-> EMLFieldReadout
```

Image field and text field encoders are validation paths, but they are not proven replacements for CNN/local text baselines. PureEML image backbones, EMLTextBackbone, CNN, and MNIST models are compatibility baselines.

Current evidence must be stated cautiously:
- `cnn_eml` is the strongest stable image baseline observed so far.
- EML clean classification head advantage is not proven. Frozen-feature and end-to-end head ablations must beat linear/MLP/cosine under matched conditions before any claim.
- End-to-end CNN plus cosine prototype is currently stronger than current EML and MERC heads on the latest CIFAR reports.
- MERC head/block is currently no-go. Do not claim MERC works as a head or residual block unless a new report reverses that result.
- Efficient EML representation trunks and field backbones are experimental and not proven replacements for CNN/local text baselines.
- Support factorization is preliminary positive; conflict/resistance factorization is weak or negative.
- Uncertainty/selective classification is the next target, not clean top-1 victory.
- Prefer falsifiable ablation claims over adding more modules.

Experimental efficient representation path:

```text
local evidence
-> support/conflict decomposition
-> EML responsibility propagation
-> precision update
-> composition
-> attractor memory
-> representation readout
```

## Durable Architecture Rules

Always preserve the drive / resistance split.

Drive means:
- evidence
- relevance
- utility
- information gain
- semantic support
- expected success

Resistance means:
- risk
- cost
- uncertainty
- noise
- ambiguity
- overwrite cost
- trust penalty

## EML Rules

Conceptual EML:

```text
eml(x, y) = exp(x) - log(y)
```

Production EML must use stable sEML:

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

Required:
- run `expm1`, `log`, and `softplus` in fp32 islands
- cast outputs back to input dtype only after the sensitive ops
- keep diagnostics available for drive, resistance, energy, and gates
- preserve responsibility/null/update diagnostics for representation modules

## Allowed Components

Allowed:
- PyTorch
- MLP
- CNN
- local convolution
- GRU / gated recurrent updates
- PureEMLImageBackbone
- EMLTextBackbone
- typed slots
- sparse routing
- sparse EML-gated message passing
- recurrent slot updates
- local reconstruction heads
- local text generation heads
- action / patch / risk heads
- optional prototype novelty head

Forbidden:
- Transformer
- self-attention
- MultiheadAttention
- Mamba
- SSM
- selective scan

Do not make GCD central.
Do not build backend or OpenClaw integration unless explicitly asked.
Optimize representation code for local windows and small attractor counts.
Keep old baselines compatible.

## This Phase Is Not

Do not build in this phase:
- backend integration
- OpenClaw integration
- web UI
- production serving stack
- GCD-specific system

## Validation Rules

Always produce durable report artifacts for experiments:
- write raw run data under `reports/runs/`
- maintain `reports/runs/summary.csv`
- regenerate `reports/EML_VALIDATION_REPORT.md` after validation runs
- generate focused reports when running focused ablations:
  - `reports/HEAD_ABLATION_REPORT.md`
  - `reports/CNN_HEAD_END_TO_END_REPORT.md`
  - `reports/MECHANISM_PROBE_REPORT.md`
  - `reports/IMAGE_REPRESENTATION_ABLATION_REPORT.md`
  - `reports/TEXT_REPRESENTATION_ABLATION_REPORT.md`
  - `reports/EML_MASTER_NEXT_STEP_REPORT.md`
  - `reports/CLAIM_STATUS.md`
  - `reports/EML_UNCERTAINTY_RESISTANCE_REPORT.md`
- never fabricate experiment results
- mark unrun experiments as `NOT RUN` and record the reason
- record failed runs with failure reasons when practical
- focus on ablation and measured diagnostics before adding new architecture
- use staged hardening only as a measured ablation, not as an assumed improvement
- keep a Claim Status table current when reports change
- use linear, MLP, and cosine prototype as primary baselines for head claims
- do not claim clean top-1 advantage unless the report directly supports it
- report calibration, selective risk, corruption AUROC, and resistance/noise or resistance/occlusion correlations for uncertainty work
- when comparing model/network performance, run each comparison until early stop; a row that only hits `max_steps` is incomplete unless a later early-stopped replacement row is recorded and cited

Always run after meaningful changes:
- `pytest`
- `python scripts/run_eml_validation_suite.py --mode smoke --device cpu`
- `python scripts/generate_eml_report.py`
- `python scripts/train_eml_image_field.py --steps 50 --device cpu`
- `python scripts/train_eml_text_field.py --steps 50 --device cpu`
- `python scripts/train_eml_field_foundation.py --steps 50 --device cpu`
- `python scripts/train_efficient_eml_image_repr.py --steps 50 --device cpu`
- `python scripts/train_efficient_eml_text_repr.py --steps 50 --device cpu`
- `python scripts/train_efficient_eml_foundation_repr.py --steps 50 --device cpu`
- `python scripts/train_image_shapes.py --steps 50`
- `python scripts/train_text_grammar.py --steps 50`
- `python scripts/train_foundation_core.py --steps 50`

Optional only if local data and dependencies exist:
- `python train_mnist.py --epochs 1`
- `python scripts/train_image_classification.py --dataset mnist --steps 50`

Focused stabilization smoke commands:
- `python scripts/run_head_ablation.py --dataset synthetic_shape --mode smoke --seeds 0 1 --num-workers 0`
- `python scripts/run_cnn_head_end_to_end_ablation.py --dataset synthetic_shape --mode smoke --seeds 0 1 --num-workers 0`
- `python scripts/run_mechanism_probes.py --mode smoke --seeds 0 1`
- `python scripts/run_image_representation_ablation.py --mode smoke --seeds 0 1 --num-workers 0`
- `python scripts/run_text_representation_ablation.py --mode smoke --seeds 0 1 --num-workers 0`
- `python scripts/generate_master_eml_report.py`
