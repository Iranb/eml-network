# EML Master Next Step Report

## Executive Summary
- Head isolation: NOT PROVEN: current artifacts include `eml_raw_ambiguity` as the best mean smoke-scale head at 0.2070, but this is not sufficient to claim EML heads beat ordinary heads under matched medium/real-data conditions.
- Efficient image synthetic gate: `EfficientEMLImageClassifier_baseline` reached `0.3750`; CIFAR medium remains gated below `0.8000`.
- Mechanism conclusions should be read from probe success and diagnostics tables, not from synthetic task accuracy alone.
- Representation trunk claims remain conditional until efficient image/text paths beat simple baselines under the same budgets.

## What Is Proven
- Completed runs in this report are backed by raw artifact directories listed below.
- The available head-ablation data isolates ordinary heads and EML prototype heads on the same CNN feature source when those runs are present.
- The synthetic mechanism probes exercise null responsibility, responsibility selection, precision updates, composition consistency, and attractor collapse checks.

## What Is Not Proven
- EML as a general backbone replacement is not proven by these artifacts.
- If the efficient image synthetic gate is below `0.8`, CIFAR claims for that path should be skipped.
- Any missing or failed rows are not interpreted as evidence.

## Run Status
| suite | completed | failed | not run |
| --- | ---: | ---: | ---: |
| head | 64 | 0 | 44 |
| mechanism | 98 | 0 | 0 |
| image | 26 | 0 | 0 |
| text | 20 | 0 | 0 |
| cifar | 0 | 0 | 1 |

## Head Isolation Results
| run_id | status | model | dataset | best | final | reason |
| --- | --- | --- | --- | ---: | ---: | --- |
| frozen_synthetic_shape_eml_raw_ambiguity_seed0 | COMPLETED | eml_raw_ambiguity | synthetic_shape | 0.2422 | 0.2109 |  |

## Mechanism Probe Results
| probe | mechanism | mean success | n |
| --- | --- | ---: | ---: |
| all_noise_should_choose_null | precision_update | 0.0000 | 2 |
| all_noise_should_choose_null | responsibility_no_null | 0.0000 | 2 |
| all_noise_should_choose_null | responsibility_null | 1.0000 | 2 |
| all_noise_should_choose_null | sigmoid_gate_mean | 0.0000 | 2 |
| all_noise_should_choose_null | sigmoid_gate_sum | 0.0000 | 2 |
| all_noise_should_choose_null | sigmoid_update | 0.0000 | 2 |
| all_noise_should_choose_null | thresholded_null | 1.0000 | 2 |
| attractor_should_not_collapse | precision_update | 1.0000 | 2 |
| attractor_should_not_collapse | responsibility_no_null | 1.0000 | 2 |
| attractor_should_not_collapse | responsibility_null | 1.0000 | 2 |
| attractor_should_not_collapse | sigmoid_gate_mean | 1.0000 | 2 |
| attractor_should_not_collapse | sigmoid_gate_sum | 1.0000 | 2 |
| attractor_should_not_collapse | sigmoid_update | 1.0000 | 2 |
| attractor_should_not_collapse | thresholded_null | 1.0000 | 2 |
| composition_requires_consistent_children | precision_update | 1.0000 | 2 |
| composition_requires_consistent_children | responsibility_no_null | 1.0000 | 2 |
| composition_requires_consistent_children | responsibility_null | 1.0000 | 2 |
| composition_requires_consistent_children | sigmoid_gate_mean | 1.0000 | 2 |
| composition_requires_consistent_children | sigmoid_gate_sum | 1.0000 | 2 |
| composition_requires_consistent_children | sigmoid_update | 1.0000 | 2 |
| composition_requires_consistent_children | thresholded_null | 1.0000 | 2 |
| conflicting_neighbors_increase_resistance | precision_update | 1.0000 | 2 |
| conflicting_neighbors_increase_resistance | responsibility_no_null | 1.0000 | 2 |
| conflicting_neighbors_increase_resistance | responsibility_null | 1.0000 | 2 |
| conflicting_neighbors_increase_resistance | sigmoid_gate_mean | 1.0000 | 2 |
| conflicting_neighbors_increase_resistance | sigmoid_gate_sum | 1.0000 | 2 |
| conflicting_neighbors_increase_resistance | sigmoid_update | 1.0000 | 2 |
| conflicting_neighbors_increase_resistance | thresholded_null | 1.0000 | 2 |
| old_state_confident_new_evidence_weak_should_not_update | precision_update | 1.0000 | 2 |
| old_state_confident_new_evidence_weak_should_not_update | responsibility_no_null | 1.0000 | 2 |
| old_state_confident_new_evidence_weak_should_not_update | responsibility_null | 1.0000 | 2 |
| old_state_confident_new_evidence_weak_should_not_update | sigmoid_gate_mean | 1.0000 | 2 |
| old_state_confident_new_evidence_weak_should_not_update | sigmoid_gate_sum | 1.0000 | 2 |
| old_state_confident_new_evidence_weak_should_not_update | sigmoid_update | 1.0000 | 2 |
| old_state_confident_new_evidence_weak_should_not_update | thresholded_null | 1.0000 | 2 |
| old_state_weak_new_evidence_strong_should_update | precision_update | 1.0000 | 2 |
| old_state_weak_new_evidence_strong_should_update | responsibility_no_null | 1.0000 | 2 |
| old_state_weak_new_evidence_strong_should_update | responsibility_null | 1.0000 | 2 |
| old_state_weak_new_evidence_strong_should_update | sigmoid_gate_mean | 1.0000 | 2 |
| old_state_weak_new_evidence_strong_should_update | sigmoid_gate_sum | 1.0000 | 2 |
| old_state_weak_new_evidence_strong_should_update | sigmoid_update | 1.0000 | 2 |
| old_state_weak_new_evidence_strong_should_update | thresholded_null | 1.0000 | 2 |
| one_strong_signal_many_weak_noise | precision_update | 1.0000 | 2 |
| one_strong_signal_many_weak_noise | responsibility_no_null | 1.0000 | 2 |
| one_strong_signal_many_weak_noise | responsibility_null | 1.0000 | 2 |
| one_strong_signal_many_weak_noise | sigmoid_gate_mean | 0.0000 | 2 |
| one_strong_signal_many_weak_noise | sigmoid_gate_sum | 0.0000 | 2 |
| one_strong_signal_many_weak_noise | sigmoid_update | 1.0000 | 2 |
| one_strong_signal_many_weak_noise | thresholded_null | 1.0000 | 2 |

## Image Representation Ablation
| run_id | status | model | dataset | best | final | reason |
| --- | --- | --- | --- | ---: | ---: | --- |
| pure_eml_workers0_seed0 | COMPLETED | pure_eml_workers0 | SyntheticShapeEnergyDataset | 0.3750 | 0.2500 |  |

## Text Representation Ablation
| run_id | status | model | dataset | best | final | reason |
| --- | --- | --- | --- | ---: | ---: | --- |
| efficient_window8_seed0 | COMPLETED | EfficientEMLTextEncoder_window8 | SyntheticTextEnergyDataset | 0.2102 | 0.2102 |  |

## CIFAR Medium Status
| run_id | status | model | dataset | best | final | reason |
| --- | --- | --- | --- | ---: | ---: | --- |
| MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | no completed rows |

## EML Diagnostics
Diagnostics are stored in each run's `diagnostics.csv` and `summary.json` under `final_diagnostics`. Key fields include drive, resistance, energy, null weight, update gate, ambiguity, and attractor diversity when a model exposes them.

## Efficiency Analysis
Per-step time, examples/sec, tokens/sec, parameter counts, and peak memory are stored in run metrics. This master report does not average incompatible task families.

## Failure Modes
- Mark all missing experiments as `NOT RUN` instead of interpreting them.
- Treat all-null behavior, never-null behavior, excessive update gates, resistance collapse, and attractor collapse as diagnostic failures requiring targeted probe review.
- Slow local-window implementations should be compared against CNN/local recurrent baselines before broad claims.

## Recommended Next Experiment
Run medium head isolation and end-to-end CNN+head ablations on CIFAR only after synthetic smoke remains stable, then compare EML centered ambiguity against cosine and MLP heads with bootstrap deltas.

## Stop/Go Decisions
- EML as head: NOT PROVEN: current artifacts include `eml_raw_ambiguity` as the best mean smoke-scale head at 0.2070, but this is not sufficient to claim EML heads beat ordinary heads under matched medium/real-data conditions.
- EML as refinement: GO only as a controlled ablation against the same CNN feature extractor and losses.
- EML as representation trunk: HOLD until synthetic image efficient path clears `0.8` and beats local baselines.
- EML as foundation architecture: HOLD until representation trunk validation improves.

## Raw Artifacts
- head report (exists): `reports/HEAD_ABLATION_REPORT.md`
- cnn_head report (exists): `reports/CNN_HEAD_END_TO_END_REPORT.md`
- mechanism report (exists): `reports/MECHANISM_PROBE_REPORT.md`
- image report (exists): `reports/IMAGE_REPRESENTATION_ABLATION_REPORT.md`
- text report (exists): `reports/TEXT_REPRESENTATION_ABLATION_REPORT.md`
- cifar report (exists): `reports/CIFAR_MEDIUM_REPORT.md`
- head summary: `reports/head_ablation/runs/summary.csv`
  - `frozen_synthetic_shape_linear_seed0`: `reports/head_ablation/runs/20260424_081433_frozen_synthetic_shape_linear_seed0`
  - `frozen_synthetic_shape_mlp_seed0`: `reports/head_ablation/runs/20260424_081433_frozen_synthetic_shape_mlp_seed0`
  - `frozen_synthetic_shape_cosine_prototype_seed0`: `reports/head_ablation/runs/20260424_081433_frozen_synthetic_shape_cosine_prototype_seed0`
  - `frozen_synthetic_shape_eml_no_ambiguity_seed0`: `reports/head_ablation/runs/20260424_081433_frozen_synthetic_shape_eml_no_ambiguity_seed0`
  - `frozen_synthetic_shape_eml_raw_ambiguity_seed0`: `reports/head_ablation/runs/20260424_081433_frozen_synthetic_shape_eml_raw_ambiguity_seed0`
  - `frozen_synthetic_shape_eml_centered_ambiguity_seed0`: `reports/head_ablation/runs/20260424_081433_frozen_synthetic_shape_eml_centered_ambiguity_seed0`
  - `frozen_synthetic_shape_linear_seed1`: `reports/head_ablation/runs/20260424_081434_frozen_synthetic_shape_linear_seed1`
  - `frozen_synthetic_shape_mlp_seed1`: `reports/head_ablation/runs/20260424_081434_frozen_synthetic_shape_mlp_seed1`
  - `frozen_synthetic_shape_cosine_prototype_seed1`: `reports/head_ablation/runs/20260424_081434_frozen_synthetic_shape_cosine_prototype_seed1`
  - `frozen_synthetic_shape_eml_no_ambiguity_seed1`: `reports/head_ablation/runs/20260424_081434_frozen_synthetic_shape_eml_no_ambiguity_seed1`
  - `frozen_synthetic_shape_eml_raw_ambiguity_seed1`: `reports/head_ablation/runs/20260424_081434_frozen_synthetic_shape_eml_raw_ambiguity_seed1`
  - `frozen_synthetic_shape_eml_centered_ambiguity_seed1`: `reports/head_ablation/runs/20260424_081434_frozen_synthetic_shape_eml_centered_ambiguity_seed1`
  - `e2e_synthetic_shape_linear_ce_seed0`: `reports/head_ablation/runs/20260424_081501_e2e_synthetic_shape_linear_ce_seed0`
  - `e2e_synthetic_shape_linear_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081503_e2e_synthetic_shape_linear_ce_pairwise_seed0`
  - `e2e_synthetic_shape_mlp_ce_seed0`: `reports/head_ablation/runs/20260424_081503_e2e_synthetic_shape_mlp_ce_seed0`
  - `e2e_synthetic_shape_mlp_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081505_e2e_synthetic_shape_mlp_ce_pairwise_seed0`
  - `e2e_synthetic_shape_cosine_prototype_ce_seed0`: `reports/head_ablation/runs/20260424_081505_e2e_synthetic_shape_cosine_prototype_ce_seed0`
  - `e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081506_e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed0`
  - `e2e_synthetic_shape_eml_no_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_081508_e2e_synthetic_shape_eml_no_ambiguity_ce_seed0`
  - `e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081510_e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0`
  - `e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_081511_e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0`
  - `e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081513_e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0`
  - `e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_081515_e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0`
  - `e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081516_e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0`
  - `e2e_synthetic_shape_linear_ce_seed1`: `reports/head_ablation/runs/20260424_081518_e2e_synthetic_shape_linear_ce_seed1`
  - `e2e_synthetic_shape_linear_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081520_e2e_synthetic_shape_linear_ce_pairwise_seed1`
  - `e2e_synthetic_shape_mlp_ce_seed1`: `reports/head_ablation/runs/20260424_081520_e2e_synthetic_shape_mlp_ce_seed1`
  - `e2e_synthetic_shape_mlp_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081521_e2e_synthetic_shape_mlp_ce_pairwise_seed1`
  - `e2e_synthetic_shape_cosine_prototype_ce_seed1`: `reports/head_ablation/runs/20260424_081521_e2e_synthetic_shape_cosine_prototype_ce_seed1`
  - `e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081523_e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed1`
  - `e2e_synthetic_shape_eml_no_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_081524_e2e_synthetic_shape_eml_no_ambiguity_ce_seed1`
  - `e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081526_e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1`
  - `e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_081528_e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1`
  - `e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081529_e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1`
  - `e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_081531_e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1`
  - `e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081533_e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1`
  - `frozen_cifar10_linear_seed0`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_linear_seed0`
  - `frozen_cifar10_mlp_seed0`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_mlp_seed0`
  - `frozen_cifar10_cosine_prototype_seed0`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_cosine_prototype_seed0`
  - `frozen_cifar10_eml_no_ambiguity_seed0`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_eml_no_ambiguity_seed0`
  - `frozen_cifar10_eml_raw_ambiguity_seed0`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_eml_raw_ambiguity_seed0`
  - `frozen_cifar10_eml_centered_ambiguity_seed0`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_eml_centered_ambiguity_seed0`
  - `frozen_cifar10_linear_seed1`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_linear_seed1`
  - `frozen_cifar10_mlp_seed1`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_mlp_seed1`
  - `frozen_cifar10_cosine_prototype_seed1`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_cosine_prototype_seed1`
  - `frozen_cifar10_eml_no_ambiguity_seed1`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_eml_no_ambiguity_seed1`
  - `frozen_cifar10_eml_raw_ambiguity_seed1`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_eml_raw_ambiguity_seed1`
  - `frozen_cifar10_eml_centered_ambiguity_seed1`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_eml_centered_ambiguity_seed1`
  - `e2e_cifar10_linear_ce_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_linear_ce_seed0`
  - `e2e_cifar10_linear_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_linear_ce_pairwise_seed0`
  - `e2e_cifar10_mlp_ce_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_mlp_ce_seed0`
  - `e2e_cifar10_mlp_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_mlp_ce_pairwise_seed0`
  - `e2e_cifar10_cosine_prototype_ce_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_cosine_prototype_ce_seed0`
  - `e2e_cifar10_cosine_prototype_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_cosine_prototype_ce_pairwise_seed0`
  - `e2e_cifar10_eml_no_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_no_ambiguity_ce_seed0`
  - `e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0`
  - `e2e_cifar10_eml_centered_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_centered_ambiguity_ce_seed0`
  - `e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0`
  - `e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0`
  - `e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0`
  - `e2e_cifar10_linear_ce_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_linear_ce_seed1`
  - `e2e_cifar10_linear_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_linear_ce_pairwise_seed1`
  - `e2e_cifar10_mlp_ce_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_mlp_ce_seed1`
  - `e2e_cifar10_mlp_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_mlp_ce_pairwise_seed1`
  - `e2e_cifar10_cosine_prototype_ce_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_cosine_prototype_ce_seed1`
  - `e2e_cifar10_cosine_prototype_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_cosine_prototype_ce_pairwise_seed1`
  - `e2e_cifar10_eml_no_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_no_ambiguity_ce_seed1`
  - `e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1`
  - `e2e_cifar10_eml_centered_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_centered_ambiguity_ce_seed1`
  - `e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1`
  - `e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1`
  - `e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1`
  - `frozen_synthetic_shape_linear_seed0`: `reports/head_ablation/runs/20260424_085043_frozen_synthetic_shape_linear_seed0`
  - `frozen_synthetic_shape_mlp_seed0`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_mlp_seed0`
  - `frozen_synthetic_shape_cosine_prototype_seed0`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_cosine_prototype_seed0`
  - `frozen_synthetic_shape_eml_no_ambiguity_seed0`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_eml_no_ambiguity_seed0`
  - `frozen_synthetic_shape_eml_raw_ambiguity_seed0`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_eml_raw_ambiguity_seed0`
  - `frozen_synthetic_shape_eml_centered_ambiguity_seed0`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_eml_centered_ambiguity_seed0`
  - `frozen_synthetic_shape_linear_seed1`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_linear_seed1`
  - `frozen_synthetic_shape_mlp_seed1`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_mlp_seed1`
  - `frozen_synthetic_shape_cosine_prototype_seed1`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_cosine_prototype_seed1`
  - `frozen_synthetic_shape_eml_no_ambiguity_seed1`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_eml_no_ambiguity_seed1`
  - `frozen_synthetic_shape_eml_raw_ambiguity_seed1`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_eml_raw_ambiguity_seed1`
  - `frozen_synthetic_shape_eml_centered_ambiguity_seed1`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_eml_centered_ambiguity_seed1`
  - `e2e_synthetic_shape_linear_ce_seed0`: `reports/head_ablation/runs/20260424_085059_e2e_synthetic_shape_linear_ce_seed0`
  - `e2e_synthetic_shape_linear_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_085101_e2e_synthetic_shape_linear_ce_pairwise_seed0`
  - `e2e_synthetic_shape_mlp_ce_seed0`: `reports/head_ablation/runs/20260424_085101_e2e_synthetic_shape_mlp_ce_seed0`
  - `e2e_synthetic_shape_mlp_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_085103_e2e_synthetic_shape_mlp_ce_pairwise_seed0`
  - `e2e_synthetic_shape_cosine_prototype_ce_seed0`: `reports/head_ablation/runs/20260424_085103_e2e_synthetic_shape_cosine_prototype_ce_seed0`
  - `e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_085104_e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed0`
  - `e2e_synthetic_shape_eml_no_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_085106_e2e_synthetic_shape_eml_no_ambiguity_ce_seed0`
  - `e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_085108_e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0`
  - `e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_085110_e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0`
  - `e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_085112_e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0`
  - `e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_085114_e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0`
  - `e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_085116_e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0`
  - `e2e_synthetic_shape_linear_ce_seed1`: `reports/head_ablation/runs/20260424_085118_e2e_synthetic_shape_linear_ce_seed1`
  - `e2e_synthetic_shape_linear_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_085119_e2e_synthetic_shape_linear_ce_pairwise_seed1`
  - `e2e_synthetic_shape_mlp_ce_seed1`: `reports/head_ablation/runs/20260424_085119_e2e_synthetic_shape_mlp_ce_seed1`
  - `e2e_synthetic_shape_mlp_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_085121_e2e_synthetic_shape_mlp_ce_pairwise_seed1`
  - `e2e_synthetic_shape_cosine_prototype_ce_seed1`: `reports/head_ablation/runs/20260424_085121_e2e_synthetic_shape_cosine_prototype_ce_seed1`
  - `e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_085123_e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed1`
  - `e2e_synthetic_shape_eml_no_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_085125_e2e_synthetic_shape_eml_no_ambiguity_ce_seed1`
  - `e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_085127_e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1`
  - `e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_085129_e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1`
  - `e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_085131_e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1`
  - `e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_085133_e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1`
  - `e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_085135_e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1`
- mechanism summary: `reports/mechanism_probes/runs/summary.csv`
  - `probe_all_noise_should_choose_null_sigmoid_gate_sum_seed0`: `reports/mechanism_probes/runs/20260424_085143_probe_all_noise_should_choose_null_sigmoid_gate_sum_seed0`
  - `probe_all_noise_should_choose_null_sigmoid_gate_mean_seed0`: `reports/mechanism_probes/runs/20260424_085143_probe_all_noise_should_choose_null_sigmoid_gate_mean_seed0`
  - `probe_all_noise_should_choose_null_responsibility_no_null_seed0`: `reports/mechanism_probes/runs/20260424_085143_probe_all_noise_should_choose_null_responsibility_no_null_seed0`
  - `probe_all_noise_should_choose_null_responsibility_null_seed0`: `reports/mechanism_probes/runs/20260424_085143_probe_all_noise_should_choose_null_responsibility_null_seed0`
  - `probe_all_noise_should_choose_null_thresholded_null_seed0`: `reports/mechanism_probes/runs/20260424_085143_probe_all_noise_should_choose_null_thresholded_null_seed0`
  - `probe_all_noise_should_choose_null_precision_update_seed0`: `reports/mechanism_probes/runs/20260424_085143_probe_all_noise_should_choose_null_precision_update_seed0`
  - `probe_all_noise_should_choose_null_sigmoid_update_seed0`: `reports/mechanism_probes/runs/20260424_085143_probe_all_noise_should_choose_null_sigmoid_update_seed0`
  - `probe_one_strong_signal_many_weak_noise_sigmoid_gate_sum_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_sigmoid_gate_sum_seed0`
  - `probe_one_strong_signal_many_weak_noise_sigmoid_gate_mean_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_sigmoid_gate_mean_seed0`
  - `probe_one_strong_signal_many_weak_noise_responsibility_no_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_responsibility_no_null_seed0`
  - `probe_one_strong_signal_many_weak_noise_responsibility_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_responsibility_null_seed0`
  - `probe_one_strong_signal_many_weak_noise_thresholded_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_thresholded_null_seed0`
  - `probe_one_strong_signal_many_weak_noise_precision_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_precision_update_seed0`
  - `probe_one_strong_signal_many_weak_noise_sigmoid_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_sigmoid_update_seed0`
  - `probe_conflicting_neighbors_increase_resistance_sigmoid_gate_sum_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_sigmoid_gate_sum_seed0`
  - `probe_conflicting_neighbors_increase_resistance_sigmoid_gate_mean_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_sigmoid_gate_mean_seed0`
  - `probe_conflicting_neighbors_increase_resistance_responsibility_no_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_responsibility_no_null_seed0`
  - `probe_conflicting_neighbors_increase_resistance_responsibility_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_responsibility_null_seed0`
  - `probe_conflicting_neighbors_increase_resistance_thresholded_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_thresholded_null_seed0`
  - `probe_conflicting_neighbors_increase_resistance_precision_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_precision_update_seed0`
  - `probe_conflicting_neighbors_increase_resistance_sigmoid_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_sigmoid_update_seed0`
  - `probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_gate_sum_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_gate_sum_seed0`
  - `probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_gate_mean_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_gate_mean_seed0`
  - `probe_old_state_confident_new_evidence_weak_should_not_update_responsibility_no_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_confident_new_evidence_weak_should_not_update_responsibility_no_null_seed0`
  - `probe_old_state_confident_new_evidence_weak_should_not_update_responsibility_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_confident_new_evidence_weak_should_not_update_responsibility_null_seed0`
  - `probe_old_state_confident_new_evidence_weak_should_not_update_thresholded_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_confident_new_evidence_weak_should_not_update_thresholded_null_seed0`
  - `probe_old_state_confident_new_evidence_weak_should_not_update_precision_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_confident_new_evidence_weak_should_not_update_precision_update_seed0`
  - `probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_update_seed0`
  - `probe_old_state_weak_new_evidence_strong_should_update_sigmoid_gate_sum_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_weak_new_evidence_strong_should_update_sigmoid_gate_sum_seed0`
  - `probe_old_state_weak_new_evidence_strong_should_update_sigmoid_gate_mean_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_weak_new_evidence_strong_should_update_sigmoid_gate_mean_seed0`
  - `probe_old_state_weak_new_evidence_strong_should_update_responsibility_no_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_weak_new_evidence_strong_should_update_responsibility_no_null_seed0`
  - `probe_old_state_weak_new_evidence_strong_should_update_responsibility_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_weak_new_evidence_strong_should_update_responsibility_null_seed0`
  - `probe_old_state_weak_new_evidence_strong_should_update_thresholded_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_weak_new_evidence_strong_should_update_thresholded_null_seed0`
  - `probe_old_state_weak_new_evidence_strong_should_update_precision_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_weak_new_evidence_strong_should_update_precision_update_seed0`
  - `probe_old_state_weak_new_evidence_strong_should_update_sigmoid_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_weak_new_evidence_strong_should_update_sigmoid_update_seed0`
  - `probe_composition_requires_consistent_children_sigmoid_gate_sum_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_composition_requires_consistent_children_sigmoid_gate_sum_seed0`
  - `probe_composition_requires_consistent_children_sigmoid_gate_mean_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_composition_requires_consistent_children_sigmoid_gate_mean_seed0`
  - `probe_composition_requires_consistent_children_responsibility_no_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_composition_requires_consistent_children_responsibility_no_null_seed0`
  - `probe_composition_requires_consistent_children_responsibility_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_composition_requires_consistent_children_responsibility_null_seed0`
  - `probe_composition_requires_consistent_children_thresholded_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_composition_requires_consistent_children_thresholded_null_seed0`
  - `probe_composition_requires_consistent_children_precision_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_composition_requires_consistent_children_precision_update_seed0`
  - `probe_composition_requires_consistent_children_sigmoid_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_composition_requires_consistent_children_sigmoid_update_seed0`
  - `probe_attractor_should_not_collapse_sigmoid_gate_sum_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_attractor_should_not_collapse_sigmoid_gate_sum_seed0`
  - `probe_attractor_should_not_collapse_sigmoid_gate_mean_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_attractor_should_not_collapse_sigmoid_gate_mean_seed0`
  - `probe_attractor_should_not_collapse_responsibility_no_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_attractor_should_not_collapse_responsibility_no_null_seed0`
  - `probe_attractor_should_not_collapse_responsibility_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_attractor_should_not_collapse_responsibility_null_seed0`
  - `probe_attractor_should_not_collapse_thresholded_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_attractor_should_not_collapse_thresholded_null_seed0`
  - `probe_attractor_should_not_collapse_precision_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_attractor_should_not_collapse_precision_update_seed0`
  - `probe_attractor_should_not_collapse_sigmoid_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_attractor_should_not_collapse_sigmoid_update_seed0`
  - `probe_all_noise_should_choose_null_sigmoid_gate_sum_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_all_noise_should_choose_null_sigmoid_gate_sum_seed1`
  - `probe_all_noise_should_choose_null_sigmoid_gate_mean_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_all_noise_should_choose_null_sigmoid_gate_mean_seed1`
  - `probe_all_noise_should_choose_null_responsibility_no_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_all_noise_should_choose_null_responsibility_no_null_seed1`
  - `probe_all_noise_should_choose_null_responsibility_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_all_noise_should_choose_null_responsibility_null_seed1`
  - `probe_all_noise_should_choose_null_thresholded_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_all_noise_should_choose_null_thresholded_null_seed1`
  - `probe_all_noise_should_choose_null_precision_update_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_all_noise_should_choose_null_precision_update_seed1`
  - `probe_all_noise_should_choose_null_sigmoid_update_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_all_noise_should_choose_null_sigmoid_update_seed1`
  - `probe_one_strong_signal_many_weak_noise_sigmoid_gate_sum_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_sigmoid_gate_sum_seed1`
  - `probe_one_strong_signal_many_weak_noise_sigmoid_gate_mean_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_sigmoid_gate_mean_seed1`
  - `probe_one_strong_signal_many_weak_noise_responsibility_no_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_responsibility_no_null_seed1`
  - `probe_one_strong_signal_many_weak_noise_responsibility_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_responsibility_null_seed1`
  - `probe_one_strong_signal_many_weak_noise_thresholded_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_thresholded_null_seed1`
  - `probe_one_strong_signal_many_weak_noise_precision_update_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_precision_update_seed1`
  - `probe_one_strong_signal_many_weak_noise_sigmoid_update_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_sigmoid_update_seed1`
  - `probe_conflicting_neighbors_increase_resistance_sigmoid_gate_sum_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_sigmoid_gate_sum_seed1`
  - `probe_conflicting_neighbors_increase_resistance_sigmoid_gate_mean_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_sigmoid_gate_mean_seed1`
  - `probe_conflicting_neighbors_increase_resistance_responsibility_no_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_responsibility_no_null_seed1`
  - `probe_conflicting_neighbors_increase_resistance_responsibility_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_responsibility_null_seed1`
  - `probe_conflicting_neighbors_increase_resistance_thresholded_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_thresholded_null_seed1`
  - `probe_conflicting_neighbors_increase_resistance_precision_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_conflicting_neighbors_increase_resistance_precision_update_seed1`
  - `probe_conflicting_neighbors_increase_resistance_sigmoid_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_conflicting_neighbors_increase_resistance_sigmoid_update_seed1`
  - `probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_gate_sum_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_gate_sum_seed1`
  - `probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_gate_mean_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_gate_mean_seed1`
  - `probe_old_state_confident_new_evidence_weak_should_not_update_responsibility_no_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_confident_new_evidence_weak_should_not_update_responsibility_no_null_seed1`
  - `probe_old_state_confident_new_evidence_weak_should_not_update_responsibility_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_confident_new_evidence_weak_should_not_update_responsibility_null_seed1`
  - `probe_old_state_confident_new_evidence_weak_should_not_update_thresholded_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_confident_new_evidence_weak_should_not_update_thresholded_null_seed1`
  - `probe_old_state_confident_new_evidence_weak_should_not_update_precision_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_confident_new_evidence_weak_should_not_update_precision_update_seed1`
  - `probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_update_seed1`
  - `probe_old_state_weak_new_evidence_strong_should_update_sigmoid_gate_sum_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_weak_new_evidence_strong_should_update_sigmoid_gate_sum_seed1`
  - `probe_old_state_weak_new_evidence_strong_should_update_sigmoid_gate_mean_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_weak_new_evidence_strong_should_update_sigmoid_gate_mean_seed1`
  - `probe_old_state_weak_new_evidence_strong_should_update_responsibility_no_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_weak_new_evidence_strong_should_update_responsibility_no_null_seed1`
  - `probe_old_state_weak_new_evidence_strong_should_update_responsibility_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_weak_new_evidence_strong_should_update_responsibility_null_seed1`
  - `probe_old_state_weak_new_evidence_strong_should_update_thresholded_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_weak_new_evidence_strong_should_update_thresholded_null_seed1`
  - `probe_old_state_weak_new_evidence_strong_should_update_precision_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_weak_new_evidence_strong_should_update_precision_update_seed1`
  - `probe_old_state_weak_new_evidence_strong_should_update_sigmoid_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_weak_new_evidence_strong_should_update_sigmoid_update_seed1`
  - `probe_composition_requires_consistent_children_sigmoid_gate_sum_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_composition_requires_consistent_children_sigmoid_gate_sum_seed1`
  - `probe_composition_requires_consistent_children_sigmoid_gate_mean_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_composition_requires_consistent_children_sigmoid_gate_mean_seed1`
  - `probe_composition_requires_consistent_children_responsibility_no_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_composition_requires_consistent_children_responsibility_no_null_seed1`
  - `probe_composition_requires_consistent_children_responsibility_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_composition_requires_consistent_children_responsibility_null_seed1`
  - `probe_composition_requires_consistent_children_thresholded_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_composition_requires_consistent_children_thresholded_null_seed1`
  - `probe_composition_requires_consistent_children_precision_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_composition_requires_consistent_children_precision_update_seed1`
  - `probe_composition_requires_consistent_children_sigmoid_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_composition_requires_consistent_children_sigmoid_update_seed1`
  - `probe_attractor_should_not_collapse_sigmoid_gate_sum_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_attractor_should_not_collapse_sigmoid_gate_sum_seed1`
  - `probe_attractor_should_not_collapse_sigmoid_gate_mean_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_attractor_should_not_collapse_sigmoid_gate_mean_seed1`
  - `probe_attractor_should_not_collapse_responsibility_no_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_attractor_should_not_collapse_responsibility_no_null_seed1`
  - `probe_attractor_should_not_collapse_responsibility_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_attractor_should_not_collapse_responsibility_null_seed1`
  - `probe_attractor_should_not_collapse_thresholded_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_attractor_should_not_collapse_thresholded_null_seed1`
  - `probe_attractor_should_not_collapse_precision_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_attractor_should_not_collapse_precision_update_seed1`
  - `probe_attractor_should_not_collapse_sigmoid_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_attractor_should_not_collapse_sigmoid_update_seed1`
- image summary: `reports/image_representation_ablation/runs/summary.csv`
  - `cnn_eml_workers0_seed0`: `reports/image_representation_ablation/runs/20260424_085157_cnn_eml_workers0_seed0`
  - `pure_eml_workers0_seed0`: `reports/image_representation_ablation/runs/20260424_085157_pure_eml_workers0_seed0`
  - `efficient_baseline_seed0`: `reports/image_representation_ablation/runs/20260424_085157_efficient_baseline_seed0`
  - `efficient_centered_ambiguity_seed0`: `reports/image_representation_ablation/runs/20260424_085158_efficient_centered_ambiguity_seed0`
  - `efficient_thresholded_null_seed0`: `reports/image_representation_ablation/runs/20260424_085158_efficient_thresholded_null_seed0`
  - `efficient_precision_identity_seed0`: `reports/image_representation_ablation/runs/20260424_085158_efficient_precision_identity_seed0`
  - `efficient_combo_seed0`: `reports/image_representation_ablation/runs/20260424_085158_efficient_combo_seed0`
  - `efficient_combo_no_composition_seed0`: `reports/image_representation_ablation/runs/20260424_085158_efficient_combo_no_composition_seed0`
  - `efficient_combo_no_attractor_seed0`: `reports/image_representation_ablation/runs/20260424_085158_efficient_combo_no_attractor_seed0`
  - `efficient_combo_sensor_bypass_seed0`: `reports/image_representation_ablation/runs/20260424_085158_efficient_combo_sensor_bypass_seed0`
  - `efficient_combo_staged_seed0`: `reports/image_representation_ablation/runs/20260424_085159_efficient_combo_staged_seed0`
  - `efficient_combo_bypass_staged_seed0`: `reports/image_representation_ablation/runs/20260424_085159_efficient_combo_bypass_staged_seed0`
  - `head_without_ambiguity_seed0`: `reports/image_representation_ablation/runs/20260424_085159_head_without_ambiguity_seed0`
  - `cnn_eml_workers0_seed1`: `reports/image_representation_ablation/runs/20260424_085159_cnn_eml_workers0_seed1`
  - `pure_eml_workers0_seed1`: `reports/image_representation_ablation/runs/20260424_085159_pure_eml_workers0_seed1`
  - `efficient_baseline_seed1`: `reports/image_representation_ablation/runs/20260424_085159_efficient_baseline_seed1`
  - `efficient_centered_ambiguity_seed1`: `reports/image_representation_ablation/runs/20260424_085159_efficient_centered_ambiguity_seed1`
  - `efficient_thresholded_null_seed1`: `reports/image_representation_ablation/runs/20260424_085159_efficient_thresholded_null_seed1`
  - `efficient_precision_identity_seed1`: `reports/image_representation_ablation/runs/20260424_085200_efficient_precision_identity_seed1`
  - `efficient_combo_seed1`: `reports/image_representation_ablation/runs/20260424_085200_efficient_combo_seed1`
  - `efficient_combo_no_composition_seed1`: `reports/image_representation_ablation/runs/20260424_085200_efficient_combo_no_composition_seed1`
  - `efficient_combo_no_attractor_seed1`: `reports/image_representation_ablation/runs/20260424_085200_efficient_combo_no_attractor_seed1`
  - `efficient_combo_sensor_bypass_seed1`: `reports/image_representation_ablation/runs/20260424_085200_efficient_combo_sensor_bypass_seed1`
  - `efficient_combo_staged_seed1`: `reports/image_representation_ablation/runs/20260424_085200_efficient_combo_staged_seed1`
  - `efficient_combo_bypass_staged_seed1`: `reports/image_representation_ablation/runs/20260424_085200_efficient_combo_bypass_staged_seed1`
  - `head_without_ambiguity_seed1`: `reports/image_representation_ablation/runs/20260424_085200_head_without_ambiguity_seed1`
- text summary: `reports/text_representation_ablation/runs/summary.csv`
  - `local_causal_conv_seed0`: `reports/text_representation_ablation/runs/20260424_085213_local_causal_conv_seed0`
  - `small_gru_seed0`: `reports/text_representation_ablation/runs/20260424_085213_small_gru_seed0`
  - `old_eml_text_backbone_seed0`: `reports/text_representation_ablation/runs/20260424_085213_old_eml_text_backbone_seed0`
  - `efficient_window8_seed0`: `reports/text_representation_ablation/runs/20260424_085213_efficient_window8_seed0`
  - `efficient_window8_thresholded_null_seed0`: `reports/text_representation_ablation/runs/20260424_085213_efficient_window8_thresholded_null_seed0`
  - `efficient_window8_precision_identity_seed0`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_precision_identity_seed0`
  - `efficient_window8_chunk_seed0`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_chunk_seed0`
  - `efficient_window8_chunk_attractor_seed0`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_chunk_attractor_seed0`
  - `efficient_window8_staged_seed0`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_staged_seed0`
  - `best_text_config_seed0`: `reports/text_representation_ablation/runs/20260424_085214_best_text_config_seed0`
  - `local_causal_conv_seed1`: `reports/text_representation_ablation/runs/20260424_085214_local_causal_conv_seed1`
  - `small_gru_seed1`: `reports/text_representation_ablation/runs/20260424_085214_small_gru_seed1`
  - `old_eml_text_backbone_seed1`: `reports/text_representation_ablation/runs/20260424_085214_old_eml_text_backbone_seed1`
  - `efficient_window8_seed1`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_seed1`
  - `efficient_window8_thresholded_null_seed1`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_thresholded_null_seed1`
  - `efficient_window8_precision_identity_seed1`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_precision_identity_seed1`
  - `efficient_window8_chunk_seed1`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_chunk_seed1`
  - `efficient_window8_chunk_attractor_seed1`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_chunk_attractor_seed1`
  - `efficient_window8_staged_seed1`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_staged_seed1`
  - `best_text_config_seed1`: `reports/text_representation_ablation/runs/20260424_085214_best_text_config_seed1`
- cifar summary: `reports/cifar_medium/runs/summary.csv`
  - `cifar_medium_synthetic_gate`: `reports/cifar_medium/runs/20260424_085232_cifar_medium_synthetic_gate`
