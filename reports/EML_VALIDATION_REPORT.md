# EML Validation and Ablation Report

Generated: 2026-04-24T08:54:14Z

## 1. Executive Summary

- Best image result: cnn_eml (0.25)
- Best text result: EfficientEMLTextEncoder (0.11274509876966476)
- Strongest baseline: cnn_eml (0.25)
- Responsibility evidence weighting: see mechanism probe and downstream ablation tables.
- Precision update: see update probe rows and text/image ablation cells; model-quality conclusions need longer runs.
- Attractor memory: no-attractor comparison is MISSING unless a completed row exists below.
- Major failure modes: 0 failed runs and 5 not-run cells are recorded in the status table.
- Recommended next step: standardize the remaining NOT RUN switches, then repeat the best image/text runs across seeds.

## 2. Repository and Environment

- git_commit: ae26ee9
- hostname: BMW-Pro.local
- python_version: 3.12.12
- torch_version: 2.10.0
- torchvision_version: unavailable
- cuda_available: False
- device: cpu
- timestamp: 2026-04-24T06:36:08Z

## 3. Experimental Scope

| run_id | status | task | model | dataset | reason |
| --- | --- | --- | --- | --- | --- |
| smoke_image_cnn_eml_baseline | COMPLETED | image_synthetic | cnn_eml | SyntheticShapeEnergyDataset |  |
| smoke_image_efficient_eml | COMPLETED | image_synthetic | EfficientEMLImageClassifier | SyntheticShapeEnergyDataset |  |
| smoke_text_efficient_eml | COMPLETED | text_synthetic | EfficientEMLTextEncoder | SyntheticTextEnergyDataset |  |
| probe_gate_compat_sigmoid_update | COMPLETED | mechanism_probe | probe_gate_compat_sigmoid_update | synthetic_probe_tensors |  |
| probe_responsibility_no_null_precision | COMPLETED | mechanism_probe | probe_responsibility_no_null_precision | synthetic_probe_tensors |  |
| probe_responsibility_with_null_precision | COMPLETED | mechanism_probe | probe_responsibility_with_null_precision | synthetic_probe_tensors |  |
| local_conv_baseline | NOT RUN | image_synthetic | LocalConvBaseline | SyntheticShapeEnergyDataset | smoke mode: not implemented |
| local_text_linear_baseline | NOT RUN | text_synthetic | LocalTextCodecLinear | SyntheticTextEnergyDataset | smoke mode: not standardized |
| cifar_medium_suite | NOT RUN | image_cifar | selected_image_models | CIFAR10 | smoke mode: not requested in this mode |
| text_medium_suite | NOT RUN | text_synthetic | selected_text_models | SyntheticTextEnergyDataset | smoke mode: not requested in this mode |
| full_seeded_ablation | NOT RUN | mechanism_ablation | all_supported_cells | mixed | smoke mode: not requested in this mode |

Failed runs: 0
Not-run entries: 5

## 4. Datasets

| dataset | synthetic/real | notes |
| --- | --- | --- |
| CIFAR10 | real/optional | requires local data/dependency |
| SyntheticShapeEnergyDataset | synthetic | offline |
| SyntheticTextEnergyDataset | synthetic | offline |
| mixed | real/optional | requires local data/dependency |
| synthetic_probe_tensors | synthetic | offline |

## 5. Models Compared

| model | parameter count | task names | key mechanisms |
| --- | ---: | --- | --- |
| EfficientEMLImageClassifier | 117076 | image_synthetic | see config artifacts |
| EfficientEMLTextEncoder | 92950 | text_synthetic | see config artifacts |
| LocalConvBaseline | 0 | image_synthetic | see config artifacts |
| LocalTextCodecLinear | 0 | text_synthetic | see config artifacts |
| all_supported_cells | 0 | mechanism_ablation | see config artifacts |
| cnn_eml | 162644 | image_synthetic | see config artifacts |
| probe_gate_compat_sigmoid_update | 14151 | mechanism_probe | see config artifacts |
| probe_responsibility_no_null_precision | 14151 | mechanism_probe | see config artifacts |
| probe_responsibility_with_null_precision | 14151 | mechanism_probe | see config artifacts |
| selected_image_models | 0 | image_cifar | see config artifacts |
| selected_text_models | 0 | text_synthetic | see config artifacts |

## 6. Main Results

### Image
| run_id | model | dataset | final metric | best metric | loss | accuracy | time sec | params |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_image_cnn_eml_baseline | cnn_eml | SyntheticShapeEnergyDataset | 0.0 | 0.25 | 1.6346 | 0.0000 | 0.05253291130065918 | 162644 |
| smoke_image_efficient_eml | EfficientEMLImageClassifier | SyntheticShapeEnergyDataset | 0.0 | 0.25 | 1.7484 | 0.0000 | 0.07413816452026367 | 117076 |

### Text
| run_id | model | dataset | final metric | best metric | loss | accuracy | time sec | params |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_text_efficient_eml | EfficientEMLTextEncoder | SyntheticTextEnergyDataset | 0.11274509876966476 | 0.11274509876966476 | 4.3839 | 0.1127 | 0.047122955322265625 | 92950 |

### Efficiency
| run_id | model | task | examples/sec | tokens/sec | step time | peak memory MB | params |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| smoke_image_cnn_eml_baseline | cnn_eml | image_synthetic | 984.4917407505208 |  | 0.008126020431518555 | 0.0 | 162644 |
| smoke_image_efficient_eml | EfficientEMLImageClassifier | image_synthetic | 491.2225800784681 |  | 0.01628589630126953 | 0.0 | 117076 |
| smoke_text_efficient_eml | EfficientEMLTextEncoder | text_synthetic |  | 23118.478722541946 | 0.00882411003112793 | 0.0 | 92950 |
| probe_gate_compat_sigmoid_update | probe_gate_compat_sigmoid_update | mechanism_probe |  |  | 0.0009789466857910156 | 0.0 | 14151 |
| probe_responsibility_no_null_precision | probe_responsibility_no_null_precision | mechanism_probe |  |  | 0.000885009765625 | 0.0 | 14151 |
| probe_responsibility_with_null_precision | probe_responsibility_with_null_precision | mechanism_probe |  |  | 0.0009799003601074219 | 0.0 | 14151 |

### Stability
NaN/Inf counts are recorded when runners emit `nan_inf_count`; otherwise MISSING.

## 7. Ablation Results

### Responsibility / Null / Update Probes
| run_id | status | model | best | final | delta vs ref | loss | time sec | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| probe_gate_compat_sigmoid_update | COMPLETED | probe_gate_compat_sigmoid_update | 1.0 | 1.0 | 0.0000 | 0.0000 | 0.0009777545928955078 |  |
| probe_responsibility_no_null_precision | COMPLETED | probe_responsibility_no_null_precision | 1.0 | 1.0 | 0.0000 | 0.0000 | 0.000885009765625 |  |
| probe_responsibility_with_null_precision | COMPLETED | probe_responsibility_with_null_precision | 1.0 | 1.0 | 0.0000 | 0.0000 | 0.0009799003601074219 |  |

Interpretation: these probes validate finite propagation and diagnostic behavior. They do not by themselves prove downstream task quality.

### Image Representation / Attractor / Warmup / Window
| run_id | status | model | best | final | delta vs ref | loss | time sec | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| smoke_image_cnn_eml_baseline | COMPLETED | cnn_eml | 0.25 | 0.0 | 0.0000 | 1.6346 | 0.05253291130065918 |  |
| smoke_image_efficient_eml | COMPLETED | EfficientEMLImageClassifier | 0.25 | 0.0 | 0.0000 | 1.7484 | 0.07413816452026367 |  |
| local_conv_baseline | NOT RUN | LocalConvBaseline |  |  |  |  |  | smoke mode: not implemented |

### Text Local Window
| run_id | status | model | best | final | delta vs ref | loss | time sec | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| smoke_text_efficient_eml | COMPLETED | EfficientEMLTextEncoder | 0.11274509876966476 | 0.11274509876966476 | 0.0000 | 4.3839 | 0.047122955322265625 |  |
| local_text_linear_baseline | NOT RUN | LocalTextCodecLinear |  |  |  |  |  | smoke mode: not standardized |
| text_medium_suite | NOT RUN | selected_text_models |  |  |  |  |  | smoke mode: not requested in this mode |

### CIFAR Medium
| run_id | status | model | best | final | delta vs ref | loss | time sec | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| cifar_medium_suite | NOT RUN | selected_image_models |  |  |  |  |  | smoke mode: not requested in this mode |

### Failed And Not Run Cells
| run_id | status | task | model | dataset | reason |
| --- | --- | --- | --- | --- | --- |
| local_conv_baseline | NOT RUN | image_synthetic | LocalConvBaseline | SyntheticShapeEnergyDataset | smoke mode: not implemented |
| local_text_linear_baseline | NOT RUN | text_synthetic | LocalTextCodecLinear | SyntheticTextEnergyDataset | smoke mode: not standardized |
| cifar_medium_suite | NOT RUN | image_cifar | selected_image_models | CIFAR10 | smoke mode: not requested in this mode |
| text_medium_suite | NOT RUN | text_synthetic | selected_text_models | SyntheticTextEnergyDataset | smoke mode: not requested in this mode |
| full_seeded_ablation | NOT RUN | mechanism_ablation | all_supported_cells | mixed | smoke mode: not requested in this mode |

### All Ablation Cells
MISSING

Other ablation axes remain `NOT RUN` when listed in the status table.

## 8. EML Diagnostics

| run_id | model | drive_mean | drive_std | resistance_mean | resistance_std | energy_mean | energy_std | null_weight_mean | responsibility_entropy_mean | update_strength_mean | update_gate_mean | attractor_diversity | ambiguity_mean | ambiguity_weight_mean | sample_uncertainty_mean | resistance_noise_corr | resistance_occlusion_corr | corruption_resistance_corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_image_cnn_eml_baseline | cnn_eml | 0.0009 | 0.3032 | 0.3478 | 0.8859 | -0.8709 | 0.3305 |  |  |  |  |  |  |  | 0.6924 |  |  |  |
| smoke_image_efficient_eml | EfficientEMLImageClassifier | 0.0090 | 0.0766 | 1.3658 | 0.1513 | -0.2375 | 0.0300 | 0.1598 |  | 0.8545 | 0.3961 | 0.8370 | 2.4728 |  | 0.6928 |  |  |  |
| smoke_text_efficient_eml | EfficientEMLTextEncoder | 0.0086 | 0.3501 | 1.5926 | 0.2265 | -0.0834 | 0.1056 | 0.5353 |  | 0.4598 | 0.2142 | 0.2319 |  |  |  |  |  |  |
| probe_gate_compat_sigmoid_update | probe_gate_compat_sigmoid_update | -0.0002 | 0.0024 | 0.0000 | 0.0018 | -0.8810 | 0.2129 |  | 1.4710 | 1.0000 | 0.2689 |  |  |  |  |  |  |  |
| probe_responsibility_no_null_precision | probe_responsibility_no_null_precision | 0.0002 | 0.0019 | -0.0002 | 0.0022 | -0.8809 | 0.2130 |  | 1.3863 | 1.0000 | 0.3113 |  |  |  |  |  |  |  |
| probe_responsibility_with_null_precision | probe_responsibility_with_null_precision | -0.0001 | 0.0020 | -0.0002 | 0.0022 | -0.8810 | 0.2129 | 0.2919 | 1.5855 | 0.7081 | 0.3113 |  |  |  |  |  |  |  |

Resistance-noise, resistance-occlusion, and resistance-corruption correlations are included when emitted by a run; otherwise MISSING.

## 9. Training Curves

### smoke_image_cnn_eml_baseline

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.5923607349395752 | 0.25 |  | 0.020364999771118164 |
| 2 | 1.6290571689605713 | 0.25 |  | 0.030501842498779297 |
| 3 | 1.5851128101348877 | 0.25 |  | 0.042111873626708984 |
| 4 | 1.6345659494400024 | 0.0 |  | 0.05194997787475586 |

### smoke_image_efficient_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.5977815389633179 | 0.125 |  | 0.019364118576049805 |
| 2 | 1.6858720779418945 | 0.0 |  | 0.037405967712402344 |
| 3 | 1.5530210733413696 | 0.25 |  | 0.05495595932006836 |
| 4 | 1.7483947277069092 | 0.0 |  | 0.0734257698059082 |

### smoke_text_efficient_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.431398868560791 |  | 0.016806723549962044 | 0.012645959854125977 |
| 2 | 4.407316207885742 |  | 0.019801979884505272 | 0.024617910385131836 |
| 3 | 4.394105911254883 |  | 0.07339449226856232 | 0.03557896614074707 |
| 4 | 4.383850574493408 |  | 0.11274509876966476 | 0.046571969985961914 |

### probe_gate_compat_sigmoid_update

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0009777545928955078 |

### probe_responsibility_no_null_precision

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.000885009765625 |

### probe_responsibility_with_null_precision

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0009799003601074219 |


## 10. Efficiency Analysis

- Runtime and throughput are available in per-run summaries and the efficiency table.
- Local-window cost and attractor count are recorded when model diagnostics expose them.
- Short smoke runs are not enough to decide whether accuracy gain justifies added cost.

## 11. Failure Modes

- gate collapse: MISSING unless gate diagnostics are emitted.
- all-null collapse: inspect `null_weight_mean`; high values indicate risk.
- never-null collapse: inspect `null_weight_mean`; near zero indicates risk.
- energy explosion: inspect `energy_mean/std` and NaN/Inf counts.
- resistance collapse: inspect `resistance_mean/std`.
- attractor collapse: inspect `attractor_diversity`.
- update gate too high at init: inspect `update_gate_mean`.
- poor causal text behavior: no-leak tests exist; training report includes only available run metrics.
- slow local-window implementation: compare seconds and throughput.

## 12. Conclusions

- Current evidence remains preliminary when runs are short or single-seed.
- Use the synthetic image and text ablation tables to identify mechanisms worth longer training before making CIFAR claims.
- The exact next experiment is repeat-seed CIFAR validation for the best efficient image setting and the strongest CNN baseline.

## 13. Raw Artifacts

- smoke_image_cnn_eml_baseline: reports/runs/20260424_063607_smoke_image_cnn_eml_baseline
  - history: reports/runs/20260424_063607_smoke_image_cnn_eml_baseline/history.json
  - metrics: reports/runs/20260424_063607_smoke_image_cnn_eml_baseline/metrics.csv
  - diagnostics: reports/runs/20260424_063607_smoke_image_cnn_eml_baseline/diagnostics.csv
  - summary: reports/runs/20260424_063607_smoke_image_cnn_eml_baseline/summary.json
- smoke_image_efficient_eml: reports/runs/20260424_063608_smoke_image_efficient_eml
  - history: reports/runs/20260424_063608_smoke_image_efficient_eml/history.json
  - metrics: reports/runs/20260424_063608_smoke_image_efficient_eml/metrics.csv
  - diagnostics: reports/runs/20260424_063608_smoke_image_efficient_eml/diagnostics.csv
  - summary: reports/runs/20260424_063608_smoke_image_efficient_eml/summary.json
- smoke_text_efficient_eml: reports/runs/20260424_063608_smoke_text_efficient_eml
  - history: reports/runs/20260424_063608_smoke_text_efficient_eml/history.json
  - metrics: reports/runs/20260424_063608_smoke_text_efficient_eml/metrics.csv
  - diagnostics: reports/runs/20260424_063608_smoke_text_efficient_eml/diagnostics.csv
  - summary: reports/runs/20260424_063608_smoke_text_efficient_eml/summary.json
- probe_gate_compat_sigmoid_update: reports/runs/20260424_063608_probe_gate_compat_sigmoid_update
  - history: reports/runs/20260424_063608_probe_gate_compat_sigmoid_update/history.json
  - metrics: reports/runs/20260424_063608_probe_gate_compat_sigmoid_update/metrics.csv
  - diagnostics: reports/runs/20260424_063608_probe_gate_compat_sigmoid_update/diagnostics.csv
  - summary: reports/runs/20260424_063608_probe_gate_compat_sigmoid_update/summary.json
- probe_responsibility_no_null_precision: reports/runs/20260424_063608_probe_responsibility_no_null_precision
  - history: reports/runs/20260424_063608_probe_responsibility_no_null_precision/history.json
  - metrics: reports/runs/20260424_063608_probe_responsibility_no_null_precision/metrics.csv
  - diagnostics: reports/runs/20260424_063608_probe_responsibility_no_null_precision/diagnostics.csv
  - summary: reports/runs/20260424_063608_probe_responsibility_no_null_precision/summary.json
- probe_responsibility_with_null_precision: reports/runs/20260424_063608_probe_responsibility_with_null_precision
  - history: reports/runs/20260424_063608_probe_responsibility_with_null_precision/history.json
  - metrics: reports/runs/20260424_063608_probe_responsibility_with_null_precision/metrics.csv
  - diagnostics: reports/runs/20260424_063608_probe_responsibility_with_null_precision/diagnostics.csv
  - summary: reports/runs/20260424_063608_probe_responsibility_with_null_precision/summary.json
- local_conv_baseline: reports/runs/20260424_063608_local_conv_baseline
  - history: reports/runs/20260424_063608_local_conv_baseline/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260424_063608_local_conv_baseline/summary.json
- local_text_linear_baseline: reports/runs/20260424_063608_local_text_linear_baseline
  - history: reports/runs/20260424_063608_local_text_linear_baseline/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260424_063608_local_text_linear_baseline/summary.json
- cifar_medium_suite: reports/runs/20260424_063608_cifar_medium_suite
  - history: reports/runs/20260424_063608_cifar_medium_suite/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260424_063608_cifar_medium_suite/summary.json
- text_medium_suite: reports/runs/20260424_063608_text_medium_suite
  - history: reports/runs/20260424_063608_text_medium_suite/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260424_063608_text_medium_suite/summary.json
- full_seeded_ablation: reports/runs/20260424_063608_full_seeded_ablation
  - history: reports/runs/20260424_063608_full_seeded_ablation/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260424_063608_full_seeded_ablation/summary.json

## 14. Appendix: Commands

```bash
pytest
python scripts/run_eml_validation_suite.py --mode smoke --device cpu
python scripts/generate_eml_report.py
python scripts/run_eml_validation_suite.py --mode ablation --device cuda
python scripts/run_eml_validation_suite.py --mode cifar-medium --device cuda
python scripts/run_eml_validation_suite.py --mode text-medium --device cuda
```
