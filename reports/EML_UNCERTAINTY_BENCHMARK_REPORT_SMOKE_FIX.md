# EML Uncertainty and Selective Classification Benchmark

## Scope

- Frozen CNN features with head-only comparisons.
- Baselines: `linear`, `mlp`, `cosine_prototype`.
- EML heads: `eml_no_ambiguity`, `eml_centered_ambiguity`, `eml_supervised_resistance`.
- Clean CIFAR accuracy claims are intentionally conservative.

## Run Status

| run_id | status | model | dataset | reason |
| --- | --- | --- | --- | --- |
| uncertainty_synthetic_shape_linear_seed0 | COMPLETED | linear | synthetic_shape |  |
| uncertainty_synthetic_shape_mlp_seed0 | COMPLETED | mlp | synthetic_shape |  |
| uncertainty_synthetic_shape_cosine_prototype_seed0 | COMPLETED | cosine_prototype | synthetic_shape |  |
| uncertainty_synthetic_shape_eml_no_ambiguity_seed0 | COMPLETED | eml_no_ambiguity | synthetic_shape |  |
| uncertainty_synthetic_shape_eml_centered_ambiguity_seed0 | COMPLETED | eml_centered_ambiguity | synthetic_shape |  |
| uncertainty_synthetic_shape_eml_supervised_resistance_seed0 | COMPLETED | eml_supervised_resistance | synthetic_shape |  |

## synthetic_shape

| model | clean acc | noisy acc | occluded acc | clean ECE | clean Brier | clean selective AURC | clean->noisy AUROC | clean->occluded AUROC | pooled resistance-noise corr | pooled resistance-occlusion corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cosine_prototype | 0.3672 | 0.4062 | 0.4141 | 0.1202 | 0.7208 | 0.3751 | 0.5448 | 0.4579 | MISSING | MISSING |
| eml_centered_ambiguity | 0.3594 | 0.4219 | 0.4219 | 0.2058 | 0.7034 | 0.3459 | 0.5234 | 0.4931 | MISSING | MISSING |
| eml_no_ambiguity | 0.3594 | 0.4219 | 0.4219 | 0.1958 | 0.7039 | 0.4915 | 0.5242 | 0.4503 | MISSING | MISSING |
| eml_supervised_resistance | 0.3203 | 0.3945 | 0.3867 | 0.1688 | 0.7206 | 0.3508 | 0.5226 | 0.5326 | MISSING | MISSING |
| linear | 0.5234 | 0.3008 | 0.5000 | 0.2879 | 0.7645 | 0.3686 | 0.5508 | 0.4554 | MISSING | MISSING |
| mlp | 0.3906 | 0.4492 | 0.4375 | 0.1108 | 0.6952 | 0.4100 | 0.5402 | 0.4492 | MISSING | MISSING |

### synthetic_shape Detailed Runs

| run_id | model | seed | best step | steps run | early stop | clean acc | clean ECE | clean AURC | noisy acc | occluded acc |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| uncertainty_synthetic_shape_cosine_prototype_seed0 | cosine_prototype | 0 | 40 | 40 | False | 0.3672 | 0.1202 | 0.3751 | 0.4062 | 0.4141 |
| uncertainty_synthetic_shape_eml_centered_ambiguity_seed0 | eml_centered_ambiguity | 0 | 40 | 40 | False | 0.3594 | 0.2058 | 0.3459 | 0.4219 | 0.4219 |
| uncertainty_synthetic_shape_eml_no_ambiguity_seed0 | eml_no_ambiguity | 0 | 40 | 40 | False | 0.3594 | 0.1958 | 0.4915 | 0.4219 | 0.4219 |
| uncertainty_synthetic_shape_eml_supervised_resistance_seed0 | eml_supervised_resistance | 0 | 40 | 40 | False | 0.3203 | 0.1688 | 0.3508 | 0.3945 | 0.3867 |
| uncertainty_synthetic_shape_linear_seed0 | linear | 0 | 20 | 40 | True | 0.5234 | 0.2879 | 0.3686 | 0.3008 | 0.5000 |
| uncertainty_synthetic_shape_mlp_seed0 | mlp | 0 | 30 | 40 | False | 0.3906 | 0.1108 | 0.4100 | 0.4492 | 0.4375 |

## Conclusions

- `synthetic_shape` clean accuracy vs cosine: EML centered 0.3594 vs cosine 0.3672. Head advantage is not supported.
- `synthetic_shape` calibration vs cosine: EML centered ECE 0.2058 vs cosine 0.1202. Calibration is not better.
- `synthetic_shape` selective prediction vs cosine: EML centered clean AURC 0.3459 vs cosine 0.3751. Selective prediction is better.
- `synthetic_shape` resistance-correlation check: noise MISSING, occlusion MISSING. Corruption correlation is not supported.
- `cifar10` was NOT RUN here because `torchvision` is unavailable in the local environment.

- If the EML rows do not beat cosine on calibration or selective risk, the benchmark does not support an EML head advantage.

## Raw Artifacts

- Runs root: `reports/uncertainty_benchmark_smoke_fix/runs`
- Summary CSV: `reports/uncertainty_benchmark_smoke_fix/runs/summary.csv`
