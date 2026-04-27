# EML as Pluggable Energy / Responsibility / Uncertainty Primitive

## 1. Executive Summary

This report focuses on EML as a plug-in uncertainty/resistance primitive on existing CNN features. It does not claim EML is a clean-accuracy backbone replacement.

- `cifar10` clean accuracy: EML centered 0.4505 vs cosine 0.4891; clean head advantage is not claimed unless EML is higher.
- `cifar10` calibration: EML centered ECE 0.0799 vs cosine 0.1299.
- `cifar10` selective prediction: EML centered AURC 0.2895 vs cosine 0.3507; lower is better.
- `cifar10` supervised resistance correlations: noise 0.0540, occlusion -0.0109.
- `synthetic_shape` clean accuracy: EML centered 0.9939 vs cosine 0.9931; clean head advantage is not claimed unless EML is higher.
- `synthetic_shape` calibration: EML centered ECE 0.3461 vs cosine 0.1207.
- `synthetic_shape` selective prediction: EML centered AURC 0.0002 vs cosine 0.0002; lower is better.
- `synthetic_shape` supervised resistance correlations: noise -0.0525, occlusion 0.4251.

Latest accepted early-stop coverage:
- Frozen feature: 48/48 latest accepted completed rows early-stopped.
- End-to-end: 42/42 latest accepted completed rows early-stopped.
- Responsibility plugin: 18/18 latest accepted completed rows early-stopped.
- Agent risk toy: 12/12 latest accepted completed rows early-stopped.

Raw CSV files may contain superseded capped rows; this report uses the latest early-stopped replacement for each dataset/model/seed key unless `--include-superseded` is passed.

## 2. Current Claim Status

| Claim | Status | Evidence |
| --- | --- | --- |
| EML as backbone | Not proven | Existing reports favor CNN baselines for image representation. |
| EML as clean classification head | Not proven unless this report shows a matched win | Linear/MLP/cosine remain primary baselines. |
| EML as uncertainty/risk primitive | Open, measured here | Use calibration, selective risk, AUROC, and resistance correlations, not clean top-1 alone. |
| MERC as head/block | No-go currently | Included only as experimental comparison when present. |

## 3. Datasets

- Synthetic shape uncertainty: available through `SyntheticShapeEnergyDataset` corruptions.
- CIFAR corruption wrapper: available when local CIFAR-10 and torchvision are available.
- Text corruption: NOT RUN in this report.
- Agent risk toy: included when `scripts/run_agent_risk_toy_benchmark.py` artifacts are present.

## 4. Models

- Baselines: `linear`, `mlp`, `cosine_prototype`.
- EML heads: `eml_no_ambiguity`, `eml_centered_ambiguity`, `eml_supervised_resistance`.
- MERC heads: `merc_linear`, `merc_energy` when present; experimental/no-go until evidence changes.

## 5. Frozen Feature Results

| dataset | model | seeds | clean acc | noisy acc | occluded acc | clean ECE | clean AURC | clean->noisy AUROC | clean->occluded AUROC | resistance-noise corr | resistance-occlusion corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar10 | cosine_prototype | 3 | 0.4891 | 0.4108 | 0.4417 | 0.1299 | 0.3507 | 0.5853 | 0.5323 | MISSING | MISSING |
| cifar10 | eml_centered_ambiguity | 3 | 0.4505 | 0.3675 | 0.4056 | 0.0799 | 0.2895 | 0.5485 | 0.5411 | 0.0411 | 0.0232 |
| cifar10 | eml_no_ambiguity | 3 | 0.3963 | 0.3281 | 0.3613 | 0.0980 | 0.5583 | 0.5212 | 0.4874 | 0.0385 | -0.0363 |
| cifar10 | eml_supervised_resistance | 3 | 0.4049 | 0.3320 | 0.3828 | 0.1342 | 0.4300 | 0.5419 | 0.5176 | 0.0540 | -0.0109 |
| cifar10 | linear | 3 | 0.4831 | 0.3952 | 0.4323 | 0.1731 | 0.3565 | 0.5909 | 0.5444 | MISSING | MISSING |
| cifar10 | merc_energy | 3 | 0.3689 | 0.3444 | 0.3503 | 0.1072 | 0.5569 | 0.5375 | 0.5283 | 0.0390 | 0.0017 |
| cifar10 | merc_linear | 3 | 0.4505 | 0.3825 | 0.4082 | 0.0593 | 0.5268 | 0.5119 | 0.4942 | 0.0227 | -0.0195 |
| cifar10 | mlp | 3 | 0.4909 | 0.3965 | 0.4463 | 0.0475 | 0.3473 | 0.5260 | 0.5037 | MISSING | MISSING |
| synthetic_shape | cosine_prototype | 3 | 0.9931 | 0.9570 | 0.9004 | 0.1207 | 0.0002 | 0.6647 | 0.6252 | MISSING | MISSING |
| synthetic_shape | eml_centered_ambiguity | 3 | 0.9939 | 0.9548 | 0.9040 | 0.3461 | 0.0002 | 0.5350 | 0.5880 | -0.0724 | 0.3919 |
| synthetic_shape | eml_no_ambiguity | 3 | 0.9944 | 0.9544 | 0.9049 | 0.3600 | 0.0057 | 0.5198 | 0.5013 | 0.0377 | -0.0443 |
| synthetic_shape | eml_supervised_resistance | 3 | 0.9944 | 0.9548 | 0.9043 | 0.3460 | 0.0005 | 0.5766 | 0.5906 | -0.0525 | 0.4251 |
| synthetic_shape | linear | 3 | 0.9926 | 0.9587 | 0.9014 | 0.0742 | 0.0002 | 0.6830 | 0.6339 | MISSING | MISSING |
| synthetic_shape | merc_energy | 3 | 0.9284 | 0.9121 | 0.8509 | 0.0659 | 0.0500 | 0.5458 | 0.5754 | -0.0355 | 0.2906 |
| synthetic_shape | merc_linear | 3 | 0.9948 | 0.9648 | 0.9033 | 0.2135 | 0.0068 | 0.4946 | 0.5036 | 0.0004 | 0.0163 |
| synthetic_shape | mlp | 3 | 0.9939 | 0.9701 | 0.9089 | 0.0190 | 0.0001 | 0.5577 | 0.5967 | MISSING | MISSING |

## 6. End-to-End Results

Artifacts: `reports/pluggable_primitive_real_20260427/end_to_end_runs` (42 rows).

| dataset | model | seeds | clean acc | noisy acc | occluded acc | clean ECE | clean AURC | clean->noisy AUROC | clean->occluded AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar10_corrupt | cnn_cosine_eml_aux_resistance | 3 | 0.4899 | 0.4212 | 0.4326 | 0.0661 | 0.5213 | 0.9488 | 0.4709 |
| cifar10_corrupt | cnn_cosine_prototype | 3 | 0.5254 | 0.4551 | 0.4486 | 0.0635 | 0.3100 | 0.5216 | 0.5241 |
| cifar10_corrupt | cnn_eml_centered_ambiguity | 3 | 0.4102 | 0.3678 | 0.3672 | 0.0897 | 0.2884 | 0.4966 | 0.5166 |
| cifar10_corrupt | cnn_eml_supervised_resistance | 3 | 0.5029 | 0.4297 | 0.4535 | 0.0481 | 0.2958 | 0.8432 | 0.4934 |
| cifar10_corrupt | cnn_linear | 3 | 0.4974 | 0.4242 | 0.4398 | 0.0632 | 0.3497 | 0.5391 | 0.5197 |
| cifar10_corrupt | cnn_merc_energy | 3 | 0.3154 | 0.2855 | 0.2793 | 0.0224 | 0.5660 | 0.5583 | 0.5415 |
| cifar10_corrupt | cnn_mlp | 3 | 0.4727 | 0.4079 | 0.4157 | 0.0518 | 0.3601 | 0.5235 | 0.5306 |
| synthetic_shape_uncertainty | cnn_cosine_eml_aux_resistance | 3 | 0.9915 | 0.9766 | 0.8831 | 0.0561 | 0.0091 | 0.9928 | 0.6321 |
| synthetic_shape_uncertainty | cnn_cosine_prototype | 3 | 0.9691 | 0.9600 | 0.8460 | 0.1115 | 0.0043 | 0.5568 | 0.5809 |
| synthetic_shape_uncertainty | cnn_eml_centered_ambiguity | 3 | 0.9899 | 0.9606 | 0.8558 | 0.1924 | 0.0001 | 0.5304 | 0.5873 |
| synthetic_shape_uncertainty | cnn_eml_supervised_resistance | 3 | 0.9925 | 0.9733 | 0.8776 | 0.1031 | 0.0003 | 0.9277 | 0.6468 |
| synthetic_shape_uncertainty | cnn_linear | 3 | 0.9925 | 0.9775 | 0.8910 | 0.0214 | 0.0005 | 0.5739 | 0.6023 |
| synthetic_shape_uncertainty | cnn_merc_energy | 3 | 0.9896 | 0.9606 | 0.8643 | 0.0192 | 0.0108 | 0.5356 | 0.5830 |
| synthetic_shape_uncertainty | cnn_mlp | 3 | 0.9889 | 0.9701 | 0.8564 | 0.0178 | 0.0009 | 0.5778 | 0.6200 |

## 7. Responsibility Plugin Results

Artifacts: `reports/pluggable_primitive_real_20260427/responsibility_plugin_runs` (18 rows).

| module | seeds | accuracy | AURC | null mean | null-severity corr | update mean | update-severity corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 3 | 0.8223 | 0.0881 | MISSING | MISSING | MISSING | MISSING |
| mlp_refinement | 3 | 0.8265 | 0.0816 | MISSING | MISSING | MISSING | MISSING |
| responsibility_no_null | 3 | 0.8177 | 0.0822 | 0.0000 | MISSING | 1.0000 | MISSING |
| sigmoid_eml_gate | 3 | 0.8200 | 0.0817 | MISSING | MISSING | 0.5528 | 0.0973 |
| thresholded_null | 3 | 0.8180 | 0.0840 | 0.3760 | -0.0918 | 0.6240 | 0.0918 |
| thresholded_null_precision | 3 | 0.8242 | 0.0897 | 0.3006 | -0.0915 | 0.6994 | 0.0915 |

## 8. Agent Risk Toy Results

Artifacts: `reports/pluggable_primitive_real_20260427/agent_risk_runs` (12 rows).

| model | seeds | action acc | unsafe rate | utility | reward | risk corr | approval precision | approval recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eml_action | 3 | 0.6764 | 0.0911 | 0.6883 | 0.1520 | 0.5366 | 0.2747 | 0.6573 |
| eml_supervised_resistance | 3 | 0.6921 | 0.1035 | 0.7041 | 0.1621 | 0.8657 | 0.2669 | 0.6386 |
| linear | 3 | 0.6530 | 0.0905 | 0.6546 | 0.1366 | 0.0011 | 0.1992 | 0.4766 |
| mlp | 3 | 0.6914 | 0.1035 | 0.6958 | 0.1613 | -0.0202 | 0.2083 | 0.4984 |


## 9. EML Diagnostics

Drive/resistance/energy diagnostics are stored in each run directory's `diagnostics.csv`. Resistance correlations are summarized in the frozen-feature table when the head exposes resistance.

## 10. Failure Modes

- EML can improve calibration or selective risk while losing clean accuracy; this is not a clean-head win.
- Resistance correlations may be weak, negative, or MISSING for non-EML heads.
- Repeated run IDs can appear when capped rows are rerun. Accepted summaries prefer the latest early-stopped replacement.

## 11. Conclusions

A. Does EML beat established backbone families? No claim; not tested here.

B. Does EML beat ordinary heads on clean classification? Only if the table shows a matched clean-accuracy win; otherwise not supported.

C. Does EML help uncertainty/corruption/selective prediction? Use ECE, AURC, AUROC, and resistance correlations above; do not infer this from clean accuracy.

D. Does EML help action/risk decision? See the agent-risk table; this is only supported if EML models improve reward or unsafe-rate under matched settings.

E. Next experiment: extend early-stop discipline to any MISSING artifact family before promoting an EML primitive claim.

## Raw Artifacts

- Runs root: `reports/pluggable_primitive_real_20260427/frozen_runs`
- End-to-end root: `reports/pluggable_primitive_real_20260427/end_to_end_runs`
- Responsibility-plugin root: `reports/pluggable_primitive_real_20260427/responsibility_plugin_runs`
- Agent-risk root: `reports/pluggable_primitive_real_20260427/agent_risk_runs`
- Benchmark report generated by base runner: `reports/EML_PLUGGABLE_PRIMITIVE_REPORT_BASE.md`
