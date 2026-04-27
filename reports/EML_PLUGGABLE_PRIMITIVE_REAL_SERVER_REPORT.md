# EML as Pluggable Energy / Responsibility / Uncertainty Primitive

## 1. Executive Summary

This report focuses on EML as a plug-in uncertainty/resistance primitive on existing CNN features. It does not claim EML is a clean-accuracy backbone replacement.

- `cifar10` clean accuracy: EML centered 0.4271 vs cosine 0.4839; clean head advantage is not claimed unless EML is higher.
- `cifar10` calibration: EML centered ECE 0.0834 vs cosine 0.1358.
- `cifar10` selective prediction: EML centered AURC 0.3391 vs cosine 0.3546; lower is better.
- `cifar10` supervised resistance correlations: noise 0.1343, occlusion -0.0454.
- `synthetic_shape` clean accuracy: EML centered 0.9952 vs cosine 0.9909; clean head advantage is not claimed unless EML is higher.
- `synthetic_shape` calibration: EML centered ECE 0.2830 vs cosine 0.1901.
- `synthetic_shape` selective prediction: EML centered AURC 0.0000 vs cosine 0.0002; lower is better.
- `synthetic_shape` supervised resistance correlations: noise 0.1057, occlusion 0.4446.

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
- Agent risk toy: NOT RUN unless `scripts/run_agent_risk_toy_benchmark.py` artifacts are supplied.

## 4. Models

- Baselines: `linear`, `mlp`, `cosine_prototype`.
- EML heads: `eml_no_ambiguity`, `eml_centered_ambiguity`, `eml_supervised_resistance`.
- MERC heads: `merc_linear`, `merc_energy` when present; experimental/no-go until evidence changes.

## 5. Frozen Feature Results

| dataset | model | seeds | clean acc | noisy acc | occluded acc | clean ECE | clean AURC | clean->noisy AUROC | clean->occluded AUROC | resistance-noise corr | resistance-occlusion corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar10 | cosine_prototype | 3 | 0.4839 | 0.4121 | 0.4401 | 0.1358 | 0.3546 | 0.5870 | 0.5350 | MISSING | MISSING |
| cifar10 | eml_centered_ambiguity | 3 | 0.4271 | 0.3558 | 0.3900 | 0.0834 | 0.3391 | 0.5286 | 0.5248 | 0.0220 | 0.0153 |
| cifar10 | eml_no_ambiguity | 3 | 0.3958 | 0.3307 | 0.3636 | 0.0968 | 0.5583 | 0.5185 | 0.4865 | 0.0338 | -0.0347 |
| cifar10 | eml_supervised_resistance | 3 | 0.4280 | 0.3561 | 0.3906 | 0.0852 | 0.3775 | 0.5928 | 0.5263 | 0.1343 | -0.0454 |
| cifar10 | linear | 3 | 0.4826 | 0.3952 | 0.4378 | 0.1723 | 0.3556 | 0.5899 | 0.5433 | MISSING | MISSING |
| cifar10 | merc_energy | 3 | 0.4852 | 0.4115 | 0.4404 | 0.0962 | 0.3757 | 0.5574 | 0.5243 | 0.0312 | -0.0021 |
| cifar10 | merc_linear | 3 | 0.4575 | 0.3910 | 0.4255 | 0.0600 | 0.5049 | 0.5179 | 0.4971 | 0.0286 | -0.0212 |
| cifar10 | mlp | 3 | 0.4913 | 0.3958 | 0.4440 | 0.0489 | 0.3458 | 0.5251 | 0.5027 | MISSING | MISSING |
| synthetic_shape | cosine_prototype | 3 | 0.9909 | 0.9329 | 0.8867 | 0.1901 | 0.0002 | 0.7387 | 0.6369 | MISSING | MISSING |
| synthetic_shape | eml_centered_ambiguity | 3 | 0.9952 | 0.9521 | 0.9030 | 0.2830 | 0.0000 | 0.6384 | 0.6304 | 0.0701 | 0.4109 |
| synthetic_shape | eml_no_ambiguity | 3 | 0.9952 | 0.9525 | 0.9030 | 0.2954 | 0.0016 | 0.5727 | 0.4936 | 0.0882 | -0.0549 |
| synthetic_shape | eml_supervised_resistance | 3 | 0.9952 | 0.9518 | 0.9030 | 0.2832 | 0.0000 | 0.6858 | 0.6339 | 0.1057 | 0.4446 |
| synthetic_shape | linear | 3 | 0.9896 | 0.9186 | 0.8831 | 0.2637 | 0.0004 | 0.7366 | 0.6297 | MISSING | MISSING |
| synthetic_shape | merc_energy | 3 | 0.9944 | 0.9538 | 0.8991 | 0.1313 | 0.0028 | 0.5401 | 0.5877 | -0.0433 | 0.3289 |
| synthetic_shape | merc_linear | 3 | 0.9952 | 0.9691 | 0.8971 | 0.2120 | 0.0075 | 0.4799 | 0.5053 | -0.0143 | 0.0135 |
| synthetic_shape | mlp | 3 | 0.9948 | 0.9557 | 0.9014 | 0.0269 | 0.0005 | 0.5864 | 0.6110 | MISSING | MISSING |

## 6. End-to-End Results

NOT RUN in this report. The current run isolates heads on frozen CNN features.

## 7. Responsibility Plugin Results

NOT RUN in this report unless separate responsibility-plugin artifacts are added.

## 8. Agent Risk Toy Results

NOT RUN in this report unless separate agent-risk artifacts are added.

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

D. Does EML help action/risk decision? NOT RUN in this report.

E. Next experiment: run the same early-stop discipline for end-to-end uncertainty and responsibility-plugin artifacts before promoting any claim.

## Raw Artifacts

- Runs root: `reports/pluggable_uncertainty_real_20260427/runs`
- Benchmark report generated by base runner: `reports/EML_PLUGGABLE_PRIMITIVE_REAL_SERVER_REPORT_BASE.md`
