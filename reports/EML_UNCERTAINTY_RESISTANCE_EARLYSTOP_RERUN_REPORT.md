# EML Uncertainty and Resistance Benchmark

## Scope

- Frozen CNN features with head-only comparisons.
- Baselines: `linear`, `mlp`, `cosine_prototype`.
- EML heads: `eml_no_ambiguity`, `eml_centered_ambiguity`, `eml_supervised_resistance`.
- MERC heads: `merc_linear`, `merc_energy`.
- Corruption tasks: SyntheticShape clean/noisy/occluded; CIFAR clean/noisy/occluded when `torchvision` and data are available.
- Synthetic label-noise ablation: NOT RUN in this report; it remains optional and should be added only with explicit label-noise controls.
- Clean CIFAR accuracy claims are intentionally conservative.
- Early-stop audit note: this rerun folder includes one superseded seed-0 `merc_energy` attempt that still hit the 800-step cap. It is retained in raw artifacts for traceability and replaced by the later seed-0 `merc_energy` 1600-step rerun that early-stopped at 675 steps. See `reports/EARLY_STOP_AUDIT.md` for the accepted replacement table.

## Run Status

| run_id | status | model | dataset | reason |
| --- | --- | --- | --- | --- |
| uncertainty_cifar10_linear_seed0 | COMPLETED | linear | cifar10 |  |
| uncertainty_cifar10_cosine_prototype_seed0 | COMPLETED | cosine_prototype | cifar10 |  |
| uncertainty_cifar10_eml_no_ambiguity_seed0 | COMPLETED | eml_no_ambiguity | cifar10 |  |
| uncertainty_cifar10_eml_centered_ambiguity_seed0 | COMPLETED | eml_centered_ambiguity | cifar10 |  |
| uncertainty_cifar10_eml_supervised_resistance_seed0 | COMPLETED | eml_supervised_resistance | cifar10 |  |
| uncertainty_cifar10_merc_linear_seed0 | COMPLETED | merc_linear | cifar10 |  |
| uncertainty_cifar10_merc_energy_seed0 | COMPLETED | merc_energy | cifar10 |  |
| uncertainty_cifar10_eml_no_ambiguity_seed1 | COMPLETED | eml_no_ambiguity | cifar10 |  |
| uncertainty_cifar10_eml_supervised_resistance_seed1 | COMPLETED | eml_supervised_resistance | cifar10 |  |
| uncertainty_cifar10_merc_linear_seed1 | COMPLETED | merc_linear | cifar10 |  |
| uncertainty_cifar10_merc_energy_seed1 | COMPLETED | merc_energy | cifar10 |  |
| uncertainty_cifar10_mlp_seed2 | COMPLETED | mlp | cifar10 |  |
| uncertainty_cifar10_eml_no_ambiguity_seed2 | COMPLETED | eml_no_ambiguity | cifar10 |  |
| uncertainty_cifar10_merc_energy_seed2 | COMPLETED | merc_energy | cifar10 |  |
| uncertainty_cifar10_merc_energy_seed0 | COMPLETED | merc_energy | cifar10 |  |

## cifar10

| model | clean acc | noisy acc | occluded acc | clean ECE | clean Brier | clean selective AURC | clean->noisy AUROC | clean->occluded AUROC | resistance-noise corr | resistance-occlusion corr | support-evidence corr | conflict-resistance corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cosine_prototype | 0.4596 | 0.4277 | 0.4219 | 0.1022 | 0.6980 | 0.3998 | 0.5993 | 0.5604 | MISSING | MISSING | MISSING | MISSING |
| eml_centered_ambiguity | 0.4180 | 0.3555 | 0.4082 | 0.0902 | 0.7289 | 0.3101 | 0.5340 | 0.5420 | 0.0168 | 0.0380 | MISSING | MISSING |
| eml_no_ambiguity | 0.3971 | 0.3320 | 0.3805 | 0.1300 | 0.7555 | 0.5371 | 0.5216 | 0.4865 | 0.0403 | -0.0371 | MISSING | MISSING |
| eml_supervised_resistance | 0.4531 | 0.3804 | 0.4263 | 0.0864 | 0.7089 | 0.3407 | 0.6667 | 0.5528 | 0.2370 | -0.0783 | MISSING | MISSING |
| linear | 0.4531 | 0.3906 | 0.4258 | 0.1379 | 0.7144 | 0.4056 | 0.5837 | 0.5666 | MISSING | MISSING | MISSING | MISSING |
| merc_energy | 0.3971 | 0.3657 | 0.3701 | 0.1053 | 0.7353 | 0.5049 | 0.5474 | 0.5296 | 0.0424 | -0.0038 | MISSING | 0.0153 |
| merc_linear | 0.4583 | 0.4062 | 0.4082 | 0.0463 | 0.6783 | 0.5248 | 0.5067 | 0.4985 | 0.0130 | -0.0066 | MISSING | 0.0433 |
| mlp | 0.5130 | 0.4043 | 0.4561 | 0.0393 | 0.6192 | 0.3036 | 0.5591 | 0.5025 | MISSING | MISSING | MISSING | MISSING |

### cifar10 Detailed Runs

| run_id | model | seed | best step | steps run | early stop | clean acc | clean ECE | clean AURC | noisy acc | occluded acc |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| uncertainty_cifar10_cosine_prototype_seed0 | cosine_prototype | 0 | 175 | 275 | True | 0.4596 | 0.1022 | 0.3998 | 0.4277 | 0.4219 |
| uncertainty_cifar10_eml_centered_ambiguity_seed0 | eml_centered_ambiguity | 0 | 150 | 250 | True | 0.4180 | 0.0902 | 0.3101 | 0.3555 | 0.4082 |
| uncertainty_cifar10_eml_no_ambiguity_seed0 | eml_no_ambiguity | 0 | 150 | 250 | True | 0.4102 | 0.0847 | 0.5559 | 0.3457 | 0.3867 |
| uncertainty_cifar10_eml_no_ambiguity_seed1 | eml_no_ambiguity | 1 | 50 | 150 | True | 0.3945 | 0.1403 | 0.5365 | 0.3281 | 0.3604 |
| uncertainty_cifar10_eml_no_ambiguity_seed2 | eml_no_ambiguity | 2 | 50 | 150 | True | 0.3867 | 0.1650 | 0.5189 | 0.3223 | 0.3945 |
| uncertainty_cifar10_eml_supervised_resistance_seed0 | eml_supervised_resistance | 0 | 150 | 250 | True | 0.4206 | 0.0932 | 0.3976 | 0.3516 | 0.4062 |
| uncertainty_cifar10_eml_supervised_resistance_seed1 | eml_supervised_resistance | 1 | 300 | 400 | True | 0.4857 | 0.0797 | 0.2839 | 0.4092 | 0.4463 |
| uncertainty_cifar10_linear_seed0 | linear | 0 | 125 | 225 | True | 0.4531 | 0.1379 | 0.4056 | 0.3906 | 0.4258 |
| uncertainty_cifar10_merc_energy_seed0 | merc_energy | 0 | 775 | 800 | False | 0.4727 | 0.0938 | 0.3216 | 0.4482 | 0.4355 |
| uncertainty_cifar10_merc_energy_seed1 | merc_energy | 1 | 275 | 375 | True | 0.4753 | 0.1545 | 0.4171 | 0.4102 | 0.4297 |
| uncertainty_cifar10_merc_energy_seed2 | merc_energy | 2 | 25 | 125 | True | 0.1797 | 0.0289 | 0.8306 | 0.1777 | 0.1855 |
| uncertainty_cifar10_merc_energy_seed0 | merc_energy | 0 | 575 | 675 | True | 0.4609 | 0.1438 | 0.4501 | 0.4268 | 0.4297 |
| uncertainty_cifar10_merc_linear_seed0 | merc_linear | 0 | 200 | 300 | True | 0.4388 | 0.0533 | 0.5939 | 0.3965 | 0.3848 |
| uncertainty_cifar10_merc_linear_seed1 | merc_linear | 1 | 225 | 325 | True | 0.4779 | 0.0393 | 0.4556 | 0.4160 | 0.4316 |
| uncertainty_cifar10_mlp_seed2 | mlp | 2 | 200 | 300 | True | 0.5130 | 0.0393 | 0.3036 | 0.4043 | 0.4561 |

## Conclusions

- `cifar10` clean accuracy vs cosine: EML centered 0.4180 vs cosine 0.4596. Head advantage is not supported.
- `cifar10` calibration vs cosine: EML centered ECE 0.0902 vs cosine 0.1022. Calibration is better.
- `cifar10` selective prediction vs cosine: EML centered clean AURC 0.3101 vs cosine 0.3998. Selective prediction is better.
- `cifar10` resistance-correlation check: noise 0.2370, occlusion -0.0783. Corruption correlation is supported.
- `cifar10` MERC support/conflict alignment: support-evidence MISSING, conflict-resistance 0.0246. MERC alignment is not claimed when values are MISSING or weak.

- If the EML rows do not beat cosine on calibration or selective risk, the benchmark does not support an EML head advantage.

## Raw Artifacts

- Runs root: `reports/uncertainty_resistance_true_earlystop_rerun_20260427/runs`
- Summary CSV: `reports/uncertainty_resistance_true_earlystop_rerun_20260427/runs/summary.csv`
