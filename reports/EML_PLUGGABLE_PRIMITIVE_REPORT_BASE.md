# EML Uncertainty and Resistance Benchmark

## Scope

- Frozen CNN features with head-only comparisons.
- Baselines: `linear`, `mlp`, `cosine_prototype`.
- EML heads: `eml_no_ambiguity`, `eml_centered_ambiguity`, `eml_supervised_resistance`.
- MERC heads: `merc_linear`, `merc_energy`.
- Corruption tasks: SyntheticShape clean/noisy/occluded; CIFAR clean/noisy/occluded when `torchvision` and data are available.
- Synthetic label-noise ablation: NOT RUN in this report; it remains optional and should be added only with explicit label-noise controls.
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
| uncertainty_synthetic_shape_merc_linear_seed0 | COMPLETED | merc_linear | synthetic_shape |  |
| uncertainty_synthetic_shape_merc_energy_seed0 | COMPLETED | merc_energy | synthetic_shape |  |
| uncertainty_synthetic_shape_linear_seed1 | COMPLETED | linear | synthetic_shape |  |
| uncertainty_synthetic_shape_mlp_seed1 | COMPLETED | mlp | synthetic_shape |  |
| uncertainty_synthetic_shape_cosine_prototype_seed1 | COMPLETED | cosine_prototype | synthetic_shape |  |
| uncertainty_synthetic_shape_eml_no_ambiguity_seed1 | COMPLETED | eml_no_ambiguity | synthetic_shape |  |
| uncertainty_synthetic_shape_eml_centered_ambiguity_seed1 | COMPLETED | eml_centered_ambiguity | synthetic_shape |  |
| uncertainty_synthetic_shape_eml_supervised_resistance_seed1 | COMPLETED | eml_supervised_resistance | synthetic_shape |  |
| uncertainty_synthetic_shape_merc_linear_seed1 | COMPLETED | merc_linear | synthetic_shape |  |
| uncertainty_synthetic_shape_merc_energy_seed1 | COMPLETED | merc_energy | synthetic_shape |  |
| uncertainty_synthetic_shape_linear_seed2 | COMPLETED | linear | synthetic_shape |  |
| uncertainty_synthetic_shape_mlp_seed2 | COMPLETED | mlp | synthetic_shape |  |
| uncertainty_synthetic_shape_cosine_prototype_seed2 | COMPLETED | cosine_prototype | synthetic_shape |  |
| uncertainty_synthetic_shape_eml_no_ambiguity_seed2 | COMPLETED | eml_no_ambiguity | synthetic_shape |  |
| uncertainty_synthetic_shape_eml_centered_ambiguity_seed2 | COMPLETED | eml_centered_ambiguity | synthetic_shape |  |
| uncertainty_synthetic_shape_eml_supervised_resistance_seed2 | COMPLETED | eml_supervised_resistance | synthetic_shape |  |
| uncertainty_synthetic_shape_merc_linear_seed2 | COMPLETED | merc_linear | synthetic_shape |  |
| uncertainty_synthetic_shape_merc_energy_seed2 | COMPLETED | merc_energy | synthetic_shape |  |
| uncertainty_cifar10_linear_seed0 | COMPLETED | linear | cifar10 |  |
| uncertainty_cifar10_mlp_seed0 | COMPLETED | mlp | cifar10 |  |
| uncertainty_cifar10_cosine_prototype_seed0 | COMPLETED | cosine_prototype | cifar10 |  |
| uncertainty_cifar10_eml_no_ambiguity_seed0 | COMPLETED | eml_no_ambiguity | cifar10 |  |
| uncertainty_cifar10_eml_centered_ambiguity_seed0 | COMPLETED | eml_centered_ambiguity | cifar10 |  |
| uncertainty_cifar10_eml_supervised_resistance_seed0 | COMPLETED | eml_supervised_resistance | cifar10 |  |
| uncertainty_cifar10_merc_linear_seed0 | COMPLETED | merc_linear | cifar10 |  |
| uncertainty_cifar10_merc_energy_seed0 | COMPLETED | merc_energy | cifar10 |  |
| uncertainty_cifar10_linear_seed1 | COMPLETED | linear | cifar10 |  |
| uncertainty_cifar10_mlp_seed1 | COMPLETED | mlp | cifar10 |  |
| uncertainty_cifar10_cosine_prototype_seed1 | COMPLETED | cosine_prototype | cifar10 |  |
| uncertainty_cifar10_eml_no_ambiguity_seed1 | COMPLETED | eml_no_ambiguity | cifar10 |  |
| uncertainty_cifar10_eml_centered_ambiguity_seed1 | COMPLETED | eml_centered_ambiguity | cifar10 |  |
| uncertainty_cifar10_eml_supervised_resistance_seed1 | COMPLETED | eml_supervised_resistance | cifar10 |  |
| uncertainty_cifar10_merc_linear_seed1 | COMPLETED | merc_linear | cifar10 |  |
| uncertainty_cifar10_merc_energy_seed1 | COMPLETED | merc_energy | cifar10 |  |
| uncertainty_cifar10_linear_seed2 | COMPLETED | linear | cifar10 |  |
| uncertainty_cifar10_mlp_seed2 | COMPLETED | mlp | cifar10 |  |
| uncertainty_cifar10_cosine_prototype_seed2 | COMPLETED | cosine_prototype | cifar10 |  |
| uncertainty_cifar10_eml_no_ambiguity_seed2 | COMPLETED | eml_no_ambiguity | cifar10 |  |
| uncertainty_cifar10_eml_centered_ambiguity_seed2 | COMPLETED | eml_centered_ambiguity | cifar10 |  |
| uncertainty_cifar10_eml_supervised_resistance_seed2 | COMPLETED | eml_supervised_resistance | cifar10 |  |
| uncertainty_cifar10_merc_linear_seed2 | COMPLETED | merc_linear | cifar10 |  |
| uncertainty_cifar10_merc_energy_seed2 | COMPLETED | merc_energy | cifar10 |  |

## cifar10

| model | clean acc | noisy acc | occluded acc | clean ECE | clean Brier | clean selective AURC | clean->noisy AUROC | clean->occluded AUROC | resistance-noise corr | resistance-occlusion corr | support-evidence corr | conflict-resistance corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cosine_prototype | 0.4891 | 0.4108 | 0.4417 | 0.1299 | 0.6751 | 0.3507 | 0.5853 | 0.5323 | MISSING | MISSING | MISSING | MISSING |
| eml_centered_ambiguity | 0.4505 | 0.3675 | 0.4056 | 0.0799 | 0.7048 | 0.2895 | 0.5485 | 0.5411 | 0.0411 | 0.0232 | MISSING | MISSING |
| eml_no_ambiguity | 0.3963 | 0.3281 | 0.3613 | 0.0980 | 0.7361 | 0.5583 | 0.5212 | 0.4874 | 0.0385 | -0.0363 | MISSING | MISSING |
| eml_supervised_resistance | 0.4049 | 0.3320 | 0.3828 | 0.1342 | 0.7542 | 0.4300 | 0.5419 | 0.5176 | 0.0540 | -0.0109 | MISSING | MISSING |
| linear | 0.4831 | 0.3952 | 0.4323 | 0.1731 | 0.7055 | 0.3565 | 0.5909 | 0.5444 | MISSING | MISSING | MISSING | MISSING |
| merc_energy | 0.3689 | 0.3444 | 0.3503 | 0.1072 | 0.7516 | 0.5569 | 0.5375 | 0.5283 | 0.0390 | 0.0017 | MISSING | -0.0013 |
| merc_linear | 0.4505 | 0.3825 | 0.4082 | 0.0593 | 0.6857 | 0.5268 | 0.5119 | 0.4942 | 0.0227 | -0.0195 | MISSING | -0.0121 |
| mlp | 0.4909 | 0.3965 | 0.4463 | 0.0475 | 0.6553 | 0.3473 | 0.5260 | 0.5037 | MISSING | MISSING | MISSING | MISSING |

### cifar10 Detailed Runs

| run_id | model | seed | best step | steps run | early stop | clean acc | clean ECE | clean AURC | noisy acc | occluded acc |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| uncertainty_cifar10_cosine_prototype_seed0 | cosine_prototype | 0 | 175 | 275 | True | 0.4648 | 0.1061 | 0.3983 | 0.4248 | 0.4326 |
| uncertainty_cifar10_cosine_prototype_seed1 | cosine_prototype | 1 | 100 | 200 | True | 0.4948 | 0.1429 | 0.3494 | 0.4121 | 0.4492 |
| uncertainty_cifar10_cosine_prototype_seed2 | cosine_prototype | 2 | 150 | 250 | True | 0.5078 | 0.1409 | 0.3045 | 0.3955 | 0.4434 |
| uncertainty_cifar10_eml_centered_ambiguity_seed0 | eml_centered_ambiguity | 0 | 150 | 250 | True | 0.4154 | 0.0824 | 0.3318 | 0.3408 | 0.3867 |
| uncertainty_cifar10_eml_centered_ambiguity_seed1 | eml_centered_ambiguity | 1 | 300 | 400 | True | 0.4935 | 0.0800 | 0.2386 | 0.4111 | 0.4414 |
| uncertainty_cifar10_eml_centered_ambiguity_seed2 | eml_centered_ambiguity | 2 | 225 | 325 | True | 0.4427 | 0.0774 | 0.2981 | 0.3506 | 0.3887 |
| uncertainty_cifar10_eml_no_ambiguity_seed0 | eml_no_ambiguity | 0 | 150 | 250 | True | 0.3984 | 0.0676 | 0.5629 | 0.3408 | 0.3682 |
| uncertainty_cifar10_eml_no_ambiguity_seed1 | eml_no_ambiguity | 1 | 50 | 150 | True | 0.3997 | 0.1448 | 0.5352 | 0.3252 | 0.3672 |
| uncertainty_cifar10_eml_no_ambiguity_seed2 | eml_no_ambiguity | 2 | 100 | 200 | True | 0.3906 | 0.0816 | 0.5769 | 0.3184 | 0.3486 |
| uncertainty_cifar10_eml_supervised_resistance_seed0 | eml_supervised_resistance | 0 | 150 | 250 | True | 0.4154 | 0.0827 | 0.4065 | 0.3418 | 0.3877 |
| uncertainty_cifar10_eml_supervised_resistance_seed1 | eml_supervised_resistance | 1 | 50 | 150 | True | 0.4036 | 0.1479 | 0.4137 | 0.3301 | 0.3682 |
| uncertainty_cifar10_eml_supervised_resistance_seed2 | eml_supervised_resistance | 2 | 50 | 150 | True | 0.3958 | 0.1720 | 0.4698 | 0.3242 | 0.3926 |
| uncertainty_cifar10_linear_seed0 | linear | 0 | 175 | 275 | True | 0.4544 | 0.0913 | 0.4048 | 0.4229 | 0.4326 |
| uncertainty_cifar10_linear_seed1 | linear | 1 | 100 | 200 | True | 0.4935 | 0.1845 | 0.3526 | 0.3896 | 0.4434 |
| uncertainty_cifar10_linear_seed2 | linear | 2 | 75 | 175 | True | 0.5013 | 0.2436 | 0.3120 | 0.3730 | 0.4209 |
| uncertainty_cifar10_merc_energy_seed0 | merc_energy | 0 | 600 | 700 | True | 0.4531 | 0.1229 | 0.4189 | 0.4463 | 0.4307 |
| uncertainty_cifar10_merc_energy_seed1 | merc_energy | 1 | 275 | 375 | True | 0.4740 | 0.1646 | 0.4194 | 0.4092 | 0.4346 |
| uncertainty_cifar10_merc_energy_seed2 | merc_energy | 2 | 25 | 125 | True | 0.1797 | 0.0341 | 0.8324 | 0.1777 | 0.1855 |
| uncertainty_cifar10_merc_linear_seed0 | merc_linear | 0 | 200 | 300 | True | 0.4440 | 0.0551 | 0.5875 | 0.3994 | 0.3926 |
| uncertainty_cifar10_merc_linear_seed1 | merc_linear | 1 | 200 | 300 | True | 0.4648 | 0.0595 | 0.4819 | 0.4131 | 0.4268 |
| uncertainty_cifar10_merc_linear_seed2 | merc_linear | 2 | 125 | 225 | True | 0.4427 | 0.0634 | 0.5109 | 0.3350 | 0.4053 |
| uncertainty_cifar10_mlp_seed0 | mlp | 0 | 150 | 250 | True | 0.4961 | 0.0337 | 0.3631 | 0.3955 | 0.4502 |
| uncertainty_cifar10_mlp_seed1 | mlp | 1 | 50 | 150 | True | 0.4609 | 0.0769 | 0.3757 | 0.3838 | 0.4307 |
| uncertainty_cifar10_mlp_seed2 | mlp | 2 | 200 | 300 | True | 0.5156 | 0.0318 | 0.3032 | 0.4102 | 0.4580 |

## synthetic_shape

| model | clean acc | noisy acc | occluded acc | clean ECE | clean Brier | clean selective AURC | clean->noisy AUROC | clean->occluded AUROC | resistance-noise corr | resistance-occlusion corr | support-evidence corr | conflict-resistance corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cosine_prototype | 0.9931 | 0.9570 | 0.9004 | 0.1207 | 0.0336 | 0.0002 | 0.6647 | 0.6252 | MISSING | MISSING | MISSING | MISSING |
| eml_centered_ambiguity | 0.9939 | 0.9548 | 0.9040 | 0.3461 | 0.2110 | 0.0002 | 0.5350 | 0.5880 | -0.0724 | 0.3919 | MISSING | MISSING |
| eml_no_ambiguity | 0.9944 | 0.9544 | 0.9049 | 0.3600 | 0.2202 | 0.0057 | 0.5198 | 0.5013 | 0.0377 | -0.0443 | MISSING | MISSING |
| eml_supervised_resistance | 0.9944 | 0.9548 | 0.9043 | 0.3460 | 0.2114 | 0.0005 | 0.5766 | 0.5906 | -0.0525 | 0.4251 | MISSING | MISSING |
| linear | 0.9926 | 0.9587 | 0.9014 | 0.0742 | 0.0212 | 0.0002 | 0.6830 | 0.6339 | MISSING | MISSING | MISSING | MISSING |
| merc_energy | 0.9284 | 0.9121 | 0.8509 | 0.0659 | 0.1351 | 0.0500 | 0.5458 | 0.5754 | -0.0355 | 0.2906 | MISSING | -0.0404 |
| merc_linear | 0.9948 | 0.9648 | 0.9033 | 0.2135 | 0.1791 | 0.0068 | 0.4946 | 0.5036 | 0.0004 | 0.0163 | MISSING | -0.0991 |
| mlp | 0.9939 | 0.9701 | 0.9089 | 0.0190 | 0.0098 | 0.0001 | 0.5577 | 0.5967 | MISSING | MISSING | MISSING | MISSING |

### synthetic_shape Detailed Runs

| run_id | model | seed | best step | steps run | early stop | clean acc | clean ECE | clean AURC | noisy acc | occluded acc |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| uncertainty_synthetic_shape_cosine_prototype_seed0 | cosine_prototype | 0 | 225 | 325 | True | 0.9961 | 0.1253 | 0.0001 | 0.9619 | 0.8936 |
| uncertainty_synthetic_shape_cosine_prototype_seed1 | cosine_prototype | 1 | 150 | 250 | True | 0.9883 | 0.0999 | 0.0003 | 0.9805 | 0.8994 |
| uncertainty_synthetic_shape_cosine_prototype_seed2 | cosine_prototype | 2 | 150 | 250 | True | 0.9948 | 0.1368 | 0.0001 | 0.9287 | 0.9082 |
| uncertainty_synthetic_shape_eml_centered_ambiguity_seed0 | eml_centered_ambiguity | 0 | 200 | 300 | True | 0.9948 | 0.2153 | 0.0001 | 0.9805 | 0.9102 |
| uncertainty_synthetic_shape_eml_centered_ambiguity_seed1 | eml_centered_ambiguity | 1 | 225 | 325 | True | 0.9935 | 0.1929 | 0.0001 | 0.9873 | 0.9043 |
| uncertainty_synthetic_shape_eml_centered_ambiguity_seed2 | eml_centered_ambiguity | 2 | 25 | 125 | True | 0.9935 | 0.6302 | 0.0006 | 0.8965 | 0.8975 |
| uncertainty_synthetic_shape_eml_no_ambiguity_seed0 | eml_no_ambiguity | 0 | 200 | 300 | True | 0.9961 | 0.2298 | 0.0019 | 0.9805 | 0.9102 |
| uncertainty_synthetic_shape_eml_no_ambiguity_seed1 | eml_no_ambiguity | 1 | 225 | 325 | True | 0.9935 | 0.2133 | 0.0058 | 0.9873 | 0.9062 |
| uncertainty_synthetic_shape_eml_no_ambiguity_seed2 | eml_no_ambiguity | 2 | 25 | 125 | True | 0.9935 | 0.6369 | 0.0094 | 0.8955 | 0.8984 |
| uncertainty_synthetic_shape_eml_supervised_resistance_seed0 | eml_supervised_resistance | 0 | 200 | 300 | True | 0.9961 | 0.2138 | 0.0001 | 0.9805 | 0.9102 |
| uncertainty_synthetic_shape_eml_supervised_resistance_seed1 | eml_supervised_resistance | 1 | 225 | 325 | True | 0.9935 | 0.1932 | 0.0001 | 0.9873 | 0.9053 |
| uncertainty_synthetic_shape_eml_supervised_resistance_seed2 | eml_supervised_resistance | 2 | 25 | 125 | True | 0.9935 | 0.6309 | 0.0013 | 0.8965 | 0.8975 |
| uncertainty_synthetic_shape_linear_seed0 | linear | 0 | 250 | 350 | True | 0.9948 | 0.0750 | 0.0001 | 0.9658 | 0.8984 |
| uncertainty_synthetic_shape_linear_seed1 | linear | 1 | 200 | 300 | True | 0.9883 | 0.0549 | 0.0003 | 0.9805 | 0.8984 |
| uncertainty_synthetic_shape_linear_seed2 | linear | 2 | 150 | 250 | True | 0.9948 | 0.0927 | 0.0001 | 0.9297 | 0.9072 |
| uncertainty_synthetic_shape_merc_energy_seed0 | merc_energy | 0 | 125 | 225 | True | 0.7969 | 0.1011 | 0.1450 | 0.7852 | 0.7354 |
| uncertainty_synthetic_shape_merc_energy_seed1 | merc_energy | 1 | 150 | 250 | True | 0.9922 | 0.0057 | 0.0031 | 0.9873 | 0.9014 |
| uncertainty_synthetic_shape_merc_energy_seed2 | merc_energy | 2 | 100 | 200 | True | 0.9961 | 0.0910 | 0.0017 | 0.9639 | 0.9160 |
| uncertainty_synthetic_shape_merc_linear_seed0 | merc_linear | 0 | 150 | 250 | True | 0.9974 | 0.0061 | 0.0014 | 0.9775 | 0.9102 |
| uncertainty_synthetic_shape_merc_linear_seed1 | merc_linear | 1 | 100 | 200 | True | 0.9935 | 0.0043 | 0.0061 | 0.9854 | 0.8994 |
| uncertainty_synthetic_shape_merc_linear_seed2 | merc_linear | 2 | 25 | 125 | True | 0.9935 | 0.6302 | 0.0130 | 0.9316 | 0.9004 |
| uncertainty_synthetic_shape_mlp_seed0 | mlp | 0 | 100 | 200 | True | 0.9935 | 0.0229 | 0.0001 | 0.9805 | 0.9092 |
| uncertainty_synthetic_shape_mlp_seed1 | mlp | 1 | 100 | 200 | True | 0.9935 | 0.0104 | 0.0001 | 0.9854 | 0.9062 |
| uncertainty_synthetic_shape_mlp_seed2 | mlp | 2 | 50 | 150 | True | 0.9948 | 0.0238 | 0.0001 | 0.9443 | 0.9111 |

## Conclusions

- `cifar10` clean accuracy vs cosine: EML centered 0.4505 vs cosine 0.4891. Head advantage is not supported.
- `cifar10` calibration vs cosine: EML centered ECE 0.0799 vs cosine 0.1299. Calibration is better.
- `cifar10` selective prediction vs cosine: EML centered clean AURC 0.2895 vs cosine 0.3507. Selective prediction is better.
- `cifar10` resistance-correlation check: noise 0.0540, occlusion -0.0109. Corruption correlation is not supported or weak.
- `cifar10` MERC support/conflict alignment: support-evidence MISSING, conflict-resistance -0.0067. MERC alignment is not claimed when values are MISSING or weak.
- `synthetic_shape` clean accuracy vs cosine: EML centered 0.9939 vs cosine 0.9931. Head advantage is supported in this benchmark only.
- `synthetic_shape` calibration vs cosine: EML centered ECE 0.3461 vs cosine 0.1207. Calibration is not better.
- `synthetic_shape` selective prediction vs cosine: EML centered clean AURC 0.0002 vs cosine 0.0002. Selective prediction is not better.
- `synthetic_shape` resistance-correlation check: noise -0.0525, occlusion 0.4251. Corruption correlation is supported.
- `synthetic_shape` MERC support/conflict alignment: support-evidence MISSING, conflict-resistance -0.0698. MERC alignment is not claimed when values are MISSING or weak.

- If the EML rows do not beat cosine on calibration or selective risk, the benchmark does not support an EML head advantage.

## Raw Artifacts

- Runs root: `reports/pluggable_primitive_real_20260427/frozen_runs`
- Summary CSV: `reports/pluggable_primitive_real_20260427/frozen_runs/summary.csv`
