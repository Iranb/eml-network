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
| cosine_prototype | 0.4839 | 0.4121 | 0.4401 | 0.1358 | 0.6814 | 0.3546 | 0.5870 | 0.5350 | MISSING | MISSING | MISSING | MISSING |
| eml_centered_ambiguity | 0.4271 | 0.3558 | 0.3900 | 0.0834 | 0.7160 | 0.3391 | 0.5286 | 0.5248 | 0.0220 | 0.0153 | MISSING | MISSING |
| eml_no_ambiguity | 0.3958 | 0.3307 | 0.3636 | 0.0968 | 0.7360 | 0.5583 | 0.5185 | 0.4865 | 0.0338 | -0.0347 | MISSING | MISSING |
| eml_supervised_resistance | 0.4280 | 0.3561 | 0.3906 | 0.0852 | 0.7160 | 0.3775 | 0.5928 | 0.5263 | 0.1343 | -0.0454 | MISSING | MISSING |
| linear | 0.4826 | 0.3952 | 0.4378 | 0.1723 | 0.7051 | 0.3556 | 0.5899 | 0.5433 | MISSING | MISSING | MISSING | MISSING |
| merc_energy | 0.4852 | 0.4115 | 0.4404 | 0.0962 | 0.6671 | 0.3757 | 0.5574 | 0.5243 | 0.0312 | -0.0021 | MISSING | -0.0001 |
| merc_linear | 0.4575 | 0.3910 | 0.4255 | 0.0600 | 0.6797 | 0.5049 | 0.5179 | 0.4971 | 0.0286 | -0.0212 | MISSING | -0.0238 |
| mlp | 0.4913 | 0.3958 | 0.4440 | 0.0489 | 0.6548 | 0.3458 | 0.5251 | 0.5027 | MISSING | MISSING | MISSING | MISSING |

### cifar10 Detailed Runs

| run_id | model | seed | best step | steps run | early stop | clean acc | clean ECE | clean AURC | noisy acc | occluded acc |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| uncertainty_cifar10_cosine_prototype_seed0 | cosine_prototype | 0 | 175 | 275 | True | 0.4583 | 0.0999 | 0.3989 | 0.4316 | 0.4336 |
| uncertainty_cifar10_cosine_prototype_seed1 | cosine_prototype | 1 | 75 | 175 | True | 0.4818 | 0.1677 | 0.3605 | 0.4072 | 0.4404 |
| uncertainty_cifar10_cosine_prototype_seed2 | cosine_prototype | 2 | 150 | 250 | True | 0.5117 | 0.1397 | 0.3044 | 0.3975 | 0.4463 |
| uncertainty_cifar10_eml_centered_ambiguity_seed0 | eml_centered_ambiguity | 0 | 150 | 250 | True | 0.4206 | 0.0872 | 0.3212 | 0.3428 | 0.3896 |
| uncertainty_cifar10_eml_centered_ambiguity_seed1 | eml_centered_ambiguity | 1 | 150 | 250 | True | 0.4557 | 0.0980 | 0.3009 | 0.3799 | 0.4160 |
| uncertainty_cifar10_eml_centered_ambiguity_seed2 | eml_centered_ambiguity | 2 | 125 | 225 | True | 0.4049 | 0.0651 | 0.3951 | 0.3447 | 0.3643 |
| uncertainty_cifar10_eml_no_ambiguity_seed0 | eml_no_ambiguity | 0 | 150 | 250 | True | 0.4010 | 0.0698 | 0.5631 | 0.3408 | 0.3730 |
| uncertainty_cifar10_eml_no_ambiguity_seed1 | eml_no_ambiguity | 1 | 50 | 150 | True | 0.4023 | 0.1479 | 0.5304 | 0.3320 | 0.3701 |
| uncertainty_cifar10_eml_no_ambiguity_seed2 | eml_no_ambiguity | 2 | 100 | 200 | True | 0.3841 | 0.0729 | 0.5815 | 0.3193 | 0.3477 |
| uncertainty_cifar10_eml_supervised_resistance_seed0 | eml_supervised_resistance | 0 | 150 | 250 | True | 0.4193 | 0.0861 | 0.3995 | 0.3447 | 0.3906 |
| uncertainty_cifar10_eml_supervised_resistance_seed1 | eml_supervised_resistance | 1 | 150 | 250 | True | 0.4609 | 0.1009 | 0.3174 | 0.3809 | 0.4160 |
| uncertainty_cifar10_eml_supervised_resistance_seed2 | eml_supervised_resistance | 2 | 125 | 225 | True | 0.4036 | 0.0685 | 0.4157 | 0.3428 | 0.3652 |
| uncertainty_cifar10_linear_seed0 | linear | 0 | 175 | 275 | True | 0.4531 | 0.0897 | 0.4046 | 0.4268 | 0.4385 |
| uncertainty_cifar10_linear_seed1 | linear | 1 | 100 | 200 | True | 0.4935 | 0.1834 | 0.3517 | 0.3867 | 0.4551 |
| uncertainty_cifar10_linear_seed2 | linear | 2 | 75 | 175 | True | 0.5013 | 0.2437 | 0.3106 | 0.3721 | 0.4199 |
| uncertainty_cifar10_merc_energy_seed0 | merc_energy | 0 | 575 | 675 | True | 0.4661 | 0.0876 | 0.4486 | 0.4219 | 0.4150 |
| uncertainty_cifar10_merc_energy_seed1 | merc_energy | 1 | 450 | 550 | True | 0.4766 | 0.1559 | 0.3814 | 0.4082 | 0.4385 |
| uncertainty_cifar10_merc_energy_seed2 | merc_energy | 2 | 575 | 675 | True | 0.5130 | 0.0452 | 0.2972 | 0.4043 | 0.4678 |
| uncertainty_cifar10_merc_linear_seed0 | merc_linear | 0 | 400 | 500 | True | 0.4740 | 0.0621 | 0.5307 | 0.4258 | 0.4395 |
| uncertainty_cifar10_merc_linear_seed1 | merc_linear | 1 | 200 | 300 | True | 0.4609 | 0.0630 | 0.4756 | 0.4111 | 0.4355 |
| uncertainty_cifar10_merc_linear_seed2 | merc_linear | 2 | 125 | 225 | True | 0.4375 | 0.0550 | 0.5084 | 0.3359 | 0.4014 |
| uncertainty_cifar10_mlp_seed0 | mlp | 0 | 150 | 250 | True | 0.4922 | 0.0469 | 0.3634 | 0.3984 | 0.4541 |
| uncertainty_cifar10_mlp_seed1 | mlp | 1 | 50 | 150 | True | 0.4648 | 0.0644 | 0.3719 | 0.3828 | 0.4229 |
| uncertainty_cifar10_mlp_seed2 | mlp | 2 | 200 | 300 | True | 0.5169 | 0.0353 | 0.3021 | 0.4062 | 0.4551 |

## synthetic_shape

| model | clean acc | noisy acc | occluded acc | clean ECE | clean Brier | clean selective AURC | clean->noisy AUROC | clean->occluded AUROC | resistance-noise corr | resistance-occlusion corr | support-evidence corr | conflict-resistance corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cosine_prototype | 0.9909 | 0.9329 | 0.8867 | 0.1901 | 0.0672 | 0.0002 | 0.7387 | 0.6369 | MISSING | MISSING | MISSING | MISSING |
| eml_centered_ambiguity | 0.9952 | 0.9521 | 0.9030 | 0.2830 | 0.1082 | 0.0000 | 0.6384 | 0.6304 | 0.0701 | 0.4109 | MISSING | MISSING |
| eml_no_ambiguity | 0.9952 | 0.9525 | 0.9030 | 0.2954 | 0.1170 | 0.0016 | 0.5727 | 0.4936 | 0.0882 | -0.0549 | MISSING | MISSING |
| eml_supervised_resistance | 0.9952 | 0.9518 | 0.9030 | 0.2832 | 0.1083 | 0.0000 | 0.6858 | 0.6339 | 0.1057 | 0.4446 | MISSING | MISSING |
| linear | 0.9896 | 0.9186 | 0.8831 | 0.2637 | 0.1174 | 0.0004 | 0.7366 | 0.6297 | MISSING | MISSING | MISSING | MISSING |
| merc_energy | 0.9944 | 0.9538 | 0.8991 | 0.1313 | 0.1319 | 0.0028 | 0.5401 | 0.5877 | -0.0433 | 0.3289 | MISSING | -0.0472 |
| merc_linear | 0.9952 | 0.9691 | 0.8971 | 0.2120 | 0.1807 | 0.0075 | 0.4799 | 0.5053 | -0.0143 | 0.0135 | MISSING | -0.1080 |
| mlp | 0.9948 | 0.9557 | 0.9014 | 0.0269 | 0.0127 | 0.0005 | 0.5864 | 0.6110 | MISSING | MISSING | MISSING | MISSING |

### synthetic_shape Detailed Runs

| run_id | model | seed | best step | steps run | early stop | clean acc | clean ECE | clean AURC | noisy acc | occluded acc |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| uncertainty_synthetic_shape_cosine_prototype_seed0 | cosine_prototype | 0 | 75 | 175 | True | 0.9857 | 0.2433 | 0.0003 | 0.8936 | 0.8730 |
| uncertainty_synthetic_shape_cosine_prototype_seed1 | cosine_prototype | 1 | 75 | 175 | True | 0.9935 | 0.2069 | 0.0001 | 0.9609 | 0.8896 |
| uncertainty_synthetic_shape_cosine_prototype_seed2 | cosine_prototype | 2 | 175 | 275 | True | 0.9935 | 0.1200 | 0.0003 | 0.9443 | 0.8975 |
| uncertainty_synthetic_shape_eml_centered_ambiguity_seed0 | eml_centered_ambiguity | 0 | 100 | 200 | True | 0.9935 | 0.2880 | 0.0001 | 0.9355 | 0.9023 |
| uncertainty_synthetic_shape_eml_centered_ambiguity_seed1 | eml_centered_ambiguity | 1 | 125 | 225 | True | 0.9961 | 0.2607 | 0.0000 | 0.9727 | 0.9043 |
| uncertainty_synthetic_shape_eml_centered_ambiguity_seed2 | eml_centered_ambiguity | 2 | 75 | 175 | True | 0.9961 | 0.3004 | 0.0000 | 0.9482 | 0.9023 |
| uncertainty_synthetic_shape_eml_no_ambiguity_seed0 | eml_no_ambiguity | 0 | 100 | 200 | True | 0.9935 | 0.3002 | 0.0026 | 0.9355 | 0.9023 |
| uncertainty_synthetic_shape_eml_no_ambiguity_seed1 | eml_no_ambiguity | 1 | 125 | 225 | True | 0.9961 | 0.2735 | 0.0013 | 0.9736 | 0.9043 |
| uncertainty_synthetic_shape_eml_no_ambiguity_seed2 | eml_no_ambiguity | 2 | 75 | 175 | True | 0.9961 | 0.3124 | 0.0009 | 0.9482 | 0.9023 |
| uncertainty_synthetic_shape_eml_supervised_resistance_seed0 | eml_supervised_resistance | 0 | 100 | 200 | True | 0.9935 | 0.2881 | 0.0001 | 0.9346 | 0.9023 |
| uncertainty_synthetic_shape_eml_supervised_resistance_seed1 | eml_supervised_resistance | 1 | 125 | 225 | True | 0.9961 | 0.2609 | 0.0000 | 0.9727 | 0.9043 |
| uncertainty_synthetic_shape_eml_supervised_resistance_seed2 | eml_supervised_resistance | 2 | 75 | 175 | True | 0.9961 | 0.3007 | 0.0000 | 0.9482 | 0.9023 |
| uncertainty_synthetic_shape_linear_seed0 | linear | 0 | 100 | 200 | True | 0.9844 | 0.1876 | 0.0002 | 0.8975 | 0.8760 |
| uncertainty_synthetic_shape_linear_seed1 | linear | 1 | 75 | 175 | True | 0.9935 | 0.2293 | 0.0001 | 0.9521 | 0.8877 |
| uncertainty_synthetic_shape_linear_seed2 | linear | 2 | 50 | 150 | True | 0.9909 | 0.3743 | 0.0009 | 0.9062 | 0.8857 |
| uncertainty_synthetic_shape_merc_energy_seed0 | merc_energy | 0 | 100 | 200 | True | 0.9909 | 0.1052 | 0.0035 | 0.9580 | 0.8857 |
| uncertainty_synthetic_shape_merc_energy_seed1 | merc_energy | 1 | 125 | 225 | True | 1.0000 | 0.1960 | 0.0000 | 0.9561 | 0.9053 |
| uncertainty_synthetic_shape_merc_energy_seed2 | merc_energy | 2 | 100 | 200 | True | 0.9922 | 0.0926 | 0.0049 | 0.9473 | 0.9062 |
| uncertainty_synthetic_shape_merc_linear_seed0 | merc_linear | 0 | 125 | 225 | True | 0.9935 | 0.0069 | 0.0037 | 0.9668 | 0.8887 |
| uncertainty_synthetic_shape_merc_linear_seed1 | merc_linear | 1 | 100 | 200 | True | 0.9974 | 0.0042 | 0.0046 | 0.9854 | 0.9092 |
| uncertainty_synthetic_shape_merc_linear_seed2 | merc_linear | 2 | 25 | 125 | True | 0.9948 | 0.6249 | 0.0142 | 0.9551 | 0.8936 |
| uncertainty_synthetic_shape_mlp_seed0 | mlp | 0 | 50 | 150 | True | 0.9922 | 0.0490 | 0.0005 | 0.9307 | 0.8926 |
| uncertainty_synthetic_shape_mlp_seed1 | mlp | 1 | 150 | 250 | True | 0.9974 | 0.0072 | 0.0007 | 0.9844 | 0.9092 |
| uncertainty_synthetic_shape_mlp_seed2 | mlp | 2 | 50 | 150 | True | 0.9948 | 0.0244 | 0.0004 | 0.9521 | 0.9023 |

## Conclusions

- `cifar10` clean accuracy vs cosine: EML centered 0.4271 vs cosine 0.4839. Head advantage is not supported.
- `cifar10` calibration vs cosine: EML centered ECE 0.0834 vs cosine 0.1358. Calibration is better.
- `cifar10` selective prediction vs cosine: EML centered clean AURC 0.3391 vs cosine 0.3546. Selective prediction is better.
- `cifar10` resistance-correlation check: noise 0.1343, occlusion -0.0454. Corruption correlation is not supported or weak.
- `cifar10` MERC support/conflict alignment: support-evidence MISSING, conflict-resistance -0.0120. MERC alignment is not claimed when values are MISSING or weak.
- `synthetic_shape` clean accuracy vs cosine: EML centered 0.9952 vs cosine 0.9909. Head advantage is supported in this benchmark only.
- `synthetic_shape` calibration vs cosine: EML centered ECE 0.2830 vs cosine 0.1901. Calibration is not better.
- `synthetic_shape` selective prediction vs cosine: EML centered clean AURC 0.0000 vs cosine 0.0002. Selective prediction is better.
- `synthetic_shape` resistance-correlation check: noise 0.1057, occlusion 0.4446. Corruption correlation is supported.
- `synthetic_shape` MERC support/conflict alignment: support-evidence MISSING, conflict-resistance -0.0776. MERC alignment is not claimed when values are MISSING or weak.

- If the EML rows do not beat cosine on calibration or selective risk, the benchmark does not support an EML head advantage.

## Raw Artifacts

- Runs root: `reports/pluggable_uncertainty_real_20260427/runs`
- Summary CSV: `reports/pluggable_uncertainty_real_20260427/runs/summary.csv`
