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
| uncertainty_synthetic_shape_linear_seed1 | COMPLETED | linear | synthetic_shape |  |
| uncertainty_synthetic_shape_mlp_seed1 | COMPLETED | mlp | synthetic_shape |  |
| uncertainty_synthetic_shape_cosine_prototype_seed1 | COMPLETED | cosine_prototype | synthetic_shape |  |
| uncertainty_synthetic_shape_eml_no_ambiguity_seed1 | COMPLETED | eml_no_ambiguity | synthetic_shape |  |
| uncertainty_synthetic_shape_eml_centered_ambiguity_seed1 | COMPLETED | eml_centered_ambiguity | synthetic_shape |  |
| uncertainty_synthetic_shape_eml_supervised_resistance_seed1 | COMPLETED | eml_supervised_resistance | synthetic_shape |  |
| uncertainty_synthetic_shape_linear_seed2 | COMPLETED | linear | synthetic_shape |  |
| uncertainty_synthetic_shape_mlp_seed2 | COMPLETED | mlp | synthetic_shape |  |
| uncertainty_synthetic_shape_cosine_prototype_seed2 | COMPLETED | cosine_prototype | synthetic_shape |  |
| uncertainty_synthetic_shape_eml_no_ambiguity_seed2 | COMPLETED | eml_no_ambiguity | synthetic_shape |  |
| uncertainty_synthetic_shape_eml_centered_ambiguity_seed2 | COMPLETED | eml_centered_ambiguity | synthetic_shape |  |
| uncertainty_synthetic_shape_eml_supervised_resistance_seed2 | COMPLETED | eml_supervised_resistance | synthetic_shape |  |
| uncertainty_cifar10_linear_seed0 | COMPLETED | linear | cifar10 |  |
| uncertainty_cifar10_mlp_seed0 | COMPLETED | mlp | cifar10 |  |
| uncertainty_cifar10_cosine_prototype_seed0 | COMPLETED | cosine_prototype | cifar10 |  |
| uncertainty_cifar10_eml_no_ambiguity_seed0 | COMPLETED | eml_no_ambiguity | cifar10 |  |
| uncertainty_cifar10_eml_centered_ambiguity_seed0 | COMPLETED | eml_centered_ambiguity | cifar10 |  |
| uncertainty_cifar10_eml_supervised_resistance_seed0 | COMPLETED | eml_supervised_resistance | cifar10 |  |
| uncertainty_cifar10_linear_seed1 | COMPLETED | linear | cifar10 |  |
| uncertainty_cifar10_mlp_seed1 | COMPLETED | mlp | cifar10 |  |
| uncertainty_cifar10_cosine_prototype_seed1 | COMPLETED | cosine_prototype | cifar10 |  |
| uncertainty_cifar10_eml_no_ambiguity_seed1 | COMPLETED | eml_no_ambiguity | cifar10 |  |
| uncertainty_cifar10_eml_centered_ambiguity_seed1 | COMPLETED | eml_centered_ambiguity | cifar10 |  |
| uncertainty_cifar10_eml_supervised_resistance_seed1 | COMPLETED | eml_supervised_resistance | cifar10 |  |
| uncertainty_cifar10_linear_seed2 | COMPLETED | linear | cifar10 |  |
| uncertainty_cifar10_mlp_seed2 | COMPLETED | mlp | cifar10 |  |
| uncertainty_cifar10_cosine_prototype_seed2 | COMPLETED | cosine_prototype | cifar10 |  |
| uncertainty_cifar10_eml_no_ambiguity_seed2 | COMPLETED | eml_no_ambiguity | cifar10 |  |
| uncertainty_cifar10_eml_centered_ambiguity_seed2 | COMPLETED | eml_centered_ambiguity | cifar10 |  |
| uncertainty_cifar10_eml_supervised_resistance_seed2 | COMPLETED | eml_supervised_resistance | cifar10 |  |

## cifar10

| model | clean acc | noisy acc | occluded acc | clean ECE | clean Brier | clean selective AURC | clean->noisy AUROC | clean->occluded AUROC | pooled resistance-noise corr | pooled resistance-occlusion corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cosine_prototype | 0.4826 | 0.4102 | 0.4378 | 0.1332 | 0.6823 | 0.3561 | 0.5887 | 0.5368 | MISSING | MISSING |
| eml_centered_ambiguity | 0.4601 | 0.3939 | 0.4232 | 0.1302 | 0.7057 | 0.2618 | 0.5510 | 0.5313 | MISSING | MISSING |
| eml_no_ambiguity | 0.4575 | 0.4010 | 0.4128 | 0.1217 | 0.7042 | 0.4487 | 0.5124 | 0.5066 | MISSING | MISSING |
| eml_supervised_resistance | 0.4583 | 0.4007 | 0.4222 | 0.1254 | 0.7055 | 0.3273 | 0.6734 | 0.5330 | MISSING | MISSING |
| linear | 0.4753 | 0.3936 | 0.4310 | 0.1801 | 0.7147 | 0.3615 | 0.5894 | 0.5469 | MISSING | MISSING |
| mlp | 0.4935 | 0.3981 | 0.4440 | 0.0551 | 0.6556 | 0.3466 | 0.5238 | 0.5044 | MISSING | MISSING |

### cifar10 Detailed Runs

| run_id | model | seed | best step | steps run | early stop | clean acc | clean ECE | clean AURC | noisy acc | occluded acc |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| uncertainty_cifar10_cosine_prototype_seed0 | cosine_prototype | 0 | 175 | 250 | False | 0.4609 | 0.1055 | 0.3990 | 0.4219 | 0.4258 |
| uncertainty_cifar10_cosine_prototype_seed1 | cosine_prototype | 1 | 75 | 175 | True | 0.4766 | 0.1567 | 0.3631 | 0.4092 | 0.4424 |
| uncertainty_cifar10_cosine_prototype_seed2 | cosine_prototype | 2 | 150 | 250 | True | 0.5104 | 0.1374 | 0.3062 | 0.3994 | 0.4453 |
| uncertainty_cifar10_eml_centered_ambiguity_seed0 | eml_centered_ambiguity | 0 | 225 | 250 | False | 0.4596 | 0.1246 | 0.2417 | 0.4160 | 0.4463 |
| uncertainty_cifar10_eml_centered_ambiguity_seed1 | eml_centered_ambiguity | 1 | 225 | 250 | False | 0.5039 | 0.1538 | 0.2006 | 0.4219 | 0.4434 |
| uncertainty_cifar10_eml_centered_ambiguity_seed2 | eml_centered_ambiguity | 2 | 100 | 200 | True | 0.4167 | 0.1123 | 0.3430 | 0.3438 | 0.3799 |
| uncertainty_cifar10_eml_no_ambiguity_seed0 | eml_no_ambiguity | 0 | 200 | 250 | False | 0.4557 | 0.1137 | 0.4420 | 0.4277 | 0.4395 |
| uncertainty_cifar10_eml_no_ambiguity_seed1 | eml_no_ambiguity | 1 | 225 | 250 | False | 0.5026 | 0.1562 | 0.4060 | 0.4229 | 0.4414 |
| uncertainty_cifar10_eml_no_ambiguity_seed2 | eml_no_ambiguity | 2 | 125 | 225 | True | 0.4141 | 0.0951 | 0.4982 | 0.3525 | 0.3574 |
| uncertainty_cifar10_eml_supervised_resistance_seed0 | eml_supervised_resistance | 0 | 200 | 250 | False | 0.4570 | 0.1132 | 0.3313 | 0.4346 | 0.4463 |
| uncertainty_cifar10_eml_supervised_resistance_seed1 | eml_supervised_resistance | 1 | 225 | 250 | False | 0.5039 | 0.1530 | 0.2819 | 0.4248 | 0.4443 |
| uncertainty_cifar10_eml_supervised_resistance_seed2 | eml_supervised_resistance | 2 | 100 | 200 | True | 0.4141 | 0.1101 | 0.3687 | 0.3428 | 0.3760 |
| uncertainty_cifar10_linear_seed0 | linear | 0 | 175 | 250 | False | 0.4544 | 0.0959 | 0.4057 | 0.4209 | 0.4375 |
| uncertainty_cifar10_linear_seed1 | linear | 1 | 75 | 175 | True | 0.4753 | 0.2058 | 0.3660 | 0.3887 | 0.4336 |
| uncertainty_cifar10_linear_seed2 | linear | 2 | 75 | 175 | True | 0.4961 | 0.2386 | 0.3127 | 0.3711 | 0.4219 |
| uncertainty_cifar10_mlp_seed0 | mlp | 0 | 150 | 250 | True | 0.4935 | 0.0403 | 0.3651 | 0.3955 | 0.4502 |
| uncertainty_cifar10_mlp_seed1 | mlp | 1 | 50 | 150 | True | 0.4688 | 0.0807 | 0.3735 | 0.3877 | 0.4229 |
| uncertainty_cifar10_mlp_seed2 | mlp | 2 | 200 | 250 | False | 0.5182 | 0.0443 | 0.3013 | 0.4111 | 0.4590 |

## synthetic_shape

| model | clean acc | noisy acc | occluded acc | clean ECE | clean Brier | clean selective AURC | clean->noisy AUROC | clean->occluded AUROC | pooled resistance-noise corr | pooled resistance-occlusion corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cosine_prototype | 0.9905 | 0.9476 | 0.8841 | 0.2442 | 0.1030 | 0.0009 | 0.6152 | 0.6118 | MISSING | MISSING |
| eml_centered_ambiguity | 0.9939 | 0.9642 | 0.8978 | 0.2920 | 0.1163 | 0.0001 | 0.5919 | 0.6327 | MISSING | MISSING |
| eml_no_ambiguity | 0.9939 | 0.9642 | 0.8975 | 0.3049 | 0.1258 | 0.0035 | 0.5486 | 0.5096 | MISSING | MISSING |
| eml_supervised_resistance | 0.9935 | 0.9642 | 0.8975 | 0.2930 | 0.1168 | 0.0001 | 0.6443 | 0.6220 | MISSING | MISSING |
| linear | 0.9874 | 0.9421 | 0.8792 | 0.4325 | 0.3036 | 0.0006 | 0.6217 | 0.6179 | MISSING | MISSING |
| mlp | 0.9939 | 0.9635 | 0.8955 | 0.0302 | 0.0137 | 0.0002 | 0.5761 | 0.6187 | MISSING | MISSING |

### synthetic_shape Detailed Runs

| run_id | model | seed | best step | steps run | early stop | clean acc | clean ECE | clean AURC | noisy acc | occluded acc |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| uncertainty_synthetic_shape_cosine_prototype_seed0 | cosine_prototype | 0 | 125 | 225 | True | 0.9844 | 0.1750 | 0.0003 | 0.9287 | 0.8535 |
| uncertainty_synthetic_shape_cosine_prototype_seed1 | cosine_prototype | 1 | 50 | 150 | True | 0.9922 | 0.3088 | 0.0001 | 0.9814 | 0.9062 |
| uncertainty_synthetic_shape_cosine_prototype_seed2 | cosine_prototype | 2 | 75 | 175 | True | 0.9948 | 0.2487 | 0.0024 | 0.9326 | 0.8926 |
| uncertainty_synthetic_shape_eml_centered_ambiguity_seed0 | eml_centered_ambiguity | 0 | 75 | 175 | True | 0.9935 | 0.3086 | 0.0001 | 0.9482 | 0.8818 |
| uncertainty_synthetic_shape_eml_centered_ambiguity_seed1 | eml_centered_ambiguity | 1 | 100 | 200 | True | 0.9948 | 0.2758 | 0.0001 | 0.9863 | 0.9053 |
| uncertainty_synthetic_shape_eml_centered_ambiguity_seed2 | eml_centered_ambiguity | 2 | 75 | 175 | True | 0.9935 | 0.2914 | 0.0001 | 0.9580 | 0.9062 |
| uncertainty_synthetic_shape_eml_no_ambiguity_seed0 | eml_no_ambiguity | 0 | 75 | 175 | True | 0.9935 | 0.3198 | 0.0025 | 0.9482 | 0.8818 |
| uncertainty_synthetic_shape_eml_no_ambiguity_seed1 | eml_no_ambiguity | 1 | 100 | 200 | True | 0.9948 | 0.2896 | 0.0022 | 0.9863 | 0.9043 |
| uncertainty_synthetic_shape_eml_no_ambiguity_seed2 | eml_no_ambiguity | 2 | 75 | 175 | True | 0.9935 | 0.3053 | 0.0059 | 0.9580 | 0.9062 |
| uncertainty_synthetic_shape_eml_supervised_resistance_seed0 | eml_supervised_resistance | 0 | 75 | 175 | True | 0.9935 | 0.3092 | 0.0001 | 0.9482 | 0.8818 |
| uncertainty_synthetic_shape_eml_supervised_resistance_seed1 | eml_supervised_resistance | 1 | 100 | 200 | True | 0.9935 | 0.2776 | 0.0001 | 0.9873 | 0.9053 |
| uncertainty_synthetic_shape_eml_supervised_resistance_seed2 | eml_supervised_resistance | 2 | 75 | 175 | True | 0.9935 | 0.2921 | 0.0001 | 0.9570 | 0.9053 |
| uncertainty_synthetic_shape_linear_seed0 | linear | 0 | 125 | 225 | True | 0.9831 | 0.1622 | 0.0003 | 0.9258 | 0.8535 |
| uncertainty_synthetic_shape_linear_seed1 | linear | 1 | 25 | 125 | True | 0.9909 | 0.5648 | 0.0001 | 0.9814 | 0.9072 |
| uncertainty_synthetic_shape_linear_seed2 | linear | 2 | 25 | 125 | True | 0.9883 | 0.5706 | 0.0013 | 0.9189 | 0.8770 |
| uncertainty_synthetic_shape_mlp_seed0 | mlp | 0 | 50 | 150 | True | 0.9922 | 0.0647 | 0.0002 | 0.9521 | 0.8740 |
| uncertainty_synthetic_shape_mlp_seed1 | mlp | 1 | 75 | 175 | True | 0.9948 | 0.0103 | 0.0001 | 0.9873 | 0.9053 |
| uncertainty_synthetic_shape_mlp_seed2 | mlp | 2 | 50 | 150 | True | 0.9948 | 0.0155 | 0.0002 | 0.9512 | 0.9072 |

## Conclusions

- `cifar10` clean accuracy vs cosine: EML centered 0.4601 vs cosine 0.4826. Head advantage is not supported.
- `cifar10` calibration vs cosine: EML centered ECE 0.1302 vs cosine 0.1332. Calibration is better.
- `cifar10` selective prediction vs cosine: EML centered clean AURC 0.2618 vs cosine 0.3561. Selective prediction is better.
- `cifar10` resistance-correlation check: noise MISSING, occlusion MISSING. Corruption correlation is not supported.
- `synthetic_shape` clean accuracy vs cosine: EML centered 0.9939 vs cosine 0.9905. Head advantage is supported in this benchmark only.
- `synthetic_shape` calibration vs cosine: EML centered ECE 0.2920 vs cosine 0.2442. Calibration is not better.
- `synthetic_shape` selective prediction vs cosine: EML centered clean AURC 0.0001 vs cosine 0.0009. Selective prediction is better.
- `synthetic_shape` resistance-correlation check: noise MISSING, occlusion MISSING. Corruption correlation is not supported.

- If the EML rows do not beat cosine on calibration or selective risk, the benchmark does not support an EML head advantage.

## Raw Artifacts

- Runs root: `reports/uncertainty_remote_20260424_175558/runs`
- Summary CSV: `reports/uncertainty_remote_20260424_175558/runs/summary.csv`
