# CNN Head Ablation Report

## 1. Executive Summary
- Completed runs: 48
- NOT RUN entries: 6
- Failed runs: 0
- Best frozen-feature result: cosine_prototype seed=2 test_accuracy=0.4746
- Best end-to-end result: eml_bank_centered_ambiguity seed=2 test_accuracy=0.4824
- Claim status: The evidence is mixed: centered EML wins 5/21 paired comparisons.

## 2. Experimental Setup
- Frozen-feature runs train one shared CNN feature extractor per dataset/seed, cache features, then train only the selected head.
- End-to-end runs train the same CNN backbone with one selected head; the EML residual-bank variant is reported separately.
- CE-only and prototype-pairwise settings are separated. Linear and MLP heads are marked NOT RUN for prototype-pairwise because that loss is not applicable.

## 3. Run Status
| run_id | status | experiment | model | dataset | seed | reason |
| --- | --- | --- | --- | --- | ---: | --- |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed0 | COMPLETED | end_to_end | cosine_prototype | cifar10 | 0 |  |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed1 | COMPLETED | end_to_end | cosine_prototype | cifar10 | 1 |  |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed2 | COMPLETED | end_to_end | cosine_prototype | cifar10 | 2 |  |
| e2e_cifar10_cosine_prototype_ce_seed0 | COMPLETED | end_to_end | cosine_prototype | cifar10 | 0 |  |
| e2e_cifar10_cosine_prototype_ce_seed1 | COMPLETED | end_to_end | cosine_prototype | cifar10 | 1 |  |
| e2e_cifar10_cosine_prototype_ce_seed2 | COMPLETED | end_to_end | cosine_prototype | cifar10 | 2 |  |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0 | COMPLETED | end_to_end | eml_bank_centered_ambiguity | cifar10 | 0 |  |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1 | COMPLETED | end_to_end | eml_bank_centered_ambiguity | cifar10 | 1 |  |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed2 | COMPLETED | end_to_end | eml_bank_centered_ambiguity | cifar10 | 2 |  |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0 | COMPLETED | end_to_end | eml_bank_centered_ambiguity | cifar10 | 0 |  |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1 | COMPLETED | end_to_end | eml_bank_centered_ambiguity | cifar10 | 1 |  |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed2 | COMPLETED | end_to_end | eml_bank_centered_ambiguity | cifar10 | 2 |  |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0 | COMPLETED | end_to_end | eml_centered_ambiguity | cifar10 | 0 |  |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1 | COMPLETED | end_to_end | eml_centered_ambiguity | cifar10 | 1 |  |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed2 | COMPLETED | end_to_end | eml_centered_ambiguity | cifar10 | 2 |  |
| e2e_cifar10_eml_centered_ambiguity_ce_seed0 | COMPLETED | end_to_end | eml_centered_ambiguity | cifar10 | 0 |  |
| e2e_cifar10_eml_centered_ambiguity_ce_seed1 | COMPLETED | end_to_end | eml_centered_ambiguity | cifar10 | 1 |  |
| e2e_cifar10_eml_centered_ambiguity_ce_seed2 | COMPLETED | end_to_end | eml_centered_ambiguity | cifar10 | 2 |  |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0 | COMPLETED | end_to_end | eml_no_ambiguity | cifar10 | 0 |  |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1 | COMPLETED | end_to_end | eml_no_ambiguity | cifar10 | 1 |  |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed2 | COMPLETED | end_to_end | eml_no_ambiguity | cifar10 | 2 |  |
| e2e_cifar10_eml_no_ambiguity_ce_seed0 | COMPLETED | end_to_end | eml_no_ambiguity | cifar10 | 0 |  |
| e2e_cifar10_eml_no_ambiguity_ce_seed1 | COMPLETED | end_to_end | eml_no_ambiguity | cifar10 | 1 |  |
| e2e_cifar10_eml_no_ambiguity_ce_seed2 | COMPLETED | end_to_end | eml_no_ambiguity | cifar10 | 2 |  |
| e2e_cifar10_linear_ce_pairwise_seed0 | NOT RUN | end_to_end | linear | cifar10 | 0 | pairwise prototype margin is not applicable |
| e2e_cifar10_linear_ce_pairwise_seed1 | NOT RUN | end_to_end | linear | cifar10 | 1 | pairwise prototype margin is not applicable |
| e2e_cifar10_linear_ce_pairwise_seed2 | NOT RUN | end_to_end | linear | cifar10 | 2 | pairwise prototype margin is not applicable |
| e2e_cifar10_linear_ce_seed0 | COMPLETED | end_to_end | linear | cifar10 | 0 |  |
| e2e_cifar10_linear_ce_seed1 | COMPLETED | end_to_end | linear | cifar10 | 1 |  |
| e2e_cifar10_linear_ce_seed2 | COMPLETED | end_to_end | linear | cifar10 | 2 |  |
| e2e_cifar10_mlp_ce_pairwise_seed0 | NOT RUN | end_to_end | mlp | cifar10 | 0 | pairwise prototype margin is not applicable |
| e2e_cifar10_mlp_ce_pairwise_seed1 | NOT RUN | end_to_end | mlp | cifar10 | 1 | pairwise prototype margin is not applicable |
| e2e_cifar10_mlp_ce_pairwise_seed2 | NOT RUN | end_to_end | mlp | cifar10 | 2 | pairwise prototype margin is not applicable |
| e2e_cifar10_mlp_ce_seed0 | COMPLETED | end_to_end | mlp | cifar10 | 0 |  |
| e2e_cifar10_mlp_ce_seed1 | COMPLETED | end_to_end | mlp | cifar10 | 1 |  |
| e2e_cifar10_mlp_ce_seed2 | COMPLETED | end_to_end | mlp | cifar10 | 2 |  |
| frozen_cifar10_cosine_prototype_seed0 | COMPLETED | frozen_features | cosine_prototype | cifar10 | 0 |  |
| frozen_cifar10_cosine_prototype_seed1 | COMPLETED | frozen_features | cosine_prototype | cifar10 | 1 |  |
| frozen_cifar10_cosine_prototype_seed2 | COMPLETED | frozen_features | cosine_prototype | cifar10 | 2 |  |
| frozen_cifar10_eml_centered_ambiguity_seed0 | COMPLETED | frozen_features | eml_centered_ambiguity | cifar10 | 0 |  |
| frozen_cifar10_eml_centered_ambiguity_seed1 | COMPLETED | frozen_features | eml_centered_ambiguity | cifar10 | 1 |  |
| frozen_cifar10_eml_centered_ambiguity_seed2 | COMPLETED | frozen_features | eml_centered_ambiguity | cifar10 | 2 |  |
| frozen_cifar10_eml_no_ambiguity_seed0 | COMPLETED | frozen_features | eml_no_ambiguity | cifar10 | 0 |  |
| frozen_cifar10_eml_no_ambiguity_seed1 | COMPLETED | frozen_features | eml_no_ambiguity | cifar10 | 1 |  |
| frozen_cifar10_eml_no_ambiguity_seed2 | COMPLETED | frozen_features | eml_no_ambiguity | cifar10 | 2 |  |
| frozen_cifar10_eml_raw_ambiguity_seed0 | COMPLETED | frozen_features | eml_raw_ambiguity | cifar10 | 0 |  |
| frozen_cifar10_eml_raw_ambiguity_seed1 | COMPLETED | frozen_features | eml_raw_ambiguity | cifar10 | 1 |  |
| frozen_cifar10_eml_raw_ambiguity_seed2 | COMPLETED | frozen_features | eml_raw_ambiguity | cifar10 | 2 |  |
| frozen_cifar10_linear_seed0 | COMPLETED | frozen_features | linear | cifar10 | 0 |  |
| frozen_cifar10_linear_seed1 | COMPLETED | frozen_features | linear | cifar10 | 1 |  |
| frozen_cifar10_linear_seed2 | COMPLETED | frozen_features | linear | cifar10 | 2 |  |
| frozen_cifar10_mlp_seed0 | COMPLETED | frozen_features | mlp | cifar10 | 0 |  |
| frozen_cifar10_mlp_seed1 | COMPLETED | frozen_features | mlp | cifar10 | 1 |  |
| frozen_cifar10_mlp_seed2 | COMPLETED | frozen_features | mlp | cifar10 | 2 |  |

## 4. Frozen Feature Results
### Frozen CNN Features
| run_id | seed | model | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen_cifar10_cosine_prototype_seed0 | 0 | cosine_prototype | 0.4121 | 0.4062 | 1.6385 | 0.0743 | 0.7304 | -0.3853 | 1.5499 |
| frozen_cifar10_cosine_prototype_seed1 | 1 | cosine_prototype | 0.4277 | 0.4219 | 1.6173 | 0.0812 | 0.7250 | -0.3480 | 1.3675 |
| frozen_cifar10_cosine_prototype_seed2 | 2 | cosine_prototype | 0.4746 | 0.4512 | 1.5104 | 0.1039 | 0.6795 | -0.2011 | 1.4450 |
| frozen_cifar10_eml_centered_ambiguity_seed0 | 0 | eml_centered_ambiguity | 0.3984 | 0.3750 | 1.6835 | 0.0702 | 0.7415 | -0.4350 | 2.9472 |
| frozen_cifar10_eml_centered_ambiguity_seed1 | 1 | eml_centered_ambiguity | 0.4004 | 0.4160 | 1.7016 | 0.0823 | 0.7476 | -0.4517 | 2.9787 |
| frozen_cifar10_eml_centered_ambiguity_seed2 | 2 | eml_centered_ambiguity | 0.4668 | 0.4160 | 1.5934 | 0.1321 | 0.7128 | -0.3363 | 3.0262 |
| frozen_cifar10_eml_no_ambiguity_seed0 | 0 | eml_no_ambiguity | 0.3965 | 0.3730 | 1.6855 | 0.0698 | 0.7424 | -0.4354 | 3.0482 |
| frozen_cifar10_eml_no_ambiguity_seed1 | 1 | eml_no_ambiguity | 0.3906 | 0.4043 | 1.7039 | 0.0803 | 0.7485 | -0.4521 | 2.7475 |
| frozen_cifar10_eml_no_ambiguity_seed2 | 2 | eml_no_ambiguity | 0.4668 | 0.4199 | 1.5962 | 0.1339 | 0.7142 | -0.3392 | 2.9885 |
| frozen_cifar10_eml_raw_ambiguity_seed0 | 0 | eml_raw_ambiguity | 0.4004 | 0.3750 | 1.6841 | 0.0727 | 0.7417 | -0.4350 | 2.9161 |
| frozen_cifar10_eml_raw_ambiguity_seed1 | 1 | eml_raw_ambiguity | 0.4004 | 0.4141 | 1.7022 | 0.0838 | 0.7478 | -0.4517 | 2.9102 |
| frozen_cifar10_eml_raw_ambiguity_seed2 | 2 | eml_raw_ambiguity | 0.4668 | 0.4180 | 1.5940 | 0.1325 | 0.7131 | -0.3367 | 3.0062 |
| frozen_cifar10_linear_seed0 | 0 | linear | 0.4355 | 0.4199 | 1.6304 | 0.0886 | 0.7285 | -0.3749 | 1.1413 |
| frozen_cifar10_linear_seed1 | 1 | linear | 0.4160 | 0.4180 | 1.6157 | 0.0908 | 0.7269 | -0.3339 | 0.8159 |
| frozen_cifar10_linear_seed2 | 2 | linear | 0.4590 | 0.4434 | 1.5095 | 0.0931 | 0.6805 | -0.1939 | 0.9505 |
| frozen_cifar10_mlp_seed0 | 0 | mlp | 0.4180 | 0.4102 | 1.6032 | 0.0568 | 0.7093 | -0.4280 | 0.9383 |
| frozen_cifar10_mlp_seed1 | 1 | mlp | 0.4121 | 0.4062 | 1.5891 | 0.0693 | 0.7177 | -0.4255 | 0.9177 |
| frozen_cifar10_mlp_seed2 | 2 | mlp | 0.4629 | 0.4883 | 1.4559 | 0.0661 | 0.6560 | -0.2455 | 0.9632 |

## 5. End-To-End Results
### CNN Plus Head
| run_id | seed | model | loss mode | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed0 | 0 | cosine_prototype | ce_pairwise | 0.4434 | 0.4121 | 1.6029 | 0.0740 | 0.6950 | -0.3479 | 4.6055 |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed1 | 1 | cosine_prototype | ce_pairwise | 0.4570 | 0.4883 | 1.5106 | 0.0646 | 0.6788 | -0.2630 | 4.6120 |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed2 | 2 | cosine_prototype | ce_pairwise | 0.4609 | 0.4727 | 1.5369 | 0.0597 | 0.6851 | -0.2835 | 4.6391 |
| e2e_cifar10_cosine_prototype_ce_seed0 | 0 | cosine_prototype | ce | 0.4434 | 0.4121 | 1.6081 | 0.0620 | 0.6970 | -0.3566 | 4.7124 |
| e2e_cifar10_cosine_prototype_ce_seed1 | 1 | cosine_prototype | ce | 0.4590 | 0.4980 | 1.5086 | 0.0499 | 0.6781 | -0.2589 | 4.6884 |
| e2e_cifar10_cosine_prototype_ce_seed2 | 2 | cosine_prototype | ce | 0.4688 | 0.4727 | 1.5240 | 0.0621 | 0.6800 | -0.2663 | 4.6079 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_bank_centered_ambiguity | ce_pairwise | 0.3867 | 0.3691 | 1.7374 | 0.0363 | 0.7540 | -0.5470 | 7.1461 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_bank_centered_ambiguity | ce_pairwise | 0.4062 | 0.3789 | 1.7084 | 0.0904 | 0.7506 | -0.4477 | 7.3892 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed2 | 2 | eml_bank_centered_ambiguity | ce_pairwise | 0.4824 | 0.4277 | 1.5528 | 0.1340 | 0.6941 | -0.2791 | 7.2749 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0 | 0 | eml_bank_centered_ambiguity | ce | 0.4355 | 0.4062 | 1.6107 | 0.0854 | 0.7175 | -0.3653 | 7.3436 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1 | 1 | eml_bank_centered_ambiguity | ce | 0.3906 | 0.4355 | 1.7005 | 0.0681 | 0.7463 | -0.4441 | 7.2667 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed2 | 2 | eml_bank_centered_ambiguity | ce | 0.4453 | 0.4277 | 1.6214 | 0.1052 | 0.7162 | -0.3739 | 7.2526 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_centered_ambiguity | ce_pairwise | 0.3379 | 0.3613 | 1.6543 | 0.0479 | 0.7398 | -0.4172 | 6.2512 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_centered_ambiguity | ce_pairwise | 0.4062 | 0.4062 | 1.6842 | 0.0636 | 0.7347 | -0.4461 | 6.3302 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed2 | 2 | eml_centered_ambiguity | ce_pairwise | 0.4277 | 0.4121 | 1.6611 | 0.0723 | 0.7273 | -0.4150 | 6.3032 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed0 | 0 | eml_centered_ambiguity | ce | 0.3691 | 0.3711 | 1.6556 | 0.0456 | 0.7361 | -0.4188 | 6.3703 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed1 | 1 | eml_centered_ambiguity | ce | 0.4062 | 0.4043 | 1.6860 | 0.0827 | 0.7393 | -0.4308 | 6.3644 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed2 | 2 | eml_centered_ambiguity | ce | 0.4805 | 0.4492 | 1.5825 | 0.1292 | 0.7021 | -0.3155 | 6.3350 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0 | 0 | eml_no_ambiguity | ce_pairwise | 0.3633 | 0.3711 | 1.6514 | 0.0362 | 0.7343 | -0.4384 | 5.9053 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1 | 1 | eml_no_ambiguity | ce_pairwise | 0.4355 | 0.4102 | 1.6471 | 0.1047 | 0.7282 | -0.4005 | 6.1300 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed2 | 2 | eml_no_ambiguity | ce_pairwise | 0.4746 | 0.4453 | 1.5485 | 0.1529 | 0.6964 | -0.2974 | 6.0611 |
| e2e_cifar10_eml_no_ambiguity_ce_seed0 | 0 | eml_no_ambiguity | ce | 0.3008 | 0.3008 | 1.7863 | 0.0283 | 0.7762 | -0.5609 | 6.1379 |
| e2e_cifar10_eml_no_ambiguity_ce_seed1 | 1 | eml_no_ambiguity | ce | 0.4160 | 0.4121 | 1.6819 | 0.0876 | 0.7401 | -0.4471 | 6.1857 |
| e2e_cifar10_eml_no_ambiguity_ce_seed2 | 2 | eml_no_ambiguity | ce | 0.4551 | 0.4277 | 1.5691 | 0.1122 | 0.6973 | -0.2903 | 6.1425 |
| e2e_cifar10_linear_ce_pairwise_seed0 | 0 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_pairwise_seed1 | 1 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_pairwise_seed2 | 2 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_seed0 | 0 | linear | ce | 0.4824 | 0.4805 | 1.4377 | 0.0781 | 0.6571 | -0.2554 | 4.3237 |
| e2e_cifar10_linear_ce_seed1 | 1 | linear | ce | 0.4551 | 0.4824 | 1.4241 | 0.0536 | 0.6631 | -0.2137 | 3.7800 |
| e2e_cifar10_linear_ce_seed2 | 2 | linear | ce | 0.4766 | 0.4805 | 1.4686 | 0.0529 | 0.6679 | -0.2456 | 3.8893 |
| e2e_cifar10_mlp_ce_pairwise_seed0 | 0 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_pairwise_seed1 | 1 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_pairwise_seed2 | 2 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_seed0 | 0 | mlp | ce | 0.4062 | 0.3574 | 1.7038 | 0.0559 | 0.7418 | -0.6262 | 4.0748 |
| e2e_cifar10_mlp_ce_seed1 | 1 | mlp | ce | 0.4082 | 0.4238 | 1.6062 | 0.0528 | 0.7199 | -0.4896 | 3.9669 |
| e2e_cifar10_mlp_ce_seed2 | 2 | mlp | ce | 0.4492 | 0.4277 | 1.5279 | 0.0453 | 0.6991 | -0.4508 | 3.9980 |

## 6. CE-Only Comparison
### End-To-End CE Only
| run_id | seed | model | loss mode | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e_cifar10_cosine_prototype_ce_seed0 | 0 | cosine_prototype | ce | 0.4434 | 0.4121 | 1.6081 | 0.0620 | 0.6970 | -0.3566 | 4.7124 |
| e2e_cifar10_cosine_prototype_ce_seed1 | 1 | cosine_prototype | ce | 0.4590 | 0.4980 | 1.5086 | 0.0499 | 0.6781 | -0.2589 | 4.6884 |
| e2e_cifar10_cosine_prototype_ce_seed2 | 2 | cosine_prototype | ce | 0.4688 | 0.4727 | 1.5240 | 0.0621 | 0.6800 | -0.2663 | 4.6079 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0 | 0 | eml_bank_centered_ambiguity | ce | 0.4355 | 0.4062 | 1.6107 | 0.0854 | 0.7175 | -0.3653 | 7.3436 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1 | 1 | eml_bank_centered_ambiguity | ce | 0.3906 | 0.4355 | 1.7005 | 0.0681 | 0.7463 | -0.4441 | 7.2667 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed2 | 2 | eml_bank_centered_ambiguity | ce | 0.4453 | 0.4277 | 1.6214 | 0.1052 | 0.7162 | -0.3739 | 7.2526 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed0 | 0 | eml_centered_ambiguity | ce | 0.3691 | 0.3711 | 1.6556 | 0.0456 | 0.7361 | -0.4188 | 6.3703 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed1 | 1 | eml_centered_ambiguity | ce | 0.4062 | 0.4043 | 1.6860 | 0.0827 | 0.7393 | -0.4308 | 6.3644 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed2 | 2 | eml_centered_ambiguity | ce | 0.4805 | 0.4492 | 1.5825 | 0.1292 | 0.7021 | -0.3155 | 6.3350 |
| e2e_cifar10_eml_no_ambiguity_ce_seed0 | 0 | eml_no_ambiguity | ce | 0.3008 | 0.3008 | 1.7863 | 0.0283 | 0.7762 | -0.5609 | 6.1379 |
| e2e_cifar10_eml_no_ambiguity_ce_seed1 | 1 | eml_no_ambiguity | ce | 0.4160 | 0.4121 | 1.6819 | 0.0876 | 0.7401 | -0.4471 | 6.1857 |
| e2e_cifar10_eml_no_ambiguity_ce_seed2 | 2 | eml_no_ambiguity | ce | 0.4551 | 0.4277 | 1.5691 | 0.1122 | 0.6973 | -0.2903 | 6.1425 |
| e2e_cifar10_linear_ce_seed0 | 0 | linear | ce | 0.4824 | 0.4805 | 1.4377 | 0.0781 | 0.6571 | -0.2554 | 4.3237 |
| e2e_cifar10_linear_ce_seed1 | 1 | linear | ce | 0.4551 | 0.4824 | 1.4241 | 0.0536 | 0.6631 | -0.2137 | 3.7800 |
| e2e_cifar10_linear_ce_seed2 | 2 | linear | ce | 0.4766 | 0.4805 | 1.4686 | 0.0529 | 0.6679 | -0.2456 | 3.8893 |
| e2e_cifar10_mlp_ce_seed0 | 0 | mlp | ce | 0.4062 | 0.3574 | 1.7038 | 0.0559 | 0.7418 | -0.6262 | 4.0748 |
| e2e_cifar10_mlp_ce_seed1 | 1 | mlp | ce | 0.4082 | 0.4238 | 1.6062 | 0.0528 | 0.7199 | -0.4896 | 3.9669 |
| e2e_cifar10_mlp_ce_seed2 | 2 | mlp | ce | 0.4492 | 0.4277 | 1.5279 | 0.0453 | 0.6991 | -0.4508 | 3.9980 |

## 7. CE + Pairwise Comparison
### End-To-End CE + Prototype Pairwise
| run_id | seed | model | loss mode | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed0 | 0 | cosine_prototype | ce_pairwise | 0.4434 | 0.4121 | 1.6029 | 0.0740 | 0.6950 | -0.3479 | 4.6055 |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed1 | 1 | cosine_prototype | ce_pairwise | 0.4570 | 0.4883 | 1.5106 | 0.0646 | 0.6788 | -0.2630 | 4.6120 |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed2 | 2 | cosine_prototype | ce_pairwise | 0.4609 | 0.4727 | 1.5369 | 0.0597 | 0.6851 | -0.2835 | 4.6391 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_bank_centered_ambiguity | ce_pairwise | 0.3867 | 0.3691 | 1.7374 | 0.0363 | 0.7540 | -0.5470 | 7.1461 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_bank_centered_ambiguity | ce_pairwise | 0.4062 | 0.3789 | 1.7084 | 0.0904 | 0.7506 | -0.4477 | 7.3892 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed2 | 2 | eml_bank_centered_ambiguity | ce_pairwise | 0.4824 | 0.4277 | 1.5528 | 0.1340 | 0.6941 | -0.2791 | 7.2749 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_centered_ambiguity | ce_pairwise | 0.3379 | 0.3613 | 1.6543 | 0.0479 | 0.7398 | -0.4172 | 6.2512 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_centered_ambiguity | ce_pairwise | 0.4062 | 0.4062 | 1.6842 | 0.0636 | 0.7347 | -0.4461 | 6.3302 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed2 | 2 | eml_centered_ambiguity | ce_pairwise | 0.4277 | 0.4121 | 1.6611 | 0.0723 | 0.7273 | -0.4150 | 6.3032 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0 | 0 | eml_no_ambiguity | ce_pairwise | 0.3633 | 0.3711 | 1.6514 | 0.0362 | 0.7343 | -0.4384 | 5.9053 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1 | 1 | eml_no_ambiguity | ce_pairwise | 0.4355 | 0.4102 | 1.6471 | 0.1047 | 0.7282 | -0.4005 | 6.1300 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed2 | 2 | eml_no_ambiguity | ce_pairwise | 0.4746 | 0.4453 | 1.5485 | 0.1529 | 0.6964 | -0.2974 | 6.0611 |
| e2e_cifar10_linear_ce_pairwise_seed0 | 0 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_pairwise_seed1 | 1 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_pairwise_seed2 | 2 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_pairwise_seed0 | 0 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_pairwise_seed1 | 1 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_pairwise_seed2 | 2 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |

## 8. Calibration Metrics
ECE and Brier score are included in the result tables. Lower is better for both.

## 9. Hard-Negative Margin Analysis
Margin is positive-logit minus hardest-negative-logit; larger is better.

## 10. EML Drive/Resistance Analysis
| run_id | model | pos drive | hard neg drive | pos resistance | hard neg resistance | uncertainty | ambiguity | noise corr | occlusion corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0 | eml_bank_centered_ambiguity | 4.6422 | 6.5500 | 0.9993 | 0.9271 | 0.0013 | 1.1290 | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1 | eml_bank_centered_ambiguity | 5.0135 | 6.4252 | 1.1606 | 1.0568 | 0.0007 | 1.2261 | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed2 | eml_bank_centered_ambiguity | 5.4174 | 6.1566 | 0.9028 | 0.9935 | 0.0009 | 1.1391 | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0 | eml_bank_centered_ambiguity | 5.1026 | 6.2208 | 1.0296 | 1.0411 | 0.0012 | 1.2014 | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1 | eml_bank_centered_ambiguity | 5.1516 | 6.5807 | 1.1562 | 1.0698 | 0.0041 | 1.2434 | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed2 | eml_bank_centered_ambiguity | 5.0523 | 6.1495 | 1.0057 | 1.0119 | 0.0015 | 1.2035 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0 | eml_centered_ambiguity | 5.1832 | 6.5111 | 1.1495 | 1.0665 | 0.0003 | 1.1980 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1 | eml_centered_ambiguity | 4.7714 | 6.1669 | 0.9855 | 0.9193 | 0.0029 | 1.0667 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed2 | eml_centered_ambiguity | 4.8419 | 6.0207 | 0.9836 | 0.9475 | 0.0005 | 1.1038 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_seed0 | eml_centered_ambiguity | 4.9112 | 6.2108 | 1.1245 | 1.0417 | 0.0007 | 1.1825 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_seed1 | eml_centered_ambiguity | 5.1423 | 6.5342 | 1.0139 | 0.9784 | 0.0006 | 1.0963 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_seed2 | eml_centered_ambiguity | 5.3489 | 6.1948 | 0.9399 | 1.0145 | 0.0005 | 1.1108 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0 | eml_no_ambiguity | 5.0665 | 6.4450 | 0.9557 | 0.9551 | 0.7549 | 0.0000 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1 | eml_no_ambiguity | 5.4009 | 6.6209 | 0.9031 | 0.9033 | 0.7025 | 0.0000 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed2 | eml_no_ambiguity | 5.5928 | 6.4759 | 2.2481 | 2.2486 | 2.0475 | 0.0000 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_seed0 | eml_no_ambiguity | 4.4397 | 6.3589 | 0.8507 | 0.8501 | 0.6499 | 0.0000 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_seed1 | eml_no_ambiguity | 5.3413 | 6.8493 | 0.9366 | 0.9370 | 0.7360 | 0.0000 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_seed2 | eml_no_ambiguity | 5.2437 | 6.0219 | 3.3770 | 3.3776 | 3.1764 | 0.0000 | MISSING | MISSING |
| frozen_cifar10_eml_centered_ambiguity_seed0 | eml_centered_ambiguity | 4.9910 | 6.4988 | 0.6332 | 0.6146 | 0.2347 | 0.3758 | MISSING | MISSING |
| frozen_cifar10_eml_centered_ambiguity_seed1 | eml_centered_ambiguity | 4.8413 | 6.1555 | 0.7092 | 0.6790 | 0.3331 | 0.3321 | MISSING | MISSING |
| frozen_cifar10_eml_centered_ambiguity_seed2 | eml_centered_ambiguity | 5.7438 | 6.7855 | 0.7184 | 0.7523 | 0.5272 | 0.1378 | MISSING | MISSING |
| frozen_cifar10_eml_no_ambiguity_seed0 | eml_no_ambiguity | 5.0081 | 6.5237 | 2.7895 | 2.7894 | 2.5885 | 0.0000 | MISSING | MISSING |
| frozen_cifar10_eml_no_ambiguity_seed1 | eml_no_ambiguity | 4.8418 | 6.1625 | 10.9724 | 10.9728 | 10.7706 | 0.0000 | MISSING | MISSING |
| frozen_cifar10_eml_no_ambiguity_seed2 | eml_no_ambiguity | 5.7407 | 6.7898 | 5.8207 | 5.8213 | 5.6203 | 0.0000 | MISSING | MISSING |
| frozen_cifar10_eml_raw_ambiguity_seed0 | eml_raw_ambiguity | 4.9910 | 6.5013 | 2.5725 | 2.5539 | 0.0004 | 2.5486 | MISSING | MISSING |
| frozen_cifar10_eml_raw_ambiguity_seed1 | eml_raw_ambiguity | 4.8379 | 6.1548 | 2.5452 | 2.5147 | 0.0004 | 2.4995 | MISSING | MISSING |
| frozen_cifar10_eml_raw_ambiguity_seed2 | eml_raw_ambiguity | 5.7410 | 6.7847 | 2.3730 | 2.4060 | 0.0004 | 2.3184 | MISSING | MISSING |

## 11. Robustness Under Noise/Occlusion
Resistance-noise and resistance-occlusion correlations are reported when synthetic metadata is available. MISSING means the head did not expose resistance or the dataset did not provide the field.

## 12. Statistical Confidence Intervals
| experiment | dataset | seed | loss mode | comparison | delta acc | 95% CI low | 95% CI high |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| end_to_end | cifar10 | 0 | ce | eml_centered_ambiguity - linear | -0.1133 | -0.1660 | -0.0605 |
| end_to_end | cifar10 | 0 | ce | eml_centered_ambiguity - mlp | -0.0371 | -0.0879 | 0.0176 |
| end_to_end | cifar10 | 0 | ce | eml_centered_ambiguity - cosine_prototype | -0.0742 | -0.1270 | -0.0176 |
| end_to_end | cifar10 | 0 | ce_pairwise | eml_centered_ambiguity - cosine_prototype | -0.1055 | -0.1602 | -0.0527 |
| end_to_end | cifar10 | 1 | ce | eml_centered_ambiguity - linear | -0.0488 | -0.0977 | 0.0000 |
| end_to_end | cifar10 | 1 | ce | eml_centered_ambiguity - mlp | -0.0020 | -0.0449 | 0.0430 |
| end_to_end | cifar10 | 1 | ce | eml_centered_ambiguity - cosine_prototype | -0.0527 | -0.0977 | -0.0059 |
| end_to_end | cifar10 | 1 | ce_pairwise | eml_centered_ambiguity - cosine_prototype | -0.0508 | -0.0918 | -0.0097 |
| end_to_end | cifar10 | 2 | ce | eml_centered_ambiguity - linear | 0.0039 | -0.0371 | 0.0508 |
| end_to_end | cifar10 | 2 | ce | eml_centered_ambiguity - mlp | 0.0312 | -0.0137 | 0.0801 |
| end_to_end | cifar10 | 2 | ce | eml_centered_ambiguity - cosine_prototype | 0.0117 | -0.0372 | 0.0645 |
| end_to_end | cifar10 | 2 | ce_pairwise | eml_centered_ambiguity - cosine_prototype | -0.0332 | -0.0820 | 0.0137 |
| frozen_features | cifar10 | 0 | ce | eml_centered_ambiguity - linear | -0.0371 | -0.0742 | -0.0059 |
| frozen_features | cifar10 | 0 | ce | eml_centered_ambiguity - mlp | -0.0195 | -0.0586 | 0.0176 |
| frozen_features | cifar10 | 0 | ce | eml_centered_ambiguity - cosine_prototype | -0.0137 | -0.0527 | 0.0215 |
| frozen_features | cifar10 | 1 | ce | eml_centered_ambiguity - linear | -0.0156 | -0.0469 | 0.0117 |
| frozen_features | cifar10 | 1 | ce | eml_centered_ambiguity - mlp | -0.0117 | -0.0410 | 0.0156 |
| frozen_features | cifar10 | 1 | ce | eml_centered_ambiguity - cosine_prototype | -0.0273 | -0.0605 | 0.0000 |
| frozen_features | cifar10 | 2 | ce | eml_centered_ambiguity - linear | 0.0078 | -0.0195 | 0.0352 |
| frozen_features | cifar10 | 2 | ce | eml_centered_ambiguity - mlp | 0.0039 | -0.0273 | 0.0332 |
| frozen_features | cifar10 | 2 | ce | eml_centered_ambiguity - cosine_prototype | -0.0078 | -0.0371 | 0.0195 |

## 13. Which Claim Is Supported
The evidence is mixed: centered EML wins 5/21 paired comparisons.

## 14. Raw Artifacts
- `e2e_cifar10_cosine_prototype_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_082740_e2e_cifar10_cosine_prototype_ce_pairwise_seed0`
- `e2e_cifar10_cosine_prototype_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_082838_e2e_cifar10_cosine_prototype_ce_pairwise_seed1`
- `e2e_cifar10_cosine_prototype_ce_pairwise_seed2`: `reports/head_ablation/runs/20260424_082936_e2e_cifar10_cosine_prototype_ce_pairwise_seed2`
- `e2e_cifar10_cosine_prototype_ce_seed0`: `reports/head_ablation/runs/20260424_082736_e2e_cifar10_cosine_prototype_ce_seed0`
- `e2e_cifar10_cosine_prototype_ce_seed1`: `reports/head_ablation/runs/20260424_082833_e2e_cifar10_cosine_prototype_ce_seed1`
- `e2e_cifar10_cosine_prototype_ce_seed2`: `reports/head_ablation/runs/20260424_082931_e2e_cifar10_cosine_prototype_ce_seed2`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_082817_e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_082915_e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed2`: `reports/head_ablation/runs/20260424_083012_e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed2`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_082810_e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_082908_e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_seed2`: `reports/head_ablation/runs/20260424_083005_e2e_cifar10_eml_bank_centered_ambiguity_ce_seed2`
- `e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_082804_e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0`
- `e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_082901_e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1`
- `e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed2`: `reports/head_ablation/runs/20260424_082959_e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed2`
- `e2e_cifar10_eml_centered_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_082757_e2e_cifar10_eml_centered_ambiguity_ce_seed0`
- `e2e_cifar10_eml_centered_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_082855_e2e_cifar10_eml_centered_ambiguity_ce_seed1`
- `e2e_cifar10_eml_centered_ambiguity_ce_seed2`: `reports/head_ablation/runs/20260424_082953_e2e_cifar10_eml_centered_ambiguity_ce_seed2`
- `e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_082751_e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0`
- `e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_082849_e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1`
- `e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed2`: `reports/head_ablation/runs/20260424_082946_e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed2`
- `e2e_cifar10_eml_no_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_082745_e2e_cifar10_eml_no_ambiguity_ce_seed0`
- `e2e_cifar10_eml_no_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_082843_e2e_cifar10_eml_no_ambiguity_ce_seed1`
- `e2e_cifar10_eml_no_ambiguity_ce_seed2`: `reports/head_ablation/runs/20260424_082940_e2e_cifar10_eml_no_ambiguity_ce_seed2`
- `e2e_cifar10_linear_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_082732_e2e_cifar10_linear_ce_pairwise_seed0`
- `e2e_cifar10_linear_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_082829_e2e_cifar10_linear_ce_pairwise_seed1`
- `e2e_cifar10_linear_ce_pairwise_seed2`: `reports/head_ablation/runs/20260424_082927_e2e_cifar10_linear_ce_pairwise_seed2`
- `e2e_cifar10_linear_ce_seed0`: `reports/head_ablation/runs/20260424_082727_e2e_cifar10_linear_ce_seed0`
- `e2e_cifar10_linear_ce_seed1`: `reports/head_ablation/runs/20260424_082825_e2e_cifar10_linear_ce_seed1`
- `e2e_cifar10_linear_ce_seed2`: `reports/head_ablation/runs/20260424_082923_e2e_cifar10_linear_ce_seed2`
- `e2e_cifar10_mlp_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_082736_e2e_cifar10_mlp_ce_pairwise_seed0`
- `e2e_cifar10_mlp_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_082833_e2e_cifar10_mlp_ce_pairwise_seed1`
- `e2e_cifar10_mlp_ce_pairwise_seed2`: `reports/head_ablation/runs/20260424_082931_e2e_cifar10_mlp_ce_pairwise_seed2`
- `e2e_cifar10_mlp_ce_seed0`: `reports/head_ablation/runs/20260424_082732_e2e_cifar10_mlp_ce_seed0`
- `e2e_cifar10_mlp_ce_seed1`: `reports/head_ablation/runs/20260424_082829_e2e_cifar10_mlp_ce_seed1`
- `e2e_cifar10_mlp_ce_seed2`: `reports/head_ablation/runs/20260424_082927_e2e_cifar10_mlp_ce_seed2`
- `frozen_cifar10_cosine_prototype_seed0`: `reports/head_ablation/runs/20260424_082642_frozen_cifar10_cosine_prototype_seed0`
- `frozen_cifar10_cosine_prototype_seed1`: `reports/head_ablation/runs/20260424_082658_frozen_cifar10_cosine_prototype_seed1`
- `frozen_cifar10_cosine_prototype_seed2`: `reports/head_ablation/runs/20260424_082712_frozen_cifar10_cosine_prototype_seed2`
- `frozen_cifar10_eml_centered_ambiguity_seed0`: `reports/head_ablation/runs/20260424_082650_frozen_cifar10_eml_centered_ambiguity_seed0`
- `frozen_cifar10_eml_centered_ambiguity_seed1`: `reports/head_ablation/runs/20260424_082705_frozen_cifar10_eml_centered_ambiguity_seed1`
- `frozen_cifar10_eml_centered_ambiguity_seed2`: `reports/head_ablation/runs/20260424_082720_frozen_cifar10_eml_centered_ambiguity_seed2`
- `frozen_cifar10_eml_no_ambiguity_seed0`: `reports/head_ablation/runs/20260424_082644_frozen_cifar10_eml_no_ambiguity_seed0`
- `frozen_cifar10_eml_no_ambiguity_seed1`: `reports/head_ablation/runs/20260424_082659_frozen_cifar10_eml_no_ambiguity_seed1`
- `frozen_cifar10_eml_no_ambiguity_seed2`: `reports/head_ablation/runs/20260424_082713_frozen_cifar10_eml_no_ambiguity_seed2`
- `frozen_cifar10_eml_raw_ambiguity_seed0`: `reports/head_ablation/runs/20260424_082647_frozen_cifar10_eml_raw_ambiguity_seed0`
- `frozen_cifar10_eml_raw_ambiguity_seed1`: `reports/head_ablation/runs/20260424_082702_frozen_cifar10_eml_raw_ambiguity_seed1`
- `frozen_cifar10_eml_raw_ambiguity_seed2`: `reports/head_ablation/runs/20260424_082717_frozen_cifar10_eml_raw_ambiguity_seed2`
- `frozen_cifar10_linear_seed0`: `reports/head_ablation/runs/20260424_082640_frozen_cifar10_linear_seed0`
- `frozen_cifar10_linear_seed1`: `reports/head_ablation/runs/20260424_082656_frozen_cifar10_linear_seed1`
- `frozen_cifar10_linear_seed2`: `reports/head_ablation/runs/20260424_082710_frozen_cifar10_linear_seed2`
- `frozen_cifar10_mlp_seed0`: `reports/head_ablation/runs/20260424_082641_frozen_cifar10_mlp_seed0`
- `frozen_cifar10_mlp_seed1`: `reports/head_ablation/runs/20260424_082657_frozen_cifar10_mlp_seed1`
- `frozen_cifar10_mlp_seed2`: `reports/head_ablation/runs/20260424_082711_frozen_cifar10_mlp_seed2`

## 15. Appendix: Commands
- `pytest`
- `python scripts/run_head_ablation.py --dataset synthetic_shape --mode smoke --seeds 0 1`
- `python scripts/run_cnn_head_end_to_end_ablation.py --dataset synthetic_shape --mode smoke --seeds 0 1`
- `python scripts/generate_head_ablation_report.py`
