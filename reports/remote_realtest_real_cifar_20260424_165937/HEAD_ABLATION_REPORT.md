# CNN Head Ablation Report

## 1. Executive Summary
- Completed runs: 48
- NOT RUN entries: 6
- Failed runs: 0
- Best frozen-feature result: mlp seed=0 test_accuracy=0.5391
- Best end-to-end result: cosine_prototype seed=2 test_accuracy=0.5645
- Claim status: The evidence is mixed: centered EML wins 4/21 paired comparisons.

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
| frozen_cifar10_cosine_prototype_seed0 | 0 | cosine_prototype | 0.5176 | 0.5234 | 1.3887 | 0.0919 | 0.6306 | -0.1395 | 3.3264 |
| frozen_cifar10_cosine_prototype_seed1 | 1 | cosine_prototype | 0.4961 | 0.5117 | 1.3982 | 0.0695 | 0.6373 | -0.0608 | 3.0186 |
| frozen_cifar10_cosine_prototype_seed2 | 2 | cosine_prototype | 0.5098 | 0.5586 | 1.3632 | 0.0771 | 0.6241 | -0.0076 | 2.9071 |
| frozen_cifar10_eml_centered_ambiguity_seed0 | 0 | eml_centered_ambiguity | 0.5137 | 0.5410 | 1.4748 | 0.0846 | 0.6464 | -0.1766 | 6.3235 |
| frozen_cifar10_eml_centered_ambiguity_seed1 | 1 | eml_centered_ambiguity | 0.4902 | 0.5156 | 1.4784 | 0.1009 | 0.6512 | -0.1577 | 6.4943 |
| frozen_cifar10_eml_centered_ambiguity_seed2 | 2 | eml_centered_ambiguity | 0.5312 | 0.5664 | 1.4519 | 0.0898 | 0.6359 | -0.0555 | 17.1657 |
| frozen_cifar10_eml_no_ambiguity_seed0 | 0 | eml_no_ambiguity | 0.5137 | 0.5371 | 1.4781 | 0.0690 | 0.6479 | -0.1799 | 6.4247 |
| frozen_cifar10_eml_no_ambiguity_seed1 | 1 | eml_no_ambiguity | 0.4922 | 0.5156 | 1.4822 | 0.1039 | 0.6524 | -0.1594 | 6.2572 |
| frozen_cifar10_eml_no_ambiguity_seed2 | 2 | eml_no_ambiguity | 0.5273 | 0.5664 | 1.4547 | 0.0930 | 0.6368 | -0.0573 | 6.0660 |
| frozen_cifar10_eml_raw_ambiguity_seed0 | 0 | eml_raw_ambiguity | 0.5137 | 0.5410 | 1.4757 | 0.0811 | 0.6468 | -0.1769 | 6.2913 |
| frozen_cifar10_eml_raw_ambiguity_seed1 | 1 | eml_raw_ambiguity | 0.4922 | 0.5176 | 1.4793 | 0.1021 | 0.6515 | -0.1576 | 6.4253 |
| frozen_cifar10_eml_raw_ambiguity_seed2 | 2 | eml_raw_ambiguity | 0.5273 | 0.5664 | 1.4528 | 0.0878 | 0.6362 | -0.0553 | 6.2351 |
| frozen_cifar10_linear_seed0 | 0 | linear | 0.5312 | 0.5391 | 1.3404 | 0.0662 | 0.6125 | -0.1184 | 2.0372 |
| frozen_cifar10_linear_seed1 | 1 | linear | 0.4922 | 0.5137 | 1.3546 | 0.0462 | 0.6280 | -0.0329 | 1.7024 |
| frozen_cifar10_linear_seed2 | 2 | linear | 0.5156 | 0.5547 | 1.3317 | 0.0597 | 0.6167 | 0.0099 | 1.7298 |
| frozen_cifar10_mlp_seed0 | 0 | mlp | 0.5391 | 0.5469 | 1.3873 | 0.1239 | 0.6263 | -0.0012 | 1.8988 |
| frozen_cifar10_mlp_seed1 | 1 | mlp | 0.5117 | 0.5391 | 1.3843 | 0.1099 | 0.6375 | -0.0691 | 1.8084 |
| frozen_cifar10_mlp_seed2 | 2 | mlp | 0.5332 | 0.5664 | 1.3793 | 0.0920 | 0.6191 | 0.0506 | 1.8425 |

## 5. End-To-End Results
### CNN Plus Head
| run_id | seed | model | loss mode | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed0 | 0 | cosine_prototype | ce_pairwise | 0.5410 | 0.5293 | 1.3448 | 0.0740 | 0.6018 | 0.0129 | 7.8820 |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed1 | 1 | cosine_prototype | ce_pairwise | 0.5312 | 0.5781 | 1.3184 | 0.0543 | 0.6059 | -0.0096 | 7.6707 |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed2 | 2 | cosine_prototype | ce_pairwise | 0.5645 | 0.5098 | 1.3460 | 0.0520 | 0.5923 | 0.0730 | 10.0450 |
| e2e_cifar10_cosine_prototype_ce_seed0 | 0 | cosine_prototype | ce | 0.5566 | 0.5352 | 1.3300 | 0.0664 | 0.5976 | 0.0267 | 7.5332 |
| e2e_cifar10_cosine_prototype_ce_seed1 | 1 | cosine_prototype | ce | 0.5352 | 0.5703 | 1.3336 | 0.0504 | 0.6111 | -0.0358 | 7.7522 |
| e2e_cifar10_cosine_prototype_ce_seed2 | 2 | cosine_prototype | ce | 0.5605 | 0.5215 | 1.3611 | 0.0688 | 0.5949 | 0.0643 | 7.6841 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_bank_centered_ambiguity | ce_pairwise | 0.4531 | 0.4277 | 1.6331 | 0.0819 | 0.7141 | -0.4492 | 13.0377 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_bank_centered_ambiguity | ce_pairwise | 0.4551 | 0.4473 | 1.5998 | 0.0713 | 0.7080 | -0.4122 | 13.2170 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed2 | 2 | eml_bank_centered_ambiguity | ce_pairwise | 0.3984 | 0.3652 | 1.7195 | 0.0519 | 0.7385 | -0.5070 | 12.4904 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0 | 0 | eml_bank_centered_ambiguity | ce | 0.3926 | 0.4043 | 1.6488 | 0.0271 | 0.7262 | -0.4621 | 12.8159 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1 | 1 | eml_bank_centered_ambiguity | ce | 0.4473 | 0.4648 | 1.5646 | 0.0952 | 0.6941 | -0.3680 | 13.1782 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed2 | 2 | eml_bank_centered_ambiguity | ce | 0.4570 | 0.3906 | 1.6226 | 0.0847 | 0.7018 | -0.4239 | 12.6022 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_centered_ambiguity | ce_pairwise | 0.4180 | 0.4004 | 1.6617 | 0.0939 | 0.7272 | -0.5059 | 11.6131 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_centered_ambiguity | ce_pairwise | 0.3965 | 0.4453 | 1.6173 | 0.0552 | 0.7228 | -0.4650 | 11.6758 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed2 | 2 | eml_centered_ambiguity | ce_pairwise | 0.4688 | 0.4453 | 1.5998 | 0.0680 | 0.6947 | -0.3642 | 11.1575 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed0 | 0 | eml_centered_ambiguity | ce | 0.4512 | 0.4277 | 1.6058 | 0.0681 | 0.7038 | -0.4335 | 11.4866 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed1 | 1 | eml_centered_ambiguity | ce | 0.3965 | 0.4258 | 1.6393 | 0.0295 | 0.7344 | -0.5055 | 11.2242 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed2 | 2 | eml_centered_ambiguity | ce | 0.4570 | 0.4258 | 1.5508 | 0.0542 | 0.6901 | -0.3584 | 11.1593 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0 | 0 | eml_no_ambiguity | ce_pairwise | 0.3965 | 0.3770 | 1.6484 | 0.0447 | 0.7276 | -0.4785 | 10.9873 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1 | 1 | eml_no_ambiguity | ce_pairwise | 0.4062 | 0.4199 | 1.6500 | 0.0517 | 0.7279 | -0.4793 | 11.2746 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed2 | 2 | eml_no_ambiguity | ce_pairwise | 0.4375 | 0.4180 | 1.6563 | 0.0818 | 0.7282 | -0.4797 | 10.9803 |
| e2e_cifar10_eml_no_ambiguity_ce_seed0 | 0 | eml_no_ambiguity | ce | 0.3477 | 0.3418 | 1.7461 | 0.0371 | 0.7599 | -0.5888 | 15.7532 |
| e2e_cifar10_eml_no_ambiguity_ce_seed1 | 1 | eml_no_ambiguity | ce | 0.4648 | 0.4316 | 1.5360 | 0.1098 | 0.6920 | -0.3724 | 11.3860 |
| e2e_cifar10_eml_no_ambiguity_ce_seed2 | 2 | eml_no_ambiguity | ce | 0.4883 | 0.4492 | 1.5179 | 0.0995 | 0.6771 | -0.3125 | 15.0676 |
| e2e_cifar10_linear_ce_pairwise_seed0 | 0 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_pairwise_seed1 | 1 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_pairwise_seed2 | 2 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_seed0 | 0 | linear | ce | 0.5156 | 0.5254 | 1.3148 | 0.0757 | 0.6075 | -0.0467 | 8.9493 |
| e2e_cifar10_linear_ce_seed1 | 1 | linear | ce | 0.5352 | 0.5469 | 1.3674 | 0.0622 | 0.6235 | -0.0884 | 6.4905 |
| e2e_cifar10_linear_ce_seed2 | 2 | linear | ce | 0.4531 | 0.4824 | 1.5549 | 0.0956 | 0.6863 | -0.3614 | 6.5023 |
| e2e_cifar10_mlp_ce_pairwise_seed0 | 0 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_pairwise_seed1 | 1 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_pairwise_seed2 | 2 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_seed0 | 0 | mlp | ce | 0.4453 | 0.4199 | 1.5127 | 0.0752 | 0.6719 | -0.3365 | 6.3922 |
| e2e_cifar10_mlp_ce_seed1 | 1 | mlp | ce | 0.4395 | 0.4785 | 1.5354 | 0.1070 | 0.6906 | -0.4441 | 6.3846 |
| e2e_cifar10_mlp_ce_seed2 | 2 | mlp | ce | 0.4629 | 0.4336 | 1.5345 | 0.0826 | 0.6695 | -0.3313 | 6.5776 |

## 6. CE-Only Comparison
### End-To-End CE Only
| run_id | seed | model | loss mode | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e_cifar10_cosine_prototype_ce_seed0 | 0 | cosine_prototype | ce | 0.5566 | 0.5352 | 1.3300 | 0.0664 | 0.5976 | 0.0267 | 7.5332 |
| e2e_cifar10_cosine_prototype_ce_seed1 | 1 | cosine_prototype | ce | 0.5352 | 0.5703 | 1.3336 | 0.0504 | 0.6111 | -0.0358 | 7.7522 |
| e2e_cifar10_cosine_prototype_ce_seed2 | 2 | cosine_prototype | ce | 0.5605 | 0.5215 | 1.3611 | 0.0688 | 0.5949 | 0.0643 | 7.6841 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0 | 0 | eml_bank_centered_ambiguity | ce | 0.3926 | 0.4043 | 1.6488 | 0.0271 | 0.7262 | -0.4621 | 12.8159 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1 | 1 | eml_bank_centered_ambiguity | ce | 0.4473 | 0.4648 | 1.5646 | 0.0952 | 0.6941 | -0.3680 | 13.1782 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed2 | 2 | eml_bank_centered_ambiguity | ce | 0.4570 | 0.3906 | 1.6226 | 0.0847 | 0.7018 | -0.4239 | 12.6022 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed0 | 0 | eml_centered_ambiguity | ce | 0.4512 | 0.4277 | 1.6058 | 0.0681 | 0.7038 | -0.4335 | 11.4866 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed1 | 1 | eml_centered_ambiguity | ce | 0.3965 | 0.4258 | 1.6393 | 0.0295 | 0.7344 | -0.5055 | 11.2242 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed2 | 2 | eml_centered_ambiguity | ce | 0.4570 | 0.4258 | 1.5508 | 0.0542 | 0.6901 | -0.3584 | 11.1593 |
| e2e_cifar10_eml_no_ambiguity_ce_seed0 | 0 | eml_no_ambiguity | ce | 0.3477 | 0.3418 | 1.7461 | 0.0371 | 0.7599 | -0.5888 | 15.7532 |
| e2e_cifar10_eml_no_ambiguity_ce_seed1 | 1 | eml_no_ambiguity | ce | 0.4648 | 0.4316 | 1.5360 | 0.1098 | 0.6920 | -0.3724 | 11.3860 |
| e2e_cifar10_eml_no_ambiguity_ce_seed2 | 2 | eml_no_ambiguity | ce | 0.4883 | 0.4492 | 1.5179 | 0.0995 | 0.6771 | -0.3125 | 15.0676 |
| e2e_cifar10_linear_ce_seed0 | 0 | linear | ce | 0.5156 | 0.5254 | 1.3148 | 0.0757 | 0.6075 | -0.0467 | 8.9493 |
| e2e_cifar10_linear_ce_seed1 | 1 | linear | ce | 0.5352 | 0.5469 | 1.3674 | 0.0622 | 0.6235 | -0.0884 | 6.4905 |
| e2e_cifar10_linear_ce_seed2 | 2 | linear | ce | 0.4531 | 0.4824 | 1.5549 | 0.0956 | 0.6863 | -0.3614 | 6.5023 |
| e2e_cifar10_mlp_ce_seed0 | 0 | mlp | ce | 0.4453 | 0.4199 | 1.5127 | 0.0752 | 0.6719 | -0.3365 | 6.3922 |
| e2e_cifar10_mlp_ce_seed1 | 1 | mlp | ce | 0.4395 | 0.4785 | 1.5354 | 0.1070 | 0.6906 | -0.4441 | 6.3846 |
| e2e_cifar10_mlp_ce_seed2 | 2 | mlp | ce | 0.4629 | 0.4336 | 1.5345 | 0.0826 | 0.6695 | -0.3313 | 6.5776 |

## 7. CE + Pairwise Comparison
### End-To-End CE + Prototype Pairwise
| run_id | seed | model | loss mode | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed0 | 0 | cosine_prototype | ce_pairwise | 0.5410 | 0.5293 | 1.3448 | 0.0740 | 0.6018 | 0.0129 | 7.8820 |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed1 | 1 | cosine_prototype | ce_pairwise | 0.5312 | 0.5781 | 1.3184 | 0.0543 | 0.6059 | -0.0096 | 7.6707 |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed2 | 2 | cosine_prototype | ce_pairwise | 0.5645 | 0.5098 | 1.3460 | 0.0520 | 0.5923 | 0.0730 | 10.0450 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_bank_centered_ambiguity | ce_pairwise | 0.4531 | 0.4277 | 1.6331 | 0.0819 | 0.7141 | -0.4492 | 13.0377 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_bank_centered_ambiguity | ce_pairwise | 0.4551 | 0.4473 | 1.5998 | 0.0713 | 0.7080 | -0.4122 | 13.2170 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed2 | 2 | eml_bank_centered_ambiguity | ce_pairwise | 0.3984 | 0.3652 | 1.7195 | 0.0519 | 0.7385 | -0.5070 | 12.4904 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_centered_ambiguity | ce_pairwise | 0.4180 | 0.4004 | 1.6617 | 0.0939 | 0.7272 | -0.5059 | 11.6131 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_centered_ambiguity | ce_pairwise | 0.3965 | 0.4453 | 1.6173 | 0.0552 | 0.7228 | -0.4650 | 11.6758 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed2 | 2 | eml_centered_ambiguity | ce_pairwise | 0.4688 | 0.4453 | 1.5998 | 0.0680 | 0.6947 | -0.3642 | 11.1575 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0 | 0 | eml_no_ambiguity | ce_pairwise | 0.3965 | 0.3770 | 1.6484 | 0.0447 | 0.7276 | -0.4785 | 10.9873 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1 | 1 | eml_no_ambiguity | ce_pairwise | 0.4062 | 0.4199 | 1.6500 | 0.0517 | 0.7279 | -0.4793 | 11.2746 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed2 | 2 | eml_no_ambiguity | ce_pairwise | 0.4375 | 0.4180 | 1.6563 | 0.0818 | 0.7282 | -0.4797 | 10.9803 |
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
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0 | eml_bank_centered_ambiguity | 5.0573 | 6.2071 | 0.9051 | 0.9154 | 0.0038 | 1.0796 | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1 | eml_bank_centered_ambiguity | 5.2133 | 6.2037 | 0.9250 | 0.9561 | 0.0004 | 1.0582 | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed2 | eml_bank_centered_ambiguity | 4.4301 | 5.9489 | 0.9941 | 0.9240 | 0.0032 | 1.1121 | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0 | eml_bank_centered_ambiguity | 4.7834 | 6.0212 | 0.9555 | 0.9565 | 0.0008 | 1.0976 | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1 | eml_bank_centered_ambiguity | 5.3440 | 6.2263 | 0.9332 | 0.9948 | 0.0002 | 1.1126 | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed2 | eml_bank_centered_ambiguity | 4.9968 | 6.1746 | 0.9372 | 0.9964 | 0.0136 | 1.1303 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0 | eml_centered_ambiguity | 5.1665 | 6.7037 | 0.9682 | 0.9552 | 0.0003 | 1.0661 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1 | eml_centered_ambiguity | 4.7966 | 6.0943 | 0.9012 | 0.8342 | 0.0012 | 0.9725 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed2 | eml_centered_ambiguity | 4.7512 | 5.6197 | 0.7598 | 0.8408 | 0.0028 | 0.9198 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_seed0 | eml_centered_ambiguity | 5.2000 | 6.4150 | 0.9574 | 0.9951 | 0.0001 | 1.1093 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_seed1 | eml_centered_ambiguity | 5.0249 | 6.4541 | 0.9715 | 0.8738 | 0.0005 | 1.0368 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_seed2 | eml_centered_ambiguity | 5.0563 | 5.9012 | 0.8379 | 0.8992 | 0.0017 | 1.0081 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0 | eml_no_ambiguity | 5.0707 | 6.4483 | 13.8149 | 13.8155 | 13.6136 | 0.0000 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1 | eml_no_ambiguity | 5.1708 | 6.6546 | 1.6668 | 1.6680 | 1.4656 | 0.0000 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed2 | eml_no_ambiguity | 4.7235 | 6.0304 | 12.4357 | 12.4366 | 12.2344 | 0.0000 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_seed0 | eml_no_ambiguity | 4.8909 | 6.7499 | 18.5949 | 18.5956 | 18.3935 | 0.0000 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_seed1 | eml_no_ambiguity | 5.7786 | 6.7242 | 1.7087 | 1.7097 | 1.5076 | 0.0000 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_seed2 | eml_no_ambiguity | 5.4915 | 6.2398 | 10.7705 | 10.7712 | 10.5691 | 0.0000 | MISSING | MISSING |
| frozen_cifar10_eml_centered_ambiguity_seed0 | eml_centered_ambiguity | 5.2465 | 5.5352 | 0.6778 | 0.7665 | 0.5562 | 0.1450 | MISSING | MISSING |
| frozen_cifar10_eml_centered_ambiguity_seed1 | eml_centered_ambiguity | 5.7124 | 5.8198 | 0.6917 | 0.7948 | 0.3878 | 0.3652 | MISSING | MISSING |
| frozen_cifar10_eml_centered_ambiguity_seed2 | eml_centered_ambiguity | 5.6274 | 5.5220 | 0.6917 | 0.7642 | 0.5552 | 0.1262 | MISSING | MISSING |
| frozen_cifar10_eml_no_ambiguity_seed0 | eml_no_ambiguity | 5.2420 | 5.5350 | 10.9031 | 10.9027 | 10.6993 | 0.0000 | MISSING | MISSING |
| frozen_cifar10_eml_no_ambiguity_seed1 | eml_no_ambiguity | 5.6662 | 5.7689 | 17.5304 | 17.5339 | 17.3255 | 0.0000 | MISSING | MISSING |
| frozen_cifar10_eml_no_ambiguity_seed2 | eml_no_ambiguity | 5.5992 | 5.5022 | 22.8380 | 22.8413 | 22.6341 | 0.0000 | MISSING | MISSING |
| frozen_cifar10_eml_raw_ambiguity_seed0 | eml_raw_ambiguity | 5.2400 | 5.5330 | 2.2648 | 2.3500 | 0.0002 | 2.2844 | MISSING | MISSING |
| frozen_cifar10_eml_raw_ambiguity_seed1 | eml_raw_ambiguity | 5.7019 | 5.8065 | 2.4265 | 2.5260 | 0.0001 | 2.4821 | MISSING | MISSING |
| frozen_cifar10_eml_raw_ambiguity_seed2 | eml_raw_ambiguity | 5.6189 | 5.5131 | 2.2867 | 2.3572 | 0.0003 | 2.2743 | MISSING | MISSING |

## 11. Robustness Under Noise/Occlusion
Resistance-noise and resistance-occlusion correlations are reported when synthetic metadata is available. MISSING means the head did not expose resistance or the dataset did not provide the field.

## 12. Statistical Confidence Intervals
| experiment | dataset | seed | loss mode | comparison | delta acc | 95% CI low | 95% CI high |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| end_to_end | cifar10 | 0 | ce | eml_centered_ambiguity - linear | -0.0645 | -0.1133 | -0.0117 |
| end_to_end | cifar10 | 0 | ce | eml_centered_ambiguity - mlp | 0.0059 | -0.0488 | 0.0605 |
| end_to_end | cifar10 | 0 | ce | eml_centered_ambiguity - cosine_prototype | -0.1055 | -0.1582 | -0.0547 |
| end_to_end | cifar10 | 0 | ce_pairwise | eml_centered_ambiguity - cosine_prototype | -0.1230 | -0.1719 | -0.0723 |
| end_to_end | cifar10 | 1 | ce | eml_centered_ambiguity - linear | -0.1387 | -0.1816 | -0.0938 |
| end_to_end | cifar10 | 1 | ce | eml_centered_ambiguity - mlp | -0.0430 | -0.0859 | 0.0020 |
| end_to_end | cifar10 | 1 | ce | eml_centered_ambiguity - cosine_prototype | -0.1387 | -0.1855 | -0.0938 |
| end_to_end | cifar10 | 1 | ce_pairwise | eml_centered_ambiguity - cosine_prototype | -0.1348 | -0.1816 | -0.0878 |
| end_to_end | cifar10 | 2 | ce | eml_centered_ambiguity - linear | 0.0039 | -0.0488 | 0.0566 |
| end_to_end | cifar10 | 2 | ce | eml_centered_ambiguity - mlp | -0.0059 | -0.0489 | 0.0430 |
| end_to_end | cifar10 | 2 | ce | eml_centered_ambiguity - cosine_prototype | -0.1035 | -0.1465 | -0.0586 |
| end_to_end | cifar10 | 2 | ce_pairwise | eml_centered_ambiguity - cosine_prototype | -0.0957 | -0.1387 | -0.0527 |
| frozen_features | cifar10 | 0 | ce | eml_centered_ambiguity - linear | -0.0176 | -0.0410 | 0.0078 |
| frozen_features | cifar10 | 0 | ce | eml_centered_ambiguity - mlp | -0.0254 | -0.0508 | 0.0020 |
| frozen_features | cifar10 | 0 | ce | eml_centered_ambiguity - cosine_prototype | -0.0039 | -0.0273 | 0.0234 |
| frozen_features | cifar10 | 1 | ce | eml_centered_ambiguity - linear | -0.0020 | -0.0293 | 0.0234 |
| frozen_features | cifar10 | 1 | ce | eml_centered_ambiguity - mlp | -0.0215 | -0.0449 | 0.0000 |
| frozen_features | cifar10 | 1 | ce | eml_centered_ambiguity - cosine_prototype | -0.0059 | -0.0312 | 0.0176 |
| frozen_features | cifar10 | 2 | ce | eml_centered_ambiguity - linear | 0.0156 | -0.0117 | 0.0430 |
| frozen_features | cifar10 | 2 | ce | eml_centered_ambiguity - mlp | -0.0020 | -0.0293 | 0.0254 |
| frozen_features | cifar10 | 2 | ce | eml_centered_ambiguity - cosine_prototype | 0.0215 | -0.0059 | 0.0508 |

## 13. Which Claim Is Supported
The evidence is mixed: centered EML wins 4/21 paired comparisons.

## 14. Raw Artifacts
- `e2e_cifar10_cosine_prototype_ce_pairwise_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_090155_e2e_cifar10_cosine_prototype_ce_pairwise_seed0`
- `e2e_cifar10_cosine_prototype_ce_pairwise_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090343_e2e_cifar10_cosine_prototype_ce_pairwise_seed1`
- `e2e_cifar10_cosine_prototype_ce_pairwise_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090531_e2e_cifar10_cosine_prototype_ce_pairwise_seed2`
- `e2e_cifar10_cosine_prototype_ce_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_090148_e2e_cifar10_cosine_prototype_ce_seed0`
- `e2e_cifar10_cosine_prototype_ce_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090336_e2e_cifar10_cosine_prototype_ce_seed1`
- `e2e_cifar10_cosine_prototype_ce_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090523_e2e_cifar10_cosine_prototype_ce_seed2`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_090306_e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090450_e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090642_e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed2`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_090253_e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090437_e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090629_e2e_cifar10_eml_bank_centered_ambiguity_ce_seed2`
- `e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_090242_e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0`
- `e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090425_e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1`
- `e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090618_e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed2`
- `e2e_cifar10_eml_centered_ambiguity_ce_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_090230_e2e_cifar10_eml_centered_ambiguity_ce_seed0`
- `e2e_cifar10_eml_centered_ambiguity_ce_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090414_e2e_cifar10_eml_centered_ambiguity_ce_seed1`
- `e2e_cifar10_eml_centered_ambiguity_ce_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090607_e2e_cifar10_eml_centered_ambiguity_ce_seed2`
- `e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_090219_e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0`
- `e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090403_e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1`
- `e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090556_e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed2`
- `e2e_cifar10_eml_no_ambiguity_ce_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_090203_e2e_cifar10_eml_no_ambiguity_ce_seed0`
- `e2e_cifar10_eml_no_ambiguity_ce_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090351_e2e_cifar10_eml_no_ambiguity_ce_seed1`
- `e2e_cifar10_eml_no_ambiguity_ce_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090541_e2e_cifar10_eml_no_ambiguity_ce_seed2`
- `e2e_cifar10_linear_ce_pairwise_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_090141_e2e_cifar10_linear_ce_pairwise_seed0`
- `e2e_cifar10_linear_ce_pairwise_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090329_e2e_cifar10_linear_ce_pairwise_seed1`
- `e2e_cifar10_linear_ce_pairwise_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090516_e2e_cifar10_linear_ce_pairwise_seed2`
- `e2e_cifar10_linear_ce_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_090132_e2e_cifar10_linear_ce_seed0`
- `e2e_cifar10_linear_ce_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090323_e2e_cifar10_linear_ce_seed1`
- `e2e_cifar10_linear_ce_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090510_e2e_cifar10_linear_ce_seed2`
- `e2e_cifar10_mlp_ce_pairwise_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_090148_e2e_cifar10_mlp_ce_pairwise_seed0`
- `e2e_cifar10_mlp_ce_pairwise_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090336_e2e_cifar10_mlp_ce_pairwise_seed1`
- `e2e_cifar10_mlp_ce_pairwise_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090523_e2e_cifar10_mlp_ce_pairwise_seed2`
- `e2e_cifar10_mlp_ce_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_090141_e2e_cifar10_mlp_ce_seed0`
- `e2e_cifar10_mlp_ce_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090329_e2e_cifar10_mlp_ce_seed1`
- `e2e_cifar10_mlp_ce_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090516_e2e_cifar10_mlp_ce_seed2`
- `frozen_cifar10_cosine_prototype_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_085950_frozen_cifar10_cosine_prototype_seed0`
- `frozen_cifar10_cosine_prototype_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090020_frozen_cifar10_cosine_prototype_seed1`
- `frozen_cifar10_cosine_prototype_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090051_frozen_cifar10_cosine_prototype_seed2`
- `frozen_cifar10_eml_centered_ambiguity_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_090006_frozen_cifar10_eml_centered_ambiguity_seed0`
- `frozen_cifar10_eml_centered_ambiguity_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090036_frozen_cifar10_eml_centered_ambiguity_seed1`
- `frozen_cifar10_eml_centered_ambiguity_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090107_frozen_cifar10_eml_centered_ambiguity_seed2`
- `frozen_cifar10_eml_no_ambiguity_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_085954_frozen_cifar10_eml_no_ambiguity_seed0`
- `frozen_cifar10_eml_no_ambiguity_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090024_frozen_cifar10_eml_no_ambiguity_seed1`
- `frozen_cifar10_eml_no_ambiguity_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090055_frozen_cifar10_eml_no_ambiguity_seed2`
- `frozen_cifar10_eml_raw_ambiguity_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_090000_frozen_cifar10_eml_raw_ambiguity_seed0`
- `frozen_cifar10_eml_raw_ambiguity_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090030_frozen_cifar10_eml_raw_ambiguity_seed1`
- `frozen_cifar10_eml_raw_ambiguity_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090101_frozen_cifar10_eml_raw_ambiguity_seed2`
- `frozen_cifar10_linear_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_085946_frozen_cifar10_linear_seed0`
- `frozen_cifar10_linear_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090017_frozen_cifar10_linear_seed1`
- `frozen_cifar10_linear_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090048_frozen_cifar10_linear_seed2`
- `frozen_cifar10_mlp_seed0`: `reports/real_cifar_20260424_165937/runs/20260424_085948_frozen_cifar10_mlp_seed0`
- `frozen_cifar10_mlp_seed1`: `reports/real_cifar_20260424_165937/runs/20260424_090019_frozen_cifar10_mlp_seed1`
- `frozen_cifar10_mlp_seed2`: `reports/real_cifar_20260424_165937/runs/20260424_090049_frozen_cifar10_mlp_seed2`

## 15. Appendix: Commands
- `pytest`
- `python scripts/run_head_ablation.py --dataset synthetic_shape --mode smoke --seeds 0 1`
- `python scripts/run_cnn_head_end_to_end_ablation.py --dataset synthetic_shape --mode smoke --seeds 0 1`
- `python scripts/generate_head_ablation_report.py`
