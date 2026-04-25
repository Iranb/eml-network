# CNN Head Ablation Report

## 1. Executive Summary
- Completed runs: 42
- NOT RUN entries: 18
- Failed runs: 0
- Best frozen-feature result: MISSING
- Best end-to-end result: cosine_prototype seed=2 test_accuracy=0.5605
- Claim status: The evidence is mixed: centered EML wins 3/12 paired comparisons.

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
| e2e_cifar10_merc_block_energy_ce_pairwise_seed0 | NOT RUN | end_to_end | merc_block_energy | cifar10 | 0 | pairwise prototype margin is not applicable |
| e2e_cifar10_merc_block_energy_ce_pairwise_seed1 | NOT RUN | end_to_end | merc_block_energy | cifar10 | 1 | pairwise prototype margin is not applicable |
| e2e_cifar10_merc_block_energy_ce_pairwise_seed2 | NOT RUN | end_to_end | merc_block_energy | cifar10 | 2 | pairwise prototype margin is not applicable |
| e2e_cifar10_merc_block_energy_ce_seed0 | COMPLETED | end_to_end | merc_block_energy | cifar10 | 0 |  |
| e2e_cifar10_merc_block_energy_ce_seed1 | COMPLETED | end_to_end | merc_block_energy | cifar10 | 1 |  |
| e2e_cifar10_merc_block_energy_ce_seed2 | COMPLETED | end_to_end | merc_block_energy | cifar10 | 2 |  |
| e2e_cifar10_merc_block_linear_ce_pairwise_seed0 | NOT RUN | end_to_end | merc_block_linear | cifar10 | 0 | pairwise prototype margin is not applicable |
| e2e_cifar10_merc_block_linear_ce_pairwise_seed1 | NOT RUN | end_to_end | merc_block_linear | cifar10 | 1 | pairwise prototype margin is not applicable |
| e2e_cifar10_merc_block_linear_ce_pairwise_seed2 | NOT RUN | end_to_end | merc_block_linear | cifar10 | 2 | pairwise prototype margin is not applicable |
| e2e_cifar10_merc_block_linear_ce_seed0 | COMPLETED | end_to_end | merc_block_linear | cifar10 | 0 |  |
| e2e_cifar10_merc_block_linear_ce_seed1 | COMPLETED | end_to_end | merc_block_linear | cifar10 | 1 |  |
| e2e_cifar10_merc_block_linear_ce_seed2 | COMPLETED | end_to_end | merc_block_linear | cifar10 | 2 |  |
| e2e_cifar10_merc_energy_ce_pairwise_seed0 | NOT RUN | end_to_end | merc_energy | cifar10 | 0 | pairwise prototype margin is not applicable |
| e2e_cifar10_merc_energy_ce_pairwise_seed1 | NOT RUN | end_to_end | merc_energy | cifar10 | 1 | pairwise prototype margin is not applicable |
| e2e_cifar10_merc_energy_ce_pairwise_seed2 | NOT RUN | end_to_end | merc_energy | cifar10 | 2 | pairwise prototype margin is not applicable |
| e2e_cifar10_merc_energy_ce_seed0 | COMPLETED | end_to_end | merc_energy | cifar10 | 0 |  |
| e2e_cifar10_merc_energy_ce_seed1 | COMPLETED | end_to_end | merc_energy | cifar10 | 1 |  |
| e2e_cifar10_merc_energy_ce_seed2 | COMPLETED | end_to_end | merc_energy | cifar10 | 2 |  |
| e2e_cifar10_merc_linear_ce_pairwise_seed0 | NOT RUN | end_to_end | merc_linear | cifar10 | 0 | pairwise prototype margin is not applicable |
| e2e_cifar10_merc_linear_ce_pairwise_seed1 | NOT RUN | end_to_end | merc_linear | cifar10 | 1 | pairwise prototype margin is not applicable |
| e2e_cifar10_merc_linear_ce_pairwise_seed2 | NOT RUN | end_to_end | merc_linear | cifar10 | 2 | pairwise prototype margin is not applicable |
| e2e_cifar10_merc_linear_ce_seed0 | COMPLETED | end_to_end | merc_linear | cifar10 | 0 |  |
| e2e_cifar10_merc_linear_ce_seed1 | COMPLETED | end_to_end | merc_linear | cifar10 | 1 |  |
| e2e_cifar10_merc_linear_ce_seed2 | COMPLETED | end_to_end | merc_linear | cifar10 | 2 |  |
| e2e_cifar10_mlp_ce_pairwise_seed0 | NOT RUN | end_to_end | mlp | cifar10 | 0 | pairwise prototype margin is not applicable |
| e2e_cifar10_mlp_ce_pairwise_seed1 | NOT RUN | end_to_end | mlp | cifar10 | 1 | pairwise prototype margin is not applicable |
| e2e_cifar10_mlp_ce_pairwise_seed2 | NOT RUN | end_to_end | mlp | cifar10 | 2 | pairwise prototype margin is not applicable |
| e2e_cifar10_mlp_ce_seed0 | COMPLETED | end_to_end | mlp | cifar10 | 0 |  |
| e2e_cifar10_mlp_ce_seed1 | COMPLETED | end_to_end | mlp | cifar10 | 1 |  |
| e2e_cifar10_mlp_ce_seed2 | COMPLETED | end_to_end | mlp | cifar10 | 2 |  |

## 4. Frozen Feature Results
### Frozen CNN Features
| run_id | seed | model | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## 5. End-To-End Results
### CNN Plus Head
| run_id | seed | model | loss mode | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed0 | 0 | cosine_prototype | ce_pairwise | 0.5430 | 0.5273 | 1.3453 | 0.0722 | 0.6014 | 0.0081 | 12.5939 |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed1 | 1 | cosine_prototype | ce_pairwise | 0.5352 | 0.5840 | 1.3208 | 0.0642 | 0.6060 | -0.0081 | 10.4521 |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed2 | 2 | cosine_prototype | ce_pairwise | 0.5605 | 0.5293 | 1.3361 | 0.0497 | 0.5906 | 0.0858 | 8.2270 |
| e2e_cifar10_cosine_prototype_ce_seed0 | 0 | cosine_prototype | ce | 0.5488 | 0.5430 | 1.3327 | 0.0711 | 0.5974 | 0.0227 | 10.9256 |
| e2e_cifar10_cosine_prototype_ce_seed1 | 1 | cosine_prototype | ce | 0.5332 | 0.5781 | 1.3178 | 0.0469 | 0.6054 | -0.0026 | 12.3218 |
| e2e_cifar10_cosine_prototype_ce_seed2 | 2 | cosine_prototype | ce | 0.5488 | 0.5059 | 1.3838 | 0.0587 | 0.6082 | 0.0226 | 8.3640 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_bank_centered_ambiguity | ce_pairwise | 0.4238 | 0.4375 | 1.6650 | 0.0380 | 0.7250 | -0.4552 | 13.4462 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_bank_centered_ambiguity | ce_pairwise | 0.4609 | 0.4688 | 1.5789 | 0.0835 | 0.6975 | -0.3688 | 14.5010 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed2 | 2 | eml_bank_centered_ambiguity | ce_pairwise | 0.4492 | 0.4180 | 1.6806 | 0.0672 | 0.7239 | -0.4735 | 14.2150 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0 | 0 | eml_bank_centered_ambiguity | ce | 0.3789 | 0.3867 | 1.7012 | 0.0433 | 0.7518 | -0.6116 | 14.3048 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1 | 1 | eml_bank_centered_ambiguity | ce | 0.4434 | 0.4453 | 1.6293 | 0.0700 | 0.7185 | -0.4602 | 13.8669 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed2 | 2 | eml_bank_centered_ambiguity | ce | 0.4453 | 0.4355 | 1.5902 | 0.0819 | 0.7053 | -0.3964 | 14.2939 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_centered_ambiguity | ce_pairwise | 0.4512 | 0.4199 | 1.5921 | 0.0789 | 0.7037 | -0.3906 | 12.8474 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_centered_ambiguity | ce_pairwise | 0.4531 | 0.4648 | 1.5510 | 0.0803 | 0.6954 | -0.3848 | 12.8281 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed2 | 2 | eml_centered_ambiguity | ce_pairwise | 0.4609 | 0.4336 | 1.6371 | 0.0751 | 0.7120 | -0.4537 | 15.3175 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed0 | 0 | eml_centered_ambiguity | ce | 0.3926 | 0.3535 | 1.7942 | 0.0419 | 0.7613 | -0.6223 | 13.1168 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed1 | 1 | eml_centered_ambiguity | ce | 0.4785 | 0.4824 | 1.5602 | 0.0988 | 0.6945 | -0.4092 | 12.7276 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed2 | 2 | eml_centered_ambiguity | ce | 0.4688 | 0.4609 | 1.6182 | 0.0967 | 0.7039 | -0.4297 | 20.7539 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0 | 0 | eml_no_ambiguity | ce_pairwise | 0.4004 | 0.3828 | 1.6369 | 0.0481 | 0.7171 | -0.4462 | 11.9863 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1 | 1 | eml_no_ambiguity | ce_pairwise | 0.4668 | 0.4902 | 1.5158 | 0.1021 | 0.6840 | -0.3076 | 24.9645 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed2 | 2 | eml_no_ambiguity | ce_pairwise | 0.4727 | 0.4258 | 1.6330 | 0.0867 | 0.7137 | -0.4288 | 13.0666 |
| e2e_cifar10_eml_no_ambiguity_ce_seed0 | 0 | eml_no_ambiguity | ce | 0.4043 | 0.3613 | 1.6180 | 0.0311 | 0.7222 | -0.4631 | 18.8835 |
| e2e_cifar10_eml_no_ambiguity_ce_seed1 | 1 | eml_no_ambiguity | ce | 0.4297 | 0.4590 | 1.6131 | 0.0658 | 0.7180 | -0.4328 | 25.3133 |
| e2e_cifar10_eml_no_ambiguity_ce_seed2 | 2 | eml_no_ambiguity | ce | 0.5020 | 0.4824 | 1.5712 | 0.1038 | 0.6897 | -0.3782 | 14.3736 |
| e2e_cifar10_linear_ce_pairwise_seed0 | 0 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_pairwise_seed1 | 1 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_pairwise_seed2 | 2 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_seed0 | 0 | linear | ce | 0.5156 | 0.5273 | 1.3054 | 0.0646 | 0.6045 | -0.0421 | 12.8890 |
| e2e_cifar10_linear_ce_seed1 | 1 | linear | ce | 0.5469 | 0.5469 | 1.3428 | 0.0582 | 0.6150 | -0.0476 | 10.2846 |
| e2e_cifar10_linear_ce_seed2 | 2 | linear | ce | 0.4297 | 0.4609 | 1.6221 | 0.1051 | 0.7064 | -0.4356 | 6.8059 |
| e2e_cifar10_merc_block_energy_ce_pairwise_seed0 | 0 | merc_block_energy | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_block_energy_ce_pairwise_seed1 | 1 | merc_block_energy | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_block_energy_ce_pairwise_seed2 | 2 | merc_block_energy | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_block_energy_ce_seed0 | 0 | merc_block_energy | ce | 0.2012 | 0.2617 | 1.9910 | 0.0321 | 0.8415 | -0.4922 | 17.6971 |
| e2e_cifar10_merc_block_energy_ce_seed1 | 1 | merc_block_energy | ce | 0.2520 | 0.2578 | 1.8922 | 0.0189 | 0.8122 | -0.4734 | 14.6659 |
| e2e_cifar10_merc_block_energy_ce_seed2 | 2 | merc_block_energy | ce | 0.1758 | 0.1699 | 1.8922 | 0.0536 | 0.8319 | -0.3144 | 14.9669 |
| e2e_cifar10_merc_block_linear_ce_pairwise_seed0 | 0 | merc_block_linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_block_linear_ce_pairwise_seed1 | 1 | merc_block_linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_block_linear_ce_pairwise_seed2 | 2 | merc_block_linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_block_linear_ce_seed0 | 0 | merc_block_linear | ce | 0.1738 | 0.2246 | 2.0320 | 0.0510 | 0.8463 | -0.5113 | 15.4942 |
| e2e_cifar10_merc_block_linear_ce_seed1 | 1 | merc_block_linear | ce | 0.1992 | 0.2246 | 1.9060 | 0.0381 | 0.8318 | -0.3807 | 15.9307 |
| e2e_cifar10_merc_block_linear_ce_seed2 | 2 | merc_block_linear | ce | 0.2070 | 0.1953 | 1.9172 | 0.0373 | 0.8338 | -0.4168 | 14.6924 |
| e2e_cifar10_merc_energy_ce_pairwise_seed0 | 0 | merc_energy | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_energy_ce_pairwise_seed1 | 1 | merc_energy | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_energy_ce_pairwise_seed2 | 2 | merc_energy | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_energy_ce_seed0 | 0 | merc_energy | ce | 0.2383 | 0.2617 | 2.1197 | 0.0691 | 0.8436 | -0.8334 | 15.3194 |
| e2e_cifar10_merc_energy_ce_seed1 | 1 | merc_energy | ce | 0.3730 | 0.3887 | 1.6023 | 0.0440 | 0.7306 | -0.4230 | 15.3817 |
| e2e_cifar10_merc_energy_ce_seed2 | 2 | merc_energy | ce | 0.3574 | 0.3203 | 1.6603 | 0.0374 | 0.7559 | -0.5407 | 12.6252 |
| e2e_cifar10_merc_linear_ce_pairwise_seed0 | 0 | merc_linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_linear_ce_pairwise_seed1 | 1 | merc_linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_linear_ce_pairwise_seed2 | 2 | merc_linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_linear_ce_seed0 | 0 | merc_linear | ce | 0.3418 | 0.3320 | 1.7623 | 0.0537 | 0.7626 | -0.5880 | 11.9258 |
| e2e_cifar10_merc_linear_ce_seed1 | 1 | merc_linear | ce | 0.3848 | 0.3867 | 1.5695 | 0.0620 | 0.7238 | -0.3520 | 14.7779 |
| e2e_cifar10_merc_linear_ce_seed2 | 2 | merc_linear | ce | 0.3398 | 0.3066 | 1.7343 | 0.0407 | 0.7692 | -0.5609 | 12.6511 |
| e2e_cifar10_mlp_ce_pairwise_seed0 | 0 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_pairwise_seed1 | 1 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_pairwise_seed2 | 2 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_seed0 | 0 | mlp | ce | 0.4570 | 0.4258 | 1.5051 | 0.0815 | 0.6679 | -0.3288 | 7.5621 |
| e2e_cifar10_mlp_ce_seed1 | 1 | mlp | ce | 0.4473 | 0.4648 | 1.5349 | 0.1035 | 0.6889 | -0.4561 | 8.5276 |
| e2e_cifar10_mlp_ce_seed2 | 2 | mlp | ce | 0.4512 | 0.4355 | 1.5197 | 0.0927 | 0.6671 | -0.3193 | 6.9578 |

## 6. CE-Only Comparison
### End-To-End CE Only
| run_id | seed | model | loss mode | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e_cifar10_cosine_prototype_ce_seed0 | 0 | cosine_prototype | ce | 0.5488 | 0.5430 | 1.3327 | 0.0711 | 0.5974 | 0.0227 | 10.9256 |
| e2e_cifar10_cosine_prototype_ce_seed1 | 1 | cosine_prototype | ce | 0.5332 | 0.5781 | 1.3178 | 0.0469 | 0.6054 | -0.0026 | 12.3218 |
| e2e_cifar10_cosine_prototype_ce_seed2 | 2 | cosine_prototype | ce | 0.5488 | 0.5059 | 1.3838 | 0.0587 | 0.6082 | 0.0226 | 8.3640 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0 | 0 | eml_bank_centered_ambiguity | ce | 0.3789 | 0.3867 | 1.7012 | 0.0433 | 0.7518 | -0.6116 | 14.3048 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1 | 1 | eml_bank_centered_ambiguity | ce | 0.4434 | 0.4453 | 1.6293 | 0.0700 | 0.7185 | -0.4602 | 13.8669 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed2 | 2 | eml_bank_centered_ambiguity | ce | 0.4453 | 0.4355 | 1.5902 | 0.0819 | 0.7053 | -0.3964 | 14.2939 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed0 | 0 | eml_centered_ambiguity | ce | 0.3926 | 0.3535 | 1.7942 | 0.0419 | 0.7613 | -0.6223 | 13.1168 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed1 | 1 | eml_centered_ambiguity | ce | 0.4785 | 0.4824 | 1.5602 | 0.0988 | 0.6945 | -0.4092 | 12.7276 |
| e2e_cifar10_eml_centered_ambiguity_ce_seed2 | 2 | eml_centered_ambiguity | ce | 0.4688 | 0.4609 | 1.6182 | 0.0967 | 0.7039 | -0.4297 | 20.7539 |
| e2e_cifar10_eml_no_ambiguity_ce_seed0 | 0 | eml_no_ambiguity | ce | 0.4043 | 0.3613 | 1.6180 | 0.0311 | 0.7222 | -0.4631 | 18.8835 |
| e2e_cifar10_eml_no_ambiguity_ce_seed1 | 1 | eml_no_ambiguity | ce | 0.4297 | 0.4590 | 1.6131 | 0.0658 | 0.7180 | -0.4328 | 25.3133 |
| e2e_cifar10_eml_no_ambiguity_ce_seed2 | 2 | eml_no_ambiguity | ce | 0.5020 | 0.4824 | 1.5712 | 0.1038 | 0.6897 | -0.3782 | 14.3736 |
| e2e_cifar10_linear_ce_seed0 | 0 | linear | ce | 0.5156 | 0.5273 | 1.3054 | 0.0646 | 0.6045 | -0.0421 | 12.8890 |
| e2e_cifar10_linear_ce_seed1 | 1 | linear | ce | 0.5469 | 0.5469 | 1.3428 | 0.0582 | 0.6150 | -0.0476 | 10.2846 |
| e2e_cifar10_linear_ce_seed2 | 2 | linear | ce | 0.4297 | 0.4609 | 1.6221 | 0.1051 | 0.7064 | -0.4356 | 6.8059 |
| e2e_cifar10_merc_block_energy_ce_seed0 | 0 | merc_block_energy | ce | 0.2012 | 0.2617 | 1.9910 | 0.0321 | 0.8415 | -0.4922 | 17.6971 |
| e2e_cifar10_merc_block_energy_ce_seed1 | 1 | merc_block_energy | ce | 0.2520 | 0.2578 | 1.8922 | 0.0189 | 0.8122 | -0.4734 | 14.6659 |
| e2e_cifar10_merc_block_energy_ce_seed2 | 2 | merc_block_energy | ce | 0.1758 | 0.1699 | 1.8922 | 0.0536 | 0.8319 | -0.3144 | 14.9669 |
| e2e_cifar10_merc_block_linear_ce_seed0 | 0 | merc_block_linear | ce | 0.1738 | 0.2246 | 2.0320 | 0.0510 | 0.8463 | -0.5113 | 15.4942 |
| e2e_cifar10_merc_block_linear_ce_seed1 | 1 | merc_block_linear | ce | 0.1992 | 0.2246 | 1.9060 | 0.0381 | 0.8318 | -0.3807 | 15.9307 |
| e2e_cifar10_merc_block_linear_ce_seed2 | 2 | merc_block_linear | ce | 0.2070 | 0.1953 | 1.9172 | 0.0373 | 0.8338 | -0.4168 | 14.6924 |
| e2e_cifar10_merc_energy_ce_seed0 | 0 | merc_energy | ce | 0.2383 | 0.2617 | 2.1197 | 0.0691 | 0.8436 | -0.8334 | 15.3194 |
| e2e_cifar10_merc_energy_ce_seed1 | 1 | merc_energy | ce | 0.3730 | 0.3887 | 1.6023 | 0.0440 | 0.7306 | -0.4230 | 15.3817 |
| e2e_cifar10_merc_energy_ce_seed2 | 2 | merc_energy | ce | 0.3574 | 0.3203 | 1.6603 | 0.0374 | 0.7559 | -0.5407 | 12.6252 |
| e2e_cifar10_merc_linear_ce_seed0 | 0 | merc_linear | ce | 0.3418 | 0.3320 | 1.7623 | 0.0537 | 0.7626 | -0.5880 | 11.9258 |
| e2e_cifar10_merc_linear_ce_seed1 | 1 | merc_linear | ce | 0.3848 | 0.3867 | 1.5695 | 0.0620 | 0.7238 | -0.3520 | 14.7779 |
| e2e_cifar10_merc_linear_ce_seed2 | 2 | merc_linear | ce | 0.3398 | 0.3066 | 1.7343 | 0.0407 | 0.7692 | -0.5609 | 12.6511 |
| e2e_cifar10_mlp_ce_seed0 | 0 | mlp | ce | 0.4570 | 0.4258 | 1.5051 | 0.0815 | 0.6679 | -0.3288 | 7.5621 |
| e2e_cifar10_mlp_ce_seed1 | 1 | mlp | ce | 0.4473 | 0.4648 | 1.5349 | 0.1035 | 0.6889 | -0.4561 | 8.5276 |
| e2e_cifar10_mlp_ce_seed2 | 2 | mlp | ce | 0.4512 | 0.4355 | 1.5197 | 0.0927 | 0.6671 | -0.3193 | 6.9578 |

## 7. CE + Pairwise Comparison
### End-To-End CE + Prototype Pairwise
| run_id | seed | model | loss mode | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed0 | 0 | cosine_prototype | ce_pairwise | 0.5430 | 0.5273 | 1.3453 | 0.0722 | 0.6014 | 0.0081 | 12.5939 |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed1 | 1 | cosine_prototype | ce_pairwise | 0.5352 | 0.5840 | 1.3208 | 0.0642 | 0.6060 | -0.0081 | 10.4521 |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed2 | 2 | cosine_prototype | ce_pairwise | 0.5605 | 0.5293 | 1.3361 | 0.0497 | 0.5906 | 0.0858 | 8.2270 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_bank_centered_ambiguity | ce_pairwise | 0.4238 | 0.4375 | 1.6650 | 0.0380 | 0.7250 | -0.4552 | 13.4462 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_bank_centered_ambiguity | ce_pairwise | 0.4609 | 0.4688 | 1.5789 | 0.0835 | 0.6975 | -0.3688 | 14.5010 |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed2 | 2 | eml_bank_centered_ambiguity | ce_pairwise | 0.4492 | 0.4180 | 1.6806 | 0.0672 | 0.7239 | -0.4735 | 14.2150 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_centered_ambiguity | ce_pairwise | 0.4512 | 0.4199 | 1.5921 | 0.0789 | 0.7037 | -0.3906 | 12.8474 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_centered_ambiguity | ce_pairwise | 0.4531 | 0.4648 | 1.5510 | 0.0803 | 0.6954 | -0.3848 | 12.8281 |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed2 | 2 | eml_centered_ambiguity | ce_pairwise | 0.4609 | 0.4336 | 1.6371 | 0.0751 | 0.7120 | -0.4537 | 15.3175 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0 | 0 | eml_no_ambiguity | ce_pairwise | 0.4004 | 0.3828 | 1.6369 | 0.0481 | 0.7171 | -0.4462 | 11.9863 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1 | 1 | eml_no_ambiguity | ce_pairwise | 0.4668 | 0.4902 | 1.5158 | 0.1021 | 0.6840 | -0.3076 | 24.9645 |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed2 | 2 | eml_no_ambiguity | ce_pairwise | 0.4727 | 0.4258 | 1.6330 | 0.0867 | 0.7137 | -0.4288 | 13.0666 |
| e2e_cifar10_linear_ce_pairwise_seed0 | 0 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_pairwise_seed1 | 1 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_pairwise_seed2 | 2 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_block_energy_ce_pairwise_seed0 | 0 | merc_block_energy | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_block_energy_ce_pairwise_seed1 | 1 | merc_block_energy | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_block_energy_ce_pairwise_seed2 | 2 | merc_block_energy | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_block_linear_ce_pairwise_seed0 | 0 | merc_block_linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_block_linear_ce_pairwise_seed1 | 1 | merc_block_linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_block_linear_ce_pairwise_seed2 | 2 | merc_block_linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_energy_ce_pairwise_seed0 | 0 | merc_energy | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_energy_ce_pairwise_seed1 | 1 | merc_energy | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_energy_ce_pairwise_seed2 | 2 | merc_energy | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_linear_ce_pairwise_seed0 | 0 | merc_linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_linear_ce_pairwise_seed1 | 1 | merc_linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_merc_linear_ce_pairwise_seed2 | 2 | merc_linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
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
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0 | eml_bank_centered_ambiguity | 4.6745 | 5.8464 | 0.9290 | 0.9100 | 0.0014 | 1.0806 | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1 | eml_bank_centered_ambiguity | 4.7700 | 5.5818 | 0.9216 | 0.9564 | 0.0006 | 1.0726 | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed2 | eml_bank_centered_ambiguity | 4.9531 | 6.4227 | 0.8683 | 0.9152 | 0.0017 | 1.0364 | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0 | eml_bank_centered_ambiguity | 4.7001 | 6.4314 | 1.0985 | 0.9428 | 0.0005 | 1.1813 | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1 | eml_bank_centered_ambiguity | 4.8963 | 6.0448 | 0.9446 | 0.9687 | 0.0006 | 1.1081 | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed2 | eml_bank_centered_ambiguity | 5.3285 | 6.4808 | 0.9446 | 0.9875 | 0.0009 | 1.1334 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0 | eml_centered_ambiguity | 5.0468 | 6.0808 | 0.9527 | 0.9775 | 0.0004 | 1.0570 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1 | eml_centered_ambiguity | 5.4149 | 6.3739 | 0.8653 | 0.8839 | 0.0003 | 0.9731 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed2 | eml_centered_ambiguity | 4.6580 | 5.7276 | 0.7807 | 0.8221 | 0.0032 | 0.9107 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_seed0 | eml_centered_ambiguity | 4.7976 | 6.8393 | 1.0999 | 1.0328 | 0.0001 | 1.1743 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_seed1 | eml_centered_ambiguity | 5.3633 | 6.3544 | 0.8551 | 0.8823 | 0.0003 | 0.9750 | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_seed2 | eml_centered_ambiguity | 4.8827 | 6.0094 | 0.7857 | 0.8627 | 0.0003 | 0.9434 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0 | eml_no_ambiguity | 5.0913 | 6.4549 | 17.5851 | 17.5855 | 17.3836 | 0.0000 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1 | eml_no_ambiguity | 5.3348 | 6.1276 | 4.0251 | 4.0264 | 3.8239 | 0.0000 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed2 | eml_no_ambiguity | 4.9270 | 5.9256 | 9.1802 | 9.1810 | 8.9788 | 0.0000 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_seed0 | eml_no_ambiguity | 5.2489 | 6.6255 | 16.0452 | 16.0454 | 15.8439 | 0.0000 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_seed1 | eml_no_ambiguity | 5.5206 | 6.7093 | 4.9166 | 4.9177 | 4.7155 | 0.0000 | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_seed2 | eml_no_ambiguity | 5.2186 | 6.0136 | 8.8909 | 8.8903 | 8.6894 | 0.0000 | MISSING | MISSING |

## 11. Robustness Under Noise/Occlusion
Resistance-noise and resistance-occlusion correlations are reported when synthetic metadata is available. MISSING means the head did not expose resistance or the dataset did not provide the field.

## 12. Statistical Confidence Intervals
| experiment | dataset | seed | loss mode | comparison | delta acc | 95% CI low | 95% CI high |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| end_to_end | cifar10 | 0 | ce | eml_centered_ambiguity - linear | -0.1230 | -0.1778 | -0.0684 |
| end_to_end | cifar10 | 0 | ce | eml_centered_ambiguity - mlp | -0.0645 | -0.1211 | -0.0059 |
| end_to_end | cifar10 | 0 | ce | eml_centered_ambiguity - cosine_prototype | -0.1562 | -0.2090 | -0.1035 |
| end_to_end | cifar10 | 0 | ce_pairwise | eml_centered_ambiguity - cosine_prototype | -0.0918 | -0.1387 | -0.0430 |
| end_to_end | cifar10 | 1 | ce | eml_centered_ambiguity - linear | -0.0684 | -0.1152 | -0.0215 |
| end_to_end | cifar10 | 1 | ce | eml_centered_ambiguity - mlp | 0.0312 | -0.0117 | 0.0762 |
| end_to_end | cifar10 | 1 | ce | eml_centered_ambiguity - cosine_prototype | -0.0547 | -0.1016 | -0.0097 |
| end_to_end | cifar10 | 1 | ce_pairwise | eml_centered_ambiguity - cosine_prototype | -0.0820 | -0.1289 | -0.0332 |
| end_to_end | cifar10 | 2 | ce | eml_centered_ambiguity - linear | 0.0391 | -0.0020 | 0.0859 |
| end_to_end | cifar10 | 2 | ce | eml_centered_ambiguity - mlp | 0.0176 | -0.0312 | 0.0703 |
| end_to_end | cifar10 | 2 | ce | eml_centered_ambiguity - cosine_prototype | -0.0801 | -0.1211 | -0.0332 |
| end_to_end | cifar10 | 2 | ce_pairwise | eml_centered_ambiguity - cosine_prototype | -0.0996 | -0.1406 | -0.0547 |

## 13. Which Claim Is Supported
The evidence is mixed: centered EML wins 3/12 paired comparisons.

## 14. Raw Artifacts
- `e2e_cifar10_cosine_prototype_ce_pairwise_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_034946_e2e_cifar10_cosine_prototype_ce_pairwise_seed0`
- `e2e_cifar10_cosine_prototype_ce_pairwise_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035352_e2e_cifar10_cosine_prototype_ce_pairwise_seed1`
- `e2e_cifar10_cosine_prototype_ce_pairwise_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035716_e2e_cifar10_cosine_prototype_ce_pairwise_seed2`
- `e2e_cifar10_cosine_prototype_ce_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_034934_e2e_cifar10_cosine_prototype_ce_seed0`
- `e2e_cifar10_cosine_prototype_ce_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035339_e2e_cifar10_cosine_prototype_ce_seed1`
- `e2e_cifar10_cosine_prototype_ce_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035708_e2e_cifar10_cosine_prototype_ce_seed2`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_035109_e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035533_e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035843_e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed2`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_035055_e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035519_e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035828_e2e_cifar10_eml_bank_centered_ambiguity_ce_seed2`
- `e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_035042_e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0`
- `e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035506_e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1`
- `e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035813_e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed2`
- `e2e_cifar10_eml_centered_ambiguity_ce_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_035029_e2e_cifar10_eml_centered_ambiguity_ce_seed0`
- `e2e_cifar10_eml_centered_ambiguity_ce_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035453_e2e_cifar10_eml_centered_ambiguity_ce_seed1`
- `e2e_cifar10_eml_centered_ambiguity_ce_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035752_e2e_cifar10_eml_centered_ambiguity_ce_seed2`
- `e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_035017_e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0`
- `e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035428_e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1`
- `e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035739_e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed2`
- `e2e_cifar10_eml_no_ambiguity_ce_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_034958_e2e_cifar10_eml_no_ambiguity_ce_seed0`
- `e2e_cifar10_eml_no_ambiguity_ce_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035403_e2e_cifar10_eml_no_ambiguity_ce_seed1`
- `e2e_cifar10_eml_no_ambiguity_ce_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035725_e2e_cifar10_eml_no_ambiguity_ce_seed2`
- `e2e_cifar10_linear_ce_pairwise_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_034927_e2e_cifar10_linear_ce_pairwise_seed0`
- `e2e_cifar10_linear_ce_pairwise_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035331_e2e_cifar10_linear_ce_pairwise_seed1`
- `e2e_cifar10_linear_ce_pairwise_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035701_e2e_cifar10_linear_ce_pairwise_seed2`
- `e2e_cifar10_linear_ce_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_034914_e2e_cifar10_linear_ce_seed0`
- `e2e_cifar10_linear_ce_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035320_e2e_cifar10_linear_ce_seed1`
- `e2e_cifar10_linear_ce_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035654_e2e_cifar10_linear_ce_seed2`
- `e2e_cifar10_merc_block_energy_ce_pairwise_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_035223_e2e_cifar10_merc_block_energy_ce_pairwise_seed0`
- `e2e_cifar10_merc_block_energy_ce_pairwise_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035649_e2e_cifar10_merc_block_energy_ce_pairwise_seed1`
- `e2e_cifar10_merc_block_energy_ce_pairwise_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035953_e2e_cifar10_merc_block_energy_ce_pairwise_seed2`
- `e2e_cifar10_merc_block_energy_ce_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_035206_e2e_cifar10_merc_block_energy_ce_seed0`
- `e2e_cifar10_merc_block_energy_ce_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035634_e2e_cifar10_merc_block_energy_ce_seed1`
- `e2e_cifar10_merc_block_energy_ce_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035937_e2e_cifar10_merc_block_energy_ce_seed2`
- `e2e_cifar10_merc_block_linear_ce_pairwise_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_035206_e2e_cifar10_merc_block_linear_ce_pairwise_seed0`
- `e2e_cifar10_merc_block_linear_ce_pairwise_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035634_e2e_cifar10_merc_block_linear_ce_pairwise_seed1`
- `e2e_cifar10_merc_block_linear_ce_pairwise_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035937_e2e_cifar10_merc_block_linear_ce_pairwise_seed2`
- `e2e_cifar10_merc_block_linear_ce_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_035150_e2e_cifar10_merc_block_linear_ce_seed0`
- `e2e_cifar10_merc_block_linear_ce_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035618_e2e_cifar10_merc_block_linear_ce_seed1`
- `e2e_cifar10_merc_block_linear_ce_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035923_e2e_cifar10_merc_block_linear_ce_seed2`
- `e2e_cifar10_merc_energy_ce_pairwise_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_035150_e2e_cifar10_merc_energy_ce_pairwise_seed0`
- `e2e_cifar10_merc_energy_ce_pairwise_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035618_e2e_cifar10_merc_energy_ce_pairwise_seed1`
- `e2e_cifar10_merc_energy_ce_pairwise_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035923_e2e_cifar10_merc_energy_ce_pairwise_seed2`
- `e2e_cifar10_merc_energy_ce_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_035135_e2e_cifar10_merc_energy_ce_seed0`
- `e2e_cifar10_merc_energy_ce_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035602_e2e_cifar10_merc_energy_ce_seed1`
- `e2e_cifar10_merc_energy_ce_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035910_e2e_cifar10_merc_energy_ce_seed2`
- `e2e_cifar10_merc_linear_ce_pairwise_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_035135_e2e_cifar10_merc_linear_ce_pairwise_seed0`
- `e2e_cifar10_merc_linear_ce_pairwise_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035602_e2e_cifar10_merc_linear_ce_pairwise_seed1`
- `e2e_cifar10_merc_linear_ce_pairwise_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035910_e2e_cifar10_merc_linear_ce_pairwise_seed2`
- `e2e_cifar10_merc_linear_ce_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_035123_e2e_cifar10_merc_linear_ce_seed0`
- `e2e_cifar10_merc_linear_ce_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035548_e2e_cifar10_merc_linear_ce_seed1`
- `e2e_cifar10_merc_linear_ce_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035857_e2e_cifar10_merc_linear_ce_seed2`
- `e2e_cifar10_mlp_ce_pairwise_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_034934_e2e_cifar10_mlp_ce_pairwise_seed0`
- `e2e_cifar10_mlp_ce_pairwise_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035339_e2e_cifar10_mlp_ce_pairwise_seed1`
- `e2e_cifar10_mlp_ce_pairwise_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035708_e2e_cifar10_mlp_ce_pairwise_seed2`
- `e2e_cifar10_mlp_ce_seed0`: `reports/merc_end_to_end_rerun/runs/20260425_034927_e2e_cifar10_mlp_ce_seed0`
- `e2e_cifar10_mlp_ce_seed1`: `reports/merc_end_to_end_rerun/runs/20260425_035331_e2e_cifar10_mlp_ce_seed1`
- `e2e_cifar10_mlp_ce_seed2`: `reports/merc_end_to_end_rerun/runs/20260425_035701_e2e_cifar10_mlp_ce_seed2`

## 15. Appendix: Commands
- `pytest`
- `python scripts/run_head_ablation.py --dataset synthetic_shape --mode smoke --seeds 0 1`
- `python scripts/run_cnn_head_end_to_end_ablation.py --dataset synthetic_shape --mode smoke --seeds 0 1`
- `python scripts/generate_head_ablation_report.py`
