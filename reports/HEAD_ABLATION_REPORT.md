# CNN Head Ablation Report

## 1. Executive Summary
- Completed runs: 64
- NOT RUN entries: 44
- Failed runs: 0
- Best frozen-feature result: eml_centered_ambiguity seed=0 test_accuracy=0.2109
- Best end-to-end result: cosine_prototype seed=0 test_accuracy=0.1953
- Claim status: The evidence is mixed: centered EML wins 5/14 paired comparisons.

## 2. Experimental Setup
- Frozen-feature runs train one shared CNN feature extractor per dataset/seed, cache features, then train only the selected head.
- End-to-end runs train the same CNN backbone with one selected head; the EML residual-bank variant is reported separately.
- CE-only and prototype-pairwise settings are separated. Linear and MLP heads are marked NOT RUN for prototype-pairwise because that loss is not applicable.

## 3. Run Status
| run_id | status | experiment | model | dataset | seed | reason |
| --- | --- | --- | --- | --- | ---: | --- |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed0 | NOT RUN | end_to_end | cosine_prototype | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed1 | NOT RUN | end_to_end | cosine_prototype | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_cosine_prototype_ce_seed0 | NOT RUN | end_to_end | cosine_prototype | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_cosine_prototype_ce_seed1 | NOT RUN | end_to_end | cosine_prototype | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0 | NOT RUN | end_to_end | eml_bank_centered_ambiguity | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1 | NOT RUN | end_to_end | eml_bank_centered_ambiguity | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0 | NOT RUN | end_to_end | eml_bank_centered_ambiguity | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1 | NOT RUN | end_to_end | eml_bank_centered_ambiguity | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0 | NOT RUN | end_to_end | eml_centered_ambiguity | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1 | NOT RUN | end_to_end | eml_centered_ambiguity | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_centered_ambiguity_ce_seed0 | NOT RUN | end_to_end | eml_centered_ambiguity | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_centered_ambiguity_ce_seed1 | NOT RUN | end_to_end | eml_centered_ambiguity | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0 | NOT RUN | end_to_end | eml_no_ambiguity | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1 | NOT RUN | end_to_end | eml_no_ambiguity | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_no_ambiguity_ce_seed0 | NOT RUN | end_to_end | eml_no_ambiguity | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_no_ambiguity_ce_seed1 | NOT RUN | end_to_end | eml_no_ambiguity | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_linear_ce_pairwise_seed0 | NOT RUN | end_to_end | linear | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_linear_ce_pairwise_seed1 | NOT RUN | end_to_end | linear | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_linear_ce_seed0 | NOT RUN | end_to_end | linear | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_linear_ce_seed1 | NOT RUN | end_to_end | linear | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_mlp_ce_pairwise_seed0 | NOT RUN | end_to_end | mlp | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_mlp_ce_pairwise_seed1 | NOT RUN | end_to_end | mlp | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_mlp_ce_seed0 | NOT RUN | end_to_end | mlp | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_mlp_ce_seed1 | NOT RUN | end_to_end | mlp | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed0 | COMPLETED | end_to_end | cosine_prototype | synthetic_shape | 0 |  |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed0 | COMPLETED | end_to_end | cosine_prototype | synthetic_shape | 0 |  |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed1 | COMPLETED | end_to_end | cosine_prototype | synthetic_shape | 1 |  |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed1 | COMPLETED | end_to_end | cosine_prototype | synthetic_shape | 1 |  |
| e2e_synthetic_shape_cosine_prototype_ce_seed0 | COMPLETED | end_to_end | cosine_prototype | synthetic_shape | 0 |  |
| e2e_synthetic_shape_cosine_prototype_ce_seed0 | COMPLETED | end_to_end | cosine_prototype | synthetic_shape | 0 |  |
| e2e_synthetic_shape_cosine_prototype_ce_seed1 | COMPLETED | end_to_end | cosine_prototype | synthetic_shape | 1 |  |
| e2e_synthetic_shape_cosine_prototype_ce_seed1 | COMPLETED | end_to_end | cosine_prototype | synthetic_shape | 1 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0 | COMPLETED | end_to_end | eml_bank_centered_ambiguity | synthetic_shape | 0 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0 | COMPLETED | end_to_end | eml_bank_centered_ambiguity | synthetic_shape | 0 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1 | COMPLETED | end_to_end | eml_bank_centered_ambiguity | synthetic_shape | 1 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1 | COMPLETED | end_to_end | eml_bank_centered_ambiguity | synthetic_shape | 1 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0 | COMPLETED | end_to_end | eml_bank_centered_ambiguity | synthetic_shape | 0 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0 | COMPLETED | end_to_end | eml_bank_centered_ambiguity | synthetic_shape | 0 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1 | COMPLETED | end_to_end | eml_bank_centered_ambiguity | synthetic_shape | 1 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1 | COMPLETED | end_to_end | eml_bank_centered_ambiguity | synthetic_shape | 1 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0 | COMPLETED | end_to_end | eml_centered_ambiguity | synthetic_shape | 0 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0 | COMPLETED | end_to_end | eml_centered_ambiguity | synthetic_shape | 0 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1 | COMPLETED | end_to_end | eml_centered_ambiguity | synthetic_shape | 1 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1 | COMPLETED | end_to_end | eml_centered_ambiguity | synthetic_shape | 1 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0 | COMPLETED | end_to_end | eml_centered_ambiguity | synthetic_shape | 0 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0 | COMPLETED | end_to_end | eml_centered_ambiguity | synthetic_shape | 0 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1 | COMPLETED | end_to_end | eml_centered_ambiguity | synthetic_shape | 1 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1 | COMPLETED | end_to_end | eml_centered_ambiguity | synthetic_shape | 1 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0 | COMPLETED | end_to_end | eml_no_ambiguity | synthetic_shape | 0 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0 | COMPLETED | end_to_end | eml_no_ambiguity | synthetic_shape | 0 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1 | COMPLETED | end_to_end | eml_no_ambiguity | synthetic_shape | 1 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1 | COMPLETED | end_to_end | eml_no_ambiguity | synthetic_shape | 1 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed0 | COMPLETED | end_to_end | eml_no_ambiguity | synthetic_shape | 0 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed0 | COMPLETED | end_to_end | eml_no_ambiguity | synthetic_shape | 0 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed1 | COMPLETED | end_to_end | eml_no_ambiguity | synthetic_shape | 1 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed1 | COMPLETED | end_to_end | eml_no_ambiguity | synthetic_shape | 1 |  |
| e2e_synthetic_shape_linear_ce_pairwise_seed0 | NOT RUN | end_to_end | linear | synthetic_shape | 0 | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_linear_ce_pairwise_seed0 | NOT RUN | end_to_end | linear | synthetic_shape | 0 | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_linear_ce_pairwise_seed1 | NOT RUN | end_to_end | linear | synthetic_shape | 1 | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_linear_ce_pairwise_seed1 | NOT RUN | end_to_end | linear | synthetic_shape | 1 | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_linear_ce_seed0 | COMPLETED | end_to_end | linear | synthetic_shape | 0 |  |
| e2e_synthetic_shape_linear_ce_seed0 | COMPLETED | end_to_end | linear | synthetic_shape | 0 |  |
| e2e_synthetic_shape_linear_ce_seed1 | COMPLETED | end_to_end | linear | synthetic_shape | 1 |  |
| e2e_synthetic_shape_linear_ce_seed1 | COMPLETED | end_to_end | linear | synthetic_shape | 1 |  |
| e2e_synthetic_shape_mlp_ce_pairwise_seed0 | NOT RUN | end_to_end | mlp | synthetic_shape | 0 | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_mlp_ce_pairwise_seed0 | NOT RUN | end_to_end | mlp | synthetic_shape | 0 | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_mlp_ce_pairwise_seed1 | NOT RUN | end_to_end | mlp | synthetic_shape | 1 | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_mlp_ce_pairwise_seed1 | NOT RUN | end_to_end | mlp | synthetic_shape | 1 | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_mlp_ce_seed0 | COMPLETED | end_to_end | mlp | synthetic_shape | 0 |  |
| e2e_synthetic_shape_mlp_ce_seed0 | COMPLETED | end_to_end | mlp | synthetic_shape | 0 |  |
| e2e_synthetic_shape_mlp_ce_seed1 | COMPLETED | end_to_end | mlp | synthetic_shape | 1 |  |
| e2e_synthetic_shape_mlp_ce_seed1 | COMPLETED | end_to_end | mlp | synthetic_shape | 1 |  |
| frozen_cifar10_cosine_prototype_seed0 | NOT RUN | frozen_features | cosine_prototype | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| frozen_cifar10_cosine_prototype_seed1 | NOT RUN | frozen_features | cosine_prototype | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| frozen_cifar10_eml_centered_ambiguity_seed0 | NOT RUN | frozen_features | eml_centered_ambiguity | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| frozen_cifar10_eml_centered_ambiguity_seed1 | NOT RUN | frozen_features | eml_centered_ambiguity | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| frozen_cifar10_eml_no_ambiguity_seed0 | NOT RUN | frozen_features | eml_no_ambiguity | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| frozen_cifar10_eml_no_ambiguity_seed1 | NOT RUN | frozen_features | eml_no_ambiguity | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| frozen_cifar10_eml_raw_ambiguity_seed0 | NOT RUN | frozen_features | eml_raw_ambiguity | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| frozen_cifar10_eml_raw_ambiguity_seed1 | NOT RUN | frozen_features | eml_raw_ambiguity | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| frozen_cifar10_linear_seed0 | NOT RUN | frozen_features | linear | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| frozen_cifar10_linear_seed1 | NOT RUN | frozen_features | linear | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| frozen_cifar10_mlp_seed0 | NOT RUN | frozen_features | mlp | cifar10 | 0 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| frozen_cifar10_mlp_seed1 | NOT RUN | frozen_features | mlp | cifar10 | 1 | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| frozen_synthetic_shape_cosine_prototype_seed0 | COMPLETED | frozen_features | cosine_prototype | synthetic_shape | 0 |  |
| frozen_synthetic_shape_cosine_prototype_seed0 | COMPLETED | frozen_features | cosine_prototype | synthetic_shape | 0 |  |
| frozen_synthetic_shape_cosine_prototype_seed1 | COMPLETED | frozen_features | cosine_prototype | synthetic_shape | 1 |  |
| frozen_synthetic_shape_cosine_prototype_seed1 | COMPLETED | frozen_features | cosine_prototype | synthetic_shape | 1 |  |
| frozen_synthetic_shape_eml_centered_ambiguity_seed0 | COMPLETED | frozen_features | eml_centered_ambiguity | synthetic_shape | 0 |  |
| frozen_synthetic_shape_eml_centered_ambiguity_seed0 | COMPLETED | frozen_features | eml_centered_ambiguity | synthetic_shape | 0 |  |
| frozen_synthetic_shape_eml_centered_ambiguity_seed1 | COMPLETED | frozen_features | eml_centered_ambiguity | synthetic_shape | 1 |  |
| frozen_synthetic_shape_eml_centered_ambiguity_seed1 | COMPLETED | frozen_features | eml_centered_ambiguity | synthetic_shape | 1 |  |
| frozen_synthetic_shape_eml_no_ambiguity_seed0 | COMPLETED | frozen_features | eml_no_ambiguity | synthetic_shape | 0 |  |
| frozen_synthetic_shape_eml_no_ambiguity_seed0 | COMPLETED | frozen_features | eml_no_ambiguity | synthetic_shape | 0 |  |
| frozen_synthetic_shape_eml_no_ambiguity_seed1 | COMPLETED | frozen_features | eml_no_ambiguity | synthetic_shape | 1 |  |
| frozen_synthetic_shape_eml_no_ambiguity_seed1 | COMPLETED | frozen_features | eml_no_ambiguity | synthetic_shape | 1 |  |
| frozen_synthetic_shape_eml_raw_ambiguity_seed0 | COMPLETED | frozen_features | eml_raw_ambiguity | synthetic_shape | 0 |  |
| frozen_synthetic_shape_eml_raw_ambiguity_seed0 | COMPLETED | frozen_features | eml_raw_ambiguity | synthetic_shape | 0 |  |
| frozen_synthetic_shape_eml_raw_ambiguity_seed1 | COMPLETED | frozen_features | eml_raw_ambiguity | synthetic_shape | 1 |  |
| frozen_synthetic_shape_eml_raw_ambiguity_seed1 | COMPLETED | frozen_features | eml_raw_ambiguity | synthetic_shape | 1 |  |
| frozen_synthetic_shape_linear_seed0 | COMPLETED | frozen_features | linear | synthetic_shape | 0 |  |
| frozen_synthetic_shape_linear_seed0 | COMPLETED | frozen_features | linear | synthetic_shape | 0 |  |
| frozen_synthetic_shape_linear_seed1 | COMPLETED | frozen_features | linear | synthetic_shape | 1 |  |
| frozen_synthetic_shape_linear_seed1 | COMPLETED | frozen_features | linear | synthetic_shape | 1 |  |
| frozen_synthetic_shape_mlp_seed0 | COMPLETED | frozen_features | mlp | synthetic_shape | 0 |  |
| frozen_synthetic_shape_mlp_seed0 | COMPLETED | frozen_features | mlp | synthetic_shape | 0 |  |
| frozen_synthetic_shape_mlp_seed1 | COMPLETED | frozen_features | mlp | synthetic_shape | 1 |  |
| frozen_synthetic_shape_mlp_seed1 | COMPLETED | frozen_features | mlp | synthetic_shape | 1 |  |

## 4. Frozen Feature Results
### Frozen CNN Features
| run_id | seed | model | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen_cifar10_cosine_prototype_seed0 | 0 | cosine_prototype | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_cifar10_cosine_prototype_seed1 | 1 | cosine_prototype | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_cifar10_eml_centered_ambiguity_seed0 | 0 | eml_centered_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_cifar10_eml_centered_ambiguity_seed1 | 1 | eml_centered_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_cifar10_eml_no_ambiguity_seed0 | 0 | eml_no_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_cifar10_eml_no_ambiguity_seed1 | 1 | eml_no_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_cifar10_eml_raw_ambiguity_seed0 | 0 | eml_raw_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_cifar10_eml_raw_ambiguity_seed1 | 1 | eml_raw_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_cifar10_linear_seed0 | 0 | linear | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_cifar10_linear_seed1 | 1 | linear | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_cifar10_mlp_seed0 | 0 | mlp | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_cifar10_mlp_seed1 | 1 | mlp | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_synthetic_shape_cosine_prototype_seed0 | 0 | cosine_prototype | 0.2031 | 0.1953 | 1.6155 | 0.0254 | 0.8024 | -0.1358 | 0.0269 |
| frozen_synthetic_shape_cosine_prototype_seed0 | 0 | cosine_prototype | 0.2031 | 0.1953 | 1.6155 | 0.0254 | 0.8024 | -0.1358 | 0.0321 |
| frozen_synthetic_shape_cosine_prototype_seed1 | 1 | cosine_prototype | 0.1953 | 0.1953 | 1.6165 | 0.0340 | 0.8027 | -0.1253 | 0.0287 |
| frozen_synthetic_shape_cosine_prototype_seed1 | 1 | cosine_prototype | 0.1953 | 0.1953 | 1.6165 | 0.0340 | 0.8027 | -0.1253 | 0.0343 |
| frozen_synthetic_shape_eml_centered_ambiguity_seed0 | 0 | eml_centered_ambiguity | 0.2109 | 0.1797 | 1.6086 | 0.0041 | 0.7997 | -0.0320 | 0.0535 |
| frozen_synthetic_shape_eml_centered_ambiguity_seed0 | 0 | eml_centered_ambiguity | 0.2109 | 0.1797 | 1.6086 | 0.0041 | 0.7997 | -0.0320 | 0.0704 |
| frozen_synthetic_shape_eml_centered_ambiguity_seed1 | 1 | eml_centered_ambiguity | 0.2031 | 0.1875 | 1.6090 | 0.0112 | 0.7998 | -0.0593 | 0.0617 |
| frozen_synthetic_shape_eml_centered_ambiguity_seed1 | 1 | eml_centered_ambiguity | 0.2031 | 0.1875 | 1.6090 | 0.0112 | 0.7998 | -0.0593 | 0.0725 |
| frozen_synthetic_shape_eml_no_ambiguity_seed0 | 0 | eml_no_ambiguity | 0.2109 | 0.1797 | 1.6087 | 0.0044 | 0.7997 | -0.0305 | 0.0542 |
| frozen_synthetic_shape_eml_no_ambiguity_seed0 | 0 | eml_no_ambiguity | 0.2109 | 0.1797 | 1.6087 | 0.0044 | 0.7997 | -0.0305 | 0.0700 |
| frozen_synthetic_shape_eml_no_ambiguity_seed1 | 1 | eml_no_ambiguity | 0.2031 | 0.1875 | 1.6090 | 0.0098 | 0.7998 | -0.0541 | 0.0581 |
| frozen_synthetic_shape_eml_no_ambiguity_seed1 | 1 | eml_no_ambiguity | 0.2031 | 0.1875 | 1.6090 | 0.0098 | 0.7998 | -0.0541 | 0.0720 |
| frozen_synthetic_shape_eml_raw_ambiguity_seed0 | 0 | eml_raw_ambiguity | 0.2109 | 0.1797 | 1.6086 | 0.0041 | 0.7997 | -0.0317 | 0.0565 |
| frozen_synthetic_shape_eml_raw_ambiguity_seed0 | 0 | eml_raw_ambiguity | 0.2109 | 0.1797 | 1.6086 | 0.0041 | 0.7997 | -0.0317 | 0.0816 |
| frozen_synthetic_shape_eml_raw_ambiguity_seed1 | 1 | eml_raw_ambiguity | 0.2031 | 0.1875 | 1.6090 | 0.0109 | 0.7998 | -0.0582 | 0.0626 |
| frozen_synthetic_shape_eml_raw_ambiguity_seed1 | 1 | eml_raw_ambiguity | 0.2031 | 0.1875 | 1.6090 | 0.0109 | 0.7998 | -0.0582 | 0.0803 |
| frozen_synthetic_shape_linear_seed0 | 0 | linear | 0.2031 | 0.1953 | 1.6111 | 0.0209 | 0.8007 | -0.1030 | 0.0245 |
| frozen_synthetic_shape_linear_seed0 | 0 | linear | 0.2031 | 0.1953 | 1.6111 | 0.0209 | 0.8007 | -0.1030 | 0.0339 |
| frozen_synthetic_shape_linear_seed1 | 1 | linear | 0.1953 | 0.1953 | 1.6103 | 0.0213 | 0.8003 | -0.0768 | 0.0236 |
| frozen_synthetic_shape_linear_seed1 | 1 | linear | 0.1953 | 0.1953 | 1.6103 | 0.0213 | 0.8003 | -0.0768 | 0.0257 |
| frozen_synthetic_shape_mlp_seed0 | 0 | mlp | 0.2109 | 0.1953 | 1.6094 | 0.0190 | 0.8000 | -0.1194 | 0.0288 |
| frozen_synthetic_shape_mlp_seed0 | 0 | mlp | 0.2109 | 0.1953 | 1.6094 | 0.0190 | 0.8000 | -0.1194 | 0.0316 |
| frozen_synthetic_shape_mlp_seed1 | 1 | mlp | 0.1953 | 0.1953 | 1.6097 | 0.0181 | 0.8001 | -0.0559 | 0.0275 |
| frozen_synthetic_shape_mlp_seed1 | 1 | mlp | 0.1953 | 0.1953 | 1.6097 | 0.0181 | 0.8001 | -0.0559 | 0.0360 |

## 5. End-To-End Results
### CNN Plus Head
| run_id | seed | model | loss mode | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed0 | 0 | cosine_prototype | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed1 | 1 | cosine_prototype | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_cosine_prototype_ce_seed0 | 0 | cosine_prototype | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_cosine_prototype_ce_seed1 | 1 | cosine_prototype | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_bank_centered_ambiguity | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_bank_centered_ambiguity | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0 | 0 | eml_bank_centered_ambiguity | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1 | 1 | eml_bank_centered_ambiguity | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_centered_ambiguity | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_centered_ambiguity | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_seed0 | 0 | eml_centered_ambiguity | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_seed1 | 1 | eml_centered_ambiguity | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0 | 0 | eml_no_ambiguity | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1 | 1 | eml_no_ambiguity | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_seed0 | 0 | eml_no_ambiguity | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_seed1 | 1 | eml_no_ambiguity | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_pairwise_seed0 | 0 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_pairwise_seed1 | 1 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_seed0 | 0 | linear | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_seed1 | 1 | linear | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_pairwise_seed0 | 0 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_pairwise_seed1 | 1 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_seed0 | 0 | mlp | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_seed1 | 1 | mlp | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed0 | 0 | cosine_prototype | ce_pairwise | 0.1953 | 0.1953 | 2.1308 | 0.4646 | 1.0727 | -1.4424 | 1.6018 |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed0 | 0 | cosine_prototype | ce_pairwise | 0.1953 | 0.1953 | 2.1308 | 0.4646 | 1.0727 | -1.4424 | 1.9076 |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed1 | 1 | cosine_prototype | ce_pairwise | 0.1953 | 0.1953 | 1.9174 | 0.3343 | 0.9475 | -1.1067 | 1.5520 |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed1 | 1 | cosine_prototype | ce_pairwise | 0.1953 | 0.1953 | 1.9174 | 0.3343 | 0.9475 | -1.1067 | 1.9017 |
| e2e_synthetic_shape_cosine_prototype_ce_seed0 | 0 | cosine_prototype | ce | 0.1953 | 0.1953 | 2.1301 | 0.4642 | 1.0722 | -1.4414 | 1.5885 |
| e2e_synthetic_shape_cosine_prototype_ce_seed0 | 0 | cosine_prototype | ce | 0.1953 | 0.1953 | 2.1301 | 0.4642 | 1.0722 | -1.4414 | 1.7917 |
| e2e_synthetic_shape_cosine_prototype_ce_seed1 | 1 | cosine_prototype | ce | 0.1953 | 0.1953 | 1.9149 | 0.3336 | 0.9467 | -1.1026 | 1.5643 |
| e2e_synthetic_shape_cosine_prototype_ce_seed1 | 1 | cosine_prototype | ce | 0.1953 | 0.1953 | 1.9149 | 0.3336 | 0.9467 | -1.1026 | 1.8419 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_bank_centered_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.8235 | 0.3146 | 0.9236 | -0.9080 | 1.6746 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_bank_centered_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.8235 | 0.3146 | 0.9236 | -0.9080 | 1.9433 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_bank_centered_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.7226 | 0.2064 | 0.8568 | -0.6922 | 1.2399 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_bank_centered_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.7226 | 0.2064 | 0.8568 | -0.6922 | 1.9458 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0 | 0 | eml_bank_centered_ambiguity | ce | 0.1953 | 0.1953 | 1.8218 | 0.3132 | 0.9226 | -0.9054 | 1.6436 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0 | 0 | eml_bank_centered_ambiguity | ce | 0.1953 | 0.1953 | 1.8218 | 0.3132 | 0.9226 | -0.9054 | 1.8784 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1 | 1 | eml_bank_centered_ambiguity | ce | 0.1953 | 0.1953 | 1.7200 | 0.2050 | 0.8555 | -0.6860 | 1.6286 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1 | 1 | eml_bank_centered_ambiguity | ce | 0.1953 | 0.1953 | 1.7200 | 0.2050 | 0.8555 | -0.6860 | 1.9610 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_centered_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.8309 | 0.3057 | 0.9199 | -0.9458 | 1.5948 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_centered_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.8309 | 0.3057 | 0.9199 | -0.9458 | 1.8337 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_centered_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.7225 | 0.2164 | 0.8598 | -0.6823 | 1.5929 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_centered_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.7225 | 0.2164 | 0.8598 | -0.6823 | 1.8119 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0 | 0 | eml_centered_ambiguity | ce | 0.1953 | 0.1953 | 1.8294 | 0.3036 | 0.9185 | -0.9432 | 1.5891 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0 | 0 | eml_centered_ambiguity | ce | 0.1953 | 0.1953 | 1.8294 | 0.3036 | 0.9185 | -0.9432 | 1.8017 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1 | 1 | eml_centered_ambiguity | ce | 0.1953 | 0.1953 | 1.7236 | 0.2188 | 0.8608 | -0.6850 | 1.6771 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1 | 1 | eml_centered_ambiguity | ce | 0.1953 | 0.1953 | 1.7236 | 0.2188 | 0.8608 | -0.6850 | 1.8427 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0 | 0 | eml_no_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.8190 | 0.2941 | 0.9118 | -0.9224 | 1.5906 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0 | 0 | eml_no_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.8190 | 0.2941 | 0.9118 | -0.9224 | 1.7656 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1 | 1 | eml_no_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.7155 | 0.2059 | 0.8549 | -0.6624 | 1.6687 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1 | 1 | eml_no_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.7155 | 0.2059 | 0.8549 | -0.6624 | 1.8576 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed0 | 0 | eml_no_ambiguity | ce | 0.1953 | 0.1953 | 1.8186 | 0.2927 | 0.9110 | -0.9216 | 1.6403 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed0 | 0 | eml_no_ambiguity | ce | 0.1953 | 0.1953 | 1.8186 | 0.2927 | 0.9110 | -0.9216 | 1.8403 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed1 | 1 | eml_no_ambiguity | ce | 0.1953 | 0.1953 | 1.7152 | 0.2073 | 0.8552 | -0.6613 | 1.5957 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed1 | 1 | eml_no_ambiguity | ce | 0.1953 | 0.1953 | 1.7152 | 0.2073 | 0.8552 | -0.6613 | 1.9124 |
| e2e_synthetic_shape_linear_ce_pairwise_seed0 | 0 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_linear_ce_pairwise_seed0 | 0 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_linear_ce_pairwise_seed1 | 1 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_linear_ce_pairwise_seed1 | 1 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_linear_ce_seed0 | 0 | linear | ce | 0.1953 | 0.1953 | 1.7710 | 0.2183 | 0.8730 | -0.7968 | 1.5546 |
| e2e_synthetic_shape_linear_ce_seed0 | 0 | linear | ce | 0.1953 | 0.1953 | 1.7710 | 0.2183 | 0.8730 | -0.7968 | 1.7110 |
| e2e_synthetic_shape_linear_ce_seed1 | 1 | linear | ce | 0.1953 | 0.1953 | 1.7118 | 0.1711 | 0.8457 | -0.6377 | 1.6229 |
| e2e_synthetic_shape_linear_ce_seed1 | 1 | linear | ce | 0.1953 | 0.1953 | 1.7118 | 0.1711 | 0.8457 | -0.6377 | 1.7398 |
| e2e_synthetic_shape_mlp_ce_pairwise_seed0 | 0 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_mlp_ce_pairwise_seed0 | 0 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_mlp_ce_pairwise_seed1 | 1 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_mlp_ce_pairwise_seed1 | 1 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_mlp_ce_seed0 | 0 | mlp | ce | 0.1953 | 0.1953 | 1.9274 | 0.2861 | 0.9303 | -1.0945 | 1.5509 |
| e2e_synthetic_shape_mlp_ce_seed0 | 0 | mlp | ce | 0.1953 | 0.1953 | 1.9274 | 0.2861 | 0.9303 | -1.0945 | 1.7102 |
| e2e_synthetic_shape_mlp_ce_seed1 | 1 | mlp | ce | 0.1953 | 0.1953 | 2.0963 | 0.3118 | 0.9736 | -1.3277 | 1.5602 |
| e2e_synthetic_shape_mlp_ce_seed1 | 1 | mlp | ce | 0.1953 | 0.1953 | 2.0963 | 0.3118 | 0.9736 | -1.3277 | 1.7917 |

## 6. CE-Only Comparison
### End-To-End CE Only
| run_id | seed | model | loss mode | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e_cifar10_cosine_prototype_ce_seed0 | 0 | cosine_prototype | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_cosine_prototype_ce_seed1 | 1 | cosine_prototype | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0 | 0 | eml_bank_centered_ambiguity | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1 | 1 | eml_bank_centered_ambiguity | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_seed0 | 0 | eml_centered_ambiguity | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_seed1 | 1 | eml_centered_ambiguity | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_seed0 | 0 | eml_no_ambiguity | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_seed1 | 1 | eml_no_ambiguity | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_seed0 | 0 | linear | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_seed1 | 1 | linear | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_seed0 | 0 | mlp | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_seed1 | 1 | mlp | ce | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_cosine_prototype_ce_seed0 | 0 | cosine_prototype | ce | 0.1953 | 0.1953 | 2.1301 | 0.4642 | 1.0722 | -1.4414 | 1.5885 |
| e2e_synthetic_shape_cosine_prototype_ce_seed0 | 0 | cosine_prototype | ce | 0.1953 | 0.1953 | 2.1301 | 0.4642 | 1.0722 | -1.4414 | 1.7917 |
| e2e_synthetic_shape_cosine_prototype_ce_seed1 | 1 | cosine_prototype | ce | 0.1953 | 0.1953 | 1.9149 | 0.3336 | 0.9467 | -1.1026 | 1.5643 |
| e2e_synthetic_shape_cosine_prototype_ce_seed1 | 1 | cosine_prototype | ce | 0.1953 | 0.1953 | 1.9149 | 0.3336 | 0.9467 | -1.1026 | 1.8419 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0 | 0 | eml_bank_centered_ambiguity | ce | 0.1953 | 0.1953 | 1.8218 | 0.3132 | 0.9226 | -0.9054 | 1.6436 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0 | 0 | eml_bank_centered_ambiguity | ce | 0.1953 | 0.1953 | 1.8218 | 0.3132 | 0.9226 | -0.9054 | 1.8784 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1 | 1 | eml_bank_centered_ambiguity | ce | 0.1953 | 0.1953 | 1.7200 | 0.2050 | 0.8555 | -0.6860 | 1.6286 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1 | 1 | eml_bank_centered_ambiguity | ce | 0.1953 | 0.1953 | 1.7200 | 0.2050 | 0.8555 | -0.6860 | 1.9610 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0 | 0 | eml_centered_ambiguity | ce | 0.1953 | 0.1953 | 1.8294 | 0.3036 | 0.9185 | -0.9432 | 1.5891 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0 | 0 | eml_centered_ambiguity | ce | 0.1953 | 0.1953 | 1.8294 | 0.3036 | 0.9185 | -0.9432 | 1.8017 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1 | 1 | eml_centered_ambiguity | ce | 0.1953 | 0.1953 | 1.7236 | 0.2188 | 0.8608 | -0.6850 | 1.6771 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1 | 1 | eml_centered_ambiguity | ce | 0.1953 | 0.1953 | 1.7236 | 0.2188 | 0.8608 | -0.6850 | 1.8427 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed0 | 0 | eml_no_ambiguity | ce | 0.1953 | 0.1953 | 1.8186 | 0.2927 | 0.9110 | -0.9216 | 1.6403 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed0 | 0 | eml_no_ambiguity | ce | 0.1953 | 0.1953 | 1.8186 | 0.2927 | 0.9110 | -0.9216 | 1.8403 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed1 | 1 | eml_no_ambiguity | ce | 0.1953 | 0.1953 | 1.7152 | 0.2073 | 0.8552 | -0.6613 | 1.5957 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed1 | 1 | eml_no_ambiguity | ce | 0.1953 | 0.1953 | 1.7152 | 0.2073 | 0.8552 | -0.6613 | 1.9124 |
| e2e_synthetic_shape_linear_ce_seed0 | 0 | linear | ce | 0.1953 | 0.1953 | 1.7710 | 0.2183 | 0.8730 | -0.7968 | 1.5546 |
| e2e_synthetic_shape_linear_ce_seed0 | 0 | linear | ce | 0.1953 | 0.1953 | 1.7710 | 0.2183 | 0.8730 | -0.7968 | 1.7110 |
| e2e_synthetic_shape_linear_ce_seed1 | 1 | linear | ce | 0.1953 | 0.1953 | 1.7118 | 0.1711 | 0.8457 | -0.6377 | 1.6229 |
| e2e_synthetic_shape_linear_ce_seed1 | 1 | linear | ce | 0.1953 | 0.1953 | 1.7118 | 0.1711 | 0.8457 | -0.6377 | 1.7398 |
| e2e_synthetic_shape_mlp_ce_seed0 | 0 | mlp | ce | 0.1953 | 0.1953 | 1.9274 | 0.2861 | 0.9303 | -1.0945 | 1.5509 |
| e2e_synthetic_shape_mlp_ce_seed0 | 0 | mlp | ce | 0.1953 | 0.1953 | 1.9274 | 0.2861 | 0.9303 | -1.0945 | 1.7102 |
| e2e_synthetic_shape_mlp_ce_seed1 | 1 | mlp | ce | 0.1953 | 0.1953 | 2.0963 | 0.3118 | 0.9736 | -1.3277 | 1.5602 |
| e2e_synthetic_shape_mlp_ce_seed1 | 1 | mlp | ce | 0.1953 | 0.1953 | 2.0963 | 0.3118 | 0.9736 | -1.3277 | 1.7917 |

## 7. CE + Pairwise Comparison
### End-To-End CE + Prototype Pairwise
| run_id | seed | model | loss mode | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed0 | 0 | cosine_prototype | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed1 | 1 | cosine_prototype | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_bank_centered_ambiguity | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_bank_centered_ambiguity | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_centered_ambiguity | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_centered_ambiguity | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0 | 0 | eml_no_ambiguity | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1 | 1 | eml_no_ambiguity | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_pairwise_seed0 | 0 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_linear_ce_pairwise_seed1 | 1 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_pairwise_seed0 | 0 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_mlp_ce_pairwise_seed1 | 1 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed0 | 0 | cosine_prototype | ce_pairwise | 0.1953 | 0.1953 | 2.1308 | 0.4646 | 1.0727 | -1.4424 | 1.6018 |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed0 | 0 | cosine_prototype | ce_pairwise | 0.1953 | 0.1953 | 2.1308 | 0.4646 | 1.0727 | -1.4424 | 1.9076 |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed1 | 1 | cosine_prototype | ce_pairwise | 0.1953 | 0.1953 | 1.9174 | 0.3343 | 0.9475 | -1.1067 | 1.5520 |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed1 | 1 | cosine_prototype | ce_pairwise | 0.1953 | 0.1953 | 1.9174 | 0.3343 | 0.9475 | -1.1067 | 1.9017 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_bank_centered_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.8235 | 0.3146 | 0.9236 | -0.9080 | 1.6746 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_bank_centered_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.8235 | 0.3146 | 0.9236 | -0.9080 | 1.9433 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_bank_centered_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.7226 | 0.2064 | 0.8568 | -0.6922 | 1.2399 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_bank_centered_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.7226 | 0.2064 | 0.8568 | -0.6922 | 1.9458 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_centered_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.8309 | 0.3057 | 0.9199 | -0.9458 | 1.5948 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0 | 0 | eml_centered_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.8309 | 0.3057 | 0.9199 | -0.9458 | 1.8337 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_centered_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.7225 | 0.2164 | 0.8598 | -0.6823 | 1.5929 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1 | 1 | eml_centered_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.7225 | 0.2164 | 0.8598 | -0.6823 | 1.8119 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0 | 0 | eml_no_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.8190 | 0.2941 | 0.9118 | -0.9224 | 1.5906 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0 | 0 | eml_no_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.8190 | 0.2941 | 0.9118 | -0.9224 | 1.7656 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1 | 1 | eml_no_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.7155 | 0.2059 | 0.8549 | -0.6624 | 1.6687 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1 | 1 | eml_no_ambiguity | ce_pairwise | 0.1953 | 0.1953 | 1.7155 | 0.2059 | 0.8549 | -0.6624 | 1.8576 |
| e2e_synthetic_shape_linear_ce_pairwise_seed0 | 0 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_linear_ce_pairwise_seed0 | 0 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_linear_ce_pairwise_seed1 | 1 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_linear_ce_pairwise_seed1 | 1 | linear | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_mlp_ce_pairwise_seed0 | 0 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_mlp_ce_pairwise_seed0 | 0 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_mlp_ce_pairwise_seed1 | 1 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_mlp_ce_pairwise_seed1 | 1 | mlp | ce_pairwise | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |

## 8. Calibration Metrics
ECE and Brier score are included in the result tables. Lower is better for both.

## 9. Hard-Negative Margin Analysis
Margin is positive-logit minus hardest-negative-logit; larger is better.

## 10. EML Drive/Resistance Analysis
| run_id | model | pos drive | hard neg drive | pos resistance | hard neg resistance | uncertainty | ambiguity | noise corr | occlusion corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0 | eml_bank_centered_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1 | eml_bank_centered_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0 | eml_bank_centered_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1 | eml_bank_centered_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0 | eml_centered_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1 | eml_centered_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_seed0 | eml_centered_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_centered_ambiguity_ce_seed1 | eml_centered_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0 | eml_no_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1 | eml_no_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_seed0 | eml_no_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_cifar10_eml_no_ambiguity_ce_seed1 | eml_no_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0 | eml_bank_centered_ambiguity | 0.4354 | 3.6704 | 2.1967 | 1.0654 | 0.4682 | 1.5237 | -0.0362 | 0.0279 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0 | eml_bank_centered_ambiguity | 0.4354 | 3.6704 | 2.1967 | 1.0654 | 0.4682 | 1.5237 | -0.0362 | 0.0279 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1 | eml_bank_centered_ambiguity | 0.4023 | 3.0827 | 1.9329 | 1.2868 | 0.5091 | 1.2221 | -0.0432 | 0.0484 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1 | eml_bank_centered_ambiguity | 0.4023 | 3.0827 | 1.9329 | 1.2868 | 0.5091 | 1.2221 | -0.0432 | 0.0484 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0 | eml_bank_centered_ambiguity | 0.4637 | 3.6715 | 2.2040 | 1.0916 | 0.4668 | 1.5324 | -0.0362 | 0.0281 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0 | eml_bank_centered_ambiguity | 0.4637 | 3.6715 | 2.2040 | 1.0916 | 0.4668 | 1.5324 | -0.0362 | 0.0281 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1 | eml_bank_centered_ambiguity | 0.4396 | 3.0728 | 1.9234 | 1.2974 | 0.5096 | 1.2134 | -0.0460 | 0.0565 |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1 | eml_bank_centered_ambiguity | 0.4396 | 3.0728 | 1.9234 | 1.2974 | 0.5096 | 1.2134 | -0.0460 | 0.0565 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0 | eml_centered_ambiguity | 0.4675 | 4.0281 | 2.3111 | 1.2873 | 0.3935 | 1.7116 | -0.0375 | 0.0302 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0 | eml_centered_ambiguity | 0.4675 | 4.0281 | 2.3111 | 1.2873 | 0.3935 | 1.7116 | -0.0375 | 0.0302 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1 | eml_centered_ambiguity | 0.3640 | 2.9222 | 1.8943 | 0.9786 | 0.4717 | 1.2176 | -0.0361 | 0.0291 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1 | eml_centered_ambiguity | 0.3640 | 2.9222 | 1.8943 | 0.9786 | 0.4717 | 1.2176 | -0.0361 | 0.0291 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0 | eml_centered_ambiguity | 0.4889 | 4.0274 | 2.3187 | 1.3122 | 0.3922 | 1.7210 | -0.0372 | 0.0304 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0 | eml_centered_ambiguity | 0.4889 | 4.0274 | 2.3187 | 1.3122 | 0.3922 | 1.7210 | -0.0372 | 0.0304 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1 | eml_centered_ambiguity | 0.3788 | 2.9223 | 1.8949 | 0.9650 | 0.4714 | 1.2184 | -0.0358 | 0.0283 |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1 | eml_centered_ambiguity | 0.3788 | 2.9223 | 1.8949 | 0.9650 | 0.4714 | 1.2184 | -0.0358 | 0.0283 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0 | eml_no_ambiguity | 0.4870 | 4.0749 | 0.9094 | 0.9100 | 0.7093 | 0.0000 | 0.0058 | 0.1363 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0 | eml_no_ambiguity | 0.4870 | 4.0749 | 0.9094 | 0.9100 | 0.7093 | 0.0000 | 0.0058 | 0.1363 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1 | eml_no_ambiguity | 0.3945 | 2.9738 | 0.9006 | 0.9009 | 0.7007 | 0.0000 | 0.0983 | 0.2022 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1 | eml_no_ambiguity | 0.3945 | 2.9738 | 0.9006 | 0.9009 | 0.7007 | 0.0000 | 0.0983 | 0.2022 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed0 | eml_no_ambiguity | 0.5088 | 4.0797 | 0.9092 | 0.9097 | 0.7090 | 0.0000 | 0.0006 | 0.1422 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed0 | eml_no_ambiguity | 0.5088 | 4.0797 | 0.9092 | 0.9097 | 0.7090 | 0.0000 | 0.0006 | 0.1422 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed1 | eml_no_ambiguity | 0.4172 | 2.9622 | 0.9007 | 0.9010 | 0.7007 | 0.0000 | 0.0955 | 0.2063 |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed1 | eml_no_ambiguity | 0.4172 | 2.9622 | 0.9007 | 0.9010 | 0.7007 | 0.0000 | 0.0955 | 0.2063 |
| frozen_cifar10_eml_centered_ambiguity_seed0 | eml_centered_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_cifar10_eml_centered_ambiguity_seed1 | eml_centered_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_cifar10_eml_no_ambiguity_seed0 | eml_no_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_cifar10_eml_no_ambiguity_seed1 | eml_no_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_cifar10_eml_raw_ambiguity_seed0 | eml_raw_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_cifar10_eml_raw_ambiguity_seed1 | eml_raw_ambiguity | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| frozen_synthetic_shape_eml_centered_ambiguity_seed0 | eml_centered_ambiguity | 0.2842 | 0.4819 | 1.2610 | 1.2155 | 0.7656 | 0.2966 | 0.0798 | 0.0295 |
| frozen_synthetic_shape_eml_centered_ambiguity_seed0 | eml_centered_ambiguity | 0.2842 | 0.4819 | 1.2610 | 1.2155 | 0.7656 | 0.2966 | 0.0798 | 0.0295 |
| frozen_synthetic_shape_eml_centered_ambiguity_seed1 | eml_centered_ambiguity | -0.4255 | 0.2000 | 0.5031 | 0.3515 | 0.6489 | -0.3415 | -0.0173 | 0.1184 |
| frozen_synthetic_shape_eml_centered_ambiguity_seed1 | eml_centered_ambiguity | -0.4255 | 0.2000 | 0.5031 | 0.3515 | 0.6489 | -0.3415 | -0.0173 | 0.1184 |
| frozen_synthetic_shape_eml_no_ambiguity_seed0 | eml_no_ambiguity | 0.2820 | 0.4795 | 0.9013 | 0.9005 | 0.7010 | 0.0000 | 0.0085 | 0.1029 |
| frozen_synthetic_shape_eml_no_ambiguity_seed0 | eml_no_ambiguity | 0.2820 | 0.4795 | 0.9013 | 0.9005 | 0.7010 | 0.0000 | 0.0085 | 0.1029 |
| frozen_synthetic_shape_eml_no_ambiguity_seed1 | eml_no_ambiguity | -0.4295 | 0.1950 | 0.8948 | 0.8938 | 0.6947 | 0.0000 | 0.0351 | -0.0605 |
| frozen_synthetic_shape_eml_no_ambiguity_seed1 | eml_no_ambiguity | -0.4295 | 0.1950 | 0.8948 | 0.8938 | 0.6947 | 0.0000 | 0.0351 | -0.0605 |
| frozen_synthetic_shape_eml_raw_ambiguity_seed0 | eml_raw_ambiguity | 0.2849 | 0.4821 | 2.6032 | 2.5580 | 0.7204 | 1.6839 | 0.0806 | 0.0277 |
| frozen_synthetic_shape_eml_raw_ambiguity_seed0 | eml_raw_ambiguity | 0.2849 | 0.4821 | 2.6032 | 2.5580 | 0.7204 | 1.6839 | 0.0806 | 0.0277 |
| frozen_synthetic_shape_eml_raw_ambiguity_seed1 | eml_raw_ambiguity | -0.4234 | 0.1970 | 2.0140 | 1.8649 | 0.7721 | 1.0462 | -0.0165 | 0.1208 |
| frozen_synthetic_shape_eml_raw_ambiguity_seed1 | eml_raw_ambiguity | -0.4234 | 0.1970 | 2.0140 | 1.8649 | 0.7721 | 1.0462 | -0.0165 | 0.1208 |

## 11. Robustness Under Noise/Occlusion
Resistance-noise and resistance-occlusion correlations are reported when synthetic metadata is available. MISSING means the head did not expose resistance or the dataset did not provide the field.

## 12. Statistical Confidence Intervals
| experiment | dataset | seed | loss mode | comparison | delta acc | 95% CI low | 95% CI high |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| end_to_end | synthetic_shape | 0 | ce | eml_centered_ambiguity - linear | 0.0000 | 0.0000 | 0.0000 |
| end_to_end | synthetic_shape | 0 | ce | eml_centered_ambiguity - mlp | 0.0000 | 0.0000 | 0.0000 |
| end_to_end | synthetic_shape | 0 | ce | eml_centered_ambiguity - cosine_prototype | 0.0000 | 0.0000 | 0.0000 |
| end_to_end | synthetic_shape | 0 | ce_pairwise | eml_centered_ambiguity - cosine_prototype | 0.0000 | 0.0000 | 0.0000 |
| end_to_end | synthetic_shape | 1 | ce | eml_centered_ambiguity - linear | 0.0000 | 0.0000 | 0.0000 |
| end_to_end | synthetic_shape | 1 | ce | eml_centered_ambiguity - mlp | 0.0000 | 0.0000 | 0.0000 |
| end_to_end | synthetic_shape | 1 | ce | eml_centered_ambiguity - cosine_prototype | 0.0000 | 0.0000 | 0.0000 |
| end_to_end | synthetic_shape | 1 | ce_pairwise | eml_centered_ambiguity - cosine_prototype | 0.0000 | 0.0000 | 0.0000 |
| frozen_features | synthetic_shape | 0 | ce | eml_centered_ambiguity - linear | 0.0078 | -0.1016 | 0.1172 |
| frozen_features | synthetic_shape | 0 | ce | eml_centered_ambiguity - mlp | 0.0000 | -0.1172 | 0.1094 |
| frozen_features | synthetic_shape | 0 | ce | eml_centered_ambiguity - cosine_prototype | 0.0078 | -0.1016 | 0.1172 |
| frozen_features | synthetic_shape | 1 | ce | eml_centered_ambiguity - linear | 0.0078 | -0.1016 | 0.1094 |
| frozen_features | synthetic_shape | 1 | ce | eml_centered_ambiguity - mlp | 0.0078 | -0.1016 | 0.1094 |
| frozen_features | synthetic_shape | 1 | ce | eml_centered_ambiguity - cosine_prototype | 0.0078 | -0.1016 | 0.1094 |

## 13. Which Claim Is Supported
The evidence is mixed: centered EML wins 5/14 paired comparisons.

## 14. Raw Artifacts
- `e2e_cifar10_cosine_prototype_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_cosine_prototype_ce_pairwise_seed0`
- `e2e_cifar10_cosine_prototype_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_cosine_prototype_ce_pairwise_seed1`
- `e2e_cifar10_cosine_prototype_ce_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_cosine_prototype_ce_seed0`
- `e2e_cifar10_cosine_prototype_ce_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_cosine_prototype_ce_seed1`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0`
- `e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1`
- `e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0`
- `e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1`
- `e2e_cifar10_eml_centered_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_centered_ambiguity_ce_seed0`
- `e2e_cifar10_eml_centered_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_centered_ambiguity_ce_seed1`
- `e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0`
- `e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1`
- `e2e_cifar10_eml_no_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_no_ambiguity_ce_seed0`
- `e2e_cifar10_eml_no_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_eml_no_ambiguity_ce_seed1`
- `e2e_cifar10_linear_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_linear_ce_pairwise_seed0`
- `e2e_cifar10_linear_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_linear_ce_pairwise_seed1`
- `e2e_cifar10_linear_ce_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_linear_ce_seed0`
- `e2e_cifar10_linear_ce_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_linear_ce_seed1`
- `e2e_cifar10_mlp_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_mlp_ce_pairwise_seed0`
- `e2e_cifar10_mlp_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_mlp_ce_pairwise_seed1`
- `e2e_cifar10_mlp_ce_seed0`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_mlp_ce_seed0`
- `e2e_cifar10_mlp_ce_seed1`: `reports/head_ablation/runs/20260424_081741_e2e_cifar10_mlp_ce_seed1`
- `e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081506_e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed0`
- `e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_085104_e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed0`
- `e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081523_e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed1`
- `e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_085123_e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed1`
- `e2e_synthetic_shape_cosine_prototype_ce_seed0`: `reports/head_ablation/runs/20260424_081505_e2e_synthetic_shape_cosine_prototype_ce_seed0`
- `e2e_synthetic_shape_cosine_prototype_ce_seed0`: `reports/head_ablation/runs/20260424_085103_e2e_synthetic_shape_cosine_prototype_ce_seed0`
- `e2e_synthetic_shape_cosine_prototype_ce_seed1`: `reports/head_ablation/runs/20260424_081521_e2e_synthetic_shape_cosine_prototype_ce_seed1`
- `e2e_synthetic_shape_cosine_prototype_ce_seed1`: `reports/head_ablation/runs/20260424_085121_e2e_synthetic_shape_cosine_prototype_ce_seed1`
- `e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081516_e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0`
- `e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_085116_e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0`
- `e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081533_e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1`
- `e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_085135_e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1`
- `e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_081515_e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0`
- `e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_085114_e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0`
- `e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_081531_e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1`
- `e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_085133_e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1`
- `e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081513_e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0`
- `e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_085112_e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0`
- `e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081529_e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1`
- `e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_085131_e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1`
- `e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_081511_e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0`
- `e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_085110_e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0`
- `e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_081528_e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1`
- `e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_085129_e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1`
- `e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081510_e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0`
- `e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_085108_e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0`
- `e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081526_e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1`
- `e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_085127_e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1`
- `e2e_synthetic_shape_eml_no_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_081508_e2e_synthetic_shape_eml_no_ambiguity_ce_seed0`
- `e2e_synthetic_shape_eml_no_ambiguity_ce_seed0`: `reports/head_ablation/runs/20260424_085106_e2e_synthetic_shape_eml_no_ambiguity_ce_seed0`
- `e2e_synthetic_shape_eml_no_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_081524_e2e_synthetic_shape_eml_no_ambiguity_ce_seed1`
- `e2e_synthetic_shape_eml_no_ambiguity_ce_seed1`: `reports/head_ablation/runs/20260424_085125_e2e_synthetic_shape_eml_no_ambiguity_ce_seed1`
- `e2e_synthetic_shape_linear_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081503_e2e_synthetic_shape_linear_ce_pairwise_seed0`
- `e2e_synthetic_shape_linear_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_085101_e2e_synthetic_shape_linear_ce_pairwise_seed0`
- `e2e_synthetic_shape_linear_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081520_e2e_synthetic_shape_linear_ce_pairwise_seed1`
- `e2e_synthetic_shape_linear_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_085119_e2e_synthetic_shape_linear_ce_pairwise_seed1`
- `e2e_synthetic_shape_linear_ce_seed0`: `reports/head_ablation/runs/20260424_081501_e2e_synthetic_shape_linear_ce_seed0`
- `e2e_synthetic_shape_linear_ce_seed0`: `reports/head_ablation/runs/20260424_085059_e2e_synthetic_shape_linear_ce_seed0`
- `e2e_synthetic_shape_linear_ce_seed1`: `reports/head_ablation/runs/20260424_081518_e2e_synthetic_shape_linear_ce_seed1`
- `e2e_synthetic_shape_linear_ce_seed1`: `reports/head_ablation/runs/20260424_085118_e2e_synthetic_shape_linear_ce_seed1`
- `e2e_synthetic_shape_mlp_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_081505_e2e_synthetic_shape_mlp_ce_pairwise_seed0`
- `e2e_synthetic_shape_mlp_ce_pairwise_seed0`: `reports/head_ablation/runs/20260424_085103_e2e_synthetic_shape_mlp_ce_pairwise_seed0`
- `e2e_synthetic_shape_mlp_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_081521_e2e_synthetic_shape_mlp_ce_pairwise_seed1`
- `e2e_synthetic_shape_mlp_ce_pairwise_seed1`: `reports/head_ablation/runs/20260424_085121_e2e_synthetic_shape_mlp_ce_pairwise_seed1`
- `e2e_synthetic_shape_mlp_ce_seed0`: `reports/head_ablation/runs/20260424_081503_e2e_synthetic_shape_mlp_ce_seed0`
- `e2e_synthetic_shape_mlp_ce_seed0`: `reports/head_ablation/runs/20260424_085101_e2e_synthetic_shape_mlp_ce_seed0`
- `e2e_synthetic_shape_mlp_ce_seed1`: `reports/head_ablation/runs/20260424_081520_e2e_synthetic_shape_mlp_ce_seed1`
- `e2e_synthetic_shape_mlp_ce_seed1`: `reports/head_ablation/runs/20260424_085119_e2e_synthetic_shape_mlp_ce_seed1`
- `frozen_cifar10_cosine_prototype_seed0`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_cosine_prototype_seed0`
- `frozen_cifar10_cosine_prototype_seed1`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_cosine_prototype_seed1`
- `frozen_cifar10_eml_centered_ambiguity_seed0`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_eml_centered_ambiguity_seed0`
- `frozen_cifar10_eml_centered_ambiguity_seed1`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_eml_centered_ambiguity_seed1`
- `frozen_cifar10_eml_no_ambiguity_seed0`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_eml_no_ambiguity_seed0`
- `frozen_cifar10_eml_no_ambiguity_seed1`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_eml_no_ambiguity_seed1`
- `frozen_cifar10_eml_raw_ambiguity_seed0`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_eml_raw_ambiguity_seed0`
- `frozen_cifar10_eml_raw_ambiguity_seed1`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_eml_raw_ambiguity_seed1`
- `frozen_cifar10_linear_seed0`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_linear_seed0`
- `frozen_cifar10_linear_seed1`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_linear_seed1`
- `frozen_cifar10_mlp_seed0`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_mlp_seed0`
- `frozen_cifar10_mlp_seed1`: `reports/head_ablation/runs/20260424_081740_frozen_cifar10_mlp_seed1`
- `frozen_synthetic_shape_cosine_prototype_seed0`: `reports/head_ablation/runs/20260424_081433_frozen_synthetic_shape_cosine_prototype_seed0`
- `frozen_synthetic_shape_cosine_prototype_seed0`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_cosine_prototype_seed0`
- `frozen_synthetic_shape_cosine_prototype_seed1`: `reports/head_ablation/runs/20260424_081434_frozen_synthetic_shape_cosine_prototype_seed1`
- `frozen_synthetic_shape_cosine_prototype_seed1`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_cosine_prototype_seed1`
- `frozen_synthetic_shape_eml_centered_ambiguity_seed0`: `reports/head_ablation/runs/20260424_081433_frozen_synthetic_shape_eml_centered_ambiguity_seed0`
- `frozen_synthetic_shape_eml_centered_ambiguity_seed0`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_eml_centered_ambiguity_seed0`
- `frozen_synthetic_shape_eml_centered_ambiguity_seed1`: `reports/head_ablation/runs/20260424_081434_frozen_synthetic_shape_eml_centered_ambiguity_seed1`
- `frozen_synthetic_shape_eml_centered_ambiguity_seed1`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_eml_centered_ambiguity_seed1`
- `frozen_synthetic_shape_eml_no_ambiguity_seed0`: `reports/head_ablation/runs/20260424_081433_frozen_synthetic_shape_eml_no_ambiguity_seed0`
- `frozen_synthetic_shape_eml_no_ambiguity_seed0`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_eml_no_ambiguity_seed0`
- `frozen_synthetic_shape_eml_no_ambiguity_seed1`: `reports/head_ablation/runs/20260424_081434_frozen_synthetic_shape_eml_no_ambiguity_seed1`
- `frozen_synthetic_shape_eml_no_ambiguity_seed1`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_eml_no_ambiguity_seed1`
- `frozen_synthetic_shape_eml_raw_ambiguity_seed0`: `reports/head_ablation/runs/20260424_081433_frozen_synthetic_shape_eml_raw_ambiguity_seed0`
- `frozen_synthetic_shape_eml_raw_ambiguity_seed0`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_eml_raw_ambiguity_seed0`
- `frozen_synthetic_shape_eml_raw_ambiguity_seed1`: `reports/head_ablation/runs/20260424_081434_frozen_synthetic_shape_eml_raw_ambiguity_seed1`
- `frozen_synthetic_shape_eml_raw_ambiguity_seed1`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_eml_raw_ambiguity_seed1`
- `frozen_synthetic_shape_linear_seed0`: `reports/head_ablation/runs/20260424_081433_frozen_synthetic_shape_linear_seed0`
- `frozen_synthetic_shape_linear_seed0`: `reports/head_ablation/runs/20260424_085043_frozen_synthetic_shape_linear_seed0`
- `frozen_synthetic_shape_linear_seed1`: `reports/head_ablation/runs/20260424_081434_frozen_synthetic_shape_linear_seed1`
- `frozen_synthetic_shape_linear_seed1`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_linear_seed1`
- `frozen_synthetic_shape_mlp_seed0`: `reports/head_ablation/runs/20260424_081433_frozen_synthetic_shape_mlp_seed0`
- `frozen_synthetic_shape_mlp_seed0`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_mlp_seed0`
- `frozen_synthetic_shape_mlp_seed1`: `reports/head_ablation/runs/20260424_081434_frozen_synthetic_shape_mlp_seed1`
- `frozen_synthetic_shape_mlp_seed1`: `reports/head_ablation/runs/20260424_085044_frozen_synthetic_shape_mlp_seed1`

## 15. Appendix: Commands
- `pytest`
- `python scripts/run_head_ablation.py --dataset synthetic_shape --mode smoke --seeds 0 1`
- `python scripts/run_cnn_head_end_to_end_ablation.py --dataset synthetic_shape --mode smoke --seeds 0 1`
- `python scripts/generate_head_ablation_report.py`
