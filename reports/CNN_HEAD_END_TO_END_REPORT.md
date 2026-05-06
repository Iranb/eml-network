# CNN Head End-to-End Report

This report is generated from end-to-end head ablation run artifacts only.

| run_id | status | model | loss mode | dataset | seed | test acc | test loss | reason |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| e2e_synthetic_shape_linear_ce_seed0 | COMPLETED | linear | ce | synthetic_shape | 0 | 0.1953 | 1.7710 |  |
| e2e_synthetic_shape_linear_ce_pairwise_seed0 | NOT RUN | linear | ce_pairwise | synthetic_shape | 0 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_mlp_ce_seed0 | COMPLETED | mlp | ce | synthetic_shape | 0 | 0.1953 | 1.9274 |  |
| e2e_synthetic_shape_mlp_ce_pairwise_seed0 | NOT RUN | mlp | ce_pairwise | synthetic_shape | 0 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_cosine_prototype_ce_seed0 | COMPLETED | cosine_prototype | ce | synthetic_shape | 0 | 0.1953 | 2.1301 |  |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed0 | COMPLETED | cosine_prototype | ce_pairwise | synthetic_shape | 0 | 0.1953 | 2.1308 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed0 | COMPLETED | eml_no_ambiguity | ce | synthetic_shape | 0 | 0.1953 | 1.8186 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0 | COMPLETED | eml_no_ambiguity | ce_pairwise | synthetic_shape | 0 | 0.1953 | 1.8190 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0 | COMPLETED | eml_centered_ambiguity | ce | synthetic_shape | 0 | 0.1953 | 1.8294 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0 | COMPLETED | eml_centered_ambiguity | ce_pairwise | synthetic_shape | 0 | 0.1953 | 1.8309 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0 | COMPLETED | eml_bank_centered_ambiguity | ce | synthetic_shape | 0 | 0.1953 | 1.8218 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0 | COMPLETED | eml_bank_centered_ambiguity | ce_pairwise | synthetic_shape | 0 | 0.1953 | 1.8235 |  |
| e2e_synthetic_shape_linear_ce_seed1 | COMPLETED | linear | ce | synthetic_shape | 1 | 0.1953 | 1.7118 |  |
| e2e_synthetic_shape_linear_ce_pairwise_seed1 | NOT RUN | linear | ce_pairwise | synthetic_shape | 1 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_mlp_ce_seed1 | COMPLETED | mlp | ce | synthetic_shape | 1 | 0.1953 | 2.0963 |  |
| e2e_synthetic_shape_mlp_ce_pairwise_seed1 | NOT RUN | mlp | ce_pairwise | synthetic_shape | 1 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_cosine_prototype_ce_seed1 | COMPLETED | cosine_prototype | ce | synthetic_shape | 1 | 0.1953 | 1.9149 |  |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed1 | COMPLETED | cosine_prototype | ce_pairwise | synthetic_shape | 1 | 0.1953 | 1.9174 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed1 | COMPLETED | eml_no_ambiguity | ce | synthetic_shape | 1 | 0.1953 | 1.7152 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1 | COMPLETED | eml_no_ambiguity | ce_pairwise | synthetic_shape | 1 | 0.1953 | 1.7155 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1 | COMPLETED | eml_centered_ambiguity | ce | synthetic_shape | 1 | 0.1953 | 1.7236 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1 | COMPLETED | eml_centered_ambiguity | ce_pairwise | synthetic_shape | 1 | 0.1953 | 1.7225 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1 | COMPLETED | eml_bank_centered_ambiguity | ce | synthetic_shape | 1 | 0.1953 | 1.7200 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1 | COMPLETED | eml_bank_centered_ambiguity | ce_pairwise | synthetic_shape | 1 | 0.1953 | 1.7226 |  |
| e2e_cifar10_linear_ce_seed0 | NOT RUN | linear | ce | cifar10 | 0 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_linear_ce_pairwise_seed0 | NOT RUN | linear | ce_pairwise | cifar10 | 0 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_mlp_ce_seed0 | NOT RUN | mlp | ce | cifar10 | 0 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_mlp_ce_pairwise_seed0 | NOT RUN | mlp | ce_pairwise | cifar10 | 0 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_cosine_prototype_ce_seed0 | NOT RUN | cosine_prototype | ce | cifar10 | 0 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed0 | NOT RUN | cosine_prototype | ce_pairwise | cifar10 | 0 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_no_ambiguity_ce_seed0 | NOT RUN | eml_no_ambiguity | ce | cifar10 | 0 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed0 | NOT RUN | eml_no_ambiguity | ce_pairwise | cifar10 | 0 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_centered_ambiguity_ce_seed0 | NOT RUN | eml_centered_ambiguity | ce | cifar10 | 0 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed0 | NOT RUN | eml_centered_ambiguity | ce_pairwise | cifar10 | 0 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed0 | NOT RUN | eml_bank_centered_ambiguity | ce | cifar10 | 0 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed0 | NOT RUN | eml_bank_centered_ambiguity | ce_pairwise | cifar10 | 0 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_linear_ce_seed1 | NOT RUN | linear | ce | cifar10 | 1 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_linear_ce_pairwise_seed1 | NOT RUN | linear | ce_pairwise | cifar10 | 1 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_mlp_ce_seed1 | NOT RUN | mlp | ce | cifar10 | 1 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_mlp_ce_pairwise_seed1 | NOT RUN | mlp | ce_pairwise | cifar10 | 1 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_cosine_prototype_ce_seed1 | NOT RUN | cosine_prototype | ce | cifar10 | 1 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_cosine_prototype_ce_pairwise_seed1 | NOT RUN | cosine_prototype | ce_pairwise | cifar10 | 1 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_no_ambiguity_ce_seed1 | NOT RUN | eml_no_ambiguity | ce | cifar10 | 1 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_no_ambiguity_ce_pairwise_seed1 | NOT RUN | eml_no_ambiguity | ce_pairwise | cifar10 | 1 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_centered_ambiguity_ce_seed1 | NOT RUN | eml_centered_ambiguity | ce | cifar10 | 1 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_centered_ambiguity_ce_pairwise_seed1 | NOT RUN | eml_centered_ambiguity | ce_pairwise | cifar10 | 1 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_seed1 | NOT RUN | eml_bank_centered_ambiguity | ce | cifar10 | 1 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_cifar10_eml_bank_centered_ambiguity_ce_pairwise_seed1 | NOT RUN | eml_bank_centered_ambiguity | ce_pairwise | cifar10 | 1 | MISSING | MISSING | OptionalDatasetDependencyError('CIFAR-10 requires a working torchvision installation') |
| e2e_synthetic_shape_linear_ce_seed0 | COMPLETED | linear | ce | synthetic_shape | 0 | 0.1953 | 1.7710 |  |
| e2e_synthetic_shape_linear_ce_pairwise_seed0 | NOT RUN | linear | ce_pairwise | synthetic_shape | 0 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_mlp_ce_seed0 | COMPLETED | mlp | ce | synthetic_shape | 0 | 0.1953 | 1.9274 |  |
| e2e_synthetic_shape_mlp_ce_pairwise_seed0 | NOT RUN | mlp | ce_pairwise | synthetic_shape | 0 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_cosine_prototype_ce_seed0 | COMPLETED | cosine_prototype | ce | synthetic_shape | 0 | 0.1953 | 2.1301 |  |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed0 | COMPLETED | cosine_prototype | ce_pairwise | synthetic_shape | 0 | 0.1953 | 2.1308 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed0 | COMPLETED | eml_no_ambiguity | ce | synthetic_shape | 0 | 0.1953 | 1.8186 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0 | COMPLETED | eml_no_ambiguity | ce_pairwise | synthetic_shape | 0 | 0.1953 | 1.8190 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0 | COMPLETED | eml_centered_ambiguity | ce | synthetic_shape | 0 | 0.1953 | 1.8294 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0 | COMPLETED | eml_centered_ambiguity | ce_pairwise | synthetic_shape | 0 | 0.1953 | 1.8309 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0 | COMPLETED | eml_bank_centered_ambiguity | ce | synthetic_shape | 0 | 0.1953 | 1.8218 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0 | COMPLETED | eml_bank_centered_ambiguity | ce_pairwise | synthetic_shape | 0 | 0.1953 | 1.8235 |  |
| e2e_synthetic_shape_linear_ce_seed1 | COMPLETED | linear | ce | synthetic_shape | 1 | 0.1953 | 1.7118 |  |
| e2e_synthetic_shape_linear_ce_pairwise_seed1 | NOT RUN | linear | ce_pairwise | synthetic_shape | 1 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_mlp_ce_seed1 | COMPLETED | mlp | ce | synthetic_shape | 1 | 0.1953 | 2.0963 |  |
| e2e_synthetic_shape_mlp_ce_pairwise_seed1 | NOT RUN | mlp | ce_pairwise | synthetic_shape | 1 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_cosine_prototype_ce_seed1 | COMPLETED | cosine_prototype | ce | synthetic_shape | 1 | 0.1953 | 1.9149 |  |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed1 | COMPLETED | cosine_prototype | ce_pairwise | synthetic_shape | 1 | 0.1953 | 1.9174 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed1 | COMPLETED | eml_no_ambiguity | ce | synthetic_shape | 1 | 0.1953 | 1.7152 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1 | COMPLETED | eml_no_ambiguity | ce_pairwise | synthetic_shape | 1 | 0.1953 | 1.7155 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1 | COMPLETED | eml_centered_ambiguity | ce | synthetic_shape | 1 | 0.1953 | 1.7236 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1 | COMPLETED | eml_centered_ambiguity | ce_pairwise | synthetic_shape | 1 | 0.1953 | 1.7225 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1 | COMPLETED | eml_bank_centered_ambiguity | ce | synthetic_shape | 1 | 0.1953 | 1.7200 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1 | COMPLETED | eml_bank_centered_ambiguity | ce_pairwise | synthetic_shape | 1 | 0.1953 | 1.7226 |  |
| e2e_synthetic_shape_linear_ce_seed0 | COMPLETED | linear | ce | synthetic_shape | 0 | 0.1953 | 1.7710 |  |
| e2e_synthetic_shape_linear_ce_pairwise_seed0 | NOT RUN | linear | ce_pairwise | synthetic_shape | 0 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_mlp_ce_seed0 | COMPLETED | mlp | ce | synthetic_shape | 0 | 0.1953 | 1.9274 |  |
| e2e_synthetic_shape_mlp_ce_pairwise_seed0 | NOT RUN | mlp | ce_pairwise | synthetic_shape | 0 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_cosine_prototype_ce_seed0 | COMPLETED | cosine_prototype | ce | synthetic_shape | 0 | 0.1953 | 2.1301 |  |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed0 | COMPLETED | cosine_prototype | ce_pairwise | synthetic_shape | 0 | 0.1953 | 2.1308 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed0 | COMPLETED | eml_no_ambiguity | ce | synthetic_shape | 0 | 0.1953 | 1.8186 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed0 | COMPLETED | eml_no_ambiguity | ce_pairwise | synthetic_shape | 0 | 0.1953 | 1.8190 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed0 | COMPLETED | eml_centered_ambiguity | ce | synthetic_shape | 0 | 0.1953 | 1.8294 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed0 | COMPLETED | eml_centered_ambiguity | ce_pairwise | synthetic_shape | 0 | 0.1953 | 1.8309 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed0 | COMPLETED | eml_bank_centered_ambiguity | ce | synthetic_shape | 0 | 0.1953 | 1.8218 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed0 | COMPLETED | eml_bank_centered_ambiguity | ce_pairwise | synthetic_shape | 0 | 0.1953 | 1.8235 |  |
| e2e_synthetic_shape_merc_linear_ce_seed0 | COMPLETED | merc_linear | ce | synthetic_shape | 0 | 0.1953 | 2.9952 |  |
| e2e_synthetic_shape_merc_linear_ce_pairwise_seed0 | NOT RUN | merc_linear | ce_pairwise | synthetic_shape | 0 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_merc_energy_ce_seed0 | COMPLETED | merc_energy | ce | synthetic_shape | 0 | 0.1953 | 2.1472 |  |
| e2e_synthetic_shape_merc_energy_ce_pairwise_seed0 | NOT RUN | merc_energy | ce_pairwise | synthetic_shape | 0 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_merc_block_linear_ce_seed0 | NOT RUN | merc_block_linear | ce | synthetic_shape | 0 | MISSING | MISSING | KeyError('gate') |
| e2e_synthetic_shape_merc_block_linear_ce_pairwise_seed0 | NOT RUN | merc_block_linear | ce_pairwise | synthetic_shape | 0 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_merc_block_energy_ce_seed0 | NOT RUN | merc_block_energy | ce | synthetic_shape | 0 | MISSING | MISSING | KeyError('gate') |
| e2e_synthetic_shape_merc_block_energy_ce_pairwise_seed0 | NOT RUN | merc_block_energy | ce_pairwise | synthetic_shape | 0 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_linear_ce_seed1 | COMPLETED | linear | ce | synthetic_shape | 1 | 0.1953 | 1.7118 |  |
| e2e_synthetic_shape_linear_ce_pairwise_seed1 | NOT RUN | linear | ce_pairwise | synthetic_shape | 1 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_mlp_ce_seed1 | COMPLETED | mlp | ce | synthetic_shape | 1 | 0.1953 | 2.0963 |  |
| e2e_synthetic_shape_mlp_ce_pairwise_seed1 | NOT RUN | mlp | ce_pairwise | synthetic_shape | 1 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_cosine_prototype_ce_seed1 | COMPLETED | cosine_prototype | ce | synthetic_shape | 1 | 0.1953 | 1.9149 |  |
| e2e_synthetic_shape_cosine_prototype_ce_pairwise_seed1 | COMPLETED | cosine_prototype | ce_pairwise | synthetic_shape | 1 | 0.1953 | 1.9174 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_seed1 | COMPLETED | eml_no_ambiguity | ce | synthetic_shape | 1 | 0.1953 | 1.7152 |  |
| e2e_synthetic_shape_eml_no_ambiguity_ce_pairwise_seed1 | COMPLETED | eml_no_ambiguity | ce_pairwise | synthetic_shape | 1 | 0.1953 | 1.7155 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_seed1 | COMPLETED | eml_centered_ambiguity | ce | synthetic_shape | 1 | 0.1953 | 1.7236 |  |
| e2e_synthetic_shape_eml_centered_ambiguity_ce_pairwise_seed1 | COMPLETED | eml_centered_ambiguity | ce_pairwise | synthetic_shape | 1 | 0.1953 | 1.7225 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_seed1 | COMPLETED | eml_bank_centered_ambiguity | ce | synthetic_shape | 1 | 0.1953 | 1.7200 |  |
| e2e_synthetic_shape_eml_bank_centered_ambiguity_ce_pairwise_seed1 | COMPLETED | eml_bank_centered_ambiguity | ce_pairwise | synthetic_shape | 1 | 0.1953 | 1.7226 |  |
| e2e_synthetic_shape_merc_linear_ce_seed1 | COMPLETED | merc_linear | ce | synthetic_shape | 1 | 0.1953 | 2.2972 |  |
| e2e_synthetic_shape_merc_linear_ce_pairwise_seed1 | NOT RUN | merc_linear | ce_pairwise | synthetic_shape | 1 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_merc_energy_ce_seed1 | COMPLETED | merc_energy | ce | synthetic_shape | 1 | 0.1953 | 2.0141 |  |
| e2e_synthetic_shape_merc_energy_ce_pairwise_seed1 | NOT RUN | merc_energy | ce_pairwise | synthetic_shape | 1 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_merc_block_linear_ce_seed1 | NOT RUN | merc_block_linear | ce | synthetic_shape | 1 | MISSING | MISSING | KeyError('gate') |
| e2e_synthetic_shape_merc_block_linear_ce_pairwise_seed1 | NOT RUN | merc_block_linear | ce_pairwise | synthetic_shape | 1 | MISSING | MISSING | pairwise prototype margin is not applicable |
| e2e_synthetic_shape_merc_block_energy_ce_seed1 | NOT RUN | merc_block_energy | ce | synthetic_shape | 1 | MISSING | MISSING | KeyError('gate') |
| e2e_synthetic_shape_merc_block_energy_ce_pairwise_seed1 | NOT RUN | merc_block_energy | ce_pairwise | synthetic_shape | 1 | MISSING | MISSING | pairwise prototype margin is not applicable |
