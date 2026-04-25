# MERC Master Report

This report only aggregates generated MERC artifacts. Missing files remain missing.

## Claim Status

- A. Does MERC beat linear? MISSING until the reports below show it.
- B. Does MERC beat MLP? MISSING until the reports below show it.
- C. Does MERC beat cosine prototype? MISSING until the reports below show it.
- D. Does MERC beat old EML head? MISSING until the reports below show it.
- E. Does MERC show support-factor alignment? MISSING until toy/synthetic evidence reports show it.
- F. Does MERC show conflict/resistance alignment? MISSING until toy/synthetic evidence reports show it.
- G. Is MERC worth using as a head? Inconclusive until the real-server report exists.
- H. Is MERC worth exploring as a representation block? Inconclusive; this task only validates the head/hypothesis cell path.

## Toy Nonlinear Tasks

Source: `reports/MERC_TOY_REPORT.md`

# MERC Toy Report

This report only states observed toy-task results.

| run_id | task | model | seed | best acc | final acc | ece | evidence corr | conflict corr | steps to 0.9 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| merc_toy_conjunctive_linear_seed0 | conjunctive | linear | 0 | 0.9688 | 0.9297 | 0.2320 | MISSING | MISSING | 88 |
| merc_toy_conjunctive_mlp_seed0 | conjunctive | mlp | 0 | 0.9727 | 0.9414 | 0.0255 | MISSING | MISSING | 25 |
| merc_toy_conjunctive_old_eml_gate_seed0 | conjunctive | old_eml_gate | 0 | 0.9727 | 0.9492 | 0.0433 | MISSING | MISSING | 11 |
| merc_toy_conjunctive_merc_seed0 | conjunctive | merc | 0 | 0.9766 | 0.9375 | 0.0472 | 0.8044 | 0.5750 | 3 |
| merc_toy_conjunctive_merc_energy_seed0 | conjunctive | merc_energy | 0 | 0.9727 | 0.9336 | 0.0509 | 0.7805 | 0.5787 | 1 |
| merc_toy_xor_linear_seed0 | xor | linear | 0 | 0.5352 | 0.4883 | 0.0207 | MISSING | MISSING | MISSING |
| merc_toy_xor_mlp_seed0 | xor | mlp | 0 | 0.5859 | 0.4961 | 0.0618 | MISSING | MISSING | MISSING |
| merc_toy_xor_old_eml_gate_seed0 | xor | old_eml_gate | 0 | 0.5664 | 0.5664 | 0.0522 | MISSING | MISSING | MISSING |
| merc_toy_xor_merc_seed0 | xor | merc | 0 | 0.6641 | 0.6055 | 0.0579 | 0.5189 | 0.0463 | MISSING |
| merc_toy_xor_merc_energy_seed0 | xor | merc_energy | 0 | 0.6055 | 0.6055 | 0.0843 | 0.1963 | 0.0230 | MISSING |
| merc_toy_conflict_suppression_linear_seed0 | conflict_suppression | linear | 0 | 0.8867 | 0.8867 | 0.2076 | MISSING | MISSING | 113 |
| merc_toy_conflict_suppression_mlp_seed0 | conflict_suppression | mlp | 0 | 0.9180 | 0.9180 | 0.0339 | MISSING | MISSING | 28 |
| merc_toy_conflict_suppression_old_eml_gate_seed0 | conflict_suppression | old_eml_gate | 0 | 0.8906 | 0.8906 | 0.0723 | MISSING | MISSING | 12 |
| merc_toy_conflict_suppression_merc_seed0 | conflict_suppression | merc | 0 | 0.9336 | 0.9336 | 0.0278 | 0.0660 | 0.1673 | 4 |
| merc_toy_conflict_suppression_merc_energy_seed0 | conflict_suppression | merc_energy | 0 | 0.9297 | 0.9297 | 0.0264 | 0.0804 | 0.2238 | 4 |
| merc_toy_conjunctive_linear_seed1 | conjunctive | linear | 1 | 0.9609 | 0.9609 | 0.2360 | MISSING | MISSING | 88 |
| merc_toy_conjunctive_mlp_seed1 | conjunctive | mlp | 1 | 0.9609 | 0.9609 | 0.0162 | MISSING | MISSING | 11 |
| merc_toy_conjunctive_old_eml_gate_seed1 | conjunctive | old_eml_gate | 1 | 0.9805 | 0.9805 | 0.0276 | MISSING | MISSING | 26 |
| merc_toy_conjunctive_merc_seed1 | conjunctive | merc | 1 | 0.9688 | 0.9570 | 0.0228 | 0.7518 | 0.5779 | 1 |
| merc_toy_conjunctive_merc_energy_seed1 | conjunctive | merc_energy | 1 | 0.9688 | 0.9570 | 0.0253 | 0.7368 | 0.6163 | 1 |
| merc_toy_xor_linear_seed1 | xor | linear | 1 | 0.5273 | 0.5273 | 0.0203 | MISSING | MISSING | MISSING |
| merc_toy_xor_mlp_seed1 | xor | mlp | 1 | 0.5234 | 0.5117 | 0.0174 | MISSING | MISSING | MISSING |
| merc_toy_xor_old_eml_gate_seed1 | xor | old_eml_gate | 1 | 0.5508 | 0.5508 | 0.0405 | MISSING | MISSING | MISSING |
| merc_toy_xor_merc_seed1 | xor | merc | 1 | 0.5273 | 0.4727 | 0.0291 | 0.0025 | -0.0342 | MISSING |
| merc_toy_xor_merc_energy_seed1 | xor | merc_energy | 1 | 0.6016 | 0.5977 | 0.0614 | 0.5864 | -0.0035 | MISSING |
| merc_toy_conflict_suppression_linear_seed1 | conflict_suppression | linear | 1 | 0.9141 | 0.9141 | 0.1715 | MISSING | MISSING | 112 |
| merc_toy_conflict_suppression_mlp_seed1 | conflict_suppression | mlp | 1 | 0.9414 | 0.9414 | 0.0220 | MISSING | MISSING | 17 |
| merc_toy_conflict_suppression_old_eml_gate_seed1 | conflict_suppression | old_eml_gate | 1 | 0.9219 | 0.9219 | 0.0405 | MISSING | MISSING | 28 |
| merc_toy_conflict_suppression_merc_seed1 | conflict_suppression | merc | 1 | 0.9375 | 0.9375 | 0.0307 | 0.1741 | 0.2086 | 1 |
| merc_toy_conflict_suppression_merc_energy_seed1 | conflict_suppression | merc_energy | 1 | 0.9414 | 0.9414 | 0.0384 | 0.4610 | 0.1896 | 1 |
| merc_toy_conjunctive_linear_seed2 | conjunctive | linear | 2 | 0.9648 | 0.9609 | 0.1733 | MISSING | MISSING | 45 |
| merc_toy_conjunctive_mlp_seed2 | conjunctive | mlp | 2 | 0.9688 | 0.9688 | 0.0266 | MISSING | MISSING | 29 |
| merc_toy_conjunctive_old_eml_gate_seed2 | conjunctive | old_eml_gate | 2 | 0.9727 | 0.9688 | 0.0202 | MISSING | MISSING | 8 |
| merc_toy_conjunctive_merc_seed2 | conjunctive | merc | 2 | 0.9688 | 0.9688 | 0.0159 | 0.7667 | 0.4504 | 4 |
| merc_toy_conjunctive_merc_energy_seed2 | conjunctive | merc_energy | 2 | 0.9688 | 0.9648 | 0.0208 | 0.7575 | 0.4839 | 1 |
| merc_toy_xor_linear_seed2 | xor | linear | 2 | 0.5156 | 0.5156 | 0.0056 | MISSING | MISSING | MISSING |
| merc_toy_xor_mlp_seed2 | xor | mlp | 2 | 0.5742 | 0.5117 | 0.0058 | MISSING | MISSING | MISSING |
| merc_toy_xor_old_eml_gate_seed2 | xor | old_eml_gate | 2 | 0.5820 | 0.5820 | 0.0780 | MISSING | MISSING | MISSING |
| merc_toy_xor_merc_seed2 | xor | merc | 2 | 0.5547 | 0.5039 | 0.0024 | -0.1186 | 0.0243 | MISSING |
| merc_toy_xor_merc_energy_seed2 | xor | merc_energy | 2 | 0.5547 | 0.5039 | 0.0004 | -0.1407 | 0.0577 | MISSING |
| merc_toy_conflict_suppression_linear_seed2 | conflict_suppression | linear | 2 | 0.9180 | 0.9062 | 0.1496 | MISSING | MISSING | 47 |
| merc_toy_conflict_suppression_mlp_seed2 | conflict_suppression | mlp | 2 | 0.9297 | 0.9219 | 0.0422 | MISSING | MISSING | 29 |
| merc_toy_conflict_suppression_old_eml_gate_seed2 | conflict_suppression | old_eml_gate | 2 | 0.9219 | 0.9219 | 0.0388 | MISSING | MISSING | 10 |
| merc_toy_conflict_suppression_merc_seed2 | conflict_suppression | merc | 2 | 0.9297 | 0.9141 | 0.0384 | 0.3194 | 0.2413 | 7 |
| merc_toy_conflict_suppression_merc_energy_seed2 | conflict_suppression | merc_energy | 2 | 0.9219 | 0.9141 | 0.0455 | 0.3537 | 0.2591 | 2 |

## Conclusion

- MERC neuron design is not yet justified.

## Frozen CNN Feature Head Isolation

Source: `reports/MERC_HEAD_ABLATION_REPORT.md`

# CNN Head Ablation Report

## 1. Executive Summary
- Completed runs: 30
- NOT RUN entries: 0
- Failed runs: 0
- Best frozen-feature result: mlp seed=0 test_accuracy=0.5371
- Best end-to-end result: MISSING
- Claim status: The evidence is mixed: centered EML wins 4/9 paired comparisons.

## 2. Experimental Setup
- Frozen-feature runs train one shared CNN feature extractor per dataset/seed, cache features, then train only the selected head.
- End-to-end runs train the same CNN backbone with one selected head; the EML residual-bank variant is reported separately.
- CE-only and prototype-pairwise settings are separated. Linear and MLP heads are marked NOT RUN for prototype-pairwise because that loss is not applicable.

## 3. Run Status
| run_id | status | experiment | model | dataset | seed | reason |
| --- | --- | --- | --- | --- | ---: | --- |
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
| frozen_cifar10_merc_energy_seed0 | COMPLETED | frozen_features | merc_energy | cifar10 | 0 |  |
| frozen_cifar10_merc_energy_seed1 | COMPLETED | frozen_features | merc_energy | cifar10 | 1 |  |
| frozen_cifar10_merc_energy_seed2 | COMPLETED | frozen_features | merc_energy | cifar10 | 2 |  |
| frozen_cifar10_merc_energy_small_seed0 | COMPLETED | frozen_features | merc_energy_small | cifar10 | 0 |  |
| frozen_cifar10_merc_energy_small_seed1 | COMPLETED | frozen_features | merc_energy_small | cifar10 | 1 |  |
| frozen_cifar10_merc_energy_small_seed2 | COMPLETED | frozen_features | merc_energy_small | cifar10 | 2 |  |
| frozen_cifar10_merc_linear_seed0 | COMPLETED | frozen_features | merc_linear | cifar10 | 0 |  |
| frozen_cifar10_merc_linear_seed1 | COMPLETED | frozen_features | merc_linear | cifar10 | 1 |  |
| frozen_cifar10_merc_linear_seed2 | COMPLETED | frozen_features | merc_linear | cifar10 | 2 |  |
| frozen_cifar10_merc_linear_small_seed0 | COMPLETED | frozen_features | merc_linear_small | cifar10 | 0 |  |
| frozen_cifar10_merc_linear_small_seed1 | COMPLETED | frozen_features | merc_linear_small | cifar10 | 1 |  |
| frozen_cifar10_merc_linear_small_seed2 | COMPLETED | frozen_features | merc_linear_small | cifar10 | 2 |  |
| frozen_cifar10_mlp_seed0 | COMPLETED | frozen_features | mlp | cifar10 | 0 |  |
| frozen_cifar10_mlp_seed1 | COMPLETED | frozen_features | mlp | cifar10 | 1 |  |
| frozen_cifar10_mlp_seed2 | COMPLETED | frozen_features | mlp | cifar10 | 2 |  |

## 4. Frozen Feature Results
### Frozen CNN Features
| run_id | seed | model | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen_cifar10_cosine_prototype_seed0 | 0 | cosine_prototype | 0.5215 | 0.5312 | 1.3883 | 0.0869 | 0.6304 | -0.1383 | 3.2458 |
| frozen_cifar10_cosine_prototype_seed1 | 1 | cosine_prototype | 0.4941 | 0.5156 | 1.3989 | 0.0880 | 0.6373 | -0.0592 | 3.2743 |
| frozen_cifar10_cosine_prototype_seed2 | 2 | cosine_prototype | 0.5117 | 0.5605 | 1.3629 | 0.0773 | 0.6241 | -0.0076 | 3.2897 |
| frozen_cifar10_eml_centered_ambiguity_seed0 | 0 | eml_centered_ambiguity | 0.5156 | 0.5391 | 1.4751 | 0.0877 | 0.6457 | -0.1716 | 6.5453 |
| frozen_cifar10_eml_centered_ambiguity_seed1 | 1 | eml_centered_ambiguity | 0.4961 | 0.5117 | 1.4776 | 0.0945 | 0.6514 | -0.1600 | 6.4321 |
| frozen_cifar10_eml_centered_ambiguity_seed2 | 2 | eml_centered_ambiguity | 0.5293 | 0.5703 | 1.4549 | 0.0808 | 0.6370 | -0.0629 | 13.6356 |
| frozen_cifar10_eml_no_ambiguity_seed0 | 0 | eml_no_ambiguity | 0.5098 | 0.5391 | 1.4785 | 0.0806 | 0.6472 | -0.1749 | 12.8142 |
| frozen_cifar10_eml_no_ambiguity_seed1 | 1 | eml_no_ambiguity | 0.4922 | 0.5137 | 1.4814 | 0.0980 | 0.6526 | -0.1620 | 7.0469 |
| frozen_cifar10_eml_no_ambiguity_seed2 | 2 | eml_no_ambiguity | 0.5293 | 0.5703 | 1.4575 | 0.0834 | 0.6379 | -0.0647 | 7.2272 |
| frozen_cifar10_eml_raw_ambiguity_seed0 | 0 | eml_raw_ambiguity | 0.5156 | 0.5391 | 1.4760 | 0.0911 | 0.6460 | -0.1716 | 6.8075 |
| frozen_cifar10_eml_raw_ambiguity_seed1 | 1 | eml_raw_ambiguity | 0.4980 | 0.5137 | 1.4785 | 0.1003 | 0.6517 | -0.1600 | 6.3345 |
| frozen_cifar10_eml_raw_ambiguity_seed2 | 2 | eml_raw_ambiguity | 0.5273 | 0.5703 | 1.4558 | 0.0794 | 0.6373 | -0.0628 | 6.7999 |
| frozen_cifar10_linear_seed0 | 0 | linear | 0.5293 | 0.5371 | 1.3410 | 0.0661 | 0.6125 | -0.1191 | 8.7570 |
| frozen_cifar10_linear_seed1 | 1 | linear | 0.4883 | 0.5156 | 1.3549 | 0.0534 | 0.6281 | -0.0320 | 1.8326 |
| frozen_cifar10_linear_seed2 | 2 | linear | 0.5117 | 0.5566 | 1.3312 | 0.0569 | 0.6167 | 0.0081 | 2.3218 |
| frozen_cifar10_merc_energy_seed0 | 0 | merc_energy | 0.5059 | 0.5195 | 1.4529 | 0.1080 | 0.6329 | -0.1726 | 12.6143 |
| frozen_cifar10_merc_energy_seed1 | 1 | merc_energy | 0.4805 | 0.5273 | 1.4619 | 0.1222 | 0.6640 | -0.1964 | 7.6669 |
| frozen_cifar10_merc_energy_seed2 | 2 | merc_energy | 0.5000 | 0.5508 | 1.4378 | 0.0832 | 0.6402 | -0.1205 | 10.0984 |
| frozen_cifar10_merc_energy_small_seed0 | 0 | merc_energy_small | 0.4883 | 0.5098 | 1.4909 | 0.0916 | 0.6640 | -0.3235 | 7.8905 |
| frozen_cifar10_merc_energy_small_seed1 | 1 | merc_energy_small | 0.4766 | 0.5312 | 1.4462 | 0.1214 | 0.6580 | -0.1554 | 6.8439 |
| frozen_cifar10_merc_energy_small_seed2 | 2 | merc_energy_small | 0.4844 | 0.5508 | 1.4370 | 0.0637 | 0.6504 | -0.2037 | 9.2154 |
| frozen_cifar10_merc_linear_seed0 | 0 | merc_linear | 0.5254 | 0.5156 | 1.4583 | 0.1344 | 0.6359 | -0.1212 | 6.6278 |
| frozen_cifar10_merc_linear_seed1 | 1 | merc_linear | 0.5020 | 0.5508 | 1.4439 | 0.1062 | 0.6507 | -0.1204 | 12.6298 |
| frozen_cifar10_merc_linear_seed2 | 2 | merc_linear | 0.5234 | 0.5527 | 1.4551 | 0.0794 | 0.6358 | -0.1437 | 6.5745 |
| frozen_cifar10_merc_linear_small_seed0 | 0 | merc_linear_small | 0.5156 | 0.5137 | 1.4206 | 0.1237 | 0.6284 | -0.0615 | 7.5501 |
| frozen_cifar10_merc_linear_small_seed1 | 1 | merc_linear_small | 0.5020 | 0.5371 | 1.4370 | 0.1069 | 0.6497 | -0.1495 | 6.8141 |
| frozen_cifar10_merc_linear_small_seed2 | 2 | merc_linear_small | 0.5059 | 0.5508 | 1.4047 | 0.0729 | 0.6232 | -0.0649 | 13.7411 |

## End-to-End CNN Plus Head

Source: `reports/MERC_END_TO_END_REPORT.md`

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

## Synthetic Evidence Diagnostics

Source: `reports/MERC_SYNTHETIC_EVIDENCE_REPORT.md`

# MERC Synthetic Evidence Report

| run_id | model | status | final metric | support evidence corr | conflict resistance corr |
| --- | --- | --- | ---: | ---: | ---: |
| frozen_synthetic_evidence_linear_seed0 | linear | COMPLETED | 0.9961 | nan | nan |
| frozen_synthetic_evidence_mlp_seed0 | mlp | COMPLETED | 0.9941 | nan | nan |
| frozen_synthetic_evidence_cosine_prototype_seed0 | cosine_prototype | COMPLETED | 0.9941 | nan | nan |
| frozen_synthetic_evidence_eml_no_ambiguity_seed0 | eml_no_ambiguity | COMPLETED | 0.9961 | nan | nan |
| frozen_synthetic_evidence_eml_raw_ambiguity_seed0 | eml_raw_ambiguity | COMPLETED | 0.9941 | nan | nan |
| frozen_synthetic_evidence_eml_centered_ambiguity_seed0 | eml_centered_ambiguity | COMPLETED | 0.9941 | nan | nan |
| frozen_synthetic_evidence_merc_linear_seed0 | merc_linear | COMPLETED | 0.9961 | 0.6356 | -0.1628 |
| frozen_synthetic_evidence_merc_energy_seed0 | merc_energy | COMPLETED | 0.9961 | -0.3842 | -0.1695 |
| frozen_synthetic_evidence_merc_linear_small_seed0 | merc_linear_small | COMPLETED | 0.9961 | 0.6611 | -0.1786 |
| frozen_synthetic_evidence_merc_energy_small_seed0 | merc_energy_small | COMPLETED | 0.9961 | 0.4231 | -0.1907 |
| frozen_synthetic_evidence_linear_seed1 | linear | COMPLETED | 0.9961 | nan | nan |
| frozen_synthetic_evidence_mlp_seed1 | mlp | COMPLETED | 0.9941 | nan | nan |
| frozen_synthetic_evidence_cosine_prototype_seed1 | cosine_prototype | COMPLETED | 0.9961 | nan | nan |
| frozen_synthetic_evidence_eml_no_ambiguity_seed1 | eml_no_ambiguity | COMPLETED | 0.9961 | nan | nan |
| frozen_synthetic_evidence_eml_raw_ambiguity_seed1 | eml_raw_ambiguity | COMPLETED | 0.9961 | nan | nan |
| frozen_synthetic_evidence_eml_centered_ambiguity_seed1 | eml_centered_ambiguity | COMPLETED | 0.9961 | nan | nan |
| frozen_synthetic_evidence_merc_linear_seed1 | merc_linear | COMPLETED | 0.9941 | 0.6111 | 0.0089 |
| frozen_synthetic_evidence_merc_energy_seed1 | merc_energy | COMPLETED | 0.9941 | 0.4963 | 0.0073 |
| frozen_synthetic_evidence_merc_linear_small_seed1 | merc_linear_small | COMPLETED | 0.9941 | 0.6032 | 0.0019 |
| frozen_synthetic_evidence_merc_energy_small_seed1 | merc_energy_small | COMPLETED | 0.9922 | 0.5673 | 0.0160 |
| frozen_synthetic_evidence_linear_seed2 | linear | COMPLETED | 0.9922 | nan | nan |
| frozen_synthetic_evidence_mlp_seed2 | mlp | COMPLETED | 0.9922 | nan | nan |
| frozen_synthetic_evidence_cosine_prototype_seed2 | cosine_prototype | COMPLETED | 0.9922 | nan | nan |
| frozen_synthetic_evidence_eml_no_ambiguity_seed2 | eml_no_ambiguity | COMPLETED | 0.9922 | nan | nan |
| frozen_synthetic_evidence_eml_raw_ambiguity_seed2 | eml_raw_ambiguity | COMPLETED | 0.9922 | nan | nan |
| frozen_synthetic_evidence_eml_centered_ambiguity_seed2 | eml_centered_ambiguity | COMPLETED | 0.9922 | nan | nan |
| frozen_synthetic_evidence_merc_linear_seed2 | merc_linear | COMPLETED | 0.9922 | 0.5730 | -0.0866 |
| frozen_synthetic_evidence_merc_energy_seed2 | merc_energy | COMPLETED | 0.9922 | 0.5406 | -0.0406 |
| frozen_synthetic_evidence_merc_linear_small_seed2 | merc_linear_small | COMPLETED | 0.9922 | -0.3045 | -0.0828 |
| frozen_synthetic_evidence_merc_energy_small_seed2 | merc_energy_small | COMPLETED | 0.9922 | 0.2088 | -0.0420 |

## CIFAR-10 Real Server Results

Source: `reports/MERC_REAL_SERVER_VALIDATION_REPORT.md`

# MERC Real Server Validation Report

## 1. Environment

- Server: `211.71.76.29`
- Remote workspace: `/data16T/hyq/eml-network-merc-20260425_105808`
- Dataset root: `/data16T/hyq/dataset/data`
- Python: `/data16T/hyq/miniconda3/envs/simgcd/bin/python`
- Device policy: `CUDA_VISIBLE_DEVICES=1`
- Visible training GPU: `NVIDIA RTX 5880 Ada Generation`
- Titan Xp usage: not used
- Seeds: `0 1 2`
- Batch size: `64`
- DataLoader workers: `0`
- Early stop settings: `patience=4`, `min_evals=3`

## 2. Commands Run

```bash
CUDA_VISIBLE_DEVICES=1 /data16T/hyq/miniconda3/envs/simgcd/bin/python scripts/run_merc_toy_experiments.py --mode medium --device cuda --seeds 0 1 2
CUDA_VISIBLE_DEVICES=1 /data16T/hyq/miniconda3/envs/simgcd/bin/python scripts/run_head_ablation.py --dataset cifar10 --mode medium --seeds 0 1 2 --device cuda --data-dir /data16T/hyq/dataset/data --num-workers 0 --batch-size 64 --include-merc --early-stop-patience 4 --early-stop-min-evals 3 --runs-root reports/merc_head_ablation/runs
CUDA_VISIBLE_DEVICES=1 /data16T/hyq/miniconda3/envs/simgcd/bin/python scripts/run_cnn_head_end_to_end_ablation.py --dataset cifar10 --mode medium --seeds 0 1 2 --device cuda --data-dir /data16T/hyq/dataset/data --num-workers 0 --batch-size 64 --include-merc --early-stop-patience 4 --early-stop-min-evals 3 --runs-root reports/merc_end_to_end_rerun/runs
CUDA_VISIBLE_DEVICES=1 /data16T/hyq/miniconda3/envs/simgcd/bin/python scripts/run_merc_synthetic_evidence.py --mode medium --device cuda --seeds 0 1 2 --num-workers 0 --data-dir /data16T/hyq/dataset/data
```

## 3. Toy Nonlinear Tasks

The toy runs support the MERC hypothesis only partially.

- Conjunctive evidence:
  - MERC reached `>= 0.9` accuracy much faster than linear and MLP in all three seeds.
  - Example: `merc_toy_conjunctive_merc_seed0` reached `0.9766` best accuracy and hit `0.9` by step `3`.
- Conflict suppression:
  - MERC and `merc_energy` were competitive with MLP and old EML gate.
  - Example bests: `merc` `0.9336`, `merc_energy` `0.9414`, `mlp` `0.9414`.
- XOR:
  - MERC did not establish an advantage.
  - Best MERC result was `0.6641` for `merc` on seed `0`, but other seeds were weak and inconsistent.

Conclusion from toy tasks:

- MERC shows useful multiplicative behavior on conjunctive evidence.
- MERC does not yet show robust superiority on harder interaction structure.
- The toy report conclusion remains valid: **MERC neuron design is not yet justified**.

## 4. Frozen CNN Feature Head Isolation on CIFAR-10

All heads used the same frozen CNN features.

### Best observed frozen-feature accuracies

| head | best test acc | mean test acc |
| --- | ---: | ---: |
| `mlp` | `0.5371` | `0.5273` |
| `merc_linear` | `0.5254` | `0.5169` |
| `eml_centered_ambiguity` | `0.5293` | `0.5137` |
| `cosine_prototype` | `0.5215` | `0.5091` |
| `linear` | `0.5293` | `0.5098` |
| `merc_energy` | `0.5059` | `0.4954` |

Observations:

- MERC did **not** beat MLP on frozen CNN features.
- `merc_linear` was slightly above cosine on mean accuracy, but below MLP and below the best linear/old-EML runs.
- `merc_energy` underperformed `merc_linear`.
- Frozen-feature claim is therefore **not strong enough to support a MERC head advantage**.

## 5. End-to-End CNN + Head on CIFAR-10

The end-to-end rerun includes the fixed `merc_block_*` variants.

### Best observed end-to-end accuracies

| model | best test acc | mean test acc |
| --- | ---: | ---: |
| `cosine_prototype` | `0.5605` | `0.5449` |
| `linear` | `0.5469` | `0.4974` |
| `eml_no_ambiguity` | `0.5020` | `0.4544` |
| `eml_centered_ambiguity` | `0.4785` | `0.4505` |
| `merc_linear` | `0.3848` | `0.3555` |
