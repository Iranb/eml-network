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
| frozen_cifar10_mlp_seed0 | 0 | mlp | 0.5371 | 0.5469 | 1.3847 | 0.1249 | 0.6244 | 0.0079 | 2.1736 |
| frozen_cifar10_mlp_seed1 | 1 | mlp | 0.5098 | 0.5352 | 1.3836 | 0.1127 | 0.6375 | -0.0664 | 1.9517 |
| frozen_cifar10_mlp_seed2 | 2 | mlp | 0.5352 | 0.5703 | 1.3757 | 0.1066 | 0.6187 | 0.0500 | 2.3504 |

## 5. End-To-End Results
### CNN Plus Head
| run_id | seed | model | loss mode | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## 6. CE-Only Comparison
### End-To-End CE Only
| run_id | seed | model | loss mode | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## 7. CE + Pairwise Comparison
### End-To-End CE + Prototype Pairwise
| run_id | seed | model | loss mode | test acc | val acc | test loss | ECE | Brier | margin | time sec |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## 8. Calibration Metrics
ECE and Brier score are included in the result tables. Lower is better for both.

## 9. Hard-Negative Margin Analysis
Margin is positive-logit minus hardest-negative-logit; larger is better.

## 10. EML Drive/Resistance Analysis
| run_id | model | pos drive | hard neg drive | pos resistance | hard neg resistance | uncertainty | ambiguity | noise corr | occlusion corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen_cifar10_eml_centered_ambiguity_seed0 | eml_centered_ambiguity | 5.2214 | 5.4990 | 0.6776 | 0.7625 | 0.5524 | 0.1466 | MISSING | MISSING |
| frozen_cifar10_eml_centered_ambiguity_seed1 | eml_centered_ambiguity | 5.7135 | 5.8377 | 0.6934 | 0.7967 | 0.3950 | 0.3581 | MISSING | MISSING |
| frozen_cifar10_eml_centered_ambiguity_seed2 | eml_centered_ambiguity | 5.5947 | 5.5191 | 0.6886 | 0.7611 | 0.5584 | 0.1204 | MISSING | MISSING |
| frozen_cifar10_eml_no_ambiguity_seed0 | eml_no_ambiguity | 5.2154 | 5.4945 | 10.9543 | 10.9539 | 10.7504 | 0.0000 | MISSING | MISSING |
| frozen_cifar10_eml_no_ambiguity_seed1 | eml_no_ambiguity | 5.6683 | 5.7910 | 15.0985 | 15.1021 | 14.8936 | 0.0000 | MISSING | MISSING |
| frozen_cifar10_eml_no_ambiguity_seed2 | eml_no_ambiguity | 5.5694 | 5.5009 | 22.9385 | 22.9419 | 22.7346 | 0.0000 | MISSING | MISSING |
| frozen_cifar10_eml_raw_ambiguity_seed0 | eml_raw_ambiguity | 5.2153 | 5.4930 | 2.2688 | 2.3498 | 0.0002 | 2.2863 | MISSING | MISSING |
| frozen_cifar10_eml_raw_ambiguity_seed1 | eml_raw_ambiguity | 5.7024 | 5.8243 | 2.4213 | 2.5197 | 0.0001 | 2.4752 | MISSING | MISSING |
| frozen_cifar10_eml_raw_ambiguity_seed2 | eml_raw_ambiguity | 5.5859 | 5.5105 | 2.2793 | 2.3498 | 0.0003 | 2.2676 | MISSING | MISSING |

## 11. Robustness Under Noise/Occlusion
Resistance-noise and resistance-occlusion correlations are reported when synthetic metadata is available. MISSING means the head did not expose resistance or the dataset did not provide the field.

## 12. Statistical Confidence Intervals
| experiment | dataset | seed | loss mode | comparison | delta acc | 95% CI low | 95% CI high |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| frozen_features | cifar10 | 0 | ce | eml_centered_ambiguity - linear | -0.0137 | -0.0371 | 0.0098 |
| frozen_features | cifar10 | 0 | ce | eml_centered_ambiguity - mlp | -0.0215 | -0.0469 | 0.0059 |
| frozen_features | cifar10 | 0 | ce | eml_centered_ambiguity - cosine_prototype | -0.0059 | -0.0293 | 0.0195 |
| frozen_features | cifar10 | 1 | ce | eml_centered_ambiguity - linear | 0.0078 | -0.0215 | 0.0352 |
| frozen_features | cifar10 | 1 | ce | eml_centered_ambiguity - mlp | -0.0137 | -0.0391 | 0.0117 |
| frozen_features | cifar10 | 1 | ce | eml_centered_ambiguity - cosine_prototype | 0.0020 | -0.0254 | 0.0273 |
| frozen_features | cifar10 | 2 | ce | eml_centered_ambiguity - linear | 0.0176 | -0.0098 | 0.0430 |
| frozen_features | cifar10 | 2 | ce | eml_centered_ambiguity - mlp | -0.0059 | -0.0333 | 0.0195 |
| frozen_features | cifar10 | 2 | ce | eml_centered_ambiguity - cosine_prototype | 0.0176 | -0.0117 | 0.0469 |

## 13. Which Claim Is Supported
The evidence is mixed: centered EML wins 4/9 paired comparisons.

## 14. Raw Artifacts
- `frozen_cifar10_cosine_prototype_seed0`: `reports/merc_head_ablation/runs/20260425_030213_frozen_cifar10_cosine_prototype_seed0`
- `frozen_cifar10_cosine_prototype_seed1`: `reports/merc_head_ablation/runs/20260425_030330_frozen_cifar10_cosine_prototype_seed1`
- `frozen_cifar10_cosine_prototype_seed2`: `reports/merc_head_ablation/runs/20260425_030455_frozen_cifar10_cosine_prototype_seed2`
- `frozen_cifar10_eml_centered_ambiguity_seed0`: `reports/merc_head_ablation/runs/20260425_030236_frozen_cifar10_eml_centered_ambiguity_seed0`
- `frozen_cifar10_eml_centered_ambiguity_seed1`: `reports/merc_head_ablation/runs/20260425_030346_frozen_cifar10_eml_centered_ambiguity_seed1`
- `frozen_cifar10_eml_centered_ambiguity_seed2`: `reports/merc_head_ablation/runs/20260425_030513_frozen_cifar10_eml_centered_ambiguity_seed2`
- `frozen_cifar10_eml_no_ambiguity_seed0`: `reports/merc_head_ablation/runs/20260425_030216_frozen_cifar10_eml_no_ambiguity_seed0`
- `frozen_cifar10_eml_no_ambiguity_seed1`: `reports/merc_head_ablation/runs/20260425_030333_frozen_cifar10_eml_no_ambiguity_seed1`
- `frozen_cifar10_eml_no_ambiguity_seed2`: `reports/merc_head_ablation/runs/20260425_030459_frozen_cifar10_eml_no_ambiguity_seed2`
- `frozen_cifar10_eml_raw_ambiguity_seed0`: `reports/merc_head_ablation/runs/20260425_030229_frozen_cifar10_eml_raw_ambiguity_seed0`
- `frozen_cifar10_eml_raw_ambiguity_seed1`: `reports/merc_head_ablation/runs/20260425_030340_frozen_cifar10_eml_raw_ambiguity_seed1`
- `frozen_cifar10_eml_raw_ambiguity_seed2`: `reports/merc_head_ablation/runs/20260425_030506_frozen_cifar10_eml_raw_ambiguity_seed2`
- `frozen_cifar10_linear_seed0`: `reports/merc_head_ablation/runs/20260425_030202_frozen_cifar10_linear_seed0`
- `frozen_cifar10_linear_seed1`: `reports/merc_head_ablation/runs/20260425_030326_frozen_cifar10_linear_seed1`
- `frozen_cifar10_linear_seed2`: `reports/merc_head_ablation/runs/20260425_030451_frozen_cifar10_linear_seed2`
- `frozen_cifar10_merc_energy_seed0`: `reports/merc_head_ablation/runs/20260425_030249_frozen_cifar10_merc_energy_seed0`
- `frozen_cifar10_merc_energy_seed1`: `reports/merc_head_ablation/runs/20260425_030405_frozen_cifar10_merc_energy_seed1`
- `frozen_cifar10_merc_energy_seed2`: `reports/merc_head_ablation/runs/20260425_030533_frozen_cifar10_merc_energy_seed2`
- `frozen_cifar10_merc_energy_small_seed0`: `reports/merc_head_ablation/runs/20260425_030309_frozen_cifar10_merc_energy_small_seed0`
- `frozen_cifar10_merc_energy_small_seed1`: `reports/merc_head_ablation/runs/20260425_030420_frozen_cifar10_merc_energy_small_seed1`
- `frozen_cifar10_merc_energy_small_seed2`: `reports/merc_head_ablation/runs/20260425_030557_frozen_cifar10_merc_energy_small_seed2`
- `frozen_cifar10_merc_linear_seed0`: `reports/merc_head_ablation/runs/20260425_030242_frozen_cifar10_merc_linear_seed0`
- `frozen_cifar10_merc_linear_seed1`: `reports/merc_head_ablation/runs/20260425_030353_frozen_cifar10_merc_linear_seed1`
- `frozen_cifar10_merc_linear_seed2`: `reports/merc_head_ablation/runs/20260425_030526_frozen_cifar10_merc_linear_seed2`
- `frozen_cifar10_merc_linear_small_seed0`: `reports/merc_head_ablation/runs/20260425_030302_frozen_cifar10_merc_linear_small_seed0`
- `frozen_cifar10_merc_linear_small_seed1`: `reports/merc_head_ablation/runs/20260425_030413_frozen_cifar10_merc_linear_small_seed1`
- `frozen_cifar10_merc_linear_small_seed2`: `reports/merc_head_ablation/runs/20260425_030543_frozen_cifar10_merc_linear_small_seed2`
- `frozen_cifar10_mlp_seed0`: `reports/merc_head_ablation/runs/20260425_030211_frozen_cifar10_mlp_seed0`
- `frozen_cifar10_mlp_seed1`: `reports/merc_head_ablation/runs/20260425_030327_frozen_cifar10_mlp_seed1`
- `frozen_cifar10_mlp_seed2`: `reports/merc_head_ablation/runs/20260425_030453_frozen_cifar10_mlp_seed2`

## 15. Appendix: Commands
- `pytest`
- `python scripts/run_head_ablation.py --dataset synthetic_shape --mode smoke --seeds 0 1`
- `python scripts/run_cnn_head_end_to_end_ablation.py --dataset synthetic_shape --mode smoke --seeds 0 1`
- `python scripts/generate_head_ablation_report.py`
