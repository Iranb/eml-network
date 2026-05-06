# EML-KAN vs MLP Fair Comparison Report

## Scope
- KANbeFair reference code: https://github.com/yu-rp/KANbeFair
- KAN paper reference: https://arxiv.org/abs/2404.19756
- EML paper reference: https://arxiv.org/html/2603.21852v2
- This benchmark compares the current KAN-style sEML edge network against MLP baselines on identical generated data and seeds.
- `mlp_same_width` matches hidden width/depth; `mlp_param_matched` widens MLP until its parameter count is closest to EML-KAN.
- `spline_kan_reference` is optional and uses the local degree-1 spline edge implementation, not external pykan.
- Rows that hit the configured step cap are marked capped and should not be used as final model-comparison evidence.
- Completeness: 27/27 rows early-stopped; 0 rows hit the step cap.

## Protocol Notes
- KANbeFair lesson used here: report parameters/FLOPs, use same data/seeds, include a capacity-matched MLP, and keep architecture-specific operators isolated.
- EML-KAN operator: KAN-style edge matrix with `silu` residual plus stable sEML drive/resistance energy.
- MLP baseline: Linear/GELU stack with the same number of hidden layers.
- Validation score drives early stopping: negative MSE for regression and accuracy for classification.

## Regression Aggregates
| task | model | n | mean test MSE | mean test RMSE | mean worst-group MSE | mean group MSE gap | mean params | mean approx FLOPs | capped rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| localized_regression | eml_kan | 3 | 0.00943 | 0.09597 | 0.00981 | 0.00076 | 37767.0 | 107327.0 | 0 |
| localized_regression | mlp_param_matched | 3 | 0.01612 | 0.12192 | 0.01777 | 0.00335 | 37801.0 | 75223.0 | 0 |
| localized_regression | mlp_same_width | 3 | 0.00534 | 0.07310 | 0.00569 | 0.00070 | 4801.0 | 9473.0 | 0 |
| symbolic_regression | eml_kan | 3 | 0.00361 | 0.06003 | 0.00371 | 0.00020 | 37767.0 | 107327.0 | 0 |
| symbolic_regression | mlp_param_matched | 3 | 0.02247 | 0.14990 | 0.02262 | 0.00029 | 37801.0 | 75223.0 | 0 |
| symbolic_regression | mlp_same_width | 3 | 0.02270 | 0.15061 | 0.02314 | 0.00088 | 4801.0 | 9473.0 | 0 |

## Classification Aggregates
| task | model | n | mean test acc | mean NLL | mean ECE | mean worst-group acc | mean acc gap | mean resistance-error AUC | mean params | mean approx FLOPs | capped rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shift_classification | eml_kan | 3 | 0.42822 | 0.82139 | 0.27109 | 0.40590 | 0.04558 | 0.38870 | 38280.0 | 108798.0 | 0 |
| shift_classification | mlp_param_matched | 3 | 0.34456 | 2.09037 | 0.42077 | 0.31987 | 0.05056 | nan | 38382.0 | 76382.0 | 0 |
| shift_classification | mlp_same_width | 3 | 0.50309 | 1.46889 | 0.24063 | 0.48702 | 0.03284 | nan | 4866.0 | 9602.0 | 0 |

## Runs
| run_id | task | model | seed | status | test score | best val score | steps | early stop | capped | params | FLOPs |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| eml_kan_mlp_localized_regression_eml_kan_seed0 | localized_regression | eml_kan | 0 | COMPLETED | -0.01247 | -0.01333 | 9650 | True | False | 37767 | 107327.0 |
| eml_kan_mlp_localized_regression_eml_kan_seed1 | localized_regression | eml_kan | 1 | COMPLETED | -0.01001 | -0.00922 | 7600 | True | False | 37767 | 107327.0 |
| eml_kan_mlp_localized_regression_eml_kan_seed2 | localized_regression | eml_kan | 2 | COMPLETED | -0.00580 | -0.00574 | 9800 | True | False | 37767 | 107327.0 |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed0 | localized_regression | mlp_param_matched | 0 | COMPLETED | -0.02851 | -0.03103 | 9650 | True | False | 37801 | 75223.0 |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed1 | localized_regression | mlp_param_matched | 1 | COMPLETED | -0.00696 | -0.00621 | 9650 | True | False | 37801 | 75223.0 |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed2 | localized_regression | mlp_param_matched | 2 | COMPLETED | -0.01288 | -0.01342 | 12000 | True | False | 37801 | 75223.0 |
| eml_kan_mlp_localized_regression_mlp_same_width_seed0 | localized_regression | mlp_same_width | 0 | COMPLETED | -0.00545 | -0.00505 | 11650 | True | False | 4801 | 9473.0 |
| eml_kan_mlp_localized_regression_mlp_same_width_seed1 | localized_regression | mlp_same_width | 1 | COMPLETED | -0.00548 | -0.00548 | 14300 | True | False | 4801 | 9473.0 |
| eml_kan_mlp_localized_regression_mlp_same_width_seed2 | localized_regression | mlp_same_width | 2 | COMPLETED | -0.00511 | -0.00468 | 14200 | True | False | 4801 | 9473.0 |
| eml_kan_mlp_shift_classification_eml_kan_seed0 | shift_classification | eml_kan | 0 | COMPLETED | 0.31543 | 0.49512 | 650 | True | False | 38280 | 108798.0 |
| eml_kan_mlp_shift_classification_eml_kan_seed1 | shift_classification | eml_kan | 1 | COMPLETED | 0.31885 | 0.47705 | 650 | True | False | 38280 | 108798.0 |
| eml_kan_mlp_shift_classification_eml_kan_seed2 | shift_classification | eml_kan | 2 | COMPLETED | 0.65039 | 0.61133 | 600 | True | False | 38280 | 108798.0 |
| eml_kan_mlp_shift_classification_mlp_param_matched_seed0 | shift_classification | mlp_param_matched | 0 | COMPLETED | 0.38770 | 0.51514 | 600 | True | False | 38382 | 76382.0 |
| eml_kan_mlp_shift_classification_mlp_param_matched_seed1 | shift_classification | mlp_param_matched | 1 | COMPLETED | 0.31689 | 0.49023 | 900 | True | False | 38382 | 76382.0 |
| eml_kan_mlp_shift_classification_mlp_param_matched_seed2 | shift_classification | mlp_param_matched | 2 | COMPLETED | 0.32910 | 0.50928 | 2300 | True | False | 38382 | 76382.0 |
| eml_kan_mlp_shift_classification_mlp_same_width_seed0 | shift_classification | mlp_same_width | 0 | COMPLETED | 0.29834 | 0.48340 | 1400 | True | False | 4866 | 9602.0 |
| eml_kan_mlp_shift_classification_mlp_same_width_seed1 | shift_classification | mlp_same_width | 1 | COMPLETED | 0.59521 | 0.60107 | 600 | True | False | 4866 | 9602.0 |
| eml_kan_mlp_shift_classification_mlp_same_width_seed2 | shift_classification | mlp_same_width | 2 | COMPLETED | 0.61572 | 0.63086 | 600 | True | False | 4866 | 9602.0 |
| eml_kan_mlp_symbolic_regression_eml_kan_seed0 | symbolic_regression | eml_kan | 0 | COMPLETED | -0.00388 | -0.00368 | 4050 | True | False | 37767 | 107327.0 |
| eml_kan_mlp_symbolic_regression_eml_kan_seed1 | symbolic_regression | eml_kan | 1 | COMPLETED | -0.00335 | -0.00325 | 4300 | True | False | 37767 | 107327.0 |
| eml_kan_mlp_symbolic_regression_eml_kan_seed2 | symbolic_regression | eml_kan | 2 | COMPLETED | -0.00360 | -0.00416 | 4350 | True | False | 37767 | 107327.0 |
| eml_kan_mlp_symbolic_regression_mlp_param_matched_seed0 | symbolic_regression | mlp_param_matched | 0 | COMPLETED | -0.02230 | -0.02126 | 3250 | True | False | 37801 | 75223.0 |
| eml_kan_mlp_symbolic_regression_mlp_param_matched_seed1 | symbolic_regression | mlp_param_matched | 1 | COMPLETED | -0.02216 | -0.02058 | 3600 | True | False | 37801 | 75223.0 |
| eml_kan_mlp_symbolic_regression_mlp_param_matched_seed2 | symbolic_regression | mlp_param_matched | 2 | COMPLETED | -0.02296 | -0.02309 | 2850 | True | False | 37801 | 75223.0 |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed0 | symbolic_regression | mlp_same_width | 0 | COMPLETED | -0.02433 | -0.02301 | 4550 | True | False | 4801 | 9473.0 |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed1 | symbolic_regression | mlp_same_width | 1 | COMPLETED | -0.02121 | -0.01988 | 6100 | True | False | 4801 | 9473.0 |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed2 | symbolic_regression | mlp_same_width | 2 | COMPLETED | -0.02256 | -0.02381 | 5300 | True | False | 4801 | 9473.0 |

## Raw Artifacts
- `eml_kan_mlp_localized_regression_eml_kan_seed0`: `reports/runs/20260506_022208_eml_kan_mlp_localized_regression_eml_kan_seed0`
- `eml_kan_mlp_localized_regression_eml_kan_seed1`: `reports/runs/20260506_022409_eml_kan_mlp_localized_regression_eml_kan_seed1`
- `eml_kan_mlp_localized_regression_eml_kan_seed2`: `reports/runs/20260506_022558_eml_kan_mlp_localized_regression_eml_kan_seed2`
- `eml_kan_mlp_localized_regression_mlp_param_matched_seed0`: `reports/runs/20260506_022353_eml_kan_mlp_localized_regression_mlp_param_matched_seed0`
- `eml_kan_mlp_localized_regression_mlp_param_matched_seed1`: `reports/runs/20260506_022541_eml_kan_mlp_localized_regression_mlp_param_matched_seed1`
- `eml_kan_mlp_localized_regression_mlp_param_matched_seed2`: `reports/runs/20260506_022748_eml_kan_mlp_localized_regression_mlp_param_matched_seed2`
- `eml_kan_mlp_localized_regression_mlp_same_width_seed0`: `reports/runs/20260506_022333_eml_kan_mlp_localized_regression_mlp_same_width_seed0`
- `eml_kan_mlp_localized_regression_mlp_same_width_seed1`: `reports/runs/20260506_022857_eml_kan_mlp_localized_regression_mlp_same_width_seed1`
- `eml_kan_mlp_localized_regression_mlp_same_width_seed2`: `reports/runs/20260506_022922_eml_kan_mlp_localized_regression_mlp_same_width_seed2`
- `eml_kan_mlp_shift_classification_eml_kan_seed0`: `reports/runs/20260506_022102_eml_kan_mlp_shift_classification_eml_kan_seed0`
- `eml_kan_mlp_shift_classification_eml_kan_seed1`: `reports/runs/20260506_022112_eml_kan_mlp_shift_classification_eml_kan_seed1`
- `eml_kan_mlp_shift_classification_eml_kan_seed2`: `reports/runs/20260506_022121_eml_kan_mlp_shift_classification_eml_kan_seed2`
- `eml_kan_mlp_shift_classification_mlp_param_matched_seed0`: `reports/runs/20260506_022111_eml_kan_mlp_shift_classification_mlp_param_matched_seed0`
- `eml_kan_mlp_shift_classification_mlp_param_matched_seed1`: `reports/runs/20260506_022120_eml_kan_mlp_shift_classification_mlp_param_matched_seed1`
- `eml_kan_mlp_shift_classification_mlp_param_matched_seed2`: `reports/runs/20260506_022129_eml_kan_mlp_shift_classification_mlp_param_matched_seed2`
- `eml_kan_mlp_shift_classification_mlp_same_width_seed0`: `reports/runs/20260506_022108_eml_kan_mlp_shift_classification_mlp_same_width_seed0`
- `eml_kan_mlp_shift_classification_mlp_same_width_seed1`: `reports/runs/20260506_022118_eml_kan_mlp_shift_classification_mlp_same_width_seed1`
- `eml_kan_mlp_shift_classification_mlp_same_width_seed2`: `reports/runs/20260506_022127_eml_kan_mlp_shift_classification_mlp_same_width_seed2`
- `eml_kan_mlp_symbolic_regression_eml_kan_seed0`: `reports/runs/20260506_021516_eml_kan_mlp_symbolic_regression_eml_kan_seed0`
- `eml_kan_mlp_symbolic_regression_eml_kan_seed1`: `reports/runs/20260506_021606_eml_kan_mlp_symbolic_regression_eml_kan_seed1`
- `eml_kan_mlp_symbolic_regression_eml_kan_seed2`: `reports/runs/20260506_021659_eml_kan_mlp_symbolic_regression_eml_kan_seed2`
- `eml_kan_mlp_symbolic_regression_mlp_param_matched_seed0`: `reports/runs/20260506_021600_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed0`
- `eml_kan_mlp_symbolic_regression_mlp_param_matched_seed1`: `reports/runs/20260506_021653_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed1`
- `eml_kan_mlp_symbolic_regression_mlp_param_matched_seed2`: `reports/runs/20260506_021748_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed2`
- `eml_kan_mlp_symbolic_regression_mlp_same_width_seed0`: `reports/runs/20260506_021552_eml_kan_mlp_symbolic_regression_mlp_same_width_seed0`
- `eml_kan_mlp_symbolic_regression_mlp_same_width_seed1`: `reports/runs/20260506_022813_eml_kan_mlp_symbolic_regression_mlp_same_width_seed1`
- `eml_kan_mlp_symbolic_regression_mlp_same_width_seed2`: `reports/runs/20260506_022823_eml_kan_mlp_symbolic_regression_mlp_same_width_seed2`
