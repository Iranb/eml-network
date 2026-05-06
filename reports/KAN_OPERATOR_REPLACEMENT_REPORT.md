# KAN Operator Replacement Report

## Scope
- This report compares a compact spline-KAN edge operator with the current stable sEML edge operator under the same KAN-style topology.
- KAN reference: https://arxiv.org/abs/2404.19756
- EML reference: https://arxiv.org/html/2603.21852v2
- The spline baseline here is a degree-1 B-spline KAN-style implementation, not the external pykan package.
- Lower MSE is better. `best_metric` in raw summaries is stored as negative MSE for compatibility with existing summary fields.
- Rows that reach the configured step cap without early stop are capped ablations, not final model-comparison evidence under the repository validation rules.

## Operator Difference
| model | edge operator | drive/resistance split | expected strength | expected weakness |
| --- | --- | --- | --- | --- |
| spline_kan | `silu(x) * base + sum_k w_k B_k(x)` | no | local univariate curve fitting through spline control points | no explicit uncertainty/resistance diagnostic |
| semL_operator_replacement | `silu(x) * base + scale * sEML(a*x+b, softplus(s)*(x-c)^2+floor)` | yes | stable EML drive/resistance diagnostics and bounded energy | less local spline resolution per edge |

## Aggregate Results
| task | model | n | mean best val MSE | mean final val MSE | mean final train MSE | mean params |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| additive_smooth | semL_operator_replacement | 2 | 0.056396 | 0.056396 | 0.058418 | 9671.000000 |
| additive_smooth | spline_kan | 2 | 0.005938 | 0.005938 | 0.000893 | 21505.000000 |
| local_bumps | semL_operator_replacement | 2 | 0.265803 | 0.265803 | 0.242057 | 9671.000000 |
| local_bumps | spline_kan | 2 | 0.033731 | 0.033902 | 0.007615 | 21505.000000 |
| mixed_composition | semL_operator_replacement | 2 | 0.093731 | 0.093731 | 0.117507 | 9671.000000 |
| mixed_composition | spline_kan | 2 | 0.023082 | 0.023082 | 0.001393 | 21505.000000 |

## Runs
| run_id | status | task | model | seed | best val MSE | final val MSE | final train MSE | steps | early stop | params | time sec |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| kan_operator_additive_smooth_semL_operator_replacement_seed0 | COMPLETED | additive_smooth | semL_operator_replacement | 0 | 0.086014 | 0.086014 | 0.088082 | 500 | False | 9671 | 4.212905 |
| kan_operator_additive_smooth_semL_operator_replacement_seed1 | COMPLETED | additive_smooth | semL_operator_replacement | 1 | 0.026779 | 0.026779 | 0.028754 | 500 | False | 9671 | 4.533869 |
| kan_operator_additive_smooth_spline_kan_seed0 | COMPLETED | additive_smooth | spline_kan | 0 | 0.006964 | 0.006964 | 0.000812 | 500 | False | 21505 | 2.617736 |
| kan_operator_additive_smooth_spline_kan_seed1 | COMPLETED | additive_smooth | spline_kan | 1 | 0.004912 | 0.004912 | 0.000974 | 500 | False | 21505 | 2.379037 |
| kan_operator_local_bumps_semL_operator_replacement_seed0 | COMPLETED | local_bumps | semL_operator_replacement | 0 | 0.242210 | 0.242210 | 0.224127 | 500 | False | 9671 | 4.703997 |
| kan_operator_local_bumps_semL_operator_replacement_seed1 | COMPLETED | local_bumps | semL_operator_replacement | 1 | 0.289397 | 0.289397 | 0.259987 | 500 | False | 9671 | 5.217521 |
| kan_operator_local_bumps_spline_kan_seed0 | COMPLETED | local_bumps | spline_kan | 0 | 0.034437 | 0.034778 | 0.010869 | 500 | False | 21505 | 2.372984 |
| kan_operator_local_bumps_spline_kan_seed1 | COMPLETED | local_bumps | spline_kan | 1 | 0.033026 | 0.033026 | 0.004361 | 500 | False | 21505 | 2.472070 |
| kan_operator_mixed_composition_semL_operator_replacement_seed0 | COMPLETED | mixed_composition | semL_operator_replacement | 0 | 0.094807 | 0.094807 | 0.081669 | 500 | False | 9671 | 4.753960 |
| kan_operator_mixed_composition_semL_operator_replacement_seed1 | COMPLETED | mixed_composition | semL_operator_replacement | 1 | 0.092655 | 0.092655 | 0.153344 | 500 | False | 9671 | 4.549353 |
| kan_operator_mixed_composition_spline_kan_seed0 | COMPLETED | mixed_composition | spline_kan | 0 | 0.021779 | 0.021779 | 0.000794 | 500 | False | 21505 | 2.448095 |
| kan_operator_mixed_composition_spline_kan_seed1 | COMPLETED | mixed_composition | spline_kan | 1 | 0.024386 | 0.024386 | 0.001992 | 500 | False | 21505 | 2.438577 |

## Raw Artifacts
- `kan_operator_additive_smooth_semL_operator_replacement_seed0`: `reports/runs/20260506_013710_kan_operator_additive_smooth_semL_operator_replacement_seed0`
- `kan_operator_additive_smooth_semL_operator_replacement_seed1`: `reports/runs/20260506_013717_kan_operator_additive_smooth_semL_operator_replacement_seed1`
- `kan_operator_additive_smooth_spline_kan_seed0`: `reports/runs/20260506_013705_kan_operator_additive_smooth_spline_kan_seed0`
- `kan_operator_additive_smooth_spline_kan_seed1`: `reports/runs/20260506_013714_kan_operator_additive_smooth_spline_kan_seed1`
- `kan_operator_local_bumps_semL_operator_replacement_seed0`: `reports/runs/20260506_013723_kan_operator_local_bumps_semL_operator_replacement_seed0`
- `kan_operator_local_bumps_semL_operator_replacement_seed1`: `reports/runs/20260506_013731_kan_operator_local_bumps_semL_operator_replacement_seed1`
- `kan_operator_local_bumps_spline_kan_seed0`: `reports/runs/20260506_013721_kan_operator_local_bumps_spline_kan_seed0`
- `kan_operator_local_bumps_spline_kan_seed1`: `reports/runs/20260506_013728_kan_operator_local_bumps_spline_kan_seed1`
- `kan_operator_mixed_composition_semL_operator_replacement_seed0`: `reports/runs/20260506_013738_kan_operator_mixed_composition_semL_operator_replacement_seed0`
- `kan_operator_mixed_composition_semL_operator_replacement_seed1`: `reports/runs/20260506_013746_kan_operator_mixed_composition_semL_operator_replacement_seed1`
- `kan_operator_mixed_composition_spline_kan_seed0`: `reports/runs/20260506_013736_kan_operator_mixed_composition_spline_kan_seed0`
- `kan_operator_mixed_composition_spline_kan_seed1`: `reports/runs/20260506_013743_kan_operator_mixed_composition_spline_kan_seed1`
