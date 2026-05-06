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
| additive_smooth | semL_operator_replacement | 3 | 0.000255 | 0.000196 | 0.000170 | 35719.000000 |
| additive_smooth | spline_kan | 3 | 0.003145 | 0.003080 | 0.000018 | 150529.000000 |
| local_bumps | semL_operator_replacement | 3 | 0.002195 | 0.002365 | 0.002086 | 35719.000000 |
| local_bumps | spline_kan | 3 | 0.009868 | 0.009858 | 0.000151 | 150529.000000 |
| mixed_composition | semL_operator_replacement | 3 | 0.000415 | 0.000657 | 0.000531 | 35719.000000 |
| mixed_composition | spline_kan | 3 | 0.006631 | 0.006628 | 0.000211 | 150529.000000 |

## Runs
| run_id | status | task | model | seed | best val MSE | final val MSE | final train MSE | steps | early stop | params | time sec |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| kan_operator_additive_smooth_semL_operator_replacement_seed0 | COMPLETED | additive_smooth | semL_operator_replacement | 0 | 0.000217 | 0.000151 | 0.000146 | 2750 | True | 35719 | 24.381772 |
| kan_operator_additive_smooth_semL_operator_replacement_seed1 | COMPLETED | additive_smooth | semL_operator_replacement | 1 | 0.000229 | 0.000161 | 0.000167 | 2950 | True | 35719 | 27.129488 |
| kan_operator_additive_smooth_semL_operator_replacement_seed2 | COMPLETED | additive_smooth | semL_operator_replacement | 2 | 0.000319 | 0.000277 | 0.000199 | 2500 | True | 35719 | 21.742487 |
| kan_operator_additive_smooth_spline_kan_seed0 | COMPLETED | additive_smooth | spline_kan | 0 | 0.001568 | 0.001497 | 0.000030 | 5000 | False | 150529 | 20.690419 |
| kan_operator_additive_smooth_spline_kan_seed1 | COMPLETED | additive_smooth | spline_kan | 1 | 0.004210 | 0.004136 | 0.000022 | 1050 | True | 150529 | 4.622183 |
| kan_operator_additive_smooth_spline_kan_seed2 | COMPLETED | additive_smooth | spline_kan | 2 | 0.003657 | 0.003607 | 0.000004 | 1100 | True | 150529 | 5.253544 |
| kan_operator_local_bumps_semL_operator_replacement_seed0 | COMPLETED | local_bumps | semL_operator_replacement | 0 | 0.001543 | 0.001874 | 0.001812 | 5000 | False | 35719 | 44.706755 |
| kan_operator_local_bumps_semL_operator_replacement_seed1 | COMPLETED | local_bumps | semL_operator_replacement | 1 | 0.002194 | 0.002194 | 0.001443 | 5000 | False | 35719 | 48.401096 |
| kan_operator_local_bumps_semL_operator_replacement_seed2 | COMPLETED | local_bumps | semL_operator_replacement | 2 | 0.002846 | 0.003025 | 0.003005 | 5000 | False | 35719 | 44.923506 |
| kan_operator_local_bumps_spline_kan_seed0 | COMPLETED | local_bumps | spline_kan | 0 | 0.008159 | 0.008197 | 0.000092 | 5000 | False | 150529 | 22.282780 |
| kan_operator_local_bumps_spline_kan_seed1 | COMPLETED | local_bumps | spline_kan | 1 | 0.010872 | 0.010841 | 0.000057 | 5000 | False | 150529 | 22.796094 |
| kan_operator_local_bumps_spline_kan_seed2 | COMPLETED | local_bumps | spline_kan | 2 | 0.010572 | 0.010536 | 0.000304 | 5000 | False | 150529 | 23.327072 |
| kan_operator_mixed_composition_semL_operator_replacement_seed0 | COMPLETED | mixed_composition | semL_operator_replacement | 0 | 0.000443 | 0.000382 | 0.000514 | 4300 | True | 35719 | 38.735192 |
| kan_operator_mixed_composition_semL_operator_replacement_seed1 | COMPLETED | mixed_composition | semL_operator_replacement | 1 | 0.000354 | 0.001107 | 0.000736 | 4300 | True | 35719 | 37.955453 |
| kan_operator_mixed_composition_semL_operator_replacement_seed2 | COMPLETED | mixed_composition | semL_operator_replacement | 2 | 0.000449 | 0.000480 | 0.000343 | 4400 | True | 35719 | 36.614188 |
| kan_operator_mixed_composition_spline_kan_seed0 | COMPLETED | mixed_composition | spline_kan | 0 | 0.007312 | 0.007240 | 0.000595 | 5000 | False | 150529 | 23.548173 |
| kan_operator_mixed_composition_spline_kan_seed1 | COMPLETED | mixed_composition | spline_kan | 1 | 0.006275 | 0.006263 | 0.000005 | 5000 | False | 150529 | 23.981278 |
| kan_operator_mixed_composition_spline_kan_seed2 | COMPLETED | mixed_composition | spline_kan | 2 | 0.006307 | 0.006382 | 0.000032 | 5000 | False | 150529 | 23.355663 |

## Raw Artifacts
- `kan_operator_additive_smooth_semL_operator_replacement_seed0`: `reports/runs/20260506_014257_kan_operator_additive_smooth_semL_operator_replacement_seed0`
- `kan_operator_additive_smooth_semL_operator_replacement_seed1`: `reports/runs/20260506_014326_kan_operator_additive_smooth_semL_operator_replacement_seed1`
- `kan_operator_additive_smooth_semL_operator_replacement_seed2`: `reports/runs/20260506_014359_kan_operator_additive_smooth_semL_operator_replacement_seed2`
- `kan_operator_additive_smooth_spline_kan_seed0`: `reports/runs/20260506_014236_kan_operator_additive_smooth_spline_kan_seed0`
- `kan_operator_additive_smooth_spline_kan_seed1`: `reports/runs/20260506_014322_kan_operator_additive_smooth_spline_kan_seed1`
- `kan_operator_additive_smooth_spline_kan_seed2`: `reports/runs/20260506_014354_kan_operator_additive_smooth_spline_kan_seed2`
- `kan_operator_local_bumps_semL_operator_replacement_seed0`: `reports/runs/20260506_014443_kan_operator_local_bumps_semL_operator_replacement_seed0`
- `kan_operator_local_bumps_semL_operator_replacement_seed1`: `reports/runs/20260506_014551_kan_operator_local_bumps_semL_operator_replacement_seed1`
- `kan_operator_local_bumps_semL_operator_replacement_seed2`: `reports/runs/20260506_014702_kan_operator_local_bumps_semL_operator_replacement_seed2`
- `kan_operator_local_bumps_spline_kan_seed0`: `reports/runs/20260506_014421_kan_operator_local_bumps_spline_kan_seed0`
- `kan_operator_local_bumps_spline_kan_seed1`: `reports/runs/20260506_014528_kan_operator_local_bumps_spline_kan_seed1`
- `kan_operator_local_bumps_spline_kan_seed2`: `reports/runs/20260506_014639_kan_operator_local_bumps_spline_kan_seed2`
- `kan_operator_mixed_composition_semL_operator_replacement_seed0`: `reports/runs/20260506_014811_kan_operator_mixed_composition_semL_operator_replacement_seed0`
- `kan_operator_mixed_composition_semL_operator_replacement_seed1`: `reports/runs/20260506_014914_kan_operator_mixed_composition_semL_operator_replacement_seed1`
- `kan_operator_mixed_composition_semL_operator_replacement_seed2`: `reports/runs/20260506_015015_kan_operator_mixed_composition_semL_operator_replacement_seed2`
- `kan_operator_mixed_composition_spline_kan_seed0`: `reports/runs/20260506_014747_kan_operator_mixed_composition_spline_kan_seed0`
- `kan_operator_mixed_composition_spline_kan_seed1`: `reports/runs/20260506_014850_kan_operator_mixed_composition_spline_kan_seed1`
- `kan_operator_mixed_composition_spline_kan_seed2`: `reports/runs/20260506_014952_kan_operator_mixed_composition_spline_kan_seed2`
