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
