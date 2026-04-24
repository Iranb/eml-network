# Mechanism Probe Report

## Executive Summary
- Completed runs: 98
- Failed runs: 0
- NOT RUN entries: 0
- Probe results are diagnostic checks, not task-level model accuracy.

## Probe Table
| probe | mechanism | n | success rate | null weight | max responsibility | update gate | update norm | resistance-conflict corr | attractor diversity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_noise_should_choose_null | precision_update | 2 | 0.0000 | 0.0000 | 0.1250 | 1.0000 | 0.0000 | nan | nan |
| all_noise_should_choose_null | responsibility_no_null | 2 | 0.0000 | 0.0000 | 0.1250 | 1.0000 | 0.0000 | nan | nan |
| all_noise_should_choose_null | responsibility_null | 2 | 1.0000 | 0.8722 | 0.0160 | 0.1278 | 0.0000 | nan | nan |
| all_noise_should_choose_null | sigmoid_gate_mean | 2 | 0.0000 | 0.0000 | 0.1250 | 0.0180 | 0.0000 | nan | nan |
| all_noise_should_choose_null | sigmoid_gate_sum | 2 | 0.0000 | 0.0000 | 0.1250 | 0.1439 | 0.0000 | nan | nan |
| all_noise_should_choose_null | sigmoid_update | 2 | 0.0000 | 0.0000 | 0.1250 | 1.0000 | 0.0000 | nan | nan |
| all_noise_should_choose_null | thresholded_null | 2 | 1.0000 | 0.8722 | 0.0160 | 0.1278 | 0.0000 | nan | nan |
| attractor_should_not_collapse | precision_update | 2 | 1.0000 | 0.0000 | 0.4463 | 0.0000 | 0.0000 | nan | 0.7584 |
| attractor_should_not_collapse | responsibility_no_null | 2 | 1.0000 | 0.0000 | 0.4463 | 0.0000 | 0.0000 | nan | 0.7584 |
| attractor_should_not_collapse | responsibility_null | 2 | 1.0000 | 0.0000 | 0.4463 | 0.0000 | 0.0000 | nan | 0.7584 |
| attractor_should_not_collapse | sigmoid_gate_mean | 2 | 1.0000 | 0.0000 | 0.4463 | 0.0000 | 0.0000 | nan | 0.7584 |
| attractor_should_not_collapse | sigmoid_gate_sum | 2 | 1.0000 | 0.0000 | 0.4463 | 0.0000 | 0.0000 | nan | 0.7584 |
| attractor_should_not_collapse | sigmoid_update | 2 | 1.0000 | 0.0000 | 0.4463 | 0.0000 | 0.0000 | nan | 0.7584 |
| attractor_should_not_collapse | thresholded_null | 2 | 1.0000 | 0.0000 | 0.4463 | 0.0000 | 0.0000 | nan | 0.7584 |
| composition_requires_consistent_children | precision_update | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan |
| composition_requires_consistent_children | responsibility_no_null | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan |
| composition_requires_consistent_children | responsibility_null | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan |
| composition_requires_consistent_children | sigmoid_gate_mean | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan |
| composition_requires_consistent_children | sigmoid_gate_sum | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan |
| composition_requires_consistent_children | sigmoid_update | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan |
| composition_requires_consistent_children | thresholded_null | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan |
| conflicting_neighbors_increase_resistance | precision_update | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan |
| conflicting_neighbors_increase_resistance | responsibility_no_null | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan |
| conflicting_neighbors_increase_resistance | responsibility_null | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan |
| conflicting_neighbors_increase_resistance | sigmoid_gate_mean | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan |
| conflicting_neighbors_increase_resistance | sigmoid_gate_sum | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan |
| conflicting_neighbors_increase_resistance | sigmoid_update | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan |
| conflicting_neighbors_increase_resistance | thresholded_null | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan |
| old_state_confident_new_evidence_weak_should_not_update | precision_update | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | nan | nan |
| old_state_confident_new_evidence_weak_should_not_update | responsibility_no_null | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | nan | nan |
| old_state_confident_new_evidence_weak_should_not_update | responsibility_null | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | nan | nan |
| old_state_confident_new_evidence_weak_should_not_update | sigmoid_gate_mean | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | nan | nan |
| old_state_confident_new_evidence_weak_should_not_update | sigmoid_gate_sum | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | nan | nan |
| old_state_confident_new_evidence_weak_should_not_update | sigmoid_update | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0003 | 0.0009 | nan | nan |
| old_state_confident_new_evidence_weak_should_not_update | thresholded_null | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | nan | nan |
| old_state_weak_new_evidence_strong_should_update | precision_update | 2 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 2.8283 | nan | nan |
| old_state_weak_new_evidence_strong_should_update | responsibility_no_null | 2 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 2.8283 | nan | nan |
| old_state_weak_new_evidence_strong_should_update | responsibility_null | 2 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 2.8283 | nan | nan |
| old_state_weak_new_evidence_strong_should_update | sigmoid_gate_mean | 2 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 2.8283 | nan | nan |
| old_state_weak_new_evidence_strong_should_update | sigmoid_gate_sum | 2 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 2.8283 | nan | nan |
| old_state_weak_new_evidence_strong_should_update | sigmoid_update | 2 | 1.0000 | 0.0000 | 0.0000 | 0.9997 | 2.8275 | nan | nan |
| old_state_weak_new_evidence_strong_should_update | thresholded_null | 2 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 2.8283 | nan | nan |
| one_strong_signal_many_weak_noise | precision_update | 2 | 1.0000 | 0.0000 | 0.9935 | 1.0000 | 0.0000 | nan | nan |
| one_strong_signal_many_weak_noise | responsibility_no_null | 2 | 1.0000 | 0.0000 | 0.9935 | 1.0000 | 0.0000 | nan | nan |
| one_strong_signal_many_weak_noise | responsibility_null | 2 | 1.0000 | 0.0066 | 0.9869 | 0.9934 | 0.0000 | nan | nan |
| one_strong_signal_many_weak_noise | sigmoid_gate_mean | 2 | 0.0000 | 0.0000 | 0.5409 | 0.2297 | 0.0000 | nan | nan |
| one_strong_signal_many_weak_noise | sigmoid_gate_sum | 2 | 0.0000 | 0.0000 | 0.5409 | 1.0000 | 0.0000 | nan | nan |
| one_strong_signal_many_weak_noise | sigmoid_update | 2 | 1.0000 | 0.0000 | 0.9935 | 1.0000 | 0.0000 | nan | nan |
| one_strong_signal_many_weak_noise | thresholded_null | 2 | 1.0000 | 0.0066 | 0.9869 | 0.9934 | 0.0000 | nan | nan |

## Missing Or Failed
| run_id | status | reason |
| --- | --- | --- |
| none | none | none |

## Raw Artifacts
- `probe_all_noise_should_choose_null_sigmoid_gate_sum_seed0`: `reports/mechanism_probes/runs/20260424_085143_probe_all_noise_should_choose_null_sigmoid_gate_sum_seed0`
- `probe_all_noise_should_choose_null_sigmoid_gate_mean_seed0`: `reports/mechanism_probes/runs/20260424_085143_probe_all_noise_should_choose_null_sigmoid_gate_mean_seed0`
- `probe_all_noise_should_choose_null_responsibility_no_null_seed0`: `reports/mechanism_probes/runs/20260424_085143_probe_all_noise_should_choose_null_responsibility_no_null_seed0`
- `probe_all_noise_should_choose_null_responsibility_null_seed0`: `reports/mechanism_probes/runs/20260424_085143_probe_all_noise_should_choose_null_responsibility_null_seed0`
- `probe_all_noise_should_choose_null_thresholded_null_seed0`: `reports/mechanism_probes/runs/20260424_085143_probe_all_noise_should_choose_null_thresholded_null_seed0`
- `probe_all_noise_should_choose_null_precision_update_seed0`: `reports/mechanism_probes/runs/20260424_085143_probe_all_noise_should_choose_null_precision_update_seed0`
- `probe_all_noise_should_choose_null_sigmoid_update_seed0`: `reports/mechanism_probes/runs/20260424_085143_probe_all_noise_should_choose_null_sigmoid_update_seed0`
- `probe_one_strong_signal_many_weak_noise_sigmoid_gate_sum_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_sigmoid_gate_sum_seed0`
- `probe_one_strong_signal_many_weak_noise_sigmoid_gate_mean_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_sigmoid_gate_mean_seed0`
- `probe_one_strong_signal_many_weak_noise_responsibility_no_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_responsibility_no_null_seed0`
- `probe_one_strong_signal_many_weak_noise_responsibility_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_responsibility_null_seed0`
- `probe_one_strong_signal_many_weak_noise_thresholded_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_thresholded_null_seed0`
- `probe_one_strong_signal_many_weak_noise_precision_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_precision_update_seed0`
- `probe_one_strong_signal_many_weak_noise_sigmoid_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_sigmoid_update_seed0`
- `probe_conflicting_neighbors_increase_resistance_sigmoid_gate_sum_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_sigmoid_gate_sum_seed0`
- `probe_conflicting_neighbors_increase_resistance_sigmoid_gate_mean_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_sigmoid_gate_mean_seed0`
- `probe_conflicting_neighbors_increase_resistance_responsibility_no_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_responsibility_no_null_seed0`
- `probe_conflicting_neighbors_increase_resistance_responsibility_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_responsibility_null_seed0`
- `probe_conflicting_neighbors_increase_resistance_thresholded_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_thresholded_null_seed0`
- `probe_conflicting_neighbors_increase_resistance_precision_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_precision_update_seed0`
- `probe_conflicting_neighbors_increase_resistance_sigmoid_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_sigmoid_update_seed0`
- `probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_gate_sum_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_gate_sum_seed0`
- `probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_gate_mean_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_gate_mean_seed0`
- `probe_old_state_confident_new_evidence_weak_should_not_update_responsibility_no_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_confident_new_evidence_weak_should_not_update_responsibility_no_null_seed0`
- `probe_old_state_confident_new_evidence_weak_should_not_update_responsibility_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_confident_new_evidence_weak_should_not_update_responsibility_null_seed0`
- `probe_old_state_confident_new_evidence_weak_should_not_update_thresholded_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_confident_new_evidence_weak_should_not_update_thresholded_null_seed0`
- `probe_old_state_confident_new_evidence_weak_should_not_update_precision_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_confident_new_evidence_weak_should_not_update_precision_update_seed0`
- `probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_update_seed0`
- `probe_old_state_weak_new_evidence_strong_should_update_sigmoid_gate_sum_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_weak_new_evidence_strong_should_update_sigmoid_gate_sum_seed0`
- `probe_old_state_weak_new_evidence_strong_should_update_sigmoid_gate_mean_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_weak_new_evidence_strong_should_update_sigmoid_gate_mean_seed0`
- `probe_old_state_weak_new_evidence_strong_should_update_responsibility_no_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_weak_new_evidence_strong_should_update_responsibility_no_null_seed0`
- `probe_old_state_weak_new_evidence_strong_should_update_responsibility_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_weak_new_evidence_strong_should_update_responsibility_null_seed0`
- `probe_old_state_weak_new_evidence_strong_should_update_thresholded_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_weak_new_evidence_strong_should_update_thresholded_null_seed0`
- `probe_old_state_weak_new_evidence_strong_should_update_precision_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_weak_new_evidence_strong_should_update_precision_update_seed0`
- `probe_old_state_weak_new_evidence_strong_should_update_sigmoid_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_old_state_weak_new_evidence_strong_should_update_sigmoid_update_seed0`
- `probe_composition_requires_consistent_children_sigmoid_gate_sum_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_composition_requires_consistent_children_sigmoid_gate_sum_seed0`
- `probe_composition_requires_consistent_children_sigmoid_gate_mean_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_composition_requires_consistent_children_sigmoid_gate_mean_seed0`
- `probe_composition_requires_consistent_children_responsibility_no_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_composition_requires_consistent_children_responsibility_no_null_seed0`
- `probe_composition_requires_consistent_children_responsibility_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_composition_requires_consistent_children_responsibility_null_seed0`
- `probe_composition_requires_consistent_children_thresholded_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_composition_requires_consistent_children_thresholded_null_seed0`
- `probe_composition_requires_consistent_children_precision_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_composition_requires_consistent_children_precision_update_seed0`
- `probe_composition_requires_consistent_children_sigmoid_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_composition_requires_consistent_children_sigmoid_update_seed0`
- `probe_attractor_should_not_collapse_sigmoid_gate_sum_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_attractor_should_not_collapse_sigmoid_gate_sum_seed0`
- `probe_attractor_should_not_collapse_sigmoid_gate_mean_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_attractor_should_not_collapse_sigmoid_gate_mean_seed0`
- `probe_attractor_should_not_collapse_responsibility_no_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_attractor_should_not_collapse_responsibility_no_null_seed0`
- `probe_attractor_should_not_collapse_responsibility_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_attractor_should_not_collapse_responsibility_null_seed0`
- `probe_attractor_should_not_collapse_thresholded_null_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_attractor_should_not_collapse_thresholded_null_seed0`
- `probe_attractor_should_not_collapse_precision_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_attractor_should_not_collapse_precision_update_seed0`
- `probe_attractor_should_not_collapse_sigmoid_update_seed0`: `reports/mechanism_probes/runs/20260424_085144_probe_attractor_should_not_collapse_sigmoid_update_seed0`
- `probe_all_noise_should_choose_null_sigmoid_gate_sum_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_all_noise_should_choose_null_sigmoid_gate_sum_seed1`
- `probe_all_noise_should_choose_null_sigmoid_gate_mean_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_all_noise_should_choose_null_sigmoid_gate_mean_seed1`
- `probe_all_noise_should_choose_null_responsibility_no_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_all_noise_should_choose_null_responsibility_no_null_seed1`
- `probe_all_noise_should_choose_null_responsibility_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_all_noise_should_choose_null_responsibility_null_seed1`
- `probe_all_noise_should_choose_null_thresholded_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_all_noise_should_choose_null_thresholded_null_seed1`
- `probe_all_noise_should_choose_null_precision_update_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_all_noise_should_choose_null_precision_update_seed1`
- `probe_all_noise_should_choose_null_sigmoid_update_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_all_noise_should_choose_null_sigmoid_update_seed1`
- `probe_one_strong_signal_many_weak_noise_sigmoid_gate_sum_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_sigmoid_gate_sum_seed1`
- `probe_one_strong_signal_many_weak_noise_sigmoid_gate_mean_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_sigmoid_gate_mean_seed1`
- `probe_one_strong_signal_many_weak_noise_responsibility_no_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_responsibility_no_null_seed1`
- `probe_one_strong_signal_many_weak_noise_responsibility_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_responsibility_null_seed1`
- `probe_one_strong_signal_many_weak_noise_thresholded_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_thresholded_null_seed1`
- `probe_one_strong_signal_many_weak_noise_precision_update_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_precision_update_seed1`
- `probe_one_strong_signal_many_weak_noise_sigmoid_update_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_one_strong_signal_many_weak_noise_sigmoid_update_seed1`
- `probe_conflicting_neighbors_increase_resistance_sigmoid_gate_sum_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_sigmoid_gate_sum_seed1`
- `probe_conflicting_neighbors_increase_resistance_sigmoid_gate_mean_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_sigmoid_gate_mean_seed1`
- `probe_conflicting_neighbors_increase_resistance_responsibility_no_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_responsibility_no_null_seed1`
- `probe_conflicting_neighbors_increase_resistance_responsibility_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_responsibility_null_seed1`
- `probe_conflicting_neighbors_increase_resistance_thresholded_null_seed1`: `reports/mechanism_probes/runs/20260424_085144_probe_conflicting_neighbors_increase_resistance_thresholded_null_seed1`
- `probe_conflicting_neighbors_increase_resistance_precision_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_conflicting_neighbors_increase_resistance_precision_update_seed1`
- `probe_conflicting_neighbors_increase_resistance_sigmoid_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_conflicting_neighbors_increase_resistance_sigmoid_update_seed1`
- `probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_gate_sum_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_gate_sum_seed1`
- `probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_gate_mean_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_gate_mean_seed1`
- `probe_old_state_confident_new_evidence_weak_should_not_update_responsibility_no_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_confident_new_evidence_weak_should_not_update_responsibility_no_null_seed1`
- `probe_old_state_confident_new_evidence_weak_should_not_update_responsibility_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_confident_new_evidence_weak_should_not_update_responsibility_null_seed1`
- `probe_old_state_confident_new_evidence_weak_should_not_update_thresholded_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_confident_new_evidence_weak_should_not_update_thresholded_null_seed1`
- `probe_old_state_confident_new_evidence_weak_should_not_update_precision_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_confident_new_evidence_weak_should_not_update_precision_update_seed1`
- `probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_confident_new_evidence_weak_should_not_update_sigmoid_update_seed1`
- `probe_old_state_weak_new_evidence_strong_should_update_sigmoid_gate_sum_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_weak_new_evidence_strong_should_update_sigmoid_gate_sum_seed1`
- `probe_old_state_weak_new_evidence_strong_should_update_sigmoid_gate_mean_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_weak_new_evidence_strong_should_update_sigmoid_gate_mean_seed1`
- `probe_old_state_weak_new_evidence_strong_should_update_responsibility_no_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_weak_new_evidence_strong_should_update_responsibility_no_null_seed1`
- `probe_old_state_weak_new_evidence_strong_should_update_responsibility_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_weak_new_evidence_strong_should_update_responsibility_null_seed1`
- `probe_old_state_weak_new_evidence_strong_should_update_thresholded_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_weak_new_evidence_strong_should_update_thresholded_null_seed1`
- `probe_old_state_weak_new_evidence_strong_should_update_precision_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_weak_new_evidence_strong_should_update_precision_update_seed1`
- `probe_old_state_weak_new_evidence_strong_should_update_sigmoid_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_old_state_weak_new_evidence_strong_should_update_sigmoid_update_seed1`
- `probe_composition_requires_consistent_children_sigmoid_gate_sum_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_composition_requires_consistent_children_sigmoid_gate_sum_seed1`
- `probe_composition_requires_consistent_children_sigmoid_gate_mean_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_composition_requires_consistent_children_sigmoid_gate_mean_seed1`
- `probe_composition_requires_consistent_children_responsibility_no_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_composition_requires_consistent_children_responsibility_no_null_seed1`
- `probe_composition_requires_consistent_children_responsibility_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_composition_requires_consistent_children_responsibility_null_seed1`
- `probe_composition_requires_consistent_children_thresholded_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_composition_requires_consistent_children_thresholded_null_seed1`
- `probe_composition_requires_consistent_children_precision_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_composition_requires_consistent_children_precision_update_seed1`
- `probe_composition_requires_consistent_children_sigmoid_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_composition_requires_consistent_children_sigmoid_update_seed1`
- `probe_attractor_should_not_collapse_sigmoid_gate_sum_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_attractor_should_not_collapse_sigmoid_gate_sum_seed1`
- `probe_attractor_should_not_collapse_sigmoid_gate_mean_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_attractor_should_not_collapse_sigmoid_gate_mean_seed1`
- `probe_attractor_should_not_collapse_responsibility_no_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_attractor_should_not_collapse_responsibility_no_null_seed1`
- `probe_attractor_should_not_collapse_responsibility_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_attractor_should_not_collapse_responsibility_null_seed1`
- `probe_attractor_should_not_collapse_thresholded_null_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_attractor_should_not_collapse_thresholded_null_seed1`
- `probe_attractor_should_not_collapse_precision_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_attractor_should_not_collapse_precision_update_seed1`
- `probe_attractor_should_not_collapse_sigmoid_update_seed1`: `reports/mechanism_probes/runs/20260424_085145_probe_attractor_should_not_collapse_sigmoid_update_seed1`
