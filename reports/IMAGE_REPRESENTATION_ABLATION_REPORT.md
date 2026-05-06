# Image Representation Ablation Report

## Summary
- Completed runs: 54
- Failed runs: 0
- NOT RUN entries: 0
- Synthetic shape accuracy above `0.8` is the gate before making CIFAR claims for the efficient representation path.

## Results
| model | n | best accuracy | mean final accuracy | mean loss | mean time sec | null weight | update gate | ambiguity | attractor diversity | noise corr | occlusion corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EMLEdgeImageClassifier_kan_style | 2 | 0.3750 | 0.1250 | 1.6089 | 0.0470 | nan | nan | nan | nan | 0.3900 | 0.1751 |
| EfficientEMLImageClassifier_baseline | 4 | 0.3750 | 0.1250 | 1.6209 | 0.1188 | 0.1596 | 0.0914 | 0.6797 | 0.0223 | -0.4359 | -0.0592 |
| EfficientEMLImageClassifier_bypass_staged | 4 | 0.3750 | 0.1250 | 1.6225 | 0.1106 | 0.1667 | 0.0906 | 0.6174 | 0.0255 | -0.1395 | 0.3110 |
| EfficientEMLImageClassifier_centered_ambiguity | 4 | 0.3750 | 0.1250 | 1.6209 | 0.1081 | 0.1596 | 0.0914 | 0.6797 | 0.0223 | -0.4359 | -0.0592 |
| EfficientEMLImageClassifier_combo | 4 | 0.3750 | 0.1250 | 1.6209 | 0.1065 | 0.1596 | 0.0914 | 0.6797 | 0.0223 | -0.4359 | -0.0592 |
| EfficientEMLImageClassifier_head_without_ambiguity | 4 | 0.3750 | 0.1250 | 1.6169 | 0.1031 | 0.1596 | 0.0914 | 2.0769 | 0.0216 | nan | nan |
| EfficientEMLImageClassifier_no_attractor | 4 | 0.3750 | 0.1250 | 1.6475 | 0.1013 | 0.1647 | 0.0901 | 0.6930 | 1.0000 | 0.0926 | -0.2230 |
| EfficientEMLImageClassifier_no_composition | 4 | 0.3750 | 0.1250 | 1.6783 | 0.1251 | 0.1426 | 0.0916 | 0.7142 | 0.0352 | 0.1623 | 0.3898 |
| EfficientEMLImageClassifier_precision_identity | 4 | 0.3750 | 0.1250 | 1.6209 | 0.1119 | 0.1596 | 0.0914 | 0.6797 | 0.0223 | -0.4359 | -0.0592 |
| EfficientEMLImageClassifier_sensor_bypass | 4 | 0.3750 | 0.1250 | 1.6234 | 0.1309 | 0.1597 | 0.0913 | 0.6196 | 0.0259 | -0.1388 | 0.3032 |
| EfficientEMLImageClassifier_staged | 4 | 0.3750 | 0.1250 | 1.6205 | 0.1167 | 0.1666 | 0.0907 | 0.6773 | 0.0226 | -0.4536 | -0.0383 |
| EfficientEMLImageClassifier_thresholded_null | 4 | 0.3750 | 0.1250 | 1.6209 | 0.1111 | 0.1596 | 0.0914 | 0.6797 | 0.0223 | -0.4359 | -0.0592 |
| cnn_eml_workers0 | 4 | 0.2500 | 0.1875 | 1.6443 | 0.0817 | nan | nan | 0.4120 | nan | 0.0607 | -0.0377 |
| pure_eml_workers0 | 4 | 0.3750 | 0.3125 | 1.6238 | 0.0355 | nan | nan | 0.5156 | nan | -0.0326 | 0.1668 |

## CIFAR Gate
- Best efficient synthetic result in this report: `0.3750`
- CIFAR medium should be skipped unless this value is at least `0.8`.

## Missing Or Failed
| run_id | status | model | reason |
| --- | --- | --- | --- |
| none | none | none | none |

## Raw Artifacts
- `cnn_eml_workers0_seed0`: `reports/image_representation_ablation/runs/20260424_085157_cnn_eml_workers0_seed0`
- `pure_eml_workers0_seed0`: `reports/image_representation_ablation/runs/20260424_085157_pure_eml_workers0_seed0`
- `efficient_baseline_seed0`: `reports/image_representation_ablation/runs/20260424_085157_efficient_baseline_seed0`
- `efficient_centered_ambiguity_seed0`: `reports/image_representation_ablation/runs/20260424_085158_efficient_centered_ambiguity_seed0`
- `efficient_thresholded_null_seed0`: `reports/image_representation_ablation/runs/20260424_085158_efficient_thresholded_null_seed0`
- `efficient_precision_identity_seed0`: `reports/image_representation_ablation/runs/20260424_085158_efficient_precision_identity_seed0`
- `efficient_combo_seed0`: `reports/image_representation_ablation/runs/20260424_085158_efficient_combo_seed0`
- `efficient_combo_no_composition_seed0`: `reports/image_representation_ablation/runs/20260424_085158_efficient_combo_no_composition_seed0`
- `efficient_combo_no_attractor_seed0`: `reports/image_representation_ablation/runs/20260424_085158_efficient_combo_no_attractor_seed0`
- `efficient_combo_sensor_bypass_seed0`: `reports/image_representation_ablation/runs/20260424_085158_efficient_combo_sensor_bypass_seed0`
- `efficient_combo_staged_seed0`: `reports/image_representation_ablation/runs/20260424_085159_efficient_combo_staged_seed0`
- `efficient_combo_bypass_staged_seed0`: `reports/image_representation_ablation/runs/20260424_085159_efficient_combo_bypass_staged_seed0`
- `head_without_ambiguity_seed0`: `reports/image_representation_ablation/runs/20260424_085159_head_without_ambiguity_seed0`
- `cnn_eml_workers0_seed1`: `reports/image_representation_ablation/runs/20260424_085159_cnn_eml_workers0_seed1`
- `pure_eml_workers0_seed1`: `reports/image_representation_ablation/runs/20260424_085159_pure_eml_workers0_seed1`
- `efficient_baseline_seed1`: `reports/image_representation_ablation/runs/20260424_085159_efficient_baseline_seed1`
- `efficient_centered_ambiguity_seed1`: `reports/image_representation_ablation/runs/20260424_085159_efficient_centered_ambiguity_seed1`
- `efficient_thresholded_null_seed1`: `reports/image_representation_ablation/runs/20260424_085159_efficient_thresholded_null_seed1`
- `efficient_precision_identity_seed1`: `reports/image_representation_ablation/runs/20260424_085200_efficient_precision_identity_seed1`
- `efficient_combo_seed1`: `reports/image_representation_ablation/runs/20260424_085200_efficient_combo_seed1`
- `efficient_combo_no_composition_seed1`: `reports/image_representation_ablation/runs/20260424_085200_efficient_combo_no_composition_seed1`
- `efficient_combo_no_attractor_seed1`: `reports/image_representation_ablation/runs/20260424_085200_efficient_combo_no_attractor_seed1`
- `efficient_combo_sensor_bypass_seed1`: `reports/image_representation_ablation/runs/20260424_085200_efficient_combo_sensor_bypass_seed1`
- `efficient_combo_staged_seed1`: `reports/image_representation_ablation/runs/20260424_085200_efficient_combo_staged_seed1`
- `efficient_combo_bypass_staged_seed1`: `reports/image_representation_ablation/runs/20260424_085200_efficient_combo_bypass_staged_seed1`
- `head_without_ambiguity_seed1`: `reports/image_representation_ablation/runs/20260424_085200_head_without_ambiguity_seed1`
- `cnn_eml_workers0_seed0`: `reports/image_representation_ablation/runs/20260505_123520_cnn_eml_workers0_seed0`
- `eml_edge_kan_seed0`: `reports/image_representation_ablation/runs/20260505_123521_eml_edge_kan_seed0`
- `pure_eml_workers0_seed0`: `reports/image_representation_ablation/runs/20260505_123521_pure_eml_workers0_seed0`
- `efficient_baseline_seed0`: `reports/image_representation_ablation/runs/20260505_123521_efficient_baseline_seed0`
- `efficient_centered_ambiguity_seed0`: `reports/image_representation_ablation/runs/20260505_123521_efficient_centered_ambiguity_seed0`
- `efficient_thresholded_null_seed0`: `reports/image_representation_ablation/runs/20260505_123521_efficient_thresholded_null_seed0`
- `efficient_precision_identity_seed0`: `reports/image_representation_ablation/runs/20260505_123521_efficient_precision_identity_seed0`
- `efficient_combo_seed0`: `reports/image_representation_ablation/runs/20260505_123522_efficient_combo_seed0`
- `efficient_combo_no_composition_seed0`: `reports/image_representation_ablation/runs/20260505_123522_efficient_combo_no_composition_seed0`
- `efficient_combo_no_attractor_seed0`: `reports/image_representation_ablation/runs/20260505_123522_efficient_combo_no_attractor_seed0`
- `efficient_combo_sensor_bypass_seed0`: `reports/image_representation_ablation/runs/20260505_123522_efficient_combo_sensor_bypass_seed0`
- `efficient_combo_staged_seed0`: `reports/image_representation_ablation/runs/20260505_123522_efficient_combo_staged_seed0`
- `efficient_combo_bypass_staged_seed0`: `reports/image_representation_ablation/runs/20260505_123522_efficient_combo_bypass_staged_seed0`
- `head_without_ambiguity_seed0`: `reports/image_representation_ablation/runs/20260505_123523_head_without_ambiguity_seed0`
- `cnn_eml_workers0_seed1`: `reports/image_representation_ablation/runs/20260505_123523_cnn_eml_workers0_seed1`
- `eml_edge_kan_seed1`: `reports/image_representation_ablation/runs/20260505_123523_eml_edge_kan_seed1`
- `pure_eml_workers0_seed1`: `reports/image_representation_ablation/runs/20260505_123523_pure_eml_workers0_seed1`
- `efficient_baseline_seed1`: `reports/image_representation_ablation/runs/20260505_123523_efficient_baseline_seed1`
- `efficient_centered_ambiguity_seed1`: `reports/image_representation_ablation/runs/20260505_123523_efficient_centered_ambiguity_seed1`
- `efficient_thresholded_null_seed1`: `reports/image_representation_ablation/runs/20260505_123523_efficient_thresholded_null_seed1`
- `efficient_precision_identity_seed1`: `reports/image_representation_ablation/runs/20260505_123523_efficient_precision_identity_seed1`
- `efficient_combo_seed1`: `reports/image_representation_ablation/runs/20260505_123523_efficient_combo_seed1`
- `efficient_combo_no_composition_seed1`: `reports/image_representation_ablation/runs/20260505_123524_efficient_combo_no_composition_seed1`
- `efficient_combo_no_attractor_seed1`: `reports/image_representation_ablation/runs/20260505_123524_efficient_combo_no_attractor_seed1`
- `efficient_combo_sensor_bypass_seed1`: `reports/image_representation_ablation/runs/20260505_123524_efficient_combo_sensor_bypass_seed1`
- `efficient_combo_staged_seed1`: `reports/image_representation_ablation/runs/20260505_123524_efficient_combo_staged_seed1`
- `efficient_combo_bypass_staged_seed1`: `reports/image_representation_ablation/runs/20260505_123524_efficient_combo_bypass_staged_seed1`
- `head_without_ambiguity_seed1`: `reports/image_representation_ablation/runs/20260505_123524_head_without_ambiguity_seed1`
