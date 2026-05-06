# Text Representation Ablation Report

## Summary
- Completed runs: 42
- Failed runs: 0
- NOT RUN entries: 0
- Window size `8` is the default efficient text path in this report.

## Results
| model | n | best token acc | mean token acc | mean loss | bits/token | null weight | update gate | entropy | attractor diversity | corruption corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EMLEdgeTextLM_kan_style | 2 | 0.2102 | 0.1447 | 4.3465 | 6.2707 | nan | nan | nan | nan | 0.0450 |
| EMLTextBackbone | 4 | 0.0357 | 0.0184 | 4.4256 | 6.3848 | nan | nan | nan | nan | -0.0493 |
| EfficientEMLTextEncoder_best_text_config | 4 | 0.1333 | 0.1206 | 4.3666 | 6.2996 | 0.5338 | 0.0499 | nan | 0.0492 | 0.0586 |
| EfficientEMLTextEncoder_window8 | 4 | 0.2102 | 0.1614 | 4.3568 | 6.2856 | 0.5548 | 0.0475 | nan | 0.7388 | -0.0140 |
| EfficientEMLTextEncoder_window8_chunk | 4 | 0.2102 | 0.1614 | 4.3568 | 6.2856 | 0.5482 | 0.0475 | nan | 0.9740 | -0.0140 |
| EfficientEMLTextEncoder_window8_chunk_attractor | 4 | 0.1333 | 0.1206 | 4.3665 | 6.2995 | 0.5294 | 0.0503 | nan | 0.0492 | 0.0592 |
| EfficientEMLTextEncoder_window8_precision_identity | 4 | 0.2102 | 0.1614 | 4.3568 | 6.2856 | 0.5548 | 0.0475 | nan | 0.7388 | -0.0140 |
| EfficientEMLTextEncoder_window8_staged | 4 | 0.1333 | 0.1206 | 4.3666 | 6.2996 | 0.5338 | 0.0499 | nan | 0.0492 | 0.0586 |
| EfficientEMLTextEncoder_window8_thresholded_null | 4 | 0.2102 | 0.1614 | 4.3568 | 6.2856 | 0.5548 | 0.0475 | nan | 0.7388 | -0.0140 |
| LocalCausalConvLM | 4 | 0.0284 | 0.0205 | 4.4235 | 6.3817 | nan | nan | nan | nan | nan |
| SmallGRULM | 4 | 0.0455 | 0.0269 | 4.4132 | 6.3668 | nan | nan | nan | nan | nan |

## Missing Or Failed
| run_id | status | model | reason |
| --- | --- | --- | --- |
| none | none | none | none |

## Raw Artifacts
- `local_causal_conv_seed0`: `reports/text_representation_ablation/runs/20260424_085213_local_causal_conv_seed0`
- `small_gru_seed0`: `reports/text_representation_ablation/runs/20260424_085213_small_gru_seed0`
- `old_eml_text_backbone_seed0`: `reports/text_representation_ablation/runs/20260424_085213_old_eml_text_backbone_seed0`
- `efficient_window8_seed0`: `reports/text_representation_ablation/runs/20260424_085213_efficient_window8_seed0`
- `efficient_window8_thresholded_null_seed0`: `reports/text_representation_ablation/runs/20260424_085213_efficient_window8_thresholded_null_seed0`
- `efficient_window8_precision_identity_seed0`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_precision_identity_seed0`
- `efficient_window8_chunk_seed0`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_chunk_seed0`
- `efficient_window8_chunk_attractor_seed0`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_chunk_attractor_seed0`
- `efficient_window8_staged_seed0`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_staged_seed0`
- `best_text_config_seed0`: `reports/text_representation_ablation/runs/20260424_085214_best_text_config_seed0`
- `local_causal_conv_seed1`: `reports/text_representation_ablation/runs/20260424_085214_local_causal_conv_seed1`
- `small_gru_seed1`: `reports/text_representation_ablation/runs/20260424_085214_small_gru_seed1`
- `old_eml_text_backbone_seed1`: `reports/text_representation_ablation/runs/20260424_085214_old_eml_text_backbone_seed1`
- `efficient_window8_seed1`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_seed1`
- `efficient_window8_thresholded_null_seed1`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_thresholded_null_seed1`
- `efficient_window8_precision_identity_seed1`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_precision_identity_seed1`
- `efficient_window8_chunk_seed1`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_chunk_seed1`
- `efficient_window8_chunk_attractor_seed1`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_chunk_attractor_seed1`
- `efficient_window8_staged_seed1`: `reports/text_representation_ablation/runs/20260424_085214_efficient_window8_staged_seed1`
- `best_text_config_seed1`: `reports/text_representation_ablation/runs/20260424_085214_best_text_config_seed1`
- `local_causal_conv_seed0`: `reports/text_representation_ablation/runs/20260505_123520_local_causal_conv_seed0`
- `eml_edge_kan_text_seed0`: `reports/text_representation_ablation/runs/20260505_123521_eml_edge_kan_text_seed0`
- `small_gru_seed0`: `reports/text_representation_ablation/runs/20260505_123521_small_gru_seed0`
- `old_eml_text_backbone_seed0`: `reports/text_representation_ablation/runs/20260505_123521_old_eml_text_backbone_seed0`
- `efficient_window8_seed0`: `reports/text_representation_ablation/runs/20260505_123521_efficient_window8_seed0`
- `efficient_window8_thresholded_null_seed0`: `reports/text_representation_ablation/runs/20260505_123521_efficient_window8_thresholded_null_seed0`
- `efficient_window8_precision_identity_seed0`: `reports/text_representation_ablation/runs/20260505_123521_efficient_window8_precision_identity_seed0`
- `efficient_window8_chunk_seed0`: `reports/text_representation_ablation/runs/20260505_123522_efficient_window8_chunk_seed0`
- `efficient_window8_chunk_attractor_seed0`: `reports/text_representation_ablation/runs/20260505_123522_efficient_window8_chunk_attractor_seed0`
- `efficient_window8_staged_seed0`: `reports/text_representation_ablation/runs/20260505_123522_efficient_window8_staged_seed0`
- `best_text_config_seed0`: `reports/text_representation_ablation/runs/20260505_123522_best_text_config_seed0`
- `local_causal_conv_seed1`: `reports/text_representation_ablation/runs/20260505_123522_local_causal_conv_seed1`
- `eml_edge_kan_text_seed1`: `reports/text_representation_ablation/runs/20260505_123522_eml_edge_kan_text_seed1`
- `small_gru_seed1`: `reports/text_representation_ablation/runs/20260505_123522_small_gru_seed1`
- `old_eml_text_backbone_seed1`: `reports/text_representation_ablation/runs/20260505_123522_old_eml_text_backbone_seed1`
- `efficient_window8_seed1`: `reports/text_representation_ablation/runs/20260505_123522_efficient_window8_seed1`
- `efficient_window8_thresholded_null_seed1`: `reports/text_representation_ablation/runs/20260505_123523_efficient_window8_thresholded_null_seed1`
- `efficient_window8_precision_identity_seed1`: `reports/text_representation_ablation/runs/20260505_123523_efficient_window8_precision_identity_seed1`
- `efficient_window8_chunk_seed1`: `reports/text_representation_ablation/runs/20260505_123523_efficient_window8_chunk_seed1`
- `efficient_window8_chunk_attractor_seed1`: `reports/text_representation_ablation/runs/20260505_123523_efficient_window8_chunk_attractor_seed1`
- `efficient_window8_staged_seed1`: `reports/text_representation_ablation/runs/20260505_123523_efficient_window8_staged_seed1`
- `best_text_config_seed1`: `reports/text_representation_ablation/runs/20260505_123523_best_text_config_seed1`
