# EML Validation and Ablation Report

Generated: 2026-05-06T05:07:43Z

## 1. Executive Summary

- Best image result: EfficientEMLImageClassifier (0.75)
- Best text result: EMLEdgeTextLM_kan_style (0.5025380849838257)
- Strongest baseline: cnn_eml (0.75)
- Responsibility evidence weighting: see mechanism probe and downstream ablation tables.
- Precision update: see update probe rows and text/image ablation cells; model-quality conclusions need longer runs.
- Attractor memory: no-attractor comparison is MISSING unless a completed row exists below.
- Major failure modes: 0 failed runs and 19 not-run cells are recorded in the status table.
- Recommended next step: standardize the remaining NOT RUN switches, then repeat the best image/text runs across seeds.

## 2. Repository and Environment

- git_commit: 9ce7b7d
- hostname: Mac-mini.local
- python_version: 3.12.7
- torch_version: 2.10.0
- torchvision_version: 0.25.0
- cuda_available: False
- device: cpu
- timestamp: 2026-05-06T05:05:35Z

## 3. Experimental Scope

| run_id | status | task | model | dataset | reason |
| --- | --- | --- | --- | --- | --- |
| smoke_image_cnn_eml_baseline | COMPLETED | image_synthetic | cnn_eml | SyntheticShapeEnergyDataset |  |
| smoke_image_efficient_eml | COMPLETED | image_synthetic | EfficientEMLImageClassifier | SyntheticShapeEnergyDataset |  |
| smoke_text_efficient_eml | COMPLETED | text_synthetic | EfficientEMLTextEncoder | SyntheticTextEnergyDataset |  |
| probe_gate_compat_sigmoid_update | COMPLETED | mechanism_probe | probe_gate_compat_sigmoid_update | synthetic_probe_tensors |  |
| probe_responsibility_no_null_precision | COMPLETED | mechanism_probe | probe_responsibility_no_null_precision | synthetic_probe_tensors |  |
| probe_responsibility_with_null_precision | COMPLETED | mechanism_probe | probe_responsibility_with_null_precision | synthetic_probe_tensors |  |
| local_conv_baseline | NOT RUN | image_synthetic | LocalConvBaseline | SyntheticShapeEnergyDataset | smoke mode: not implemented |
| local_text_linear_baseline | NOT RUN | text_synthetic | LocalTextCodecLinear | SyntheticTextEnergyDataset | smoke mode: not standardized |
| cifar_medium_suite | NOT RUN | image_cifar | selected_image_models | CIFAR10 | smoke mode: not requested in this mode |
| text_medium_suite | NOT RUN | text_synthetic | selected_text_models | SyntheticTextEnergyDataset | smoke mode: not requested in this mode |
| full_seeded_ablation | NOT RUN | mechanism_ablation | all_supported_cells | mixed | smoke mode: not requested in this mode |
| kan_paper_reference | NOT RUN | paper_reference | KAN_arxiv_2404_19756 | KAN paper AI+Science tasks | User requested no local KAN experiment; comparing against reported paper results only. |
| kan_compare_image_cnn_eml | COMPLETED | image_synthetic | cnn_eml | SyntheticShapeEnergyDataset |  |
| kan_compare_image_efficient_eml | COMPLETED | image_synthetic | EfficientEMLImageClassifier | SyntheticShapeEnergyDataset |  |
| kan_compare_image_eml_edge | COMPLETED | image_synthetic | EMLEdgeImageClassifier_kan_style | SyntheticShapeEnergyDataset |  |
| kan_compare_text_local_conv | COMPLETED | text_synthetic | LocalCausalConvLM | SyntheticTextEnergyDataset |  |
| kan_compare_text_small_gru | COMPLETED | text_synthetic | SmallGRULM | SyntheticTextEnergyDataset |  |
| kan_compare_text_efficient_eml | COMPLETED | text_synthetic | EfficientEMLTextEncoder | SyntheticTextEnergyDataset |  |
| kan_compare_text_eml_edge | COMPLETED | text_synthetic | EMLEdgeTextLM_kan_style | SyntheticTextEnergyDataset |  |
| kan_paper_reference | NOT RUN | paper_reference | KAN_arxiv_2404_19756 | KAN paper AI+Science tasks | User requested no local KAN experiment; comparing against reported paper results only. |
| kan_compare_image_cnn_eml | COMPLETED | image_synthetic | cnn_eml | SyntheticShapeEnergyDataset |  |
| kan_compare_image_efficient_eml | COMPLETED | image_synthetic | EfficientEMLImageClassifier | SyntheticShapeEnergyDataset |  |
| kan_compare_image_eml_edge | COMPLETED | image_synthetic | EMLEdgeImageClassifier_kan_style | SyntheticShapeEnergyDataset |  |
| kan_compare_text_local_conv | COMPLETED | text_synthetic | LocalCausalConvLM | SyntheticTextEnergyDataset |  |
| kan_compare_text_small_gru | COMPLETED | text_synthetic | SmallGRULM | SyntheticTextEnergyDataset |  |
| kan_compare_text_efficient_eml | COMPLETED | text_synthetic | EfficientEMLTextEncoder | SyntheticTextEnergyDataset |  |
| kan_compare_text_eml_edge | COMPLETED | text_synthetic | EMLEdgeTextLM_kan_style | SyntheticTextEnergyDataset |  |
| kan_paper_reference | NOT RUN | paper_reference | KAN_arxiv_2404_19756 | KAN paper AI+Science tasks | User requested no local KAN experiment; comparing against reported paper results only. |
| kan_compare_image_cnn_eml | COMPLETED | image_synthetic | cnn_eml | SyntheticShapeEnergyDataset |  |
| kan_compare_image_efficient_eml | COMPLETED | image_synthetic | EfficientEMLImageClassifier | SyntheticShapeEnergyDataset |  |
| kan_compare_image_eml_edge | COMPLETED | image_synthetic | EMLEdgeImageClassifier_kan_style | SyntheticShapeEnergyDataset |  |
| kan_compare_text_local_conv | COMPLETED | text_synthetic | LocalCausalConvLM | SyntheticTextEnergyDataset |  |
| kan_compare_text_small_gru | COMPLETED | text_synthetic | SmallGRULM | SyntheticTextEnergyDataset |  |
| kan_compare_text_efficient_eml | COMPLETED | text_synthetic | EfficientEMLTextEncoder | SyntheticTextEnergyDataset |  |
| kan_compare_text_eml_edge | COMPLETED | text_synthetic | EMLEdgeTextLM_kan_style | SyntheticTextEnergyDataset |  |
| kan_paper_reference | NOT RUN | paper_reference | KAN_arxiv_2404_19756 | KAN paper AI+Science tasks | User requested no local KAN experiment; comparing against reported paper results only. |
| kan_compare_image_cnn_eml | COMPLETED | image_synthetic | cnn_eml | SyntheticShapeEnergyDataset |  |
| kan_compare_image_efficient_eml | COMPLETED | image_synthetic | EfficientEMLImageClassifier | SyntheticShapeEnergyDataset |  |
| kan_compare_image_eml_edge | COMPLETED | image_synthetic | EMLEdgeImageClassifier_kan_style | SyntheticShapeEnergyDataset |  |
| kan_compare_text_local_conv | COMPLETED | text_synthetic | LocalCausalConvLM | SyntheticTextEnergyDataset |  |
| kan_compare_text_small_gru | COMPLETED | text_synthetic | SmallGRULM | SyntheticTextEnergyDataset |  |
| kan_compare_text_efficient_eml | COMPLETED | text_synthetic | EfficientEMLTextEncoder | SyntheticTextEnergyDataset |  |
| kan_compare_text_eml_edge | COMPLETED | text_synthetic | EMLEdgeTextLM_kan_style | SyntheticTextEnergyDataset |  |
| smoke_image_cnn_eml_baseline | COMPLETED | image_synthetic | cnn_eml | SyntheticShapeEnergyDataset |  |
| smoke_image_efficient_eml | COMPLETED | image_synthetic | EfficientEMLImageClassifier | SyntheticShapeEnergyDataset |  |
| smoke_text_efficient_eml | COMPLETED | text_synthetic | EfficientEMLTextEncoder | SyntheticTextEnergyDataset |  |
| probe_gate_compat_sigmoid_update | COMPLETED | mechanism_probe | probe_gate_compat_sigmoid_update | synthetic_probe_tensors |  |
| probe_responsibility_no_null_precision | COMPLETED | mechanism_probe | probe_responsibility_no_null_precision | synthetic_probe_tensors |  |
| probe_responsibility_with_null_precision | COMPLETED | mechanism_probe | probe_responsibility_with_null_precision | synthetic_probe_tensors |  |
| probe_thresholded_null | COMPLETED | mechanism_probe | all_noise_should_choose_null | synthetic_probe_tensors |  |
| local_conv_baseline | NOT RUN | image_synthetic | LocalConvBaseline | SyntheticShapeEnergyDataset | smoke mode: not implemented |
| local_text_linear_baseline | NOT RUN | text_synthetic | LocalTextCodecLinear | SyntheticTextEnergyDataset | smoke mode: not standardized |
| cifar_medium_suite | NOT RUN | image_cifar | selected_image_models | CIFAR10 | smoke mode: not requested in this mode |
| text_medium_suite | NOT RUN | text_synthetic | selected_text_models | SyntheticTextEnergyDataset | smoke mode: not requested in this mode |
| full_seeded_ablation | NOT RUN | mechanism_ablation | all_supported_cells | mixed | smoke mode: not requested in this mode |
| kan_operator_additive_smooth_spline_kan_seed0 | COMPLETED | kan_operator_replacement | spline_kan | additive_smooth |  |
| kan_operator_additive_smooth_semL_operator_replacement_seed0 | COMPLETED | kan_operator_replacement | semL_operator_replacement | additive_smooth |  |
| kan_operator_additive_smooth_spline_kan_seed0 | COMPLETED | kan_operator_replacement | spline_kan | additive_smooth |  |
| kan_operator_additive_smooth_semL_operator_replacement_seed0 | COMPLETED | kan_operator_replacement | semL_operator_replacement | additive_smooth |  |
| kan_operator_additive_smooth_spline_kan_seed1 | COMPLETED | kan_operator_replacement | spline_kan | additive_smooth |  |
| kan_operator_additive_smooth_semL_operator_replacement_seed1 | COMPLETED | kan_operator_replacement | semL_operator_replacement | additive_smooth |  |
| kan_operator_local_bumps_spline_kan_seed0 | COMPLETED | kan_operator_replacement | spline_kan | local_bumps |  |
| kan_operator_local_bumps_semL_operator_replacement_seed0 | COMPLETED | kan_operator_replacement | semL_operator_replacement | local_bumps |  |
| kan_operator_local_bumps_spline_kan_seed1 | COMPLETED | kan_operator_replacement | spline_kan | local_bumps |  |
| kan_operator_local_bumps_semL_operator_replacement_seed1 | COMPLETED | kan_operator_replacement | semL_operator_replacement | local_bumps |  |
| kan_operator_mixed_composition_spline_kan_seed0 | COMPLETED | kan_operator_replacement | spline_kan | mixed_composition |  |
| kan_operator_mixed_composition_semL_operator_replacement_seed0 | COMPLETED | kan_operator_replacement | semL_operator_replacement | mixed_composition |  |
| kan_operator_mixed_composition_spline_kan_seed1 | COMPLETED | kan_operator_replacement | spline_kan | mixed_composition |  |
| kan_operator_mixed_composition_semL_operator_replacement_seed1 | COMPLETED | kan_operator_replacement | semL_operator_replacement | mixed_composition |  |
| kan_operator_additive_smooth_spline_kan_seed0 | COMPLETED | kan_operator_replacement | spline_kan | additive_smooth |  |
| kan_operator_additive_smooth_semL_operator_replacement_seed0 | COMPLETED | kan_operator_replacement | semL_operator_replacement | additive_smooth |  |
| kan_operator_additive_smooth_spline_kan_seed1 | COMPLETED | kan_operator_replacement | spline_kan | additive_smooth |  |
| kan_operator_additive_smooth_semL_operator_replacement_seed1 | COMPLETED | kan_operator_replacement | semL_operator_replacement | additive_smooth |  |
| kan_operator_additive_smooth_spline_kan_seed2 | COMPLETED | kan_operator_replacement | spline_kan | additive_smooth |  |
| kan_operator_additive_smooth_semL_operator_replacement_seed2 | COMPLETED | kan_operator_replacement | semL_operator_replacement | additive_smooth |  |
| kan_operator_local_bumps_spline_kan_seed0 | COMPLETED | kan_operator_replacement | spline_kan | local_bumps |  |
| kan_operator_local_bumps_semL_operator_replacement_seed0 | COMPLETED | kan_operator_replacement | semL_operator_replacement | local_bumps |  |
| kan_operator_local_bumps_spline_kan_seed1 | COMPLETED | kan_operator_replacement | spline_kan | local_bumps |  |
| kan_operator_local_bumps_semL_operator_replacement_seed1 | COMPLETED | kan_operator_replacement | semL_operator_replacement | local_bumps |  |
| kan_operator_local_bumps_spline_kan_seed2 | COMPLETED | kan_operator_replacement | spline_kan | local_bumps |  |
| kan_operator_local_bumps_semL_operator_replacement_seed2 | COMPLETED | kan_operator_replacement | semL_operator_replacement | local_bumps |  |
| kan_operator_mixed_composition_spline_kan_seed0 | COMPLETED | kan_operator_replacement | spline_kan | mixed_composition |  |
| kan_operator_mixed_composition_semL_operator_replacement_seed0 | COMPLETED | kan_operator_replacement | semL_operator_replacement | mixed_composition |  |
| kan_operator_mixed_composition_spline_kan_seed1 | COMPLETED | kan_operator_replacement | spline_kan | mixed_composition |  |
| kan_operator_mixed_composition_semL_operator_replacement_seed1 | COMPLETED | kan_operator_replacement | semL_operator_replacement | mixed_composition |  |
| kan_operator_mixed_composition_spline_kan_seed2 | COMPLETED | kan_operator_replacement | spline_kan | mixed_composition |  |
| kan_operator_mixed_composition_semL_operator_replacement_seed2 | COMPLETED | kan_operator_replacement | semL_operator_replacement | mixed_composition |  |
| eml_kan_mlp_symbolic_regression_eml_kan_seed0 | COMPLETED | eml_kan_mlp_fair_comparison | eml_kan | symbolic_regression |  |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed0 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_same_width | symbolic_regression |  |
| eml_kan_mlp_symbolic_regression_mlp_param_matched_seed0 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_param_matched | symbolic_regression |  |
| eml_kan_mlp_symbolic_regression_eml_kan_seed1 | COMPLETED | eml_kan_mlp_fair_comparison | eml_kan | symbolic_regression |  |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed1 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_same_width | symbolic_regression |  |
| eml_kan_mlp_symbolic_regression_mlp_param_matched_seed1 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_param_matched | symbolic_regression |  |
| eml_kan_mlp_symbolic_regression_eml_kan_seed2 | COMPLETED | eml_kan_mlp_fair_comparison | eml_kan | symbolic_regression |  |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed2 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_same_width | symbolic_regression |  |
| eml_kan_mlp_symbolic_regression_mlp_param_matched_seed2 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_param_matched | symbolic_regression |  |
| eml_kan_mlp_localized_regression_eml_kan_seed0 | COMPLETED | eml_kan_mlp_fair_comparison | eml_kan | localized_regression |  |
| eml_kan_mlp_localized_regression_mlp_same_width_seed0 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_same_width | localized_regression |  |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed0 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_param_matched | localized_regression |  |
| eml_kan_mlp_localized_regression_eml_kan_seed1 | COMPLETED | eml_kan_mlp_fair_comparison | eml_kan | localized_regression |  |
| eml_kan_mlp_localized_regression_mlp_same_width_seed1 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_same_width | localized_regression |  |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed1 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_param_matched | localized_regression |  |
| eml_kan_mlp_localized_regression_eml_kan_seed2 | COMPLETED | eml_kan_mlp_fair_comparison | eml_kan | localized_regression |  |
| eml_kan_mlp_localized_regression_mlp_same_width_seed2 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_same_width | localized_regression |  |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed2 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_param_matched | localized_regression |  |
| eml_kan_mlp_shift_classification_eml_kan_seed0 | COMPLETED | eml_kan_mlp_fair_comparison | eml_kan | shift_classification |  |
| eml_kan_mlp_shift_classification_mlp_same_width_seed0 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_same_width | shift_classification |  |
| eml_kan_mlp_shift_classification_mlp_param_matched_seed0 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_param_matched | shift_classification |  |
| eml_kan_mlp_shift_classification_eml_kan_seed1 | COMPLETED | eml_kan_mlp_fair_comparison | eml_kan | shift_classification |  |
| eml_kan_mlp_shift_classification_mlp_same_width_seed1 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_same_width | shift_classification |  |
| eml_kan_mlp_shift_classification_mlp_param_matched_seed1 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_param_matched | shift_classification |  |
| eml_kan_mlp_shift_classification_eml_kan_seed2 | COMPLETED | eml_kan_mlp_fair_comparison | eml_kan | shift_classification |  |
| eml_kan_mlp_shift_classification_mlp_same_width_seed2 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_same_width | shift_classification |  |
| eml_kan_mlp_shift_classification_mlp_param_matched_seed2 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_param_matched | shift_classification |  |
| eml_kan_mlp_localized_regression_eml_kan_seed0 | COMPLETED | eml_kan_mlp_fair_comparison | eml_kan | localized_regression |  |
| eml_kan_mlp_localized_regression_mlp_same_width_seed0 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_same_width | localized_regression |  |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed0 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_param_matched | localized_regression |  |
| eml_kan_mlp_localized_regression_eml_kan_seed1 | COMPLETED | eml_kan_mlp_fair_comparison | eml_kan | localized_regression |  |
| eml_kan_mlp_localized_regression_mlp_same_width_seed1 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_same_width | localized_regression |  |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed1 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_param_matched | localized_regression |  |
| eml_kan_mlp_localized_regression_eml_kan_seed2 | COMPLETED | eml_kan_mlp_fair_comparison | eml_kan | localized_regression |  |
| eml_kan_mlp_localized_regression_mlp_same_width_seed2 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_same_width | localized_regression |  |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed2 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_param_matched | localized_regression |  |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed1 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_same_width | symbolic_regression |  |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed2 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_same_width | symbolic_regression |  |
| eml_kan_mlp_localized_regression_mlp_same_width_seed1 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_same_width | localized_regression |  |
| eml_kan_mlp_localized_regression_mlp_same_width_seed2 | COMPLETED | eml_kan_mlp_fair_comparison | mlp_same_width | localized_regression |  |
| smoke_image_cnn_eml_baseline | COMPLETED | image_synthetic | cnn_eml | SyntheticShapeEnergyDataset |  |
| smoke_image_efficient_eml | COMPLETED | image_synthetic | EfficientEMLImageClassifier | SyntheticShapeEnergyDataset |  |
| smoke_text_efficient_eml | COMPLETED | text_synthetic | EfficientEMLTextEncoder | SyntheticTextEnergyDataset |  |
| probe_gate_compat_sigmoid_update | COMPLETED | mechanism_probe | probe_gate_compat_sigmoid_update | synthetic_probe_tensors |  |
| probe_responsibility_no_null_precision | COMPLETED | mechanism_probe | probe_responsibility_no_null_precision | synthetic_probe_tensors |  |
| probe_responsibility_with_null_precision | COMPLETED | mechanism_probe | probe_responsibility_with_null_precision | synthetic_probe_tensors |  |
| probe_thresholded_null | COMPLETED | mechanism_probe | all_noise_should_choose_null | synthetic_probe_tensors |  |
| local_conv_baseline | NOT RUN | image_synthetic | LocalConvBaseline | SyntheticShapeEnergyDataset | smoke mode: not implemented |
| local_text_linear_baseline | NOT RUN | text_synthetic | LocalTextCodecLinear | SyntheticTextEnergyDataset | smoke mode: not standardized |
| cifar_medium_suite | NOT RUN | image_cifar | selected_image_models | CIFAR10 | smoke mode: not requested in this mode |
| text_medium_suite | NOT RUN | text_synthetic | selected_text_models | SyntheticTextEnergyDataset | smoke mode: not requested in this mode |
| full_seeded_ablation | NOT RUN | mechanism_ablation | all_supported_cells | mixed | smoke mode: not requested in this mode |

Failed runs: 0
Not-run entries: 19

## 4. Datasets

| dataset | synthetic/real | notes |
| --- | --- | --- |
| CIFAR10 | real/optional | requires local data/dependency |
| KAN paper AI+Science tasks | real/optional | requires local data/dependency |
| SyntheticShapeEnergyDataset | synthetic | offline |
| SyntheticTextEnergyDataset | synthetic | offline |
| additive_smooth | real/optional | requires local data/dependency |
| local_bumps | real/optional | requires local data/dependency |
| localized_regression | real/optional | requires local data/dependency |
| mixed | real/optional | requires local data/dependency |
| mixed_composition | real/optional | requires local data/dependency |
| shift_classification | real/optional | requires local data/dependency |
| symbolic_regression | real/optional | requires local data/dependency |
| synthetic_probe_tensors | synthetic | offline |

## 5. Models Compared

| model | parameter count | task names | key mechanisms |
| --- | ---: | --- | --- |
| EMLEdgeImageClassifier_kan_style | 10889 | image_synthetic | see config artifacts |
| EMLEdgeTextLM_kan_style | 53736 | text_synthetic | see config artifacts |
| EfficientEMLImageClassifier | 117076 | image_synthetic | see config artifacts |
| EfficientEMLTextEncoder | 92950 | text_synthetic | see config artifacts |
| KAN_arxiv_2404_19756 | 0 | paper_reference | see config artifacts |
| LocalCausalConvLM | 11668 | text_synthetic | see config artifacts |
| LocalConvBaseline | 0 | image_synthetic | see config artifacts |
| LocalTextCodecLinear | 0 | text_synthetic | see config artifacts |
| SmallGRULM | 11796 | text_synthetic | see config artifacts |
| all_noise_should_choose_null | 0 | mechanism_probe | see config artifacts |
| all_supported_cells | 0 | mechanism_ablation | see config artifacts |
| cnn_eml | 162644 | image_synthetic | see config artifacts |
| eml_kan | 37767 | eml_kan_mlp_fair_comparison | see config artifacts |
| mlp_param_matched | 37801 | eml_kan_mlp_fair_comparison | see config artifacts |
| mlp_same_width | 4801 | eml_kan_mlp_fair_comparison | see config artifacts |
| probe_gate_compat_sigmoid_update | 14151 | mechanism_probe | see config artifacts |
| probe_responsibility_no_null_precision | 14151 | mechanism_probe | see config artifacts |
| probe_responsibility_with_null_precision | 14151 | mechanism_probe | see config artifacts |
| selected_image_models | 0 | image_cifar | see config artifacts |
| selected_text_models | 0 | text_synthetic | see config artifacts |
| semL_operator_replacement | 9671 | kan_operator_replacement | see config artifacts |
| spline_kan | 21505 | kan_operator_replacement | see config artifacts |

## 6. Main Results

### Image
| run_id | model | dataset | final metric | best metric | loss | accuracy | time sec | params |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_image_cnn_eml_baseline | cnn_eml | SyntheticShapeEnergyDataset | 0.0 | 0.25 | 1.6346 | 0.0000 | 0.05253291130065918 | 162644 |
| smoke_image_efficient_eml | EfficientEMLImageClassifier | SyntheticShapeEnergyDataset | 0.0 | 0.25 | 1.7484 | 0.0000 | 0.07413816452026367 | 117076 |
| kan_compare_image_cnn_eml | cnn_eml | SyntheticShapeEnergyDataset | 0.5 | 0.5 | 1.5468 | 0.5000 | 0.052607059478759766 | 162644 |
| kan_compare_image_efficient_eml | EfficientEMLImageClassifier | SyntheticShapeEnergyDataset | 0.0 | 0.0 | 1.6426 | 0.0000 | 0.05013298988342285 | 115573 |
| kan_compare_image_eml_edge | EMLEdgeImageClassifier_kan_style | SyntheticShapeEnergyDataset | 0.25 | 0.25 | 1.6112 | 0.2500 | 0.011314868927001953 | 10889 |
| kan_compare_image_cnn_eml | cnn_eml | SyntheticShapeEnergyDataset | 0.25 | 0.625 | 1.6268 | 0.2500 | 0.7324850559234619 | 162644 |
| kan_compare_image_efficient_eml | EfficientEMLImageClassifier | SyntheticShapeEnergyDataset | 0.25 | 0.75 | 1.5906 | 0.2500 | 1.3546850681304932 | 115573 |
| kan_compare_image_eml_edge | EMLEdgeImageClassifier_kan_style | SyntheticShapeEnergyDataset | 0.125 | 0.5 | 1.6313 | 0.1250 | 0.35624098777770996 | 10889 |
| kan_compare_image_cnn_eml | cnn_eml | SyntheticShapeEnergyDataset | 0.25 | 0.75 | 1.6007 | 0.2500 | 0.32799720764160156 | 162644 |
| kan_compare_image_efficient_eml | EfficientEMLImageClassifier | SyntheticShapeEnergyDataset | 0.375 | 0.375 | 1.5922 | 0.3750 | 0.3134782314300537 | 115573 |
| kan_compare_image_eml_edge | EMLEdgeImageClassifier_kan_style | SyntheticShapeEnergyDataset | 0.5 | 0.5 | 1.6206 | 0.5000 | 0.1622910499572754 | 10889 |
| kan_compare_image_cnn_eml | cnn_eml | SyntheticShapeEnergyDataset | 0.25 | 0.625 | 1.5949 | 0.2500 | 0.24678707122802734 | 162644 |
| kan_compare_image_efficient_eml | EfficientEMLImageClassifier | SyntheticShapeEnergyDataset | 0.25 | 0.375 | 1.6057 | 0.2500 | 0.5140008926391602 | 115573 |
| kan_compare_image_eml_edge | EMLEdgeImageClassifier_kan_style | SyntheticShapeEnergyDataset | 0.125 | 0.375 | 1.6120 | 0.1250 | 0.07409095764160156 | 10889 |
| smoke_image_cnn_eml_baseline | cnn_eml | SyntheticShapeEnergyDataset | 0.375 | 0.5 | 1.5745 | 0.3750 | 0.8756721019744873 | 162644 |
| smoke_image_efficient_eml | EfficientEMLImageClassifier | SyntheticShapeEnergyDataset | 0.0 | 0.375 | 1.7369 | 0.0000 | 0.2402501106262207 | 115573 |
| smoke_image_cnn_eml_baseline | cnn_eml | SyntheticShapeEnergyDataset | 0.25 | 0.5 | 1.6227 | 0.2500 | 0.07447671890258789 | 162644 |
| smoke_image_efficient_eml | EfficientEMLImageClassifier | SyntheticShapeEnergyDataset | 0.0 | 0.375 | 1.7369 | 0.0000 | 0.09504485130310059 | 115573 |

### Text
| run_id | model | dataset | final metric | best metric | loss | accuracy | time sec | params |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_text_efficient_eml | EfficientEMLTextEncoder | SyntheticTextEnergyDataset | 0.11274509876966476 | 0.11274509876966476 | 4.3839 | 0.1127 | 0.047122955322265625 | 92950 |
| kan_compare_text_local_conv | LocalCausalConvLM | SyntheticTextEnergyDataset | 0.0 | 0.0 | 4.4488 | 0.0000 | 0.006283998489379883 | 11668 |
| kan_compare_text_small_gru | SmallGRULM | SyntheticTextEnergyDataset | 0.05000000074505806 | 0.05000000074505806 | 4.4340 | 0.0500 | 0.013735055923461914 | 11796 |
| kan_compare_text_efficient_eml | EfficientEMLTextEncoder | SyntheticTextEnergyDataset | 0.07446808367967606 | 0.07446808367967606 | 4.3975 | 0.0745 | 0.01978278160095215 | 92950 |
| kan_compare_text_eml_edge | EMLEdgeTextLM_kan_style | SyntheticTextEnergyDataset | 0.0 | 0.04854368790984154 | 4.4340 | 0.0000 | 0.07332515716552734 | 53736 |
| kan_compare_text_local_conv | LocalCausalConvLM | SyntheticTextEnergyDataset | 0.17766498029232025 | 0.22404371201992035 | 3.6288 | 0.1777 | 0.15865302085876465 | 11668 |
| kan_compare_text_small_gru | SmallGRULM | SyntheticTextEnergyDataset | 0.08888889104127884 | 0.12921348214149475 | 3.9181 | 0.0889 | 0.2814500331878662 | 11796 |
| kan_compare_text_efficient_eml | EfficientEMLTextEncoder | SyntheticTextEnergyDataset | 0.3787234127521515 | 0.40963855385780334 | 3.8218 | 0.3787 | 0.7171981334686279 | 92950 |
| kan_compare_text_eml_edge | EMLEdgeTextLM_kan_style | SyntheticTextEnergyDataset | 0.4296296238899231 | 0.4296296238899231 | 3.9839 | 0.4296 | 2.393962860107422 | 53736 |
| kan_compare_text_local_conv | LocalCausalConvLM | SyntheticTextEnergyDataset | 0.3365853726863861 | 0.4334975481033325 | 2.5168 | 0.3366 | 0.3350667953491211 | 11668 |
| kan_compare_text_small_gru | SmallGRULM | SyntheticTextEnergyDataset | 0.3368421196937561 | 0.4188481569290161 | 2.3333 | 0.3368 | 1.5402390956878662 | 11796 |
| kan_compare_text_efficient_eml | EfficientEMLTextEncoder | SyntheticTextEnergyDataset | 0.36138615012168884 | 0.4950000047683716 | 2.6294 | 0.3614 | 3.2707719802856445 | 92950 |
| kan_compare_text_eml_edge | EMLEdgeTextLM_kan_style | SyntheticTextEnergyDataset | 0.322429895401001 | 0.5025380849838257 | 3.0322 | 0.3224 | 10.280304908752441 | 53736 |
| kan_compare_text_local_conv | LocalCausalConvLM | SyntheticTextEnergyDataset | 0.39086294174194336 | 0.45049506425857544 | 2.3376 | 0.3909 | 0.3432021141052246 | 11668 |
| kan_compare_text_small_gru | SmallGRULM | SyntheticTextEnergyDataset | 0.10526315867900848 | 0.22267206013202667 | 3.7472 | 0.1053 | 0.43062829971313477 | 11796 |
| kan_compare_text_efficient_eml | EfficientEMLTextEncoder | SyntheticTextEnergyDataset | 0.34296029806137085 | 0.42944785952568054 | 3.1884 | 0.3430 | 2.843071937561035 | 92950 |
| kan_compare_text_eml_edge | EMLEdgeTextLM_kan_style | SyntheticTextEnergyDataset | 0.29949238896369934 | 0.460829496383667 | 3.9456 | 0.2995 | 3.3603527545928955 | 53736 |
| smoke_text_efficient_eml | EfficientEMLTextEncoder | SyntheticTextEnergyDataset | 0.03431372717022896 | 0.04128440469503403 | 4.3897 | 0.0343 | 0.11359572410583496 | 92950 |
| smoke_text_efficient_eml | EfficientEMLTextEncoder | SyntheticTextEnergyDataset | 0.03431372717022896 | 0.04128440469503403 | 4.3897 | 0.0343 | 0.04859304428100586 | 92950 |

### Efficiency
| run_id | model | task | examples/sec | tokens/sec | step time | peak memory MB | params |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| smoke_image_cnn_eml_baseline | cnn_eml | image_synthetic | 984.4917407505208 |  | 0.008126020431518555 | 0.0 | 162644 |
| smoke_image_efficient_eml | EfficientEMLImageClassifier | image_synthetic | 491.2225800784681 |  | 0.01628589630126953 | 0.0 | 117076 |
| smoke_text_efficient_eml | EfficientEMLTextEncoder | text_synthetic |  | 23118.478722541946 | 0.00882411003112793 | 0.0 | 92950 |
| probe_gate_compat_sigmoid_update | probe_gate_compat_sigmoid_update | mechanism_probe |  |  | 0.0009789466857910156 | 0.0 | 14151 |
| probe_responsibility_no_null_precision | probe_responsibility_no_null_precision | mechanism_probe |  |  | 0.000885009765625 | 0.0 | 14151 |
| probe_responsibility_with_null_precision | probe_responsibility_with_null_precision | mechanism_probe |  |  | 0.0009799003601074219 | 0.0 | 14151 |
| kan_compare_image_cnn_eml | cnn_eml | image_synthetic | 507.61598741339145 |  | 0.007879972457885742 | 0.0 | 162644 |
| kan_compare_image_efficient_eml | EfficientEMLImageClassifier | image_synthetic | 205.0528116941053 |  | 0.019507169723510742 | 0.0 | 115573 |
| kan_compare_image_eml_edge | EMLEdgeImageClassifier_kan_style | image_synthetic | 1085.7633963241005 |  | 0.0036840438842773438 | 0.0 | 10889 |
| kan_compare_text_local_conv | LocalCausalConvLM | text_synthetic |  | 78587.74493804775 | 0.0007889270782470703 | 0.0 | 11668 |
| kan_compare_text_small_gru | SmallGRULM | text_synthetic |  | 19293.02667893284 | 0.0031099319458007812 | 0.0 | 11796 |
| kan_compare_text_efficient_eml | EfficientEMLTextEncoder | text_synthetic |  | 11344.763790176388 | 0.008285760879516602 | 0.0 | 92950 |
| kan_compare_text_eml_edge | EMLEdgeTextLM_kan_style | text_synthetic |  | 2451.3537620486263 | 0.026515960693359375 | 0.0 | 53736 |
| kan_compare_image_cnn_eml | cnn_eml | image_synthetic | 872.2233428645698 |  | 0.00917196273803711 | 0.0 | 162644 |
| kan_compare_image_efficient_eml | EfficientEMLImageClassifier | image_synthetic | 376.86365065816074 |  | 0.02122783660888672 | 0.0 | 115573 |
| kan_compare_image_eml_edge | EMLEdgeImageClassifier_kan_style | image_synthetic | 1656.6817418781475 |  | 0.004828929901123047 | 0.0 | 10889 |
| kan_compare_text_local_conv | LocalCausalConvLM | text_synthetic |  | 244316.34772324067 | 0.0008063316345214844 | 0.0 | 11668 |
| kan_compare_text_small_gru | SmallGRULM | text_synthetic |  | 43492.66763960366 | 0.003103971481323242 | 0.0 | 11796 |
| kan_compare_text_efficient_eml | EfficientEMLTextEncoder | text_synthetic |  | 26732.701581188467 | 0.008790731430053711 | 0.0 | 92950 |
| kan_compare_text_eml_edge | EMLEdgeTextLM_kan_style | text_synthetic |  | 4510.8306579461 | 0.029927968978881836 | 0.0 | 53736 |
| kan_compare_image_cnn_eml | cnn_eml | image_synthetic | 843.6060842237587 |  | 0.009483098983764648 | 0.0 | 162644 |
| kan_compare_image_efficient_eml | EfficientEMLImageClassifier | image_synthetic | 397.9321173596452 |  | 0.020103931427001953 | 0.0 | 115573 |
| kan_compare_image_eml_edge | EMLEdgeImageClassifier_kan_style | image_synthetic | 2041.5205646142613 |  | 0.003918647766113281 | 0.0 | 10889 |
| kan_compare_text_local_conv | LocalCausalConvLM | text_synthetic |  | 232198.84418039428 | 0.0008828639984130859 | 0.0 | 11668 |
| kan_compare_text_small_gru | SmallGRULM | text_synthetic |  | 61211.90260388663 | 0.003103971481323242 | 0.0 | 11796 |
| kan_compare_text_efficient_eml | EfficientEMLTextEncoder | text_synthetic |  | 24165.699030233885 | 0.008358955383300781 | 0.0 | 92950 |
| kan_compare_text_eml_edge | EMLEdgeTextLM_kan_style | text_synthetic |  | 4891.316618075802 | 0.04375100135803223 | 0.0 | 53736 |
| kan_compare_image_cnn_eml | cnn_eml | image_synthetic | 847.1843865983286 |  | 0.009443044662475586 | 0.0 | 162644 |
| kan_compare_image_efficient_eml | EfficientEMLImageClassifier | image_synthetic | 389.96376314719043 |  | 0.020514726638793945 | 0.0 | 115573 |
| kan_compare_image_eml_edge | EMLEdgeImageClassifier_kan_style | image_synthetic | 2285.101607191501 |  | 0.0035009384155273438 | 0.0 | 10889 |
| kan_compare_text_local_conv | LocalCausalConvLM | text_synthetic |  | 273240.0423280423 | 0.000720977783203125 | 0.0 | 11668 |
| kan_compare_text_small_gru | SmallGRULM | text_synthetic |  | 59283.44876325088 | 0.0038459300994873047 | 0.0 | 11796 |
| kan_compare_text_efficient_eml | EfficientEMLTextEncoder | text_synthetic |  | 29880.721362069853 | 0.009270191192626953 | 0.0 | 92950 |
| kan_compare_text_eml_edge | EMLEdgeTextLM_kan_style | text_synthetic |  | 5042.369043187463 | 0.03906893730163574 | 0.0 | 53736 |
| smoke_image_cnn_eml_baseline | cnn_eml | image_synthetic | 299.5262843115376 |  | 0.02670884132385254 | 0.0 | 162644 |
| smoke_image_efficient_eml | EfficientEMLImageClassifier | image_synthetic | 125.72712387075984 |  | 0.0636298656463623 | 0.0 | 115573 |
| smoke_text_efficient_eml | EfficientEMLTextEncoder | text_synthetic |  | 12240.361872880992 | 0.016666173934936523 | 0.0 | 92950 |
| probe_gate_compat_sigmoid_update | probe_gate_compat_sigmoid_update | mechanism_probe |  |  | 0.0013370513916015625 | 0.0 | 14151 |
| probe_responsibility_no_null_precision | probe_responsibility_no_null_precision | mechanism_probe |  |  | 0.0033092498779296875 | 0.0 | 14151 |
| probe_responsibility_with_null_precision | probe_responsibility_with_null_precision | mechanism_probe |  |  | 0.001207113265991211 | 0.0 | 14151 |
| probe_thresholded_null | all_noise_should_choose_null | mechanism_probe |  |  | 0.0010590553283691406 | 0.0 | 0 |
| kan_operator_additive_smooth_spline_kan_seed0 | spline_kan | kan_operator_replacement |  |  |  |  | 21505 |
| kan_operator_additive_smooth_semL_operator_replacement_seed0 | semL_operator_replacement | kan_operator_replacement |  |  |  |  | 9671 |
| kan_operator_additive_smooth_spline_kan_seed0 | spline_kan | kan_operator_replacement |  |  |  |  | 21505 |
| kan_operator_additive_smooth_semL_operator_replacement_seed0 | semL_operator_replacement | kan_operator_replacement |  |  |  |  | 9671 |
| kan_operator_additive_smooth_spline_kan_seed1 | spline_kan | kan_operator_replacement |  |  |  |  | 21505 |
| kan_operator_additive_smooth_semL_operator_replacement_seed1 | semL_operator_replacement | kan_operator_replacement |  |  |  |  | 9671 |
| kan_operator_local_bumps_spline_kan_seed0 | spline_kan | kan_operator_replacement |  |  |  |  | 21505 |
| kan_operator_local_bumps_semL_operator_replacement_seed0 | semL_operator_replacement | kan_operator_replacement |  |  |  |  | 9671 |
| kan_operator_local_bumps_spline_kan_seed1 | spline_kan | kan_operator_replacement |  |  |  |  | 21505 |
| kan_operator_local_bumps_semL_operator_replacement_seed1 | semL_operator_replacement | kan_operator_replacement |  |  |  |  | 9671 |
| kan_operator_mixed_composition_spline_kan_seed0 | spline_kan | kan_operator_replacement |  |  |  |  | 21505 |
| kan_operator_mixed_composition_semL_operator_replacement_seed0 | semL_operator_replacement | kan_operator_replacement |  |  |  |  | 9671 |
| kan_operator_mixed_composition_spline_kan_seed1 | spline_kan | kan_operator_replacement |  |  |  |  | 21505 |
| kan_operator_mixed_composition_semL_operator_replacement_seed1 | semL_operator_replacement | kan_operator_replacement |  |  |  |  | 9671 |
| kan_operator_additive_smooth_spline_kan_seed0 | spline_kan | kan_operator_replacement |  |  |  |  | 150529 |
| kan_operator_additive_smooth_semL_operator_replacement_seed0 | semL_operator_replacement | kan_operator_replacement |  |  |  |  | 35719 |
| kan_operator_additive_smooth_spline_kan_seed1 | spline_kan | kan_operator_replacement |  |  |  |  | 150529 |
| kan_operator_additive_smooth_semL_operator_replacement_seed1 | semL_operator_replacement | kan_operator_replacement |  |  |  |  | 35719 |
| kan_operator_additive_smooth_spline_kan_seed2 | spline_kan | kan_operator_replacement |  |  |  |  | 150529 |
| kan_operator_additive_smooth_semL_operator_replacement_seed2 | semL_operator_replacement | kan_operator_replacement |  |  |  |  | 35719 |
| kan_operator_local_bumps_spline_kan_seed0 | spline_kan | kan_operator_replacement |  |  |  |  | 150529 |
| kan_operator_local_bumps_semL_operator_replacement_seed0 | semL_operator_replacement | kan_operator_replacement |  |  |  |  | 35719 |
| kan_operator_local_bumps_spline_kan_seed1 | spline_kan | kan_operator_replacement |  |  |  |  | 150529 |
| kan_operator_local_bumps_semL_operator_replacement_seed1 | semL_operator_replacement | kan_operator_replacement |  |  |  |  | 35719 |
| kan_operator_local_bumps_spline_kan_seed2 | spline_kan | kan_operator_replacement |  |  |  |  | 150529 |
| kan_operator_local_bumps_semL_operator_replacement_seed2 | semL_operator_replacement | kan_operator_replacement |  |  |  |  | 35719 |
| kan_operator_mixed_composition_spline_kan_seed0 | spline_kan | kan_operator_replacement |  |  |  |  | 150529 |
| kan_operator_mixed_composition_semL_operator_replacement_seed0 | semL_operator_replacement | kan_operator_replacement |  |  |  |  | 35719 |
| kan_operator_mixed_composition_spline_kan_seed1 | spline_kan | kan_operator_replacement |  |  |  |  | 150529 |
| kan_operator_mixed_composition_semL_operator_replacement_seed1 | semL_operator_replacement | kan_operator_replacement |  |  |  |  | 35719 |
| kan_operator_mixed_composition_spline_kan_seed2 | spline_kan | kan_operator_replacement |  |  |  |  | 150529 |
| kan_operator_mixed_composition_semL_operator_replacement_seed2 | semL_operator_replacement | kan_operator_replacement |  |  |  |  | 35719 |
| eml_kan_mlp_symbolic_regression_eml_kan_seed0 | eml_kan | eml_kan_mlp_fair_comparison |  |  |  |  | 37767 |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed0 | mlp_same_width | eml_kan_mlp_fair_comparison |  |  |  |  | 4801 |
| eml_kan_mlp_symbolic_regression_mlp_param_matched_seed0 | mlp_param_matched | eml_kan_mlp_fair_comparison |  |  |  |  | 37801 |
| eml_kan_mlp_symbolic_regression_eml_kan_seed1 | eml_kan | eml_kan_mlp_fair_comparison |  |  |  |  | 37767 |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed1 | mlp_same_width | eml_kan_mlp_fair_comparison |  |  |  |  | 4801 |
| eml_kan_mlp_symbolic_regression_mlp_param_matched_seed1 | mlp_param_matched | eml_kan_mlp_fair_comparison |  |  |  |  | 37801 |
| eml_kan_mlp_symbolic_regression_eml_kan_seed2 | eml_kan | eml_kan_mlp_fair_comparison |  |  |  |  | 37767 |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed2 | mlp_same_width | eml_kan_mlp_fair_comparison |  |  |  |  | 4801 |
| eml_kan_mlp_symbolic_regression_mlp_param_matched_seed2 | mlp_param_matched | eml_kan_mlp_fair_comparison |  |  |  |  | 37801 |
| eml_kan_mlp_localized_regression_eml_kan_seed0 | eml_kan | eml_kan_mlp_fair_comparison |  |  |  |  | 37767 |
| eml_kan_mlp_localized_regression_mlp_same_width_seed0 | mlp_same_width | eml_kan_mlp_fair_comparison |  |  |  |  | 4801 |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed0 | mlp_param_matched | eml_kan_mlp_fair_comparison |  |  |  |  | 37801 |
| eml_kan_mlp_localized_regression_eml_kan_seed1 | eml_kan | eml_kan_mlp_fair_comparison |  |  |  |  | 37767 |
| eml_kan_mlp_localized_regression_mlp_same_width_seed1 | mlp_same_width | eml_kan_mlp_fair_comparison |  |  |  |  | 4801 |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed1 | mlp_param_matched | eml_kan_mlp_fair_comparison |  |  |  |  | 37801 |
| eml_kan_mlp_localized_regression_eml_kan_seed2 | eml_kan | eml_kan_mlp_fair_comparison |  |  |  |  | 37767 |
| eml_kan_mlp_localized_regression_mlp_same_width_seed2 | mlp_same_width | eml_kan_mlp_fair_comparison |  |  |  |  | 4801 |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed2 | mlp_param_matched | eml_kan_mlp_fair_comparison |  |  |  |  | 37801 |
| eml_kan_mlp_shift_classification_eml_kan_seed0 | eml_kan | eml_kan_mlp_fair_comparison |  |  |  |  | 38280 |
| eml_kan_mlp_shift_classification_mlp_same_width_seed0 | mlp_same_width | eml_kan_mlp_fair_comparison |  |  |  |  | 4866 |
| eml_kan_mlp_shift_classification_mlp_param_matched_seed0 | mlp_param_matched | eml_kan_mlp_fair_comparison |  |  |  |  | 38382 |
| eml_kan_mlp_shift_classification_eml_kan_seed1 | eml_kan | eml_kan_mlp_fair_comparison |  |  |  |  | 38280 |
| eml_kan_mlp_shift_classification_mlp_same_width_seed1 | mlp_same_width | eml_kan_mlp_fair_comparison |  |  |  |  | 4866 |
| eml_kan_mlp_shift_classification_mlp_param_matched_seed1 | mlp_param_matched | eml_kan_mlp_fair_comparison |  |  |  |  | 38382 |
| eml_kan_mlp_shift_classification_eml_kan_seed2 | eml_kan | eml_kan_mlp_fair_comparison |  |  |  |  | 38280 |
| eml_kan_mlp_shift_classification_mlp_same_width_seed2 | mlp_same_width | eml_kan_mlp_fair_comparison |  |  |  |  | 4866 |
| eml_kan_mlp_shift_classification_mlp_param_matched_seed2 | mlp_param_matched | eml_kan_mlp_fair_comparison |  |  |  |  | 38382 |
| eml_kan_mlp_localized_regression_eml_kan_seed0 | eml_kan | eml_kan_mlp_fair_comparison |  |  |  |  | 37767 |
| eml_kan_mlp_localized_regression_mlp_same_width_seed0 | mlp_same_width | eml_kan_mlp_fair_comparison |  |  |  |  | 4801 |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed0 | mlp_param_matched | eml_kan_mlp_fair_comparison |  |  |  |  | 37801 |
| eml_kan_mlp_localized_regression_eml_kan_seed1 | eml_kan | eml_kan_mlp_fair_comparison |  |  |  |  | 37767 |
| eml_kan_mlp_localized_regression_mlp_same_width_seed1 | mlp_same_width | eml_kan_mlp_fair_comparison |  |  |  |  | 4801 |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed1 | mlp_param_matched | eml_kan_mlp_fair_comparison |  |  |  |  | 37801 |
| eml_kan_mlp_localized_regression_eml_kan_seed2 | eml_kan | eml_kan_mlp_fair_comparison |  |  |  |  | 37767 |
| eml_kan_mlp_localized_regression_mlp_same_width_seed2 | mlp_same_width | eml_kan_mlp_fair_comparison |  |  |  |  | 4801 |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed2 | mlp_param_matched | eml_kan_mlp_fair_comparison |  |  |  |  | 37801 |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed1 | mlp_same_width | eml_kan_mlp_fair_comparison |  |  |  |  | 4801 |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed2 | mlp_same_width | eml_kan_mlp_fair_comparison |  |  |  |  | 4801 |
| eml_kan_mlp_localized_regression_mlp_same_width_seed1 | mlp_same_width | eml_kan_mlp_fair_comparison |  |  |  |  | 4801 |
| eml_kan_mlp_localized_regression_mlp_same_width_seed2 | mlp_same_width | eml_kan_mlp_fair_comparison |  |  |  |  | 4801 |
| smoke_image_cnn_eml_baseline | cnn_eml | image_synthetic | 841.4060533112665 |  | 0.009507894515991211 | 0.0 | 162644 |
| smoke_image_efficient_eml | EfficientEMLImageClassifier | image_synthetic | 393.31432858214555 |  | 0.0203399658203125 | 0.0 | 115573 |
| smoke_text_efficient_eml | EfficientEMLTextEncoder | text_synthetic |  | 22789.059180738295 | 0.008951663970947266 | 0.0 | 92950 |
| probe_gate_compat_sigmoid_update | probe_gate_compat_sigmoid_update | mechanism_probe |  |  | 0.0009140968322753906 | 0.0 | 14151 |
| probe_responsibility_no_null_precision | probe_responsibility_no_null_precision | mechanism_probe |  |  | 0.0008800029754638672 | 0.0 | 14151 |
| probe_responsibility_with_null_precision | probe_responsibility_with_null_precision | mechanism_probe |  |  | 0.0009679794311523438 | 0.0 | 14151 |
| probe_thresholded_null | all_noise_should_choose_null | mechanism_probe |  |  | 0.0005881786346435547 | 0.0 | 0 |

### Stability
NaN/Inf counts are recorded when runners emit `nan_inf_count`; otherwise MISSING.

## 7. Ablation Results

### Responsibility / Null / Update Probes
| run_id | status | model | best | final | delta vs ref | loss | time sec | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| probe_gate_compat_sigmoid_update | COMPLETED | probe_gate_compat_sigmoid_update | 1.0 | 1.0 | 0.0000 | 0.0000 | 0.0009777545928955078 |  |
| probe_responsibility_no_null_precision | COMPLETED | probe_responsibility_no_null_precision | 1.0 | 1.0 | 0.0000 | 0.0000 | 0.000885009765625 |  |
| probe_responsibility_with_null_precision | COMPLETED | probe_responsibility_with_null_precision | 1.0 | 1.0 | 0.0000 | 0.0000 | 0.0009799003601074219 |  |
| probe_gate_compat_sigmoid_update | COMPLETED | probe_gate_compat_sigmoid_update | 1.0 | 1.0 | 0.0000 | 0.0000 | 0.0013370513916015625 |  |
| probe_responsibility_no_null_precision | COMPLETED | probe_responsibility_no_null_precision | 1.0 | 1.0 | 0.0000 | 0.0000 | 0.0033080577850341797 |  |
| probe_responsibility_with_null_precision | COMPLETED | probe_responsibility_with_null_precision | 1.0 | 1.0 | 0.0000 | 0.0000 | 0.0012061595916748047 |  |
| probe_thresholded_null | COMPLETED | all_noise_should_choose_null | 1.0 | 1.0 | 0.0000 | 0.0000 | 0.0010590553283691406 |  |
| probe_gate_compat_sigmoid_update | COMPLETED | probe_gate_compat_sigmoid_update | 1.0 | 1.0 | 0.0000 | 0.0000 | 0.0009140968322753906 |  |
| probe_responsibility_no_null_precision | COMPLETED | probe_responsibility_no_null_precision | 1.0 | 1.0 | 0.0000 | 0.0000 | 0.0008790493011474609 |  |
| probe_responsibility_with_null_precision | COMPLETED | probe_responsibility_with_null_precision | 1.0 | 1.0 | 0.0000 | 0.0000 | 0.0009679794311523438 |  |
| probe_thresholded_null | COMPLETED | all_noise_should_choose_null | 1.0 | 1.0 | 0.0000 | 0.0000 | 0.0005881786346435547 |  |

Interpretation: these probes validate finite propagation and diagnostic behavior. They do not by themselves prove downstream task quality.

### Image Representation / Attractor / Warmup / Window
| run_id | status | model | best | final | delta vs ref | loss | time sec | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| smoke_image_cnn_eml_baseline | COMPLETED | cnn_eml | 0.25 | 0.0 | 0.0000 | 1.6346 | 0.05253291130065918 |  |
| smoke_image_efficient_eml | COMPLETED | EfficientEMLImageClassifier | 0.25 | 0.0 | 0.0000 | 1.7484 | 0.07413816452026367 |  |
| local_conv_baseline | NOT RUN | LocalConvBaseline |  |  |  |  |  | smoke mode: not implemented |
| kan_compare_image_cnn_eml | COMPLETED | cnn_eml | 0.5 | 0.5 | 0.2500 | 1.5468 | 0.052607059478759766 | early_stop=False |
| kan_compare_image_efficient_eml | COMPLETED | EfficientEMLImageClassifier | 0.0 | 0.0 | -0.2500 | 1.6426 | 0.05013298988342285 | early_stop=False |
| kan_compare_image_eml_edge | COMPLETED | EMLEdgeImageClassifier_kan_style | 0.25 | 0.25 | 0.0000 | 1.6112 | 0.011314868927001953 | early_stop=False |
| kan_compare_image_cnn_eml | COMPLETED | cnn_eml | 0.625 | 0.25 | 0.3750 | 1.6268 | 0.7324850559234619 | early_stop=False |
| kan_compare_image_efficient_eml | COMPLETED | EfficientEMLImageClassifier | 0.75 | 0.25 | 0.5000 | 1.5906 | 1.3546850681304932 | early_stop=False |
| kan_compare_image_eml_edge | COMPLETED | EMLEdgeImageClassifier_kan_style | 0.5 | 0.125 | 0.2500 | 1.6313 | 0.35624098777770996 | early_stop=False |
| kan_compare_image_cnn_eml | COMPLETED | cnn_eml | 0.75 | 0.25 | 0.5000 | 1.6007 | 0.32799720764160156 | early_stop=True |
| kan_compare_image_efficient_eml | COMPLETED | EfficientEMLImageClassifier | 0.375 | 0.375 | 0.1250 | 1.5922 | 0.3134782314300537 | early_stop=True |
| kan_compare_image_eml_edge | COMPLETED | EMLEdgeImageClassifier_kan_style | 0.5 | 0.5 | 0.2500 | 1.6206 | 0.1622910499572754 | early_stop=True |
| kan_compare_image_cnn_eml | COMPLETED | cnn_eml | 0.625 | 0.25 | 0.3750 | 1.5949 | 0.24678707122802734 | early_stop=True |
| kan_compare_image_efficient_eml | COMPLETED | EfficientEMLImageClassifier | 0.375 | 0.25 | 0.1250 | 1.6057 | 0.5140008926391602 | early_stop=True |
| kan_compare_image_eml_edge | COMPLETED | EMLEdgeImageClassifier_kan_style | 0.375 | 0.125 | 0.1250 | 1.6120 | 0.07409095764160156 | early_stop=True |
| smoke_image_cnn_eml_baseline | COMPLETED | cnn_eml | 0.5 | 0.375 | 0.2500 | 1.5745 | 0.8756721019744873 | early_stop=False |
| smoke_image_efficient_eml | COMPLETED | EfficientEMLImageClassifier | 0.375 | 0.0 | 0.1250 | 1.7369 | 0.2402501106262207 | early_stop=False |
| local_conv_baseline | NOT RUN | LocalConvBaseline |  |  |  |  |  | smoke mode: not implemented |
| smoke_image_cnn_eml_baseline | COMPLETED | cnn_eml | 0.5 | 0.25 | 0.2500 | 1.6227 | 0.07447671890258789 | early_stop=False |
| smoke_image_efficient_eml | COMPLETED | EfficientEMLImageClassifier | 0.375 | 0.0 | 0.1250 | 1.7369 | 0.09504485130310059 | early_stop=False |
| local_conv_baseline | NOT RUN | LocalConvBaseline |  |  |  |  |  | smoke mode: not implemented |

### Text Local Window
| run_id | status | model | best | final | delta vs ref | loss | time sec | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| smoke_text_efficient_eml | COMPLETED | EfficientEMLTextEncoder | 0.11274509876966476 | 0.11274509876966476 | 0.0000 | 4.3839 | 0.047122955322265625 |  |
| local_text_linear_baseline | NOT RUN | LocalTextCodecLinear |  |  |  |  |  | smoke mode: not standardized |
| text_medium_suite | NOT RUN | selected_text_models |  |  |  |  |  | smoke mode: not requested in this mode |
| kan_compare_text_local_conv | COMPLETED | LocalCausalConvLM | 0.0 | 0.0 | -0.1127 | 4.4488 | 0.006283998489379883 | early_stop=False |
| kan_compare_text_small_gru | COMPLETED | SmallGRULM | 0.05000000074505806 | 0.05000000074505806 | -0.0627 | 4.4340 | 0.013735055923461914 | early_stop=False |
| kan_compare_text_efficient_eml | COMPLETED | EfficientEMLTextEncoder | 0.07446808367967606 | 0.07446808367967606 | -0.0383 | 4.3975 | 0.01978278160095215 | early_stop=False |
| kan_compare_text_eml_edge | COMPLETED | EMLEdgeTextLM_kan_style | 0.04854368790984154 | 0.0 | -0.0642 | 4.4340 | 0.07332515716552734 | early_stop=False |
| kan_compare_text_local_conv | COMPLETED | LocalCausalConvLM | 0.22404371201992035 | 0.17766498029232025 | 0.1113 | 3.6288 | 0.15865302085876465 | early_stop=False |
| kan_compare_text_small_gru | COMPLETED | SmallGRULM | 0.12921348214149475 | 0.08888889104127884 | 0.0165 | 3.9181 | 0.2814500331878662 | early_stop=False |
| kan_compare_text_efficient_eml | COMPLETED | EfficientEMLTextEncoder | 0.40963855385780334 | 0.3787234127521515 | 0.2969 | 3.8218 | 0.7171981334686279 | early_stop=False |
| kan_compare_text_eml_edge | COMPLETED | EMLEdgeTextLM_kan_style | 0.4296296238899231 | 0.4296296238899231 | 0.3169 | 3.9839 | 2.393962860107422 | early_stop=False |
| kan_compare_text_local_conv | COMPLETED | LocalCausalConvLM | 0.4334975481033325 | 0.3365853726863861 | 0.3208 | 2.5168 | 0.3350667953491211 | early_stop=True |
| kan_compare_text_small_gru | COMPLETED | SmallGRULM | 0.4188481569290161 | 0.3368421196937561 | 0.3061 | 2.3333 | 1.5402390956878662 | early_stop=False |
| kan_compare_text_efficient_eml | COMPLETED | EfficientEMLTextEncoder | 0.4950000047683716 | 0.36138615012168884 | 0.3823 | 2.6294 | 3.2707719802856445 | early_stop=False |
| kan_compare_text_eml_edge | COMPLETED | EMLEdgeTextLM_kan_style | 0.5025380849838257 | 0.322429895401001 | 0.3898 | 3.0322 | 10.280304908752441 | early_stop=False |
| kan_compare_text_local_conv | COMPLETED | LocalCausalConvLM | 0.45049506425857544 | 0.39086294174194336 | 0.3377 | 2.3376 | 0.3432021141052246 | early_stop=True |
| kan_compare_text_small_gru | COMPLETED | SmallGRULM | 0.22267206013202667 | 0.10526315867900848 | 0.1099 | 3.7472 | 0.43062829971313477 | early_stop=True |
| kan_compare_text_efficient_eml | COMPLETED | EfficientEMLTextEncoder | 0.42944785952568054 | 0.34296029806137085 | 0.3167 | 3.1884 | 2.843071937561035 | early_stop=True |
| kan_compare_text_eml_edge | COMPLETED | EMLEdgeTextLM_kan_style | 0.460829496383667 | 0.29949238896369934 | 0.3481 | 3.9456 | 3.3603527545928955 | early_stop=True |
| smoke_text_efficient_eml | COMPLETED | EfficientEMLTextEncoder | 0.04128440469503403 | 0.03431372717022896 | -0.0715 | 4.3897 | 0.11359572410583496 | early_stop=False |
| local_text_linear_baseline | NOT RUN | LocalTextCodecLinear |  |  |  |  |  | smoke mode: not standardized |
| text_medium_suite | NOT RUN | selected_text_models |  |  |  |  |  | smoke mode: not requested in this mode |
| smoke_text_efficient_eml | COMPLETED | EfficientEMLTextEncoder | 0.04128440469503403 | 0.03431372717022896 | -0.0715 | 4.3897 | 0.04859304428100586 | early_stop=False |
| local_text_linear_baseline | NOT RUN | LocalTextCodecLinear |  |  |  |  |  | smoke mode: not standardized |
| text_medium_suite | NOT RUN | selected_text_models |  |  |  |  |  | smoke mode: not requested in this mode |

### CIFAR Medium
| run_id | status | model | best | final | delta vs ref | loss | time sec | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| cifar_medium_suite | NOT RUN | selected_image_models |  |  |  |  |  | smoke mode: not requested in this mode |
| cifar_medium_suite | NOT RUN | selected_image_models |  |  |  |  |  | smoke mode: not requested in this mode |
| cifar_medium_suite | NOT RUN | selected_image_models |  |  |  |  |  | smoke mode: not requested in this mode |

### Failed And Not Run Cells
| run_id | status | task | model | dataset | reason |
| --- | --- | --- | --- | --- | --- |
| local_conv_baseline | NOT RUN | image_synthetic | LocalConvBaseline | SyntheticShapeEnergyDataset | smoke mode: not implemented |
| local_text_linear_baseline | NOT RUN | text_synthetic | LocalTextCodecLinear | SyntheticTextEnergyDataset | smoke mode: not standardized |
| cifar_medium_suite | NOT RUN | image_cifar | selected_image_models | CIFAR10 | smoke mode: not requested in this mode |
| text_medium_suite | NOT RUN | text_synthetic | selected_text_models | SyntheticTextEnergyDataset | smoke mode: not requested in this mode |
| full_seeded_ablation | NOT RUN | mechanism_ablation | all_supported_cells | mixed | smoke mode: not requested in this mode |
| kan_paper_reference | NOT RUN | paper_reference | KAN_arxiv_2404_19756 | KAN paper AI+Science tasks | User requested no local KAN experiment; comparing against reported paper results only. |
| kan_paper_reference | NOT RUN | paper_reference | KAN_arxiv_2404_19756 | KAN paper AI+Science tasks | User requested no local KAN experiment; comparing against reported paper results only. |
| kan_paper_reference | NOT RUN | paper_reference | KAN_arxiv_2404_19756 | KAN paper AI+Science tasks | User requested no local KAN experiment; comparing against reported paper results only. |
| kan_paper_reference | NOT RUN | paper_reference | KAN_arxiv_2404_19756 | KAN paper AI+Science tasks | User requested no local KAN experiment; comparing against reported paper results only. |
| local_conv_baseline | NOT RUN | image_synthetic | LocalConvBaseline | SyntheticShapeEnergyDataset | smoke mode: not implemented |
| local_text_linear_baseline | NOT RUN | text_synthetic | LocalTextCodecLinear | SyntheticTextEnergyDataset | smoke mode: not standardized |
| cifar_medium_suite | NOT RUN | image_cifar | selected_image_models | CIFAR10 | smoke mode: not requested in this mode |
| text_medium_suite | NOT RUN | text_synthetic | selected_text_models | SyntheticTextEnergyDataset | smoke mode: not requested in this mode |
| full_seeded_ablation | NOT RUN | mechanism_ablation | all_supported_cells | mixed | smoke mode: not requested in this mode |
| local_conv_baseline | NOT RUN | image_synthetic | LocalConvBaseline | SyntheticShapeEnergyDataset | smoke mode: not implemented |
| local_text_linear_baseline | NOT RUN | text_synthetic | LocalTextCodecLinear | SyntheticTextEnergyDataset | smoke mode: not standardized |
| cifar_medium_suite | NOT RUN | image_cifar | selected_image_models | CIFAR10 | smoke mode: not requested in this mode |
| text_medium_suite | NOT RUN | text_synthetic | selected_text_models | SyntheticTextEnergyDataset | smoke mode: not requested in this mode |
| full_seeded_ablation | NOT RUN | mechanism_ablation | all_supported_cells | mixed | smoke mode: not requested in this mode |

### All Ablation Cells
| run_id | status | task | model | key settings | best | final | loss | reason |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| kan_paper_reference | NOT RUN | paper_reference | KAN_arxiv_2404_19756 | see config |  |  |  | User requested no local KAN experiment; comparing against reported paper results only. |
| kan_compare_image_cnn_eml | COMPLETED | image_synthetic | cnn_eml | warmup_enabled=True, early_stop=False, patience=20 | 0.625 | 0.25 | 1.6268 |  |
| kan_compare_image_efficient_eml | COMPLETED | image_synthetic | EfficientEMLImageClassifier | warmup_enabled=True, early_stop=False, patience=20 | 0.75 | 0.25 | 1.5906 |  |
| kan_compare_image_eml_edge | COMPLETED | image_synthetic | EMLEdgeImageClassifier_kan_style | warmup_enabled=True, early_stop=False, patience=20 | 0.5 | 0.125 | 1.6313 |  |
| kan_compare_text_local_conv | COMPLETED | text_synthetic | LocalCausalConvLM | early_stop=False, patience=20, seq_len=48 | 0.22404371201992035 | 0.17766498029232025 | 3.6288 |  |
| kan_compare_text_small_gru | COMPLETED | text_synthetic | SmallGRULM | early_stop=False, patience=20, seq_len=48 | 0.12921348214149475 | 0.08888889104127884 | 3.9181 |  |
| kan_compare_text_efficient_eml | COMPLETED | text_synthetic | EfficientEMLTextEncoder | early_stop=False, patience=20, seq_len=48 | 0.40963855385780334 | 0.3787234127521515 | 3.8218 |  |
| kan_compare_text_eml_edge | COMPLETED | text_synthetic | EMLEdgeTextLM_kan_style | early_stop=False, patience=20, seq_len=48 | 0.4296296238899231 | 0.4296296238899231 | 3.9839 |  |
| kan_paper_reference | NOT RUN | paper_reference | KAN_arxiv_2404_19756 | see config |  |  |  | User requested no local KAN experiment; comparing against reported paper results only. |
| kan_compare_image_cnn_eml | COMPLETED | image_synthetic | cnn_eml | warmup_enabled=True, early_stop=True, patience=12 | 0.75 | 0.25 | 1.6007 |  |
| kan_compare_image_efficient_eml | COMPLETED | image_synthetic | EfficientEMLImageClassifier | warmup_enabled=True, early_stop=True, patience=12 | 0.375 | 0.375 | 1.5922 |  |
| kan_compare_image_eml_edge | COMPLETED | image_synthetic | EMLEdgeImageClassifier_kan_style | warmup_enabled=True, early_stop=True, patience=12 | 0.5 | 0.5 | 1.6206 |  |
| kan_compare_text_local_conv | COMPLETED | text_synthetic | LocalCausalConvLM | early_stop=True, patience=12, seq_len=48 | 0.4334975481033325 | 0.3365853726863861 | 2.5168 |  |
| kan_compare_text_small_gru | COMPLETED | text_synthetic | SmallGRULM | early_stop=True, patience=12, seq_len=48 | 0.4188481569290161 | 0.3368421196937561 | 2.3333 |  |
| kan_compare_text_efficient_eml | COMPLETED | text_synthetic | EfficientEMLTextEncoder | early_stop=True, patience=12, seq_len=48 | 0.4950000047683716 | 0.36138615012168884 | 2.6294 |  |
| kan_compare_text_eml_edge | COMPLETED | text_synthetic | EMLEdgeTextLM_kan_style | early_stop=True, patience=12, seq_len=48 | 0.5025380849838257 | 0.322429895401001 | 3.0322 |  |
| kan_paper_reference | NOT RUN | paper_reference | KAN_arxiv_2404_19756 | see config |  |  |  | User requested no local KAN experiment; comparing against reported paper results only. |
| kan_compare_image_cnn_eml | COMPLETED | image_synthetic | cnn_eml | warmup_enabled=True, early_stop=True, patience=10 | 0.625 | 0.25 | 1.5949 |  |
| kan_compare_image_efficient_eml | COMPLETED | image_synthetic | EfficientEMLImageClassifier | warmup_enabled=True, early_stop=True, patience=10 | 0.375 | 0.25 | 1.6057 |  |
| kan_compare_image_eml_edge | COMPLETED | image_synthetic | EMLEdgeImageClassifier_kan_style | warmup_enabled=True, early_stop=True, patience=10 | 0.375 | 0.125 | 1.6120 |  |
| kan_compare_text_local_conv | COMPLETED | text_synthetic | LocalCausalConvLM | early_stop=True, patience=10, seq_len=48 | 0.45049506425857544 | 0.39086294174194336 | 2.3376 |  |
| kan_compare_text_small_gru | COMPLETED | text_synthetic | SmallGRULM | early_stop=True, patience=10, seq_len=48 | 0.22267206013202667 | 0.10526315867900848 | 3.7472 |  |
| kan_compare_text_efficient_eml | COMPLETED | text_synthetic | EfficientEMLTextEncoder | early_stop=True, patience=10, seq_len=48 | 0.42944785952568054 | 0.34296029806137085 | 3.1884 |  |
| kan_compare_text_eml_edge | COMPLETED | text_synthetic | EMLEdgeTextLM_kan_style | early_stop=True, patience=10, seq_len=48 | 0.460829496383667 | 0.29949238896369934 | 3.9456 |  |
| kan_operator_additive_smooth_spline_kan_seed0 | COMPLETED | kan_operator_replacement | spline_kan | see config | -0.006964400410652161 | -0.006964400410652161 |  |  |
| kan_operator_additive_smooth_semL_operator_replacement_seed0 | COMPLETED | kan_operator_replacement | semL_operator_replacement | see config | -0.08601398766040802 | -0.08601398766040802 |  |  |
| kan_operator_additive_smooth_spline_kan_seed1 | COMPLETED | kan_operator_replacement | spline_kan | see config | -0.004911794327199459 | -0.004911794327199459 |  |  |
| kan_operator_additive_smooth_semL_operator_replacement_seed1 | COMPLETED | kan_operator_replacement | semL_operator_replacement | see config | -0.026778995990753174 | -0.026778995990753174 |  |  |
| kan_operator_local_bumps_spline_kan_seed0 | COMPLETED | kan_operator_replacement | spline_kan | see config | -0.034437134861946106 | -0.03477754816412926 |  |  |
| kan_operator_local_bumps_semL_operator_replacement_seed0 | COMPLETED | kan_operator_replacement | semL_operator_replacement | see config | -0.24220982193946838 | -0.24220982193946838 |  |  |
| kan_operator_local_bumps_spline_kan_seed1 | COMPLETED | kan_operator_replacement | spline_kan | see config | -0.033025797456502914 | -0.033025797456502914 |  |  |
| kan_operator_local_bumps_semL_operator_replacement_seed1 | COMPLETED | kan_operator_replacement | semL_operator_replacement | see config | -0.2893967032432556 | -0.2893967032432556 |  |  |
| kan_operator_mixed_composition_spline_kan_seed0 | COMPLETED | kan_operator_replacement | spline_kan | see config | -0.021778786554932594 | -0.021778786554932594 |  |  |
| kan_operator_mixed_composition_semL_operator_replacement_seed0 | COMPLETED | kan_operator_replacement | semL_operator_replacement | see config | -0.09480735659599304 | -0.09480735659599304 |  |  |
| kan_operator_mixed_composition_spline_kan_seed1 | COMPLETED | kan_operator_replacement | spline_kan | see config | -0.024385713040828705 | -0.024385713040828705 |  |  |
| kan_operator_mixed_composition_semL_operator_replacement_seed1 | COMPLETED | kan_operator_replacement | semL_operator_replacement | see config | -0.09265530109405518 | -0.09265530109405518 |  |  |
| kan_operator_additive_smooth_spline_kan_seed0 | COMPLETED | kan_operator_replacement | spline_kan | see config | -0.001567840576171875 | -0.001497386023402214 |  |  |
| kan_operator_additive_smooth_semL_operator_replacement_seed0 | COMPLETED | kan_operator_replacement | semL_operator_replacement | see config | -0.00021667647524736822 | -0.00015080087177921087 |  |  |
| kan_operator_additive_smooth_spline_kan_seed1 | COMPLETED | kan_operator_replacement | spline_kan | see config | -0.004209687002003193 | -0.004136369097977877 |  |  |
| kan_operator_additive_smooth_semL_operator_replacement_seed1 | COMPLETED | kan_operator_replacement | semL_operator_replacement | see config | -0.00022901204647496343 | -0.00016085902461782098 |  |  |
| kan_operator_additive_smooth_spline_kan_seed2 | COMPLETED | kan_operator_replacement | spline_kan | see config | -0.0036571063101291656 | -0.003607383230701089 |  |  |
| kan_operator_additive_smooth_semL_operator_replacement_seed2 | COMPLETED | kan_operator_replacement | semL_operator_replacement | see config | -0.00031892131664790213 | -0.00027717393822968006 |  |  |
| kan_operator_local_bumps_spline_kan_seed0 | COMPLETED | kan_operator_replacement | spline_kan | see config | -0.00815877690911293 | -0.008197469636797905 |  |  |
| kan_operator_local_bumps_semL_operator_replacement_seed0 | COMPLETED | kan_operator_replacement | semL_operator_replacement | see config | -0.0015432039508596063 | -0.0018741983221843839 |  |  |
| kan_operator_local_bumps_spline_kan_seed1 | COMPLETED | kan_operator_replacement | spline_kan | see config | -0.010872142389416695 | -0.010840952396392822 |  |  |
| kan_operator_local_bumps_semL_operator_replacement_seed1 | COMPLETED | kan_operator_replacement | semL_operator_replacement | see config | -0.002194314496591687 | -0.002194314496591687 |  |  |
| kan_operator_local_bumps_spline_kan_seed2 | COMPLETED | kan_operator_replacement | spline_kan | see config | -0.010572157800197601 | -0.01053609512746334 |  |  |
| kan_operator_local_bumps_semL_operator_replacement_seed2 | COMPLETED | kan_operator_replacement | semL_operator_replacement | see config | -0.002846091752871871 | -0.0030250814743340015 |  |  |
| kan_operator_mixed_composition_spline_kan_seed0 | COMPLETED | kan_operator_replacement | spline_kan | see config | -0.00731165986508131 | -0.007239641156047583 |  |  |
| kan_operator_mixed_composition_semL_operator_replacement_seed0 | COMPLETED | kan_operator_replacement | semL_operator_replacement | see config | -0.00044302723836153746 | -0.0003822749713435769 |  |  |
| kan_operator_mixed_composition_spline_kan_seed1 | COMPLETED | kan_operator_replacement | spline_kan | see config | -0.006274871062487364 | -0.006263264454901218 |  |  |
| kan_operator_mixed_composition_semL_operator_replacement_seed1 | COMPLETED | kan_operator_replacement | semL_operator_replacement | see config | -0.00035359367029741406 | -0.0011069782776758075 |  |  |
| kan_operator_mixed_composition_spline_kan_seed2 | COMPLETED | kan_operator_replacement | spline_kan | see config | -0.006307478062808514 | -0.006381517741829157 |  |  |
| kan_operator_mixed_composition_semL_operator_replacement_seed2 | COMPLETED | kan_operator_replacement | semL_operator_replacement | see config | -0.0004493622400332242 | -0.0004802474577445537 |  |  |

Other ablation axes remain `NOT RUN` when listed in the status table.

## 8. EML Diagnostics

| run_id | model | drive_mean | drive_std | resistance_mean | resistance_std | energy_mean | energy_std | null_weight_mean | responsibility_entropy_mean | update_strength_mean | update_gate_mean | attractor_diversity | ambiguity_mean | ambiguity_weight_mean | sample_uncertainty_mean | resistance_noise_corr | resistance_occlusion_corr | corruption_resistance_corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_image_cnn_eml_baseline | cnn_eml | 0.0009 | 0.3032 | 0.3478 | 0.8859 | -0.8709 | 0.3305 |  |  |  |  |  |  |  | 0.6924 |  |  |  |
| smoke_image_efficient_eml | EfficientEMLImageClassifier | 0.0090 | 0.0766 | 1.3658 | 0.1513 | -0.2375 | 0.0300 | 0.1598 |  | 0.8545 | 0.3961 | 0.8370 | 2.4728 |  | 0.6928 |  |  |  |
| smoke_text_efficient_eml | EfficientEMLTextEncoder | 0.0086 | 0.3501 | 1.5926 | 0.2265 | -0.0834 | 0.1056 | 0.5353 |  | 0.4598 | 0.2142 | 0.2319 |  |  |  |  |  |  |
| probe_gate_compat_sigmoid_update | probe_gate_compat_sigmoid_update | -0.0002 | 0.0024 | 0.0000 | 0.0018 | -0.8810 | 0.2129 |  | 1.4710 | 1.0000 | 0.2689 |  |  |  |  |  |  |  |
| probe_responsibility_no_null_precision | probe_responsibility_no_null_precision | 0.0002 | 0.0019 | -0.0002 | 0.0022 | -0.8809 | 0.2130 |  | 1.3863 | 1.0000 | 0.3113 |  |  |  |  |  |  |  |
| probe_responsibility_with_null_precision | probe_responsibility_with_null_precision | -0.0001 | 0.0020 | -0.0002 | 0.0022 | -0.8810 | 0.2129 | 0.2919 | 1.5855 | 0.7081 | 0.3113 |  |  |  |  |  |  |  |
| kan_compare_image_cnn_eml | cnn_eml | 0.0015 | 0.2778 | 0.1487 | 0.4116 | -0.8653 | 0.3434 |  |  |  |  |  | 0.2105 | 1.0000 | 0.6920 | -0.9110 | nan |  |
| kan_compare_image_efficient_eml | EfficientEMLImageClassifier | 0.0034 | 0.0239 | 1.3633 | 0.1196 | -0.2375 | 0.0261 | 0.1598 |  | 0.8545 | 0.0912 | 0.0207 | 0.3049 | 1.0000 | 0.6936 | -0.3508 | -0.8127 |  |
| kan_compare_image_eml_edge | EMLEdgeImageClassifier_kan_style | 0.0043 | 0.1848 | 0.2503 | 0.3393 | -0.0054 | 0.0217 |  |  |  |  |  |  |  |  | -0.6018 | -0.4725 |  |
| kan_compare_text_local_conv | LocalCausalConvLM |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | nan |
| kan_compare_text_small_gru | SmallGRULM |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | nan |
| kan_compare_text_efficient_eml | EfficientEMLTextEncoder | -0.0399 | 0.3244 | 1.6211 | 0.2286 | -0.0850 | 0.1049 | 0.5669 |  | 0.4272 | 0.0459 | 0.0313 |  |  |  |  |  | 0.0488 |
| kan_compare_text_eml_edge | EMLEdgeTextLM_kan_style | 0.0003 | 0.0899 | 0.1252 | 0.1703 | -0.0033 | 0.0103 |  |  |  |  |  |  |  |  |  |  | 0.0913 |
| kan_compare_image_cnn_eml | cnn_eml | -0.0092 | 0.3172 | 0.1427 | 0.3892 | -0.8633 | 0.3395 |  |  |  |  |  | 0.1802 | 1.0000 | 0.7034 | -0.0152 | -0.4578 |  |
| kan_compare_image_efficient_eml | EfficientEMLImageClassifier | 0.0050 | 0.0831 | 1.3632 | 0.1197 | -0.2361 | 0.0281 | 0.1596 |  | 0.8546 | 0.0913 | 0.4278 | 0.7864 | 1.0000 | 0.7137 | -0.0019 | 0.1222 |  |
| kan_compare_image_eml_edge | EMLEdgeImageClassifier_kan_style | 0.0076 | 0.1729 | 0.2558 | 0.4646 | -0.0050 | 0.0235 |  |  |  |  |  |  |  |  | -0.3560 | -0.0436 |  |
| kan_compare_text_local_conv | LocalCausalConvLM |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | nan |
| kan_compare_text_small_gru | SmallGRULM |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | nan |
| kan_compare_text_efficient_eml | EfficientEMLTextEncoder | -0.6329 | 1.2086 | 1.4756 | 0.5498 | -0.1051 | 0.2371 | 0.4815 |  | 0.5137 | 0.0482 | 0.0312 |  |  |  |  |  | -0.0295 |
| kan_compare_text_eml_edge | EMLEdgeTextLM_kan_style | -0.0084 | 0.1659 | 0.2586 | 0.2904 | -0.0157 | 0.0502 |  |  |  |  |  |  |  |  |  |  | 0.0677 |
| kan_compare_image_cnn_eml | cnn_eml | -0.0152 | 0.4622 | 0.1310 | 0.3317 | -0.8693 | 0.3342 |  |  |  |  |  | 0.5483 | 0.1350 | 0.6937 | 0.0403 | 0.0449 |  |
| kan_compare_image_efficient_eml | EfficientEMLImageClassifier | 0.0124 | 0.0425 | 1.3599 | 0.1172 | -0.2351 | 0.0241 | 0.1602 |  | 0.8548 | 0.0914 | 0.1971 | 0.2842 | 0.0650 | 0.6963 | -0.1483 | 0.3061 |  |
| kan_compare_image_eml_edge | EMLEdgeImageClassifier_kan_style | 0.0035 | 0.1680 | 0.2522 | 0.5739 | -0.0064 | 0.0225 |  |  |  |  |  |  |  |  | 0.1732 | 0.6889 |  |
| kan_compare_text_local_conv | LocalCausalConvLM |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | nan |
| kan_compare_text_small_gru | SmallGRULM |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | nan |
| kan_compare_text_efficient_eml | EfficientEMLTextEncoder | -0.6525 | 1.8983 | 2.2168 | 2.4198 | -0.0366 | 0.4876 | 0.5346 |  | 0.4636 | 0.0623 | 0.0312 |  |  |  |  |  | -0.1524 |
| kan_compare_text_eml_edge | EMLEdgeTextLM_kan_style | -0.0457 | 0.4128 | 0.4239 | 0.5495 | -0.0548 | 0.2304 |  |  |  |  |  |  |  |  |  |  | 0.1064 |
| kan_compare_image_cnn_eml | cnn_eml | 0.0268 | 0.5047 | 0.1297 | 0.3273 | -0.8666 | 0.3424 |  |  |  |  |  | 0.9752 | 0.0667 | 0.6914 | 0.5542 | 0.0761 |  |
| kan_compare_image_efficient_eml | EfficientEMLImageClassifier | 0.0299 | 0.0641 | 1.3567 | 0.1322 | -0.2303 | 0.0291 | 0.1590 |  | 0.8554 | 0.0920 | 0.2822 | 0.8966 | 0.0700 | 0.6958 | 0.4057 | 0.5227 |  |
| kan_compare_image_eml_edge | EMLEdgeImageClassifier_kan_style | -0.0055 | 0.1795 | 0.2512 | 0.3616 | -0.0079 | 0.0203 |  |  |  |  |  |  |  |  | 0.1689 | -0.3084 |  |
| kan_compare_text_local_conv | LocalCausalConvLM |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | nan |
| kan_compare_text_small_gru | SmallGRULM |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | nan |
| kan_compare_text_efficient_eml | EfficientEMLTextEncoder | -0.6421 | 1.9280 | 1.7362 | 1.5657 | -0.0990 | 0.3837 | 0.3794 |  | 0.6162 | 0.0684 | 0.0290 |  |  |  |  |  | -0.0425 |
| kan_compare_text_eml_edge | EMLEdgeTextLM_kan_style | -0.0174 | 0.1959 | 0.2791 | 0.3286 | -0.0235 | 0.0735 |  |  |  |  |  |  |  |  |  |  | 0.0081 |
| smoke_image_cnn_eml_baseline | cnn_eml | 0.0558 | 0.4641 | 0.2509 | 0.6590 | -0.8540 | 0.3797 |  |  |  |  |  | 0.9635 | 1.0000 | 0.6925 | 0.1248 | 0.1718 |  |
| smoke_image_efficient_eml | EfficientEMLImageClassifier | 0.0000 | 0.0721 | 1.3670 | 0.1218 | -0.2391 | 0.0296 | 0.1597 |  | 0.8542 | 0.0911 | 0.0277 | 0.9784 | 1.0000 | 0.6932 | 0.5407 | -0.6726 |  |
| smoke_text_efficient_eml | EfficientEMLTextEncoder | -0.0024 | 0.3676 | 1.5983 | 0.2349 | -0.0839 | 0.1052 | 0.5352 |  | 0.4599 | 0.0496 | 0.0353 |  |  |  |  |  | -0.0836 |
| probe_gate_compat_sigmoid_update | probe_gate_compat_sigmoid_update | -0.0002 | 0.0024 | 0.0000 | 0.0018 | -0.8810 | 0.2129 |  | 1.4710 | 1.0000 | 0.2689 |  |  |  |  |  |  |  |
| probe_responsibility_no_null_precision | probe_responsibility_no_null_precision | 0.0002 | 0.0019 | -0.0002 | 0.0022 | -0.8809 | 0.2130 |  | 1.3863 | 1.0000 | 0.0589 |  |  |  |  |  |  |  |
| probe_responsibility_with_null_precision | probe_responsibility_with_null_precision | -0.0001 | 0.0020 | -0.0002 | 0.0022 | -0.8810 | 0.2129 | 0.2919 | 1.5855 | 0.7081 | 0.0589 |  |  |  |  |  |  |  |
| probe_thresholded_null | all_noise_should_choose_null |  |  |  |  |  |  | 0.9010 |  | 0.0990 |  |  |  |  |  |  |  |  |
| kan_operator_additive_smooth_spline_kan_seed0 | spline_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_additive_smooth_semL_operator_replacement_seed0 | semL_operator_replacement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_additive_smooth_spline_kan_seed0 | spline_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_additive_smooth_semL_operator_replacement_seed0 | semL_operator_replacement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_additive_smooth_spline_kan_seed1 | spline_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_additive_smooth_semL_operator_replacement_seed1 | semL_operator_replacement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_local_bumps_spline_kan_seed0 | spline_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_local_bumps_semL_operator_replacement_seed0 | semL_operator_replacement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_local_bumps_spline_kan_seed1 | spline_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_local_bumps_semL_operator_replacement_seed1 | semL_operator_replacement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_mixed_composition_spline_kan_seed0 | spline_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_mixed_composition_semL_operator_replacement_seed0 | semL_operator_replacement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_mixed_composition_spline_kan_seed1 | spline_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_mixed_composition_semL_operator_replacement_seed1 | semL_operator_replacement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_additive_smooth_spline_kan_seed0 | spline_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_additive_smooth_semL_operator_replacement_seed0 | semL_operator_replacement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_additive_smooth_spline_kan_seed1 | spline_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_additive_smooth_semL_operator_replacement_seed1 | semL_operator_replacement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_additive_smooth_spline_kan_seed2 | spline_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_additive_smooth_semL_operator_replacement_seed2 | semL_operator_replacement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_local_bumps_spline_kan_seed0 | spline_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_local_bumps_semL_operator_replacement_seed0 | semL_operator_replacement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_local_bumps_spline_kan_seed1 | spline_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_local_bumps_semL_operator_replacement_seed1 | semL_operator_replacement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_local_bumps_spline_kan_seed2 | spline_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_local_bumps_semL_operator_replacement_seed2 | semL_operator_replacement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_mixed_composition_spline_kan_seed0 | spline_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_mixed_composition_semL_operator_replacement_seed0 | semL_operator_replacement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_mixed_composition_spline_kan_seed1 | spline_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_mixed_composition_semL_operator_replacement_seed1 | semL_operator_replacement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_mixed_composition_spline_kan_seed2 | spline_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kan_operator_mixed_composition_semL_operator_replacement_seed2 | semL_operator_replacement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_symbolic_regression_eml_kan_seed0 | eml_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed0 | mlp_same_width |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_symbolic_regression_mlp_param_matched_seed0 | mlp_param_matched |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_symbolic_regression_eml_kan_seed1 | eml_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed1 | mlp_same_width |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_symbolic_regression_mlp_param_matched_seed1 | mlp_param_matched |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_symbolic_regression_eml_kan_seed2 | eml_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed2 | mlp_same_width |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_symbolic_regression_mlp_param_matched_seed2 | mlp_param_matched |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_eml_kan_seed0 | eml_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_mlp_same_width_seed0 | mlp_same_width |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed0 | mlp_param_matched |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_eml_kan_seed1 | eml_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_mlp_same_width_seed1 | mlp_same_width |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed1 | mlp_param_matched |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_eml_kan_seed2 | eml_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_mlp_same_width_seed2 | mlp_same_width |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed2 | mlp_param_matched |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_shift_classification_eml_kan_seed0 | eml_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_shift_classification_mlp_same_width_seed0 | mlp_same_width |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_shift_classification_mlp_param_matched_seed0 | mlp_param_matched |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_shift_classification_eml_kan_seed1 | eml_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_shift_classification_mlp_same_width_seed1 | mlp_same_width |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_shift_classification_mlp_param_matched_seed1 | mlp_param_matched |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_shift_classification_eml_kan_seed2 | eml_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_shift_classification_mlp_same_width_seed2 | mlp_same_width |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_shift_classification_mlp_param_matched_seed2 | mlp_param_matched |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_eml_kan_seed0 | eml_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_mlp_same_width_seed0 | mlp_same_width |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed0 | mlp_param_matched |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_eml_kan_seed1 | eml_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_mlp_same_width_seed1 | mlp_same_width |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed1 | mlp_param_matched |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_eml_kan_seed2 | eml_kan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_mlp_same_width_seed2 | mlp_same_width |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_mlp_param_matched_seed2 | mlp_param_matched |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed1 | mlp_same_width |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_symbolic_regression_mlp_same_width_seed2 | mlp_same_width |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_mlp_same_width_seed1 | mlp_same_width |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| eml_kan_mlp_localized_regression_mlp_same_width_seed2 | mlp_same_width |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| smoke_image_cnn_eml_baseline | cnn_eml | -0.0463 | 0.3544 | 0.1236 | 0.3345 | -0.8685 | 0.3361 |  |  |  |  |  | 0.0172 | 1.0000 | 0.6951 | 0.2856 | -0.0997 |  |
| smoke_image_efficient_eml | EfficientEMLImageClassifier | 0.0000 | 0.0721 | 1.3670 | 0.1218 | -0.2391 | 0.0296 | 0.1597 |  | 0.8542 | 0.0911 | 0.0277 | 0.9784 | 1.0000 | 0.6932 | 0.5407 | -0.6726 |  |
| smoke_text_efficient_eml | EfficientEMLTextEncoder | -0.0024 | 0.3676 | 1.5983 | 0.2349 | -0.0839 | 0.1052 | 0.5352 |  | 0.4599 | 0.0496 | 0.0353 |  |  |  |  |  | -0.0836 |
| probe_gate_compat_sigmoid_update | probe_gate_compat_sigmoid_update | -0.0002 | 0.0024 | 0.0000 | 0.0018 | -0.8810 | 0.2129 |  | 1.4710 | 1.0000 | 0.2689 |  |  |  |  |  |  |  |
| probe_responsibility_no_null_precision | probe_responsibility_no_null_precision | 0.0002 | 0.0019 | -0.0002 | 0.0022 | -0.8809 | 0.2130 |  | 1.3863 | 1.0000 | 0.0589 |  |  |  |  |  |  |  |
| probe_responsibility_with_null_precision | probe_responsibility_with_null_precision | -0.0001 | 0.0020 | -0.0002 | 0.0022 | -0.8810 | 0.2129 | 0.2919 | 1.5855 | 0.7081 | 0.0589 |  |  |  |  |  |  |  |
| probe_thresholded_null | all_noise_should_choose_null |  |  |  |  |  |  | 0.9010 |  | 0.0990 |  |  |  |  |  |  |  |  |

Resistance-noise, resistance-occlusion, and resistance-corruption correlations are included when emitted by a run; otherwise MISSING.

## 9. Training Curves

### smoke_image_cnn_eml_baseline

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.5923607349395752 | 0.25 |  | 0.020364999771118164 |
| 2 | 1.6290571689605713 | 0.25 |  | 0.030501842498779297 |
| 3 | 1.5851128101348877 | 0.25 |  | 0.042111873626708984 |
| 4 | 1.6345659494400024 | 0.0 |  | 0.05194997787475586 |

### smoke_image_efficient_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.5977815389633179 | 0.125 |  | 0.019364118576049805 |
| 2 | 1.6858720779418945 | 0.0 |  | 0.037405967712402344 |
| 3 | 1.5530210733413696 | 0.25 |  | 0.05495595932006836 |
| 4 | 1.7483947277069092 | 0.0 |  | 0.0734257698059082 |

### smoke_text_efficient_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.431398868560791 |  | 0.016806723549962044 | 0.012645959854125977 |
| 2 | 4.407316207885742 |  | 0.019801979884505272 | 0.024617910385131836 |
| 3 | 4.394105911254883 |  | 0.07339449226856232 | 0.03557896614074707 |
| 4 | 4.383850574493408 |  | 0.11274509876966476 | 0.046571969985961914 |

### probe_gate_compat_sigmoid_update

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0009777545928955078 |

### probe_responsibility_no_null_precision

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.000885009765625 |

### probe_responsibility_with_null_precision

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0009799003601074219 |

### kan_compare_image_cnn_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.5628628730773926 | 0.25 |  | 0.04241585731506348 |
| 2 | 1.5467561483383179 | 0.5 |  | 0.05198192596435547 |

### kan_compare_image_efficient_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.6439141035079956 | 0.0 |  | 0.027865886688232422 |
| 2 | 1.6425502300262451 | 0.0 |  | 0.049458980560302734 |

### kan_compare_image_eml_edge

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.599924921989441 | 0.25 |  | 0.005707979202270508 |
| 2 | 1.6111668348312378 | 0.25 |  | 0.010751962661743164 |

### kan_compare_text_local_conv

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.409704685211182 |  | 0.0 | 0.004072904586791992 |
| 2 | 4.448815822601318 |  | 0.0 | 0.0059452056884765625 |

### kan_compare_text_small_gru

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.427419662475586 |  | 0.011627906933426857 | 0.00896000862121582 |
| 2 | 4.433996677398682 |  | 0.05000000074505806 | 0.013354063034057617 |

### kan_compare_text_efficient_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.422602653503418 |  | 0.027027027681469917 | 0.009437799453735352 |
| 2 | 4.397518634796143 |  | 0.07446808367967606 | 0.019237756729125977 |

### kan_compare_text_eml_edge

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.421546936035156 |  | 0.04854368790984154 | 0.03317523002624512 |
| 2 | 4.434030532836914 |  | 0.0 | 0.06552815437316895 |

### kan_compare_image_cnn_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 46 | 1.6103496551513672 | 0.125 |  | 0.660585880279541 |
| 47 | 1.6076658964157104 | 0.25 |  | 0.6799581050872803 |
| 48 | 1.5635167360305786 | 0.125 |  | 0.6970851421356201 |
| 49 | 1.5304291248321533 | 0.625 |  | 0.7137279510498047 |
| 50 | 1.6267756223678589 | 0.25 |  | 0.7268199920654297 |

### kan_compare_image_efficient_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 46 | 1.5247690677642822 | 0.375 |  | 1.2413229942321777 |
| 47 | 1.7741392850875854 | 0.0 |  | 1.2753608226776123 |
| 48 | 1.7396924495697021 | 0.0 |  | 1.301192045211792 |
| 49 | 1.6405495405197144 | 0.125 |  | 1.3252220153808594 |
| 50 | 1.590557336807251 | 0.25 |  | 1.3513450622558594 |

### kan_compare_image_eml_edge

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 46 | 1.6681686639785767 | 0.125 |  | 0.32407498359680176 |
| 47 | 1.5549373626708984 | 0.25 |  | 0.330625057220459 |
| 48 | 1.7009707689285278 | 0.0 |  | 0.33866286277770996 |
| 49 | 1.607787847518921 | 0.25 |  | 0.3460109233856201 |
| 50 | 1.631274938583374 | 0.125 |  | 0.3540618419647217 |

### kan_compare_text_local_conv

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 46 | 3.691526412963867 |  | 0.15469613671302795 | 0.14316797256469727 |
| 47 | 3.604872703552246 |  | 0.17142857611179352 | 0.14633393287658691 |
| 48 | 3.4914917945861816 |  | 0.21621622145175934 | 0.14969801902770996 |
| 49 | 3.451582908630371 |  | 0.21621622145175934 | 0.15340495109558105 |
| 50 | 3.628776788711548 |  | 0.17766498029232025 | 0.1570439338684082 |

### kan_compare_text_small_gru

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 46 | 3.9320425987243652 |  | 0.12921348214149475 | 0.25673699378967285 |
| 47 | 3.877420663833618 |  | 0.10614524781703949 | 0.2624659538269043 |
| 48 | 4.001971244812012 |  | 0.04280155524611473 | 0.26819920539855957 |
| 49 | 3.909634828567505 |  | 0.05369127541780472 | 0.27408885955810547 |
| 50 | 3.918095350265503 |  | 0.08888889104127884 | 0.27997684478759766 |

### kan_compare_text_efficient_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 46 | 3.922830104827881 |  | 0.3321167826652527 | 0.6591281890869141 |
| 47 | 3.9094979763031006 |  | 0.35164836049079895 | 0.6729512214660645 |
| 48 | 3.9052507877349854 |  | 0.31877729296684265 | 0.6865239143371582 |
| 49 | 3.847784996032715 |  | 0.3787878751754761 | 0.7006411552429199 |
| 50 | 3.8218295574188232 |  | 0.3787234127521515 | 0.7141709327697754 |

### kan_compare_text_eml_edge

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 46 | 4.006764888763428 |  | 0.41081079840660095 | 2.2146401405334473 |
| 47 | 4.069974899291992 |  | 0.2977527976036072 | 2.262888193130493 |
| 48 | 4.0221452713012695 |  | 0.3499999940395355 | 2.307803153991699 |
| 49 | 3.9839744567871094 |  | 0.42060086131095886 | 2.3489530086517334 |
| 50 | 3.9839038848876953 |  | 0.4296296238899231 | 2.3915281295776367 |

### kan_compare_image_cnn_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 23 | 1.585457682609558 | 0.375 |  | 0.2764170169830322 |
| 24 | 1.5534437894821167 | 0.375 |  | 0.2890942096710205 |
| 25 | 1.6439775228500366 | 0.0 |  | 0.3019411563873291 |
| 26 | 1.5642563104629517 | 0.375 |  | 0.31379032135009766 |
| 27 | 1.60074782371521 | 0.25 |  | 0.32626914978027344 |

### kan_compare_image_efficient_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 9 | 1.6165344715118408 | 0.25 |  | 0.22101187705993652 |
| 10 | 1.655361533164978 | 0.0 |  | 0.24374699592590332 |
| 11 | 1.6900988817214966 | 0.0 |  | 0.26584601402282715 |
| 12 | 1.5923455953598022 | 0.25 |  | 0.2892160415649414 |
| 13 | 1.5922088623046875 | 0.375 |  | 0.3121180534362793 |

### kan_compare_image_eml_edge

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 21 | 1.589087724685669 | 0.375 |  | 0.13548994064331055 |
| 22 | 1.5753777027130127 | 0.25 |  | 0.14211106300354004 |
| 23 | 1.6388455629348755 | 0.125 |  | 0.1476149559020996 |
| 24 | 1.608884572982788 | 0.0 |  | 0.15477371215820312 |
| 25 | 1.6206371784210205 | 0.5 |  | 0.16106200218200684 |

### kan_compare_text_local_conv

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 88 | 3.083014488220215 |  | 0.25217390060424805 | 0.3138597011566162 |
| 89 | 2.6777443885803223 |  | 0.28651684522628784 | 0.3186030387878418 |
| 90 | 2.3011510372161865 |  | 0.4192139804363251 | 0.3229789733886719 |
| 91 | 2.4295194149017334 |  | 0.3968254029750824 | 0.32770299911499023 |
| 92 | 2.516831159591675 |  | 0.3365853726863861 | 0.3324620723724365 |

### kan_compare_text_small_gru

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 196 | 2.381460189819336 |  | 0.4188481569290161 | 1.4959580898284912 |
| 197 | 2.546915054321289 |  | 0.3063829839229584 | 1.5058679580688477 |
| 198 | 2.4616732597351074 |  | 0.32773110270500183 | 1.5153160095214844 |
| 199 | 2.444298028945923 |  | 0.38461539149284363 | 1.5254061222076416 |
| 200 | 2.3332512378692627 |  | 0.3368421196937561 | 1.5350041389465332 |

### kan_compare_text_efficient_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 196 | 2.7074179649353027 |  | 0.30612245202064514 | 3.171792984008789 |
| 197 | 2.5588738918304443 |  | 0.36263737082481384 | 3.1921560764312744 |
| 198 | 2.609968900680542 |  | 0.3529411852359772 | 3.2133519649505615 |
| 199 | 2.5640792846679688 |  | 0.4337349534034729 | 3.240557909011841 |
| 200 | 2.629408121109009 |  | 0.36138615012168884 | 3.2605841159820557 |

### kan_compare_text_eml_edge

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 196 | 2.8199644088745117 |  | 0.4507042169570923 | 10.017499923706055 |
| 197 | 2.8755598068237305 |  | 0.45306122303009033 | 10.068306922912598 |
| 198 | 2.8699302673339844 |  | 0.39181286096572876 | 10.146469831466675 |
| 199 | 3.037623405456543 |  | 0.3257918655872345 | 10.211424827575684 |
| 200 | 3.032203197479248 |  | 0.322429895401001 | 10.272602796554565 |

### kan_compare_image_cnn_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1.6010459661483765 | 0.375 |  | 0.19815802574157715 |
| 17 | 1.661698341369629 | 0.25 |  | 0.21091794967651367 |
| 18 | 1.6283984184265137 | 0.125 |  | 0.22206878662109375 |
| 19 | 1.5271775722503662 | 0.5 |  | 0.23340296745300293 |
| 20 | 1.594855785369873 | 0.25 |  | 0.2453169822692871 |

### kan_compare_image_efficient_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 17 | 1.6757079362869263 | 0.125 |  | 0.41646885871887207 |
| 18 | 1.6170878410339355 | 0.125 |  | 0.4396347999572754 |
| 19 | 1.5840191841125488 | 0.375 |  | 0.4623398780822754 |
| 20 | 1.6858166456222534 | 0.0 |  | 0.4874267578125 |
| 21 | 1.6056783199310303 | 0.25 |  | 0.5119438171386719 |

### kan_compare_image_eml_edge

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 1.6332391500473022 | 0.25 |  | 0.04998779296875 |
| 8 | 1.6860649585723877 | 0.0 |  | 0.05553269386291504 |
| 9 | 1.6242141723632812 | 0.125 |  | 0.06218981742858887 |
| 10 | 1.5764676332473755 | 0.375 |  | 0.0679478645324707 |
| 11 | 1.6119627952575684 | 0.125 |  | 0.07330894470214844 |

### kan_compare_text_local_conv

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 90 | 2.4061214923858643 |  | 0.37962964177131653 | 0.3217630386352539 |
| 91 | 2.8715386390686035 |  | 0.27927929162979126 | 0.3262321949005127 |
| 92 | 2.4471981525421143 |  | 0.3316326439380646 | 0.33130717277526855 |
| 93 | 2.416154146194458 |  | 0.3772242069244385 | 0.33615803718566895 |
| 94 | 2.337569236755371 |  | 0.39086294174194336 | 0.3406369686126709 |

### kan_compare_text_small_gru

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 3.768479347229004 |  | 0.11176470667123795 | 0.40064525604248047 |
| 65 | 3.762120485305786 |  | 0.13080169260501862 | 0.4072442054748535 |
| 66 | 3.815319776535034 |  | 0.09944751113653183 | 0.4145522117614746 |
| 67 | 3.6340620517730713 |  | 0.15789473056793213 | 0.4209253787994385 |
| 68 | 3.7472052574157715 |  | 0.10526315867900848 | 0.42826318740844727 |

### kan_compare_text_efficient_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 173 | 3.165757417678833 |  | 0.3219696879386902 | 2.750706911087036 |
| 174 | 3.2160580158233643 |  | 0.19285714626312256 | 2.7716598510742188 |
| 175 | 3.1719672679901123 |  | 0.27108433842658997 | 2.7930097579956055 |
| 176 | 3.145674228668213 |  | 0.347328245639801 | 2.8132119178771973 |
| 177 | 3.1883533000946045 |  | 0.34296029806137085 | 2.833984851837158 |

### kan_compare_text_eml_edge

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 66 | 3.9224796295166016 |  | 0.3059210479259491 | 3.154376745223999 |
| 67 | 3.902308702468872 |  | 0.37142857909202576 | 3.203023672103882 |
| 68 | 3.929304361343384 |  | 0.28110599517822266 | 3.2527077198028564 |
| 69 | 3.8974690437316895 |  | 0.3125 | 3.2996668815612793 |
| 70 | 3.945603370666504 |  | 0.29949238896369934 | 3.356712818145752 |

### smoke_image_cnn_eml_baseline

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.5547345876693726 | 0.5 |  | 0.10369205474853516 |
| 2 | 1.5958995819091797 | 0.125 |  | 0.16701412200927734 |
| 3 | 1.604356050491333 | 0.25 |  | 0.2360842227935791 |
| 4 | 1.574501395225525 | 0.375 |  | 0.8743391036987305 |

### smoke_image_efficient_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.5968564748764038 | 0.375 |  | 0.05912137031555176 |
| 2 | 1.696264624595642 | 0.0 |  | 0.11571311950683594 |
| 3 | 1.5503305196762085 | 0.375 |  | 0.17114806175231934 |
| 4 | 1.7369440793991089 | 0.0 |  | 0.23904800415039062 |

### smoke_text_efficient_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.430697917938232 |  | 0.021008403971791267 | 0.03295588493347168 |
| 2 | 4.411949157714844 |  | 0.009900989942252636 | 0.06084179878234863 |
| 3 | 4.394505977630615 |  | 0.04128440469503403 | 0.09088873863220215 |
| 4 | 4.389736175537109 |  | 0.03431372717022896 | 0.11258482933044434 |

### probe_gate_compat_sigmoid_update

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0013370513916015625 |

### probe_responsibility_no_null_precision

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0033080577850341797 |

### probe_responsibility_with_null_precision

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0012061595916748047 |

### probe_thresholded_null

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0010590553283691406 |

### kan_operator_additive_smooth_spline_kan_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |

### kan_operator_additive_smooth_semL_operator_replacement_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |

### kan_operator_additive_smooth_spline_kan_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 400 |  |  |  |  |
| 425 |  |  |  |  |
| 450 |  |  |  |  |
| 475 |  |  |  |  |
| 500 |  |  |  |  |

### kan_operator_additive_smooth_semL_operator_replacement_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 400 |  |  |  |  |
| 425 |  |  |  |  |
| 450 |  |  |  |  |
| 475 |  |  |  |  |
| 500 |  |  |  |  |

### kan_operator_additive_smooth_spline_kan_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 400 |  |  |  |  |
| 425 |  |  |  |  |
| 450 |  |  |  |  |
| 475 |  |  |  |  |
| 500 |  |  |  |  |

### kan_operator_additive_smooth_semL_operator_replacement_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 400 |  |  |  |  |
| 425 |  |  |  |  |
| 450 |  |  |  |  |
| 475 |  |  |  |  |
| 500 |  |  |  |  |

### kan_operator_local_bumps_spline_kan_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 400 |  |  |  |  |
| 425 |  |  |  |  |
| 450 |  |  |  |  |
| 475 |  |  |  |  |
| 500 |  |  |  |  |

### kan_operator_local_bumps_semL_operator_replacement_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 400 |  |  |  |  |
| 425 |  |  |  |  |
| 450 |  |  |  |  |
| 475 |  |  |  |  |
| 500 |  |  |  |  |

### kan_operator_local_bumps_spline_kan_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 400 |  |  |  |  |
| 425 |  |  |  |  |
| 450 |  |  |  |  |
| 475 |  |  |  |  |
| 500 |  |  |  |  |

### kan_operator_local_bumps_semL_operator_replacement_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 400 |  |  |  |  |
| 425 |  |  |  |  |
| 450 |  |  |  |  |
| 475 |  |  |  |  |
| 500 |  |  |  |  |

### kan_operator_mixed_composition_spline_kan_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 400 |  |  |  |  |
| 425 |  |  |  |  |
| 450 |  |  |  |  |
| 475 |  |  |  |  |
| 500 |  |  |  |  |

### kan_operator_mixed_composition_semL_operator_replacement_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 400 |  |  |  |  |
| 425 |  |  |  |  |
| 450 |  |  |  |  |
| 475 |  |  |  |  |
| 500 |  |  |  |  |

### kan_operator_mixed_composition_spline_kan_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 400 |  |  |  |  |
| 425 |  |  |  |  |
| 450 |  |  |  |  |
| 475 |  |  |  |  |
| 500 |  |  |  |  |

### kan_operator_mixed_composition_semL_operator_replacement_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 400 |  |  |  |  |
| 425 |  |  |  |  |
| 450 |  |  |  |  |
| 475 |  |  |  |  |
| 500 |  |  |  |  |

### kan_operator_additive_smooth_spline_kan_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 |  |  |  |  |
| 4850 |  |  |  |  |
| 4900 |  |  |  |  |
| 4950 |  |  |  |  |
| 5000 |  |  |  |  |

### kan_operator_additive_smooth_semL_operator_replacement_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 2550 |  |  |  |  |
| 2600 |  |  |  |  |
| 2650 |  |  |  |  |
| 2700 |  |  |  |  |
| 2750 |  |  |  |  |

### kan_operator_additive_smooth_spline_kan_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 850 |  |  |  |  |
| 900 |  |  |  |  |
| 950 |  |  |  |  |
| 1000 |  |  |  |  |
| 1050 |  |  |  |  |

### kan_operator_additive_smooth_semL_operator_replacement_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 2750 |  |  |  |  |
| 2800 |  |  |  |  |
| 2850 |  |  |  |  |
| 2900 |  |  |  |  |
| 2950 |  |  |  |  |

### kan_operator_additive_smooth_spline_kan_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 900 |  |  |  |  |
| 950 |  |  |  |  |
| 1000 |  |  |  |  |
| 1050 |  |  |  |  |
| 1100 |  |  |  |  |

### kan_operator_additive_smooth_semL_operator_replacement_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 2300 |  |  |  |  |
| 2350 |  |  |  |  |
| 2400 |  |  |  |  |
| 2450 |  |  |  |  |
| 2500 |  |  |  |  |

### kan_operator_local_bumps_spline_kan_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 |  |  |  |  |
| 4850 |  |  |  |  |
| 4900 |  |  |  |  |
| 4950 |  |  |  |  |
| 5000 |  |  |  |  |

### kan_operator_local_bumps_semL_operator_replacement_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 |  |  |  |  |
| 4850 |  |  |  |  |
| 4900 |  |  |  |  |
| 4950 |  |  |  |  |
| 5000 |  |  |  |  |

### kan_operator_local_bumps_spline_kan_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 |  |  |  |  |
| 4850 |  |  |  |  |
| 4900 |  |  |  |  |
| 4950 |  |  |  |  |
| 5000 |  |  |  |  |

### kan_operator_local_bumps_semL_operator_replacement_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 |  |  |  |  |
| 4850 |  |  |  |  |
| 4900 |  |  |  |  |
| 4950 |  |  |  |  |
| 5000 |  |  |  |  |

### kan_operator_local_bumps_spline_kan_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 |  |  |  |  |
| 4850 |  |  |  |  |
| 4900 |  |  |  |  |
| 4950 |  |  |  |  |
| 5000 |  |  |  |  |

### kan_operator_local_bumps_semL_operator_replacement_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 |  |  |  |  |
| 4850 |  |  |  |  |
| 4900 |  |  |  |  |
| 4950 |  |  |  |  |
| 5000 |  |  |  |  |

### kan_operator_mixed_composition_spline_kan_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 |  |  |  |  |
| 4850 |  |  |  |  |
| 4900 |  |  |  |  |
| 4950 |  |  |  |  |
| 5000 |  |  |  |  |

### kan_operator_mixed_composition_semL_operator_replacement_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4100 |  |  |  |  |
| 4150 |  |  |  |  |
| 4200 |  |  |  |  |
| 4250 |  |  |  |  |
| 4300 |  |  |  |  |

### kan_operator_mixed_composition_spline_kan_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 |  |  |  |  |
| 4850 |  |  |  |  |
| 4900 |  |  |  |  |
| 4950 |  |  |  |  |
| 5000 |  |  |  |  |

### kan_operator_mixed_composition_semL_operator_replacement_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4100 |  |  |  |  |
| 4150 |  |  |  |  |
| 4200 |  |  |  |  |
| 4250 |  |  |  |  |
| 4300 |  |  |  |  |

### kan_operator_mixed_composition_spline_kan_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 |  |  |  |  |
| 4850 |  |  |  |  |
| 4900 |  |  |  |  |
| 4950 |  |  |  |  |
| 5000 |  |  |  |  |

### kan_operator_mixed_composition_semL_operator_replacement_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4200 |  |  |  |  |
| 4250 |  |  |  |  |
| 4300 |  |  |  |  |
| 4350 |  |  |  |  |
| 4400 |  |  |  |  |

### eml_kan_mlp_symbolic_regression_eml_kan_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 3850 | 0.002507486380636692 |  |  |  |
| 3900 | 0.0022041085176169872 |  |  |  |
| 3950 | 0.003029043786227703 |  |  |  |
| 4000 | 0.0017612749943509698 |  |  |  |
| 4050 | 0.002111723180860281 |  |  |  |

### eml_kan_mlp_symbolic_regression_mlp_same_width_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4350 | 0.020426839590072632 |  |  |  |
| 4400 | 0.023509591817855835 |  |  |  |
| 4450 | 0.023889506235718727 |  |  |  |
| 4500 | 0.022258691489696503 |  |  |  |
| 4550 | 0.020249901339411736 |  |  |  |

### eml_kan_mlp_symbolic_regression_mlp_param_matched_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 3050 | 0.018597381189465523 |  |  |  |
| 3100 | 0.01913418248295784 |  |  |  |
| 3150 | 0.01719062402844429 |  |  |  |
| 3200 | 0.019303184002637863 |  |  |  |
| 3250 | 0.017987672239542007 |  |  |  |

### eml_kan_mlp_symbolic_regression_eml_kan_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4100 | 0.00235185120254755 |  |  |  |
| 4150 | 0.0017040546517819166 |  |  |  |
| 4200 | 0.0021030085626989603 |  |  |  |
| 4250 | 0.002118308562785387 |  |  |  |
| 4300 | 0.002200216520577669 |  |  |  |

### eml_kan_mlp_symbolic_regression_mlp_same_width_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 | 0.019414901733398438 |  |  |  |
| 4850 | 0.019280578941106796 |  |  |  |
| 4900 | 0.019911184906959534 |  |  |  |
| 4950 | 0.020556773990392685 |  |  |  |
| 5000 | 0.018226727843284607 |  |  |  |

### eml_kan_mlp_symbolic_regression_mlp_param_matched_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 3400 | 0.02016715332865715 |  |  |  |
| 3450 | 0.019024252891540527 |  |  |  |
| 3500 | 0.019177276641130447 |  |  |  |
| 3550 | 0.01735461875796318 |  |  |  |
| 3600 | 0.021044159308075905 |  |  |  |

### eml_kan_mlp_symbolic_regression_eml_kan_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4150 | 0.002632774878293276 |  |  |  |
| 4200 | 0.002693034941330552 |  |  |  |
| 4250 | 0.0032095052301883698 |  |  |  |
| 4300 | 0.0021231595892459154 |  |  |  |
| 4350 | 0.001707810559310019 |  |  |  |

### eml_kan_mlp_symbolic_regression_mlp_same_width_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 | 0.020603932440280914 |  |  |  |
| 4850 | 0.020959559828042984 |  |  |  |
| 4900 | 0.019097955897450447 |  |  |  |
| 4950 | 0.02262871339917183 |  |  |  |
| 5000 | 0.021534357219934464 |  |  |  |

### eml_kan_mlp_symbolic_regression_mlp_param_matched_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 2650 | 0.0183548741042614 |  |  |  |
| 2700 | 0.019803304225206375 |  |  |  |
| 2750 | 0.01946919597685337 |  |  |  |
| 2800 | 0.01889410801231861 |  |  |  |
| 2850 | 0.018709374591708183 |  |  |  |

### eml_kan_mlp_localized_regression_eml_kan_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 | 0.03689996525645256 |  |  |  |
| 4850 | 0.04928579181432724 |  |  |  |
| 4900 | 0.01983785256743431 |  |  |  |
| 4950 | 0.0392586849629879 |  |  |  |
| 5000 | 0.05768575891852379 |  |  |  |

### eml_kan_mlp_localized_regression_mlp_same_width_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 | 0.045244473963975906 |  |  |  |
| 4850 | 0.07321307063102722 |  |  |  |
| 4900 | 0.020275937393307686 |  |  |  |
| 4950 | 0.07482622563838959 |  |  |  |
| 5000 | 0.10814527422189713 |  |  |  |

### eml_kan_mlp_localized_regression_mlp_param_matched_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 | 0.03996490687131882 |  |  |  |
| 4850 | 0.05349508672952652 |  |  |  |
| 4900 | 0.020789340138435364 |  |  |  |
| 4950 | 0.060749731957912445 |  |  |  |
| 5000 | 0.07755398750305176 |  |  |  |

### eml_kan_mlp_localized_regression_eml_kan_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 | 0.01636769436299801 |  |  |  |
| 4850 | 0.016615189611911774 |  |  |  |
| 4900 | 0.010159431025385857 |  |  |  |
| 4950 | 0.016581157222390175 |  |  |  |
| 5000 | 0.010782435536384583 |  |  |  |

### eml_kan_mlp_localized_regression_mlp_same_width_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 | 0.08045947551727295 |  |  |  |
| 4850 | 0.09437637776136398 |  |  |  |
| 4900 | 0.03747875988483429 |  |  |  |
| 4950 | 0.10175862163305283 |  |  |  |
| 5000 | 0.03265078365802765 |  |  |  |

### eml_kan_mlp_localized_regression_mlp_param_matched_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 | 0.059581659734249115 |  |  |  |
| 4850 | 0.07864321023225784 |  |  |  |
| 4900 | 0.03651617094874382 |  |  |  |
| 4950 | 0.08105899393558502 |  |  |  |
| 5000 | 0.0351288877427578 |  |  |  |

### eml_kan_mlp_localized_regression_eml_kan_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 | 0.02670905739068985 |  |  |  |
| 4850 | 0.042398933321237564 |  |  |  |
| 4900 | 0.03992340713739395 |  |  |  |
| 4950 | 0.03755402937531471 |  |  |  |
| 5000 | 0.01873033121228218 |  |  |  |

### eml_kan_mlp_localized_regression_mlp_same_width_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 | 0.07103559374809265 |  |  |  |
| 4850 | 0.060146525502204895 |  |  |  |
| 4900 | 0.10713726282119751 |  |  |  |
| 4950 | 0.08520470559597015 |  |  |  |
| 5000 | 0.03975493460893631 |  |  |  |

### eml_kan_mlp_localized_regression_mlp_param_matched_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 4800 | 0.04011169821023941 |  |  |  |
| 4850 | 0.045952074229717255 |  |  |  |
| 4900 | 0.06959228217601776 |  |  |  |
| 4950 | 0.05364307761192322 |  |  |  |
| 5000 | 0.03128372132778168 |  |  |  |

### eml_kan_mlp_shift_classification_eml_kan_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 450 | 0.07506512105464935 |  |  |  |
| 500 | 0.0917583554983139 |  |  |  |
| 550 | 0.12099932879209518 |  |  |  |
| 600 | 0.10587986558675766 |  |  |  |
| 650 | 0.12185569107532501 |  |  |  |

### eml_kan_mlp_shift_classification_mlp_same_width_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1200 | 0.07745309174060822 |  |  |  |
| 1250 | 0.06299763172864914 |  |  |  |
| 1300 | 0.06443610787391663 |  |  |  |
| 1350 | 0.07494357973337173 |  |  |  |
| 1400 | 0.07252384722232819 |  |  |  |

### eml_kan_mlp_shift_classification_mlp_param_matched_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 400 | 0.060691602528095245 |  |  |  |
| 450 | 0.06506450474262238 |  |  |  |
| 500 | 0.07431457191705704 |  |  |  |
| 550 | 0.1165575161576271 |  |  |  |
| 600 | 0.09906929731369019 |  |  |  |

### eml_kan_mlp_shift_classification_eml_kan_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 450 | 0.1396530121564865 |  |  |  |
| 500 | 0.10325945913791656 |  |  |  |
| 550 | 0.08776912838220596 |  |  |  |
| 600 | 0.038123343139886856 |  |  |  |
| 650 | 0.08009971678256989 |  |  |  |

### eml_kan_mlp_shift_classification_mlp_same_width_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 400 | 0.105427086353302 |  |  |  |
| 450 | 0.13698159158229828 |  |  |  |
| 500 | 0.10107249766588211 |  |  |  |
| 550 | 0.09583783894777298 |  |  |  |
| 600 | 0.04389059171080589 |  |  |  |

### eml_kan_mlp_shift_classification_mlp_param_matched_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 700 | 0.07096883654594421 |  |  |  |
| 750 | 0.08305733650922775 |  |  |  |
| 800 | 0.08137533068656921 |  |  |  |
| 850 | 0.0732206329703331 |  |  |  |
| 900 | 0.08086087554693222 |  |  |  |

### eml_kan_mlp_shift_classification_eml_kan_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 400 | 0.10285639762878418 |  |  |  |
| 450 | 0.07932274788618088 |  |  |  |
| 500 | 0.07782729715108871 |  |  |  |
| 550 | 0.09211955219507217 |  |  |  |
| 600 | 0.09196434915065765 |  |  |  |

### eml_kan_mlp_shift_classification_mlp_same_width_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 400 | 0.09059014171361923 |  |  |  |
| 450 | 0.06711505353450775 |  |  |  |
| 500 | 0.08471523970365524 |  |  |  |
| 550 | 0.08551741391420364 |  |  |  |
| 600 | 0.07475699484348297 |  |  |  |

### eml_kan_mlp_shift_classification_mlp_param_matched_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 2100 | 0.10771068930625916 |  |  |  |
| 2150 | 0.04843614622950554 |  |  |  |
| 2200 | 0.07918534427881241 |  |  |  |
| 2250 | 0.08545206487178802 |  |  |  |
| 2300 | 0.07093613594770432 |  |  |  |

### eml_kan_mlp_localized_regression_eml_kan_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 9450 | 0.006484571378678083 |  |  |  |
| 9500 | 0.009946317411959171 |  |  |  |
| 9550 | 0.03349820524454117 |  |  |  |
| 9600 | 0.005899014882743359 |  |  |  |
| 9650 | 0.008377845399081707 |  |  |  |

### eml_kan_mlp_localized_regression_mlp_same_width_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 11450 | 0.0037203803658485413 |  |  |  |
| 11500 | 0.003310202620923519 |  |  |  |
| 11550 | 0.003389519639313221 |  |  |  |
| 11600 | 0.0037923604249954224 |  |  |  |
| 11650 | 0.004148297943174839 |  |  |  |

### eml_kan_mlp_localized_regression_mlp_param_matched_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 9450 | 0.019139474257826805 |  |  |  |
| 9500 | 0.02111375331878662 |  |  |  |
| 9550 | 0.02887178212404251 |  |  |  |
| 9600 | 0.01061328686773777 |  |  |  |
| 9650 | 0.015163149684667587 |  |  |  |

### eml_kan_mlp_localized_regression_eml_kan_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 7400 | 0.005860721692442894 |  |  |  |
| 7450 | 0.0062055448070168495 |  |  |  |
| 7500 | 0.004994899034500122 |  |  |  |
| 7550 | 0.003530280664563179 |  |  |  |
| 7600 | 0.006626038812100887 |  |  |  |

### eml_kan_mlp_localized_regression_mlp_same_width_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 11800 | 0.008821330964565277 |  |  |  |
| 11850 | 0.006913686171174049 |  |  |  |
| 11900 | 0.005833130329847336 |  |  |  |
| 11950 | 0.00658815260976553 |  |  |  |
| 12000 | 0.00728017371147871 |  |  |  |

### eml_kan_mlp_localized_regression_mlp_param_matched_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 9450 | 0.0033294279128313065 |  |  |  |
| 9500 | 0.0043352823704481125 |  |  |  |
| 9550 | 0.0024071368388831615 |  |  |  |
| 9600 | 0.0050450777634978294 |  |  |  |
| 9650 | 0.002967269392684102 |  |  |  |

### eml_kan_mlp_localized_regression_eml_kan_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 9600 | 0.004717834293842316 |  |  |  |
| 9650 | 0.0023108627647161484 |  |  |  |
| 9700 | 0.004278196953237057 |  |  |  |
| 9750 | 0.002503151074051857 |  |  |  |
| 9800 | 0.0025537051260471344 |  |  |  |

### eml_kan_mlp_localized_regression_mlp_same_width_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 11800 | 0.011111035943031311 |  |  |  |
| 11850 | 0.007187731564044952 |  |  |  |
| 11900 | 0.00916634313762188 |  |  |  |
| 11950 | 0.006771525368094444 |  |  |  |
| 12000 | 0.008785336278378963 |  |  |  |

### eml_kan_mlp_localized_regression_mlp_param_matched_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 11800 | 0.004443229176104069 |  |  |  |
| 11850 | 0.005583114456385374 |  |  |  |
| 11900 | 0.00540479552000761 |  |  |  |
| 11950 | 0.004510065540671349 |  |  |  |
| 12000 | 0.004651517141610384 |  |  |  |

### eml_kan_mlp_symbolic_regression_mlp_same_width_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 5900 | 0.018065165728330612 |  |  |  |
| 5950 | 0.018845070153474808 |  |  |  |
| 6000 | 0.020139707252383232 |  |  |  |
| 6050 | 0.018001409247517586 |  |  |  |
| 6100 | 0.019850745797157288 |  |  |  |

### eml_kan_mlp_symbolic_regression_mlp_same_width_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 5100 | 0.022711539641022682 |  |  |  |
| 5150 | 0.01997009664773941 |  |  |  |
| 5200 | 0.020214732736349106 |  |  |  |
| 5250 | 0.020169250667095184 |  |  |  |
| 5300 | 0.02251410111784935 |  |  |  |

### eml_kan_mlp_localized_regression_mlp_same_width_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 14100 | 0.0037667094729840755 |  |  |  |
| 14150 | 0.0035441741347312927 |  |  |  |
| 14200 | 0.004485076293349266 |  |  |  |
| 14250 | 0.003843573620542884 |  |  |  |
| 14300 | 0.003977940417826176 |  |  |  |

### eml_kan_mlp_localized_regression_mlp_same_width_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 14000 | 0.003245810279622674 |  |  |  |
| 14050 | 0.002883214270696044 |  |  |  |
| 14100 | 0.0031219967640936375 |  |  |  |
| 14150 | 0.0025425967760384083 |  |  |  |
| 14200 | 0.0031025914940983057 |  |  |  |

### smoke_image_cnn_eml_baseline

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.6235870122909546 | 0.125 |  | 0.036470890045166016 |
| 2 | 1.6331273317337036 | 0.125 |  | 0.0476229190826416 |
| 3 | 1.5680978298187256 | 0.5 |  | 0.06187701225280762 |
| 4 | 1.622673749923706 | 0.25 |  | 0.07350397109985352 |

### smoke_image_efficient_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.5968564748764038 | 0.375 |  | 0.024529695510864258 |
| 2 | 1.696264624595642 | 0.0 |  | 0.04795074462890625 |
| 3 | 1.5503305196762085 | 0.375 |  | 0.07147860527038574 |
| 4 | 1.7369440793991089 | 0.0 |  | 0.09427666664123535 |

### smoke_text_efficient_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.430697917938232 |  | 0.021008403971791267 | 0.013514995574951172 |
| 2 | 4.411949157714844 |  | 0.009900989942252636 | 0.024879932403564453 |
| 3 | 4.394505977630615 |  | 0.04128440469503403 | 0.036360979080200195 |
| 4 | 4.389736175537109 |  | 0.03431372717022896 | 0.047914981842041016 |

### probe_gate_compat_sigmoid_update

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0009140968322753906 |

### probe_responsibility_no_null_precision

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0008790493011474609 |

### probe_responsibility_with_null_precision

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0009679794311523438 |

### probe_thresholded_null

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0005881786346435547 |


## 10. Efficiency Analysis

- Runtime and throughput are available in per-run summaries and the efficiency table.
- Local-window cost and attractor count are recorded when model diagnostics expose them.
- Short smoke runs are not enough to decide whether accuracy gain justifies added cost.

## 11. Failure Modes

- gate collapse: MISSING unless gate diagnostics are emitted.
- all-null collapse: inspect `null_weight_mean`; high values indicate risk.
- never-null collapse: inspect `null_weight_mean`; near zero indicates risk.
- energy explosion: inspect `energy_mean/std` and NaN/Inf counts.
- resistance collapse: inspect `resistance_mean/std`.
- attractor collapse: inspect `attractor_diversity`.
- update gate too high at init: inspect `update_gate_mean`.
- poor causal text behavior: no-leak tests exist; training report includes only available run metrics.
- slow local-window implementation: compare seconds and throughput.

## 12. Conclusions

- Current evidence remains preliminary when runs are short or single-seed.
- Use the synthetic image and text ablation tables to identify mechanisms worth longer training before making CIFAR claims.
- The exact next experiment is repeat-seed CIFAR validation for the best efficient image setting and the strongest CNN baseline.

## 13. Raw Artifacts

- smoke_image_cnn_eml_baseline: reports/runs/20260424_063607_smoke_image_cnn_eml_baseline
  - history: reports/runs/20260424_063607_smoke_image_cnn_eml_baseline/history.json
  - metrics: reports/runs/20260424_063607_smoke_image_cnn_eml_baseline/metrics.csv
  - diagnostics: reports/runs/20260424_063607_smoke_image_cnn_eml_baseline/diagnostics.csv
  - summary: reports/runs/20260424_063607_smoke_image_cnn_eml_baseline/summary.json
- smoke_image_efficient_eml: reports/runs/20260424_063608_smoke_image_efficient_eml
  - history: reports/runs/20260424_063608_smoke_image_efficient_eml/history.json
  - metrics: reports/runs/20260424_063608_smoke_image_efficient_eml/metrics.csv
  - diagnostics: reports/runs/20260424_063608_smoke_image_efficient_eml/diagnostics.csv
  - summary: reports/runs/20260424_063608_smoke_image_efficient_eml/summary.json
- smoke_text_efficient_eml: reports/runs/20260424_063608_smoke_text_efficient_eml
  - history: reports/runs/20260424_063608_smoke_text_efficient_eml/history.json
  - metrics: reports/runs/20260424_063608_smoke_text_efficient_eml/metrics.csv
  - diagnostics: reports/runs/20260424_063608_smoke_text_efficient_eml/diagnostics.csv
  - summary: reports/runs/20260424_063608_smoke_text_efficient_eml/summary.json
- probe_gate_compat_sigmoid_update: reports/runs/20260424_063608_probe_gate_compat_sigmoid_update
  - history: reports/runs/20260424_063608_probe_gate_compat_sigmoid_update/history.json
  - metrics: reports/runs/20260424_063608_probe_gate_compat_sigmoid_update/metrics.csv
  - diagnostics: reports/runs/20260424_063608_probe_gate_compat_sigmoid_update/diagnostics.csv
  - summary: reports/runs/20260424_063608_probe_gate_compat_sigmoid_update/summary.json
- probe_responsibility_no_null_precision: reports/runs/20260424_063608_probe_responsibility_no_null_precision
  - history: reports/runs/20260424_063608_probe_responsibility_no_null_precision/history.json
  - metrics: reports/runs/20260424_063608_probe_responsibility_no_null_precision/metrics.csv
  - diagnostics: reports/runs/20260424_063608_probe_responsibility_no_null_precision/diagnostics.csv
  - summary: reports/runs/20260424_063608_probe_responsibility_no_null_precision/summary.json
- probe_responsibility_with_null_precision: reports/runs/20260424_063608_probe_responsibility_with_null_precision
  - history: reports/runs/20260424_063608_probe_responsibility_with_null_precision/history.json
  - metrics: reports/runs/20260424_063608_probe_responsibility_with_null_precision/metrics.csv
  - diagnostics: reports/runs/20260424_063608_probe_responsibility_with_null_precision/diagnostics.csv
  - summary: reports/runs/20260424_063608_probe_responsibility_with_null_precision/summary.json
- local_conv_baseline: reports/runs/20260424_063608_local_conv_baseline
  - history: reports/runs/20260424_063608_local_conv_baseline/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260424_063608_local_conv_baseline/summary.json
- local_text_linear_baseline: reports/runs/20260424_063608_local_text_linear_baseline
  - history: reports/runs/20260424_063608_local_text_linear_baseline/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260424_063608_local_text_linear_baseline/summary.json
- cifar_medium_suite: reports/runs/20260424_063608_cifar_medium_suite
  - history: reports/runs/20260424_063608_cifar_medium_suite/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260424_063608_cifar_medium_suite/summary.json
- text_medium_suite: reports/runs/20260424_063608_text_medium_suite
  - history: reports/runs/20260424_063608_text_medium_suite/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260424_063608_text_medium_suite/summary.json
- full_seeded_ablation: reports/runs/20260424_063608_full_seeded_ablation
  - history: reports/runs/20260424_063608_full_seeded_ablation/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260424_063608_full_seeded_ablation/summary.json
- kan_paper_reference: reports/runs/20260505_123105_kan_paper_reference
  - history: reports/runs/20260505_123105_kan_paper_reference/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260505_123105_kan_paper_reference/summary.json
- kan_compare_image_cnn_eml: reports/runs/20260505_123106_kan_compare_image_cnn_eml
  - history: reports/runs/20260505_123106_kan_compare_image_cnn_eml/history.json
  - metrics: reports/runs/20260505_123106_kan_compare_image_cnn_eml/metrics.csv
  - diagnostics: reports/runs/20260505_123106_kan_compare_image_cnn_eml/diagnostics.csv
  - summary: reports/runs/20260505_123106_kan_compare_image_cnn_eml/summary.json
- kan_compare_image_efficient_eml: reports/runs/20260505_123106_kan_compare_image_efficient_eml
  - history: reports/runs/20260505_123106_kan_compare_image_efficient_eml/history.json
  - metrics: reports/runs/20260505_123106_kan_compare_image_efficient_eml/metrics.csv
  - diagnostics: reports/runs/20260505_123106_kan_compare_image_efficient_eml/diagnostics.csv
  - summary: reports/runs/20260505_123106_kan_compare_image_efficient_eml/summary.json
- kan_compare_image_eml_edge: reports/runs/20260505_123107_kan_compare_image_eml_edge
  - history: reports/runs/20260505_123107_kan_compare_image_eml_edge/history.json
  - metrics: reports/runs/20260505_123107_kan_compare_image_eml_edge/metrics.csv
  - diagnostics: reports/runs/20260505_123107_kan_compare_image_eml_edge/diagnostics.csv
  - summary: reports/runs/20260505_123107_kan_compare_image_eml_edge/summary.json
- kan_compare_text_local_conv: reports/runs/20260505_123107_kan_compare_text_local_conv
  - history: reports/runs/20260505_123107_kan_compare_text_local_conv/history.json
  - metrics: reports/runs/20260505_123107_kan_compare_text_local_conv/metrics.csv
  - diagnostics: reports/runs/20260505_123107_kan_compare_text_local_conv/diagnostics.csv
  - summary: reports/runs/20260505_123107_kan_compare_text_local_conv/summary.json
- kan_compare_text_small_gru: reports/runs/20260505_123107_kan_compare_text_small_gru
  - history: reports/runs/20260505_123107_kan_compare_text_small_gru/history.json
  - metrics: reports/runs/20260505_123107_kan_compare_text_small_gru/metrics.csv
  - diagnostics: reports/runs/20260505_123107_kan_compare_text_small_gru/diagnostics.csv
  - summary: reports/runs/20260505_123107_kan_compare_text_small_gru/summary.json
- kan_compare_text_efficient_eml: reports/runs/20260505_123107_kan_compare_text_efficient_eml
  - history: reports/runs/20260505_123107_kan_compare_text_efficient_eml/history.json
  - metrics: reports/runs/20260505_123107_kan_compare_text_efficient_eml/metrics.csv
  - diagnostics: reports/runs/20260505_123107_kan_compare_text_efficient_eml/diagnostics.csv
  - summary: reports/runs/20260505_123107_kan_compare_text_efficient_eml/summary.json
- kan_compare_text_eml_edge: reports/runs/20260505_123107_kan_compare_text_eml_edge
  - history: reports/runs/20260505_123107_kan_compare_text_eml_edge/history.json
  - metrics: reports/runs/20260505_123107_kan_compare_text_eml_edge/metrics.csv
  - diagnostics: reports/runs/20260505_123107_kan_compare_text_eml_edge/diagnostics.csv
  - summary: reports/runs/20260505_123107_kan_compare_text_eml_edge/summary.json
- kan_paper_reference: reports/runs/20260505_123136_kan_paper_reference
  - history: reports/runs/20260505_123136_kan_paper_reference/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260505_123136_kan_paper_reference/summary.json
- kan_compare_image_cnn_eml: reports/runs/20260505_123136_kan_compare_image_cnn_eml
  - history: reports/runs/20260505_123136_kan_compare_image_cnn_eml/history.json
  - metrics: reports/runs/20260505_123136_kan_compare_image_cnn_eml/metrics.csv
  - diagnostics: reports/runs/20260505_123136_kan_compare_image_cnn_eml/diagnostics.csv
  - summary: reports/runs/20260505_123136_kan_compare_image_cnn_eml/summary.json
- kan_compare_image_efficient_eml: reports/runs/20260505_123137_kan_compare_image_efficient_eml
  - history: reports/runs/20260505_123137_kan_compare_image_efficient_eml/history.json
  - metrics: reports/runs/20260505_123137_kan_compare_image_efficient_eml/metrics.csv
  - diagnostics: reports/runs/20260505_123137_kan_compare_image_efficient_eml/diagnostics.csv
  - summary: reports/runs/20260505_123137_kan_compare_image_efficient_eml/summary.json
- kan_compare_image_eml_edge: reports/runs/20260505_123138_kan_compare_image_eml_edge
  - history: reports/runs/20260505_123138_kan_compare_image_eml_edge/history.json
  - metrics: reports/runs/20260505_123138_kan_compare_image_eml_edge/metrics.csv
  - diagnostics: reports/runs/20260505_123138_kan_compare_image_eml_edge/diagnostics.csv
  - summary: reports/runs/20260505_123138_kan_compare_image_eml_edge/summary.json
- kan_compare_text_local_conv: reports/runs/20260505_123139_kan_compare_text_local_conv
  - history: reports/runs/20260505_123139_kan_compare_text_local_conv/history.json
  - metrics: reports/runs/20260505_123139_kan_compare_text_local_conv/metrics.csv
  - diagnostics: reports/runs/20260505_123139_kan_compare_text_local_conv/diagnostics.csv
  - summary: reports/runs/20260505_123139_kan_compare_text_local_conv/summary.json
- kan_compare_text_small_gru: reports/runs/20260505_123139_kan_compare_text_small_gru
  - history: reports/runs/20260505_123139_kan_compare_text_small_gru/history.json
  - metrics: reports/runs/20260505_123139_kan_compare_text_small_gru/metrics.csv
  - diagnostics: reports/runs/20260505_123139_kan_compare_text_small_gru/diagnostics.csv
  - summary: reports/runs/20260505_123139_kan_compare_text_small_gru/summary.json
- kan_compare_text_efficient_eml: reports/runs/20260505_123139_kan_compare_text_efficient_eml
  - history: reports/runs/20260505_123139_kan_compare_text_efficient_eml/history.json
  - metrics: reports/runs/20260505_123139_kan_compare_text_efficient_eml/metrics.csv
  - diagnostics: reports/runs/20260505_123139_kan_compare_text_efficient_eml/diagnostics.csv
  - summary: reports/runs/20260505_123139_kan_compare_text_efficient_eml/summary.json
- kan_compare_text_eml_edge: reports/runs/20260505_123140_kan_compare_text_eml_edge
  - history: reports/runs/20260505_123140_kan_compare_text_eml_edge/history.json
  - metrics: reports/runs/20260505_123140_kan_compare_text_eml_edge/metrics.csv
  - diagnostics: reports/runs/20260505_123140_kan_compare_text_eml_edge/diagnostics.csv
  - summary: reports/runs/20260505_123140_kan_compare_text_eml_edge/summary.json
- kan_paper_reference: reports/runs/20260505_123159_kan_paper_reference
  - history: reports/runs/20260505_123159_kan_paper_reference/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260505_123159_kan_paper_reference/summary.json
- kan_compare_image_cnn_eml: reports/runs/20260505_123200_kan_compare_image_cnn_eml
  - history: reports/runs/20260505_123200_kan_compare_image_cnn_eml/history.json
  - metrics: reports/runs/20260505_123200_kan_compare_image_cnn_eml/metrics.csv
  - diagnostics: reports/runs/20260505_123200_kan_compare_image_cnn_eml/diagnostics.csv
  - summary: reports/runs/20260505_123200_kan_compare_image_cnn_eml/summary.json
- kan_compare_image_efficient_eml: reports/runs/20260505_123200_kan_compare_image_efficient_eml
  - history: reports/runs/20260505_123200_kan_compare_image_efficient_eml/history.json
  - metrics: reports/runs/20260505_123200_kan_compare_image_efficient_eml/metrics.csv
  - diagnostics: reports/runs/20260505_123200_kan_compare_image_efficient_eml/diagnostics.csv
  - summary: reports/runs/20260505_123200_kan_compare_image_efficient_eml/summary.json
- kan_compare_image_eml_edge: reports/runs/20260505_123201_kan_compare_image_eml_edge
  - history: reports/runs/20260505_123201_kan_compare_image_eml_edge/history.json
  - metrics: reports/runs/20260505_123201_kan_compare_image_eml_edge/metrics.csv
  - diagnostics: reports/runs/20260505_123201_kan_compare_image_eml_edge/diagnostics.csv
  - summary: reports/runs/20260505_123201_kan_compare_image_eml_edge/summary.json
- kan_compare_text_local_conv: reports/runs/20260505_123201_kan_compare_text_local_conv
  - history: reports/runs/20260505_123201_kan_compare_text_local_conv/history.json
  - metrics: reports/runs/20260505_123201_kan_compare_text_local_conv/metrics.csv
  - diagnostics: reports/runs/20260505_123201_kan_compare_text_local_conv/diagnostics.csv
  - summary: reports/runs/20260505_123201_kan_compare_text_local_conv/summary.json
- kan_compare_text_small_gru: reports/runs/20260505_123201_kan_compare_text_small_gru
  - history: reports/runs/20260505_123201_kan_compare_text_small_gru/history.json
  - metrics: reports/runs/20260505_123201_kan_compare_text_small_gru/metrics.csv
  - diagnostics: reports/runs/20260505_123201_kan_compare_text_small_gru/diagnostics.csv
  - summary: reports/runs/20260505_123201_kan_compare_text_small_gru/summary.json
- kan_compare_text_efficient_eml: reports/runs/20260505_123203_kan_compare_text_efficient_eml
  - history: reports/runs/20260505_123203_kan_compare_text_efficient_eml/history.json
  - metrics: reports/runs/20260505_123203_kan_compare_text_efficient_eml/metrics.csv
  - diagnostics: reports/runs/20260505_123203_kan_compare_text_efficient_eml/diagnostics.csv
  - summary: reports/runs/20260505_123203_kan_compare_text_efficient_eml/summary.json
- kan_compare_text_eml_edge: reports/runs/20260505_123206_kan_compare_text_eml_edge
  - history: reports/runs/20260505_123206_kan_compare_text_eml_edge/history.json
  - metrics: reports/runs/20260505_123206_kan_compare_text_eml_edge/metrics.csv
  - diagnostics: reports/runs/20260505_123206_kan_compare_text_eml_edge/diagnostics.csv
  - summary: reports/runs/20260505_123206_kan_compare_text_eml_edge/summary.json
- kan_paper_reference: reports/runs/20260505_123237_kan_paper_reference
  - history: reports/runs/20260505_123237_kan_paper_reference/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260505_123237_kan_paper_reference/summary.json
- kan_compare_image_cnn_eml: reports/runs/20260505_123238_kan_compare_image_cnn_eml
  - history: reports/runs/20260505_123238_kan_compare_image_cnn_eml/history.json
  - metrics: reports/runs/20260505_123238_kan_compare_image_cnn_eml/metrics.csv
  - diagnostics: reports/runs/20260505_123238_kan_compare_image_cnn_eml/diagnostics.csv
  - summary: reports/runs/20260505_123238_kan_compare_image_cnn_eml/summary.json
- kan_compare_image_efficient_eml: reports/runs/20260505_123238_kan_compare_image_efficient_eml
  - history: reports/runs/20260505_123238_kan_compare_image_efficient_eml/history.json
  - metrics: reports/runs/20260505_123238_kan_compare_image_efficient_eml/metrics.csv
  - diagnostics: reports/runs/20260505_123238_kan_compare_image_efficient_eml/diagnostics.csv
  - summary: reports/runs/20260505_123238_kan_compare_image_efficient_eml/summary.json
- kan_compare_image_eml_edge: reports/runs/20260505_123239_kan_compare_image_eml_edge
  - history: reports/runs/20260505_123239_kan_compare_image_eml_edge/history.json
  - metrics: reports/runs/20260505_123239_kan_compare_image_eml_edge/metrics.csv
  - diagnostics: reports/runs/20260505_123239_kan_compare_image_eml_edge/diagnostics.csv
  - summary: reports/runs/20260505_123239_kan_compare_image_eml_edge/summary.json
- kan_compare_text_local_conv: reports/runs/20260505_123239_kan_compare_text_local_conv
  - history: reports/runs/20260505_123239_kan_compare_text_local_conv/history.json
  - metrics: reports/runs/20260505_123239_kan_compare_text_local_conv/metrics.csv
  - diagnostics: reports/runs/20260505_123239_kan_compare_text_local_conv/diagnostics.csv
  - summary: reports/runs/20260505_123239_kan_compare_text_local_conv/summary.json
- kan_compare_text_small_gru: reports/runs/20260505_123239_kan_compare_text_small_gru
  - history: reports/runs/20260505_123239_kan_compare_text_small_gru/history.json
  - metrics: reports/runs/20260505_123239_kan_compare_text_small_gru/metrics.csv
  - diagnostics: reports/runs/20260505_123239_kan_compare_text_small_gru/diagnostics.csv
  - summary: reports/runs/20260505_123239_kan_compare_text_small_gru/summary.json
- kan_compare_text_efficient_eml: reports/runs/20260505_123240_kan_compare_text_efficient_eml
  - history: reports/runs/20260505_123240_kan_compare_text_efficient_eml/history.json
  - metrics: reports/runs/20260505_123240_kan_compare_text_efficient_eml/metrics.csv
  - diagnostics: reports/runs/20260505_123240_kan_compare_text_efficient_eml/diagnostics.csv
  - summary: reports/runs/20260505_123240_kan_compare_text_efficient_eml/summary.json
- kan_compare_text_eml_edge: reports/runs/20260505_123242_kan_compare_text_eml_edge
  - history: reports/runs/20260505_123242_kan_compare_text_eml_edge/history.json
  - metrics: reports/runs/20260505_123242_kan_compare_text_eml_edge/metrics.csv
  - diagnostics: reports/runs/20260505_123242_kan_compare_text_eml_edge/diagnostics.csv
  - summary: reports/runs/20260505_123242_kan_compare_text_eml_edge/summary.json
- smoke_image_cnn_eml_baseline: reports/runs/20260505_123422_smoke_image_cnn_eml_baseline
  - history: reports/runs/20260505_123422_smoke_image_cnn_eml_baseline/history.json
  - metrics: reports/runs/20260505_123422_smoke_image_cnn_eml_baseline/metrics.csv
  - diagnostics: reports/runs/20260505_123422_smoke_image_cnn_eml_baseline/diagnostics.csv
  - summary: reports/runs/20260505_123422_smoke_image_cnn_eml_baseline/summary.json
- smoke_image_efficient_eml: reports/runs/20260505_123425_smoke_image_efficient_eml
  - history: reports/runs/20260505_123425_smoke_image_efficient_eml/history.json
  - metrics: reports/runs/20260505_123425_smoke_image_efficient_eml/metrics.csv
  - diagnostics: reports/runs/20260505_123425_smoke_image_efficient_eml/diagnostics.csv
  - summary: reports/runs/20260505_123425_smoke_image_efficient_eml/summary.json
- smoke_text_efficient_eml: reports/runs/20260505_123425_smoke_text_efficient_eml
  - history: reports/runs/20260505_123425_smoke_text_efficient_eml/history.json
  - metrics: reports/runs/20260505_123425_smoke_text_efficient_eml/metrics.csv
  - diagnostics: reports/runs/20260505_123425_smoke_text_efficient_eml/diagnostics.csv
  - summary: reports/runs/20260505_123425_smoke_text_efficient_eml/summary.json
- probe_gate_compat_sigmoid_update: reports/runs/20260505_123425_probe_gate_compat_sigmoid_update
  - history: reports/runs/20260505_123425_probe_gate_compat_sigmoid_update/history.json
  - metrics: reports/runs/20260505_123425_probe_gate_compat_sigmoid_update/metrics.csv
  - diagnostics: reports/runs/20260505_123425_probe_gate_compat_sigmoid_update/diagnostics.csv
  - summary: reports/runs/20260505_123425_probe_gate_compat_sigmoid_update/summary.json
- probe_responsibility_no_null_precision: reports/runs/20260505_123425_probe_responsibility_no_null_precision
  - history: reports/runs/20260505_123425_probe_responsibility_no_null_precision/history.json
  - metrics: reports/runs/20260505_123425_probe_responsibility_no_null_precision/metrics.csv
  - diagnostics: reports/runs/20260505_123425_probe_responsibility_no_null_precision/diagnostics.csv
  - summary: reports/runs/20260505_123425_probe_responsibility_no_null_precision/summary.json
- probe_responsibility_with_null_precision: reports/runs/20260505_123425_probe_responsibility_with_null_precision
  - history: reports/runs/20260505_123425_probe_responsibility_with_null_precision/history.json
  - metrics: reports/runs/20260505_123425_probe_responsibility_with_null_precision/metrics.csv
  - diagnostics: reports/runs/20260505_123425_probe_responsibility_with_null_precision/diagnostics.csv
  - summary: reports/runs/20260505_123425_probe_responsibility_with_null_precision/summary.json
- probe_thresholded_null: reports/runs/20260505_123425_probe_thresholded_null
  - history: reports/runs/20260505_123425_probe_thresholded_null/history.json
  - metrics: reports/runs/20260505_123425_probe_thresholded_null/metrics.csv
  - diagnostics: reports/runs/20260505_123425_probe_thresholded_null/diagnostics.csv
  - summary: reports/runs/20260505_123425_probe_thresholded_null/summary.json
- local_conv_baseline: reports/runs/20260505_123425_local_conv_baseline
  - history: reports/runs/20260505_123425_local_conv_baseline/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260505_123425_local_conv_baseline/summary.json
- local_text_linear_baseline: reports/runs/20260505_123425_local_text_linear_baseline
  - history: reports/runs/20260505_123425_local_text_linear_baseline/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260505_123425_local_text_linear_baseline/summary.json
- cifar_medium_suite: reports/runs/20260505_123425_cifar_medium_suite
  - history: reports/runs/20260505_123425_cifar_medium_suite/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260505_123425_cifar_medium_suite/summary.json
- text_medium_suite: reports/runs/20260505_123425_text_medium_suite
  - history: reports/runs/20260505_123425_text_medium_suite/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260505_123425_text_medium_suite/summary.json
- full_seeded_ablation: reports/runs/20260505_123425_full_seeded_ablation
  - history: reports/runs/20260505_123425_full_seeded_ablation/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260505_123425_full_seeded_ablation/summary.json
- kan_operator_additive_smooth_spline_kan_seed0: reports/runs/20260506_013633_kan_operator_additive_smooth_spline_kan_seed0
  - history: reports/runs/20260506_013633_kan_operator_additive_smooth_spline_kan_seed0/history.json
  - metrics: reports/runs/20260506_013633_kan_operator_additive_smooth_spline_kan_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_013633_kan_operator_additive_smooth_spline_kan_seed0/diagnostics.csv
  - summary: reports/runs/20260506_013633_kan_operator_additive_smooth_spline_kan_seed0/summary.json
- kan_operator_additive_smooth_semL_operator_replacement_seed0: reports/runs/20260506_013634_kan_operator_additive_smooth_semL_operator_replacement_seed0
  - history: reports/runs/20260506_013634_kan_operator_additive_smooth_semL_operator_replacement_seed0/history.json
  - metrics: reports/runs/20260506_013634_kan_operator_additive_smooth_semL_operator_replacement_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_013634_kan_operator_additive_smooth_semL_operator_replacement_seed0/diagnostics.csv
  - summary: reports/runs/20260506_013634_kan_operator_additive_smooth_semL_operator_replacement_seed0/summary.json
- kan_operator_additive_smooth_spline_kan_seed0: reports/runs/20260506_013705_kan_operator_additive_smooth_spline_kan_seed0
  - history: reports/runs/20260506_013705_kan_operator_additive_smooth_spline_kan_seed0/history.json
  - metrics: reports/runs/20260506_013705_kan_operator_additive_smooth_spline_kan_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_013705_kan_operator_additive_smooth_spline_kan_seed0/diagnostics.csv
  - summary: reports/runs/20260506_013705_kan_operator_additive_smooth_spline_kan_seed0/summary.json
- kan_operator_additive_smooth_semL_operator_replacement_seed0: reports/runs/20260506_013710_kan_operator_additive_smooth_semL_operator_replacement_seed0
  - history: reports/runs/20260506_013710_kan_operator_additive_smooth_semL_operator_replacement_seed0/history.json
  - metrics: reports/runs/20260506_013710_kan_operator_additive_smooth_semL_operator_replacement_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_013710_kan_operator_additive_smooth_semL_operator_replacement_seed0/diagnostics.csv
  - summary: reports/runs/20260506_013710_kan_operator_additive_smooth_semL_operator_replacement_seed0/summary.json
- kan_operator_additive_smooth_spline_kan_seed1: reports/runs/20260506_013714_kan_operator_additive_smooth_spline_kan_seed1
  - history: reports/runs/20260506_013714_kan_operator_additive_smooth_spline_kan_seed1/history.json
  - metrics: reports/runs/20260506_013714_kan_operator_additive_smooth_spline_kan_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_013714_kan_operator_additive_smooth_spline_kan_seed1/diagnostics.csv
  - summary: reports/runs/20260506_013714_kan_operator_additive_smooth_spline_kan_seed1/summary.json
- kan_operator_additive_smooth_semL_operator_replacement_seed1: reports/runs/20260506_013717_kan_operator_additive_smooth_semL_operator_replacement_seed1
  - history: reports/runs/20260506_013717_kan_operator_additive_smooth_semL_operator_replacement_seed1/history.json
  - metrics: reports/runs/20260506_013717_kan_operator_additive_smooth_semL_operator_replacement_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_013717_kan_operator_additive_smooth_semL_operator_replacement_seed1/diagnostics.csv
  - summary: reports/runs/20260506_013717_kan_operator_additive_smooth_semL_operator_replacement_seed1/summary.json
- kan_operator_local_bumps_spline_kan_seed0: reports/runs/20260506_013721_kan_operator_local_bumps_spline_kan_seed0
  - history: reports/runs/20260506_013721_kan_operator_local_bumps_spline_kan_seed0/history.json
  - metrics: reports/runs/20260506_013721_kan_operator_local_bumps_spline_kan_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_013721_kan_operator_local_bumps_spline_kan_seed0/diagnostics.csv
  - summary: reports/runs/20260506_013721_kan_operator_local_bumps_spline_kan_seed0/summary.json
- kan_operator_local_bumps_semL_operator_replacement_seed0: reports/runs/20260506_013723_kan_operator_local_bumps_semL_operator_replacement_seed0
  - history: reports/runs/20260506_013723_kan_operator_local_bumps_semL_operator_replacement_seed0/history.json
  - metrics: reports/runs/20260506_013723_kan_operator_local_bumps_semL_operator_replacement_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_013723_kan_operator_local_bumps_semL_operator_replacement_seed0/diagnostics.csv
  - summary: reports/runs/20260506_013723_kan_operator_local_bumps_semL_operator_replacement_seed0/summary.json
- kan_operator_local_bumps_spline_kan_seed1: reports/runs/20260506_013728_kan_operator_local_bumps_spline_kan_seed1
  - history: reports/runs/20260506_013728_kan_operator_local_bumps_spline_kan_seed1/history.json
  - metrics: reports/runs/20260506_013728_kan_operator_local_bumps_spline_kan_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_013728_kan_operator_local_bumps_spline_kan_seed1/diagnostics.csv
  - summary: reports/runs/20260506_013728_kan_operator_local_bumps_spline_kan_seed1/summary.json
- kan_operator_local_bumps_semL_operator_replacement_seed1: reports/runs/20260506_013731_kan_operator_local_bumps_semL_operator_replacement_seed1
  - history: reports/runs/20260506_013731_kan_operator_local_bumps_semL_operator_replacement_seed1/history.json
  - metrics: reports/runs/20260506_013731_kan_operator_local_bumps_semL_operator_replacement_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_013731_kan_operator_local_bumps_semL_operator_replacement_seed1/diagnostics.csv
  - summary: reports/runs/20260506_013731_kan_operator_local_bumps_semL_operator_replacement_seed1/summary.json
- kan_operator_mixed_composition_spline_kan_seed0: reports/runs/20260506_013736_kan_operator_mixed_composition_spline_kan_seed0
  - history: reports/runs/20260506_013736_kan_operator_mixed_composition_spline_kan_seed0/history.json
  - metrics: reports/runs/20260506_013736_kan_operator_mixed_composition_spline_kan_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_013736_kan_operator_mixed_composition_spline_kan_seed0/diagnostics.csv
  - summary: reports/runs/20260506_013736_kan_operator_mixed_composition_spline_kan_seed0/summary.json
- kan_operator_mixed_composition_semL_operator_replacement_seed0: reports/runs/20260506_013738_kan_operator_mixed_composition_semL_operator_replacement_seed0
  - history: reports/runs/20260506_013738_kan_operator_mixed_composition_semL_operator_replacement_seed0/history.json
  - metrics: reports/runs/20260506_013738_kan_operator_mixed_composition_semL_operator_replacement_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_013738_kan_operator_mixed_composition_semL_operator_replacement_seed0/diagnostics.csv
  - summary: reports/runs/20260506_013738_kan_operator_mixed_composition_semL_operator_replacement_seed0/summary.json
- kan_operator_mixed_composition_spline_kan_seed1: reports/runs/20260506_013743_kan_operator_mixed_composition_spline_kan_seed1
  - history: reports/runs/20260506_013743_kan_operator_mixed_composition_spline_kan_seed1/history.json
  - metrics: reports/runs/20260506_013743_kan_operator_mixed_composition_spline_kan_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_013743_kan_operator_mixed_composition_spline_kan_seed1/diagnostics.csv
  - summary: reports/runs/20260506_013743_kan_operator_mixed_composition_spline_kan_seed1/summary.json
- kan_operator_mixed_composition_semL_operator_replacement_seed1: reports/runs/20260506_013746_kan_operator_mixed_composition_semL_operator_replacement_seed1
  - history: reports/runs/20260506_013746_kan_operator_mixed_composition_semL_operator_replacement_seed1/history.json
  - metrics: reports/runs/20260506_013746_kan_operator_mixed_composition_semL_operator_replacement_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_013746_kan_operator_mixed_composition_semL_operator_replacement_seed1/diagnostics.csv
  - summary: reports/runs/20260506_013746_kan_operator_mixed_composition_semL_operator_replacement_seed1/summary.json
- kan_operator_additive_smooth_spline_kan_seed0: reports/runs/20260506_014236_kan_operator_additive_smooth_spline_kan_seed0
  - history: reports/runs/20260506_014236_kan_operator_additive_smooth_spline_kan_seed0/history.json
  - metrics: reports/runs/20260506_014236_kan_operator_additive_smooth_spline_kan_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_014236_kan_operator_additive_smooth_spline_kan_seed0/diagnostics.csv
  - summary: reports/runs/20260506_014236_kan_operator_additive_smooth_spline_kan_seed0/summary.json
- kan_operator_additive_smooth_semL_operator_replacement_seed0: reports/runs/20260506_014257_kan_operator_additive_smooth_semL_operator_replacement_seed0
  - history: reports/runs/20260506_014257_kan_operator_additive_smooth_semL_operator_replacement_seed0/history.json
  - metrics: reports/runs/20260506_014257_kan_operator_additive_smooth_semL_operator_replacement_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_014257_kan_operator_additive_smooth_semL_operator_replacement_seed0/diagnostics.csv
  - summary: reports/runs/20260506_014257_kan_operator_additive_smooth_semL_operator_replacement_seed0/summary.json
- kan_operator_additive_smooth_spline_kan_seed1: reports/runs/20260506_014322_kan_operator_additive_smooth_spline_kan_seed1
  - history: reports/runs/20260506_014322_kan_operator_additive_smooth_spline_kan_seed1/history.json
  - metrics: reports/runs/20260506_014322_kan_operator_additive_smooth_spline_kan_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_014322_kan_operator_additive_smooth_spline_kan_seed1/diagnostics.csv
  - summary: reports/runs/20260506_014322_kan_operator_additive_smooth_spline_kan_seed1/summary.json
- kan_operator_additive_smooth_semL_operator_replacement_seed1: reports/runs/20260506_014326_kan_operator_additive_smooth_semL_operator_replacement_seed1
  - history: reports/runs/20260506_014326_kan_operator_additive_smooth_semL_operator_replacement_seed1/history.json
  - metrics: reports/runs/20260506_014326_kan_operator_additive_smooth_semL_operator_replacement_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_014326_kan_operator_additive_smooth_semL_operator_replacement_seed1/diagnostics.csv
  - summary: reports/runs/20260506_014326_kan_operator_additive_smooth_semL_operator_replacement_seed1/summary.json
- kan_operator_additive_smooth_spline_kan_seed2: reports/runs/20260506_014354_kan_operator_additive_smooth_spline_kan_seed2
  - history: reports/runs/20260506_014354_kan_operator_additive_smooth_spline_kan_seed2/history.json
  - metrics: reports/runs/20260506_014354_kan_operator_additive_smooth_spline_kan_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_014354_kan_operator_additive_smooth_spline_kan_seed2/diagnostics.csv
  - summary: reports/runs/20260506_014354_kan_operator_additive_smooth_spline_kan_seed2/summary.json
- kan_operator_additive_smooth_semL_operator_replacement_seed2: reports/runs/20260506_014359_kan_operator_additive_smooth_semL_operator_replacement_seed2
  - history: reports/runs/20260506_014359_kan_operator_additive_smooth_semL_operator_replacement_seed2/history.json
  - metrics: reports/runs/20260506_014359_kan_operator_additive_smooth_semL_operator_replacement_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_014359_kan_operator_additive_smooth_semL_operator_replacement_seed2/diagnostics.csv
  - summary: reports/runs/20260506_014359_kan_operator_additive_smooth_semL_operator_replacement_seed2/summary.json
- kan_operator_local_bumps_spline_kan_seed0: reports/runs/20260506_014421_kan_operator_local_bumps_spline_kan_seed0
  - history: reports/runs/20260506_014421_kan_operator_local_bumps_spline_kan_seed0/history.json
  - metrics: reports/runs/20260506_014421_kan_operator_local_bumps_spline_kan_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_014421_kan_operator_local_bumps_spline_kan_seed0/diagnostics.csv
  - summary: reports/runs/20260506_014421_kan_operator_local_bumps_spline_kan_seed0/summary.json
- kan_operator_local_bumps_semL_operator_replacement_seed0: reports/runs/20260506_014443_kan_operator_local_bumps_semL_operator_replacement_seed0
  - history: reports/runs/20260506_014443_kan_operator_local_bumps_semL_operator_replacement_seed0/history.json
  - metrics: reports/runs/20260506_014443_kan_operator_local_bumps_semL_operator_replacement_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_014443_kan_operator_local_bumps_semL_operator_replacement_seed0/diagnostics.csv
  - summary: reports/runs/20260506_014443_kan_operator_local_bumps_semL_operator_replacement_seed0/summary.json
- kan_operator_local_bumps_spline_kan_seed1: reports/runs/20260506_014528_kan_operator_local_bumps_spline_kan_seed1
  - history: reports/runs/20260506_014528_kan_operator_local_bumps_spline_kan_seed1/history.json
  - metrics: reports/runs/20260506_014528_kan_operator_local_bumps_spline_kan_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_014528_kan_operator_local_bumps_spline_kan_seed1/diagnostics.csv
  - summary: reports/runs/20260506_014528_kan_operator_local_bumps_spline_kan_seed1/summary.json
- kan_operator_local_bumps_semL_operator_replacement_seed1: reports/runs/20260506_014551_kan_operator_local_bumps_semL_operator_replacement_seed1
  - history: reports/runs/20260506_014551_kan_operator_local_bumps_semL_operator_replacement_seed1/history.json
  - metrics: reports/runs/20260506_014551_kan_operator_local_bumps_semL_operator_replacement_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_014551_kan_operator_local_bumps_semL_operator_replacement_seed1/diagnostics.csv
  - summary: reports/runs/20260506_014551_kan_operator_local_bumps_semL_operator_replacement_seed1/summary.json
- kan_operator_local_bumps_spline_kan_seed2: reports/runs/20260506_014639_kan_operator_local_bumps_spline_kan_seed2
  - history: reports/runs/20260506_014639_kan_operator_local_bumps_spline_kan_seed2/history.json
  - metrics: reports/runs/20260506_014639_kan_operator_local_bumps_spline_kan_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_014639_kan_operator_local_bumps_spline_kan_seed2/diagnostics.csv
  - summary: reports/runs/20260506_014639_kan_operator_local_bumps_spline_kan_seed2/summary.json
- kan_operator_local_bumps_semL_operator_replacement_seed2: reports/runs/20260506_014702_kan_operator_local_bumps_semL_operator_replacement_seed2
  - history: reports/runs/20260506_014702_kan_operator_local_bumps_semL_operator_replacement_seed2/history.json
  - metrics: reports/runs/20260506_014702_kan_operator_local_bumps_semL_operator_replacement_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_014702_kan_operator_local_bumps_semL_operator_replacement_seed2/diagnostics.csv
  - summary: reports/runs/20260506_014702_kan_operator_local_bumps_semL_operator_replacement_seed2/summary.json
- kan_operator_mixed_composition_spline_kan_seed0: reports/runs/20260506_014747_kan_operator_mixed_composition_spline_kan_seed0
  - history: reports/runs/20260506_014747_kan_operator_mixed_composition_spline_kan_seed0/history.json
  - metrics: reports/runs/20260506_014747_kan_operator_mixed_composition_spline_kan_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_014747_kan_operator_mixed_composition_spline_kan_seed0/diagnostics.csv
  - summary: reports/runs/20260506_014747_kan_operator_mixed_composition_spline_kan_seed0/summary.json
- kan_operator_mixed_composition_semL_operator_replacement_seed0: reports/runs/20260506_014811_kan_operator_mixed_composition_semL_operator_replacement_seed0
  - history: reports/runs/20260506_014811_kan_operator_mixed_composition_semL_operator_replacement_seed0/history.json
  - metrics: reports/runs/20260506_014811_kan_operator_mixed_composition_semL_operator_replacement_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_014811_kan_operator_mixed_composition_semL_operator_replacement_seed0/diagnostics.csv
  - summary: reports/runs/20260506_014811_kan_operator_mixed_composition_semL_operator_replacement_seed0/summary.json
- kan_operator_mixed_composition_spline_kan_seed1: reports/runs/20260506_014850_kan_operator_mixed_composition_spline_kan_seed1
  - history: reports/runs/20260506_014850_kan_operator_mixed_composition_spline_kan_seed1/history.json
  - metrics: reports/runs/20260506_014850_kan_operator_mixed_composition_spline_kan_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_014850_kan_operator_mixed_composition_spline_kan_seed1/diagnostics.csv
  - summary: reports/runs/20260506_014850_kan_operator_mixed_composition_spline_kan_seed1/summary.json
- kan_operator_mixed_composition_semL_operator_replacement_seed1: reports/runs/20260506_014914_kan_operator_mixed_composition_semL_operator_replacement_seed1
  - history: reports/runs/20260506_014914_kan_operator_mixed_composition_semL_operator_replacement_seed1/history.json
  - metrics: reports/runs/20260506_014914_kan_operator_mixed_composition_semL_operator_replacement_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_014914_kan_operator_mixed_composition_semL_operator_replacement_seed1/diagnostics.csv
  - summary: reports/runs/20260506_014914_kan_operator_mixed_composition_semL_operator_replacement_seed1/summary.json
- kan_operator_mixed_composition_spline_kan_seed2: reports/runs/20260506_014952_kan_operator_mixed_composition_spline_kan_seed2
  - history: reports/runs/20260506_014952_kan_operator_mixed_composition_spline_kan_seed2/history.json
  - metrics: reports/runs/20260506_014952_kan_operator_mixed_composition_spline_kan_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_014952_kan_operator_mixed_composition_spline_kan_seed2/diagnostics.csv
  - summary: reports/runs/20260506_014952_kan_operator_mixed_composition_spline_kan_seed2/summary.json
- kan_operator_mixed_composition_semL_operator_replacement_seed2: reports/runs/20260506_015015_kan_operator_mixed_composition_semL_operator_replacement_seed2
  - history: reports/runs/20260506_015015_kan_operator_mixed_composition_semL_operator_replacement_seed2/history.json
  - metrics: reports/runs/20260506_015015_kan_operator_mixed_composition_semL_operator_replacement_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_015015_kan_operator_mixed_composition_semL_operator_replacement_seed2/diagnostics.csv
  - summary: reports/runs/20260506_015015_kan_operator_mixed_composition_semL_operator_replacement_seed2/summary.json
- eml_kan_mlp_symbolic_regression_eml_kan_seed0: reports/runs/20260506_021516_eml_kan_mlp_symbolic_regression_eml_kan_seed0
  - history: reports/runs/20260506_021516_eml_kan_mlp_symbolic_regression_eml_kan_seed0/history.json
  - metrics: reports/runs/20260506_021516_eml_kan_mlp_symbolic_regression_eml_kan_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_021516_eml_kan_mlp_symbolic_regression_eml_kan_seed0/diagnostics.csv
  - summary: reports/runs/20260506_021516_eml_kan_mlp_symbolic_regression_eml_kan_seed0/summary.json
- eml_kan_mlp_symbolic_regression_mlp_same_width_seed0: reports/runs/20260506_021552_eml_kan_mlp_symbolic_regression_mlp_same_width_seed0
  - history: reports/runs/20260506_021552_eml_kan_mlp_symbolic_regression_mlp_same_width_seed0/history.json
  - metrics: reports/runs/20260506_021552_eml_kan_mlp_symbolic_regression_mlp_same_width_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_021552_eml_kan_mlp_symbolic_regression_mlp_same_width_seed0/diagnostics.csv
  - summary: reports/runs/20260506_021552_eml_kan_mlp_symbolic_regression_mlp_same_width_seed0/summary.json
- eml_kan_mlp_symbolic_regression_mlp_param_matched_seed0: reports/runs/20260506_021600_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed0
  - history: reports/runs/20260506_021600_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed0/history.json
  - metrics: reports/runs/20260506_021600_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_021600_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed0/diagnostics.csv
  - summary: reports/runs/20260506_021600_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed0/summary.json
- eml_kan_mlp_symbolic_regression_eml_kan_seed1: reports/runs/20260506_021606_eml_kan_mlp_symbolic_regression_eml_kan_seed1
  - history: reports/runs/20260506_021606_eml_kan_mlp_symbolic_regression_eml_kan_seed1/history.json
  - metrics: reports/runs/20260506_021606_eml_kan_mlp_symbolic_regression_eml_kan_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_021606_eml_kan_mlp_symbolic_regression_eml_kan_seed1/diagnostics.csv
  - summary: reports/runs/20260506_021606_eml_kan_mlp_symbolic_regression_eml_kan_seed1/summary.json
- eml_kan_mlp_symbolic_regression_mlp_same_width_seed1: reports/runs/20260506_021645_eml_kan_mlp_symbolic_regression_mlp_same_width_seed1
  - history: reports/runs/20260506_021645_eml_kan_mlp_symbolic_regression_mlp_same_width_seed1/history.json
  - metrics: reports/runs/20260506_021645_eml_kan_mlp_symbolic_regression_mlp_same_width_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_021645_eml_kan_mlp_symbolic_regression_mlp_same_width_seed1/diagnostics.csv
  - summary: reports/runs/20260506_021645_eml_kan_mlp_symbolic_regression_mlp_same_width_seed1/summary.json
- eml_kan_mlp_symbolic_regression_mlp_param_matched_seed1: reports/runs/20260506_021653_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed1
  - history: reports/runs/20260506_021653_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed1/history.json
  - metrics: reports/runs/20260506_021653_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_021653_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed1/diagnostics.csv
  - summary: reports/runs/20260506_021653_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed1/summary.json
- eml_kan_mlp_symbolic_regression_eml_kan_seed2: reports/runs/20260506_021659_eml_kan_mlp_symbolic_regression_eml_kan_seed2
  - history: reports/runs/20260506_021659_eml_kan_mlp_symbolic_regression_eml_kan_seed2/history.json
  - metrics: reports/runs/20260506_021659_eml_kan_mlp_symbolic_regression_eml_kan_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_021659_eml_kan_mlp_symbolic_regression_eml_kan_seed2/diagnostics.csv
  - summary: reports/runs/20260506_021659_eml_kan_mlp_symbolic_regression_eml_kan_seed2/summary.json
- eml_kan_mlp_symbolic_regression_mlp_same_width_seed2: reports/runs/20260506_021740_eml_kan_mlp_symbolic_regression_mlp_same_width_seed2
  - history: reports/runs/20260506_021740_eml_kan_mlp_symbolic_regression_mlp_same_width_seed2/history.json
  - metrics: reports/runs/20260506_021740_eml_kan_mlp_symbolic_regression_mlp_same_width_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_021740_eml_kan_mlp_symbolic_regression_mlp_same_width_seed2/diagnostics.csv
  - summary: reports/runs/20260506_021740_eml_kan_mlp_symbolic_regression_mlp_same_width_seed2/summary.json
- eml_kan_mlp_symbolic_regression_mlp_param_matched_seed2: reports/runs/20260506_021748_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed2
  - history: reports/runs/20260506_021748_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed2/history.json
  - metrics: reports/runs/20260506_021748_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_021748_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed2/diagnostics.csv
  - summary: reports/runs/20260506_021748_eml_kan_mlp_symbolic_regression_mlp_param_matched_seed2/summary.json
- eml_kan_mlp_localized_regression_eml_kan_seed0: reports/runs/20260506_021753_eml_kan_mlp_localized_regression_eml_kan_seed0
  - history: reports/runs/20260506_021753_eml_kan_mlp_localized_regression_eml_kan_seed0/history.json
  - metrics: reports/runs/20260506_021753_eml_kan_mlp_localized_regression_eml_kan_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_021753_eml_kan_mlp_localized_regression_eml_kan_seed0/diagnostics.csv
  - summary: reports/runs/20260506_021753_eml_kan_mlp_localized_regression_eml_kan_seed0/summary.json
- eml_kan_mlp_localized_regression_mlp_same_width_seed0: reports/runs/20260506_021839_eml_kan_mlp_localized_regression_mlp_same_width_seed0
  - history: reports/runs/20260506_021839_eml_kan_mlp_localized_regression_mlp_same_width_seed0/history.json
  - metrics: reports/runs/20260506_021839_eml_kan_mlp_localized_regression_mlp_same_width_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_021839_eml_kan_mlp_localized_regression_mlp_same_width_seed0/diagnostics.csv
  - summary: reports/runs/20260506_021839_eml_kan_mlp_localized_regression_mlp_same_width_seed0/summary.json
- eml_kan_mlp_localized_regression_mlp_param_matched_seed0: reports/runs/20260506_021847_eml_kan_mlp_localized_regression_mlp_param_matched_seed0
  - history: reports/runs/20260506_021847_eml_kan_mlp_localized_regression_mlp_param_matched_seed0/history.json
  - metrics: reports/runs/20260506_021847_eml_kan_mlp_localized_regression_mlp_param_matched_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_021847_eml_kan_mlp_localized_regression_mlp_param_matched_seed0/diagnostics.csv
  - summary: reports/runs/20260506_021847_eml_kan_mlp_localized_regression_mlp_param_matched_seed0/summary.json
- eml_kan_mlp_localized_regression_eml_kan_seed1: reports/runs/20260506_021856_eml_kan_mlp_localized_regression_eml_kan_seed1
  - history: reports/runs/20260506_021856_eml_kan_mlp_localized_regression_eml_kan_seed1/history.json
  - metrics: reports/runs/20260506_021856_eml_kan_mlp_localized_regression_eml_kan_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_021856_eml_kan_mlp_localized_regression_eml_kan_seed1/diagnostics.csv
  - summary: reports/runs/20260506_021856_eml_kan_mlp_localized_regression_eml_kan_seed1/summary.json
- eml_kan_mlp_localized_regression_mlp_same_width_seed1: reports/runs/20260506_021942_eml_kan_mlp_localized_regression_mlp_same_width_seed1
  - history: reports/runs/20260506_021942_eml_kan_mlp_localized_regression_mlp_same_width_seed1/history.json
  - metrics: reports/runs/20260506_021942_eml_kan_mlp_localized_regression_mlp_same_width_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_021942_eml_kan_mlp_localized_regression_mlp_same_width_seed1/diagnostics.csv
  - summary: reports/runs/20260506_021942_eml_kan_mlp_localized_regression_mlp_same_width_seed1/summary.json
- eml_kan_mlp_localized_regression_mlp_param_matched_seed1: reports/runs/20260506_021950_eml_kan_mlp_localized_regression_mlp_param_matched_seed1
  - history: reports/runs/20260506_021950_eml_kan_mlp_localized_regression_mlp_param_matched_seed1/history.json
  - metrics: reports/runs/20260506_021950_eml_kan_mlp_localized_regression_mlp_param_matched_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_021950_eml_kan_mlp_localized_regression_mlp_param_matched_seed1/diagnostics.csv
  - summary: reports/runs/20260506_021950_eml_kan_mlp_localized_regression_mlp_param_matched_seed1/summary.json
- eml_kan_mlp_localized_regression_eml_kan_seed2: reports/runs/20260506_021959_eml_kan_mlp_localized_regression_eml_kan_seed2
  - history: reports/runs/20260506_021959_eml_kan_mlp_localized_regression_eml_kan_seed2/history.json
  - metrics: reports/runs/20260506_021959_eml_kan_mlp_localized_regression_eml_kan_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_021959_eml_kan_mlp_localized_regression_eml_kan_seed2/diagnostics.csv
  - summary: reports/runs/20260506_021959_eml_kan_mlp_localized_regression_eml_kan_seed2/summary.json
- eml_kan_mlp_localized_regression_mlp_same_width_seed2: reports/runs/20260506_022043_eml_kan_mlp_localized_regression_mlp_same_width_seed2
  - history: reports/runs/20260506_022043_eml_kan_mlp_localized_regression_mlp_same_width_seed2/history.json
  - metrics: reports/runs/20260506_022043_eml_kan_mlp_localized_regression_mlp_same_width_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_022043_eml_kan_mlp_localized_regression_mlp_same_width_seed2/diagnostics.csv
  - summary: reports/runs/20260506_022043_eml_kan_mlp_localized_regression_mlp_same_width_seed2/summary.json
- eml_kan_mlp_localized_regression_mlp_param_matched_seed2: reports/runs/20260506_022052_eml_kan_mlp_localized_regression_mlp_param_matched_seed2
  - history: reports/runs/20260506_022052_eml_kan_mlp_localized_regression_mlp_param_matched_seed2/history.json
  - metrics: reports/runs/20260506_022052_eml_kan_mlp_localized_regression_mlp_param_matched_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_022052_eml_kan_mlp_localized_regression_mlp_param_matched_seed2/diagnostics.csv
  - summary: reports/runs/20260506_022052_eml_kan_mlp_localized_regression_mlp_param_matched_seed2/summary.json
- eml_kan_mlp_shift_classification_eml_kan_seed0: reports/runs/20260506_022102_eml_kan_mlp_shift_classification_eml_kan_seed0
  - history: reports/runs/20260506_022102_eml_kan_mlp_shift_classification_eml_kan_seed0/history.json
  - metrics: reports/runs/20260506_022102_eml_kan_mlp_shift_classification_eml_kan_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_022102_eml_kan_mlp_shift_classification_eml_kan_seed0/diagnostics.csv
  - summary: reports/runs/20260506_022102_eml_kan_mlp_shift_classification_eml_kan_seed0/summary.json
- eml_kan_mlp_shift_classification_mlp_same_width_seed0: reports/runs/20260506_022108_eml_kan_mlp_shift_classification_mlp_same_width_seed0
  - history: reports/runs/20260506_022108_eml_kan_mlp_shift_classification_mlp_same_width_seed0/history.json
  - metrics: reports/runs/20260506_022108_eml_kan_mlp_shift_classification_mlp_same_width_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_022108_eml_kan_mlp_shift_classification_mlp_same_width_seed0/diagnostics.csv
  - summary: reports/runs/20260506_022108_eml_kan_mlp_shift_classification_mlp_same_width_seed0/summary.json
- eml_kan_mlp_shift_classification_mlp_param_matched_seed0: reports/runs/20260506_022111_eml_kan_mlp_shift_classification_mlp_param_matched_seed0
  - history: reports/runs/20260506_022111_eml_kan_mlp_shift_classification_mlp_param_matched_seed0/history.json
  - metrics: reports/runs/20260506_022111_eml_kan_mlp_shift_classification_mlp_param_matched_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_022111_eml_kan_mlp_shift_classification_mlp_param_matched_seed0/diagnostics.csv
  - summary: reports/runs/20260506_022111_eml_kan_mlp_shift_classification_mlp_param_matched_seed0/summary.json
- eml_kan_mlp_shift_classification_eml_kan_seed1: reports/runs/20260506_022112_eml_kan_mlp_shift_classification_eml_kan_seed1
  - history: reports/runs/20260506_022112_eml_kan_mlp_shift_classification_eml_kan_seed1/history.json
  - metrics: reports/runs/20260506_022112_eml_kan_mlp_shift_classification_eml_kan_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_022112_eml_kan_mlp_shift_classification_eml_kan_seed1/diagnostics.csv
  - summary: reports/runs/20260506_022112_eml_kan_mlp_shift_classification_eml_kan_seed1/summary.json
- eml_kan_mlp_shift_classification_mlp_same_width_seed1: reports/runs/20260506_022118_eml_kan_mlp_shift_classification_mlp_same_width_seed1
  - history: reports/runs/20260506_022118_eml_kan_mlp_shift_classification_mlp_same_width_seed1/history.json
  - metrics: reports/runs/20260506_022118_eml_kan_mlp_shift_classification_mlp_same_width_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_022118_eml_kan_mlp_shift_classification_mlp_same_width_seed1/diagnostics.csv
  - summary: reports/runs/20260506_022118_eml_kan_mlp_shift_classification_mlp_same_width_seed1/summary.json
- eml_kan_mlp_shift_classification_mlp_param_matched_seed1: reports/runs/20260506_022120_eml_kan_mlp_shift_classification_mlp_param_matched_seed1
  - history: reports/runs/20260506_022120_eml_kan_mlp_shift_classification_mlp_param_matched_seed1/history.json
  - metrics: reports/runs/20260506_022120_eml_kan_mlp_shift_classification_mlp_param_matched_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_022120_eml_kan_mlp_shift_classification_mlp_param_matched_seed1/diagnostics.csv
  - summary: reports/runs/20260506_022120_eml_kan_mlp_shift_classification_mlp_param_matched_seed1/summary.json
- eml_kan_mlp_shift_classification_eml_kan_seed2: reports/runs/20260506_022121_eml_kan_mlp_shift_classification_eml_kan_seed2
  - history: reports/runs/20260506_022121_eml_kan_mlp_shift_classification_eml_kan_seed2/history.json
  - metrics: reports/runs/20260506_022121_eml_kan_mlp_shift_classification_eml_kan_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_022121_eml_kan_mlp_shift_classification_eml_kan_seed2/diagnostics.csv
  - summary: reports/runs/20260506_022121_eml_kan_mlp_shift_classification_eml_kan_seed2/summary.json
- eml_kan_mlp_shift_classification_mlp_same_width_seed2: reports/runs/20260506_022127_eml_kan_mlp_shift_classification_mlp_same_width_seed2
  - history: reports/runs/20260506_022127_eml_kan_mlp_shift_classification_mlp_same_width_seed2/history.json
  - metrics: reports/runs/20260506_022127_eml_kan_mlp_shift_classification_mlp_same_width_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_022127_eml_kan_mlp_shift_classification_mlp_same_width_seed2/diagnostics.csv
  - summary: reports/runs/20260506_022127_eml_kan_mlp_shift_classification_mlp_same_width_seed2/summary.json
- eml_kan_mlp_shift_classification_mlp_param_matched_seed2: reports/runs/20260506_022129_eml_kan_mlp_shift_classification_mlp_param_matched_seed2
  - history: reports/runs/20260506_022129_eml_kan_mlp_shift_classification_mlp_param_matched_seed2/history.json
  - metrics: reports/runs/20260506_022129_eml_kan_mlp_shift_classification_mlp_param_matched_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_022129_eml_kan_mlp_shift_classification_mlp_param_matched_seed2/diagnostics.csv
  - summary: reports/runs/20260506_022129_eml_kan_mlp_shift_classification_mlp_param_matched_seed2/summary.json
- eml_kan_mlp_localized_regression_eml_kan_seed0: reports/runs/20260506_022208_eml_kan_mlp_localized_regression_eml_kan_seed0
  - history: reports/runs/20260506_022208_eml_kan_mlp_localized_regression_eml_kan_seed0/history.json
  - metrics: reports/runs/20260506_022208_eml_kan_mlp_localized_regression_eml_kan_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_022208_eml_kan_mlp_localized_regression_eml_kan_seed0/diagnostics.csv
  - summary: reports/runs/20260506_022208_eml_kan_mlp_localized_regression_eml_kan_seed0/summary.json
- eml_kan_mlp_localized_regression_mlp_same_width_seed0: reports/runs/20260506_022333_eml_kan_mlp_localized_regression_mlp_same_width_seed0
  - history: reports/runs/20260506_022333_eml_kan_mlp_localized_regression_mlp_same_width_seed0/history.json
  - metrics: reports/runs/20260506_022333_eml_kan_mlp_localized_regression_mlp_same_width_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_022333_eml_kan_mlp_localized_regression_mlp_same_width_seed0/diagnostics.csv
  - summary: reports/runs/20260506_022333_eml_kan_mlp_localized_regression_mlp_same_width_seed0/summary.json
- eml_kan_mlp_localized_regression_mlp_param_matched_seed0: reports/runs/20260506_022353_eml_kan_mlp_localized_regression_mlp_param_matched_seed0
  - history: reports/runs/20260506_022353_eml_kan_mlp_localized_regression_mlp_param_matched_seed0/history.json
  - metrics: reports/runs/20260506_022353_eml_kan_mlp_localized_regression_mlp_param_matched_seed0/metrics.csv
  - diagnostics: reports/runs/20260506_022353_eml_kan_mlp_localized_regression_mlp_param_matched_seed0/diagnostics.csv
  - summary: reports/runs/20260506_022353_eml_kan_mlp_localized_regression_mlp_param_matched_seed0/summary.json
- eml_kan_mlp_localized_regression_eml_kan_seed1: reports/runs/20260506_022409_eml_kan_mlp_localized_regression_eml_kan_seed1
  - history: reports/runs/20260506_022409_eml_kan_mlp_localized_regression_eml_kan_seed1/history.json
  - metrics: reports/runs/20260506_022409_eml_kan_mlp_localized_regression_eml_kan_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_022409_eml_kan_mlp_localized_regression_eml_kan_seed1/diagnostics.csv
  - summary: reports/runs/20260506_022409_eml_kan_mlp_localized_regression_eml_kan_seed1/summary.json
- eml_kan_mlp_localized_regression_mlp_same_width_seed1: reports/runs/20260506_022521_eml_kan_mlp_localized_regression_mlp_same_width_seed1
  - history: reports/runs/20260506_022521_eml_kan_mlp_localized_regression_mlp_same_width_seed1/history.json
  - metrics: reports/runs/20260506_022521_eml_kan_mlp_localized_regression_mlp_same_width_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_022521_eml_kan_mlp_localized_regression_mlp_same_width_seed1/diagnostics.csv
  - summary: reports/runs/20260506_022521_eml_kan_mlp_localized_regression_mlp_same_width_seed1/summary.json
- eml_kan_mlp_localized_regression_mlp_param_matched_seed1: reports/runs/20260506_022541_eml_kan_mlp_localized_regression_mlp_param_matched_seed1
  - history: reports/runs/20260506_022541_eml_kan_mlp_localized_regression_mlp_param_matched_seed1/history.json
  - metrics: reports/runs/20260506_022541_eml_kan_mlp_localized_regression_mlp_param_matched_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_022541_eml_kan_mlp_localized_regression_mlp_param_matched_seed1/diagnostics.csv
  - summary: reports/runs/20260506_022541_eml_kan_mlp_localized_regression_mlp_param_matched_seed1/summary.json
- eml_kan_mlp_localized_regression_eml_kan_seed2: reports/runs/20260506_022558_eml_kan_mlp_localized_regression_eml_kan_seed2
  - history: reports/runs/20260506_022558_eml_kan_mlp_localized_regression_eml_kan_seed2/history.json
  - metrics: reports/runs/20260506_022558_eml_kan_mlp_localized_regression_eml_kan_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_022558_eml_kan_mlp_localized_regression_eml_kan_seed2/diagnostics.csv
  - summary: reports/runs/20260506_022558_eml_kan_mlp_localized_regression_eml_kan_seed2/summary.json
- eml_kan_mlp_localized_regression_mlp_same_width_seed2: reports/runs/20260506_022727_eml_kan_mlp_localized_regression_mlp_same_width_seed2
  - history: reports/runs/20260506_022727_eml_kan_mlp_localized_regression_mlp_same_width_seed2/history.json
  - metrics: reports/runs/20260506_022727_eml_kan_mlp_localized_regression_mlp_same_width_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_022727_eml_kan_mlp_localized_regression_mlp_same_width_seed2/diagnostics.csv
  - summary: reports/runs/20260506_022727_eml_kan_mlp_localized_regression_mlp_same_width_seed2/summary.json
- eml_kan_mlp_localized_regression_mlp_param_matched_seed2: reports/runs/20260506_022748_eml_kan_mlp_localized_regression_mlp_param_matched_seed2
  - history: reports/runs/20260506_022748_eml_kan_mlp_localized_regression_mlp_param_matched_seed2/history.json
  - metrics: reports/runs/20260506_022748_eml_kan_mlp_localized_regression_mlp_param_matched_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_022748_eml_kan_mlp_localized_regression_mlp_param_matched_seed2/diagnostics.csv
  - summary: reports/runs/20260506_022748_eml_kan_mlp_localized_regression_mlp_param_matched_seed2/summary.json
- eml_kan_mlp_symbolic_regression_mlp_same_width_seed1: reports/runs/20260506_022813_eml_kan_mlp_symbolic_regression_mlp_same_width_seed1
  - history: reports/runs/20260506_022813_eml_kan_mlp_symbolic_regression_mlp_same_width_seed1/history.json
  - metrics: reports/runs/20260506_022813_eml_kan_mlp_symbolic_regression_mlp_same_width_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_022813_eml_kan_mlp_symbolic_regression_mlp_same_width_seed1/diagnostics.csv
  - summary: reports/runs/20260506_022813_eml_kan_mlp_symbolic_regression_mlp_same_width_seed1/summary.json
- eml_kan_mlp_symbolic_regression_mlp_same_width_seed2: reports/runs/20260506_022823_eml_kan_mlp_symbolic_regression_mlp_same_width_seed2
  - history: reports/runs/20260506_022823_eml_kan_mlp_symbolic_regression_mlp_same_width_seed2/history.json
  - metrics: reports/runs/20260506_022823_eml_kan_mlp_symbolic_regression_mlp_same_width_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_022823_eml_kan_mlp_symbolic_regression_mlp_same_width_seed2/diagnostics.csv
  - summary: reports/runs/20260506_022823_eml_kan_mlp_symbolic_regression_mlp_same_width_seed2/summary.json
- eml_kan_mlp_localized_regression_mlp_same_width_seed1: reports/runs/20260506_022857_eml_kan_mlp_localized_regression_mlp_same_width_seed1
  - history: reports/runs/20260506_022857_eml_kan_mlp_localized_regression_mlp_same_width_seed1/history.json
  - metrics: reports/runs/20260506_022857_eml_kan_mlp_localized_regression_mlp_same_width_seed1/metrics.csv
  - diagnostics: reports/runs/20260506_022857_eml_kan_mlp_localized_regression_mlp_same_width_seed1/diagnostics.csv
  - summary: reports/runs/20260506_022857_eml_kan_mlp_localized_regression_mlp_same_width_seed1/summary.json
- eml_kan_mlp_localized_regression_mlp_same_width_seed2: reports/runs/20260506_022922_eml_kan_mlp_localized_regression_mlp_same_width_seed2
  - history: reports/runs/20260506_022922_eml_kan_mlp_localized_regression_mlp_same_width_seed2/history.json
  - metrics: reports/runs/20260506_022922_eml_kan_mlp_localized_regression_mlp_same_width_seed2/metrics.csv
  - diagnostics: reports/runs/20260506_022922_eml_kan_mlp_localized_regression_mlp_same_width_seed2/diagnostics.csv
  - summary: reports/runs/20260506_022922_eml_kan_mlp_localized_regression_mlp_same_width_seed2/summary.json
- smoke_image_cnn_eml_baseline: reports/runs/20260506_050534_smoke_image_cnn_eml_baseline
  - history: reports/runs/20260506_050534_smoke_image_cnn_eml_baseline/history.json
  - metrics: reports/runs/20260506_050534_smoke_image_cnn_eml_baseline/metrics.csv
  - diagnostics: reports/runs/20260506_050534_smoke_image_cnn_eml_baseline/diagnostics.csv
  - summary: reports/runs/20260506_050534_smoke_image_cnn_eml_baseline/summary.json
- smoke_image_efficient_eml: reports/runs/20260506_050535_smoke_image_efficient_eml
  - history: reports/runs/20260506_050535_smoke_image_efficient_eml/history.json
  - metrics: reports/runs/20260506_050535_smoke_image_efficient_eml/metrics.csv
  - diagnostics: reports/runs/20260506_050535_smoke_image_efficient_eml/diagnostics.csv
  - summary: reports/runs/20260506_050535_smoke_image_efficient_eml/summary.json
- smoke_text_efficient_eml: reports/runs/20260506_050535_smoke_text_efficient_eml
  - history: reports/runs/20260506_050535_smoke_text_efficient_eml/history.json
  - metrics: reports/runs/20260506_050535_smoke_text_efficient_eml/metrics.csv
  - diagnostics: reports/runs/20260506_050535_smoke_text_efficient_eml/diagnostics.csv
  - summary: reports/runs/20260506_050535_smoke_text_efficient_eml/summary.json
- probe_gate_compat_sigmoid_update: reports/runs/20260506_050535_probe_gate_compat_sigmoid_update
  - history: reports/runs/20260506_050535_probe_gate_compat_sigmoid_update/history.json
  - metrics: reports/runs/20260506_050535_probe_gate_compat_sigmoid_update/metrics.csv
  - diagnostics: reports/runs/20260506_050535_probe_gate_compat_sigmoid_update/diagnostics.csv
  - summary: reports/runs/20260506_050535_probe_gate_compat_sigmoid_update/summary.json
- probe_responsibility_no_null_precision: reports/runs/20260506_050535_probe_responsibility_no_null_precision
  - history: reports/runs/20260506_050535_probe_responsibility_no_null_precision/history.json
  - metrics: reports/runs/20260506_050535_probe_responsibility_no_null_precision/metrics.csv
  - diagnostics: reports/runs/20260506_050535_probe_responsibility_no_null_precision/diagnostics.csv
  - summary: reports/runs/20260506_050535_probe_responsibility_no_null_precision/summary.json
- probe_responsibility_with_null_precision: reports/runs/20260506_050535_probe_responsibility_with_null_precision
  - history: reports/runs/20260506_050535_probe_responsibility_with_null_precision/history.json
  - metrics: reports/runs/20260506_050535_probe_responsibility_with_null_precision/metrics.csv
  - diagnostics: reports/runs/20260506_050535_probe_responsibility_with_null_precision/diagnostics.csv
  - summary: reports/runs/20260506_050535_probe_responsibility_with_null_precision/summary.json
- probe_thresholded_null: reports/runs/20260506_050535_probe_thresholded_null
  - history: reports/runs/20260506_050535_probe_thresholded_null/history.json
  - metrics: reports/runs/20260506_050535_probe_thresholded_null/metrics.csv
  - diagnostics: reports/runs/20260506_050535_probe_thresholded_null/diagnostics.csv
  - summary: reports/runs/20260506_050535_probe_thresholded_null/summary.json
- local_conv_baseline: reports/runs/20260506_050535_local_conv_baseline
  - history: reports/runs/20260506_050535_local_conv_baseline/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260506_050535_local_conv_baseline/summary.json
- local_text_linear_baseline: reports/runs/20260506_050535_local_text_linear_baseline
  - history: reports/runs/20260506_050535_local_text_linear_baseline/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260506_050535_local_text_linear_baseline/summary.json
- cifar_medium_suite: reports/runs/20260506_050535_cifar_medium_suite
  - history: reports/runs/20260506_050535_cifar_medium_suite/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260506_050535_cifar_medium_suite/summary.json
- text_medium_suite: reports/runs/20260506_050535_text_medium_suite
  - history: reports/runs/20260506_050535_text_medium_suite/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260506_050535_text_medium_suite/summary.json
- full_seeded_ablation: reports/runs/20260506_050535_full_seeded_ablation
  - history: reports/runs/20260506_050535_full_seeded_ablation/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs/20260506_050535_full_seeded_ablation/summary.json

## 14. Appendix: Commands

```bash
pytest
python scripts/run_eml_validation_suite.py --mode smoke --device cpu
python scripts/generate_eml_report.py
python scripts/run_eml_validation_suite.py --mode ablation --device cuda
python scripts/run_eml_validation_suite.py --mode cifar-medium --device cuda
python scripts/run_eml_validation_suite.py --mode text-medium --device cuda
```
