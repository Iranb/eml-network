# EML Validation and Ablation Report

Generated: 2026-04-24T07:13:35Z

## 1. Executive Summary

- Best image result: cnn_eml_workers0 (1.0)
- Best text result: EfficientEMLTextEncoder_window8_workers0 (0.4983498454093933)
- Strongest baseline: cnn_eml_workers0 (1.0)
- Responsibility evidence weighting: see mechanism probe and ablation tables; unrun cells remain marked.
- Precision update: see update probe rows; model-quality conclusions need medium runs.
- Attractor memory: present in efficient paths; no-attractor comparison is not yet standardized unless a run row exists.
- Major failure modes: 5 failed runs and 5 not-run cells are recorded in the status table.
- Recommended next step: standardize the remaining NOT RUN switches, then repeat the best CIFAR and text runs across seeds.

## 2. Repository and Environment

- git_commit: ae26ee9
- hostname: user-NF5468M6
- python_version: 3.10.20
- torch_version: 2.7.1+cu118
- torchvision_version: 0.22.1+cu118
- cuda_available: True
- device: cuda
- timestamp: 2026-04-24T07:12:28Z

## 3. Experimental Scope

| run_id | status | task | model | dataset | reason |
| --- | --- | --- | --- | --- | --- |
| ablation_gate_sigmoid_seed0 | COMPLETED | mechanism_probe | ablation_gate_sigmoid_seed0 | synthetic_probe_tensors |  |
| ablation_resp_no_null_seed0 | COMPLETED | mechanism_probe | ablation_resp_no_null_seed0 | synthetic_probe_tensors |  |
| ablation_resp_null_seed0 | COMPLETED | mechanism_probe | ablation_resp_null_seed0 | synthetic_probe_tensors |  |
| ablation_gate_sigmoid_seed1 | COMPLETED | mechanism_probe | ablation_gate_sigmoid_seed1 | synthetic_probe_tensors |  |
| ablation_resp_no_null_seed1 | COMPLETED | mechanism_probe | ablation_resp_no_null_seed1 | synthetic_probe_tensors |  |
| ablation_resp_null_seed1 | COMPLETED | mechanism_probe | ablation_resp_null_seed1 | synthetic_probe_tensors |  |
| ablation_gate_sigmoid_seed2 | COMPLETED | mechanism_probe | ablation_gate_sigmoid_seed2 | synthetic_probe_tensors |  |
| ablation_resp_no_null_seed2 | COMPLETED | mechanism_probe | ablation_resp_no_null_seed2 | synthetic_probe_tensors |  |
| ablation_resp_null_seed2 | COMPLETED | mechanism_probe | ablation_resp_null_seed2 | synthetic_probe_tensors |  |
| ablation_image_cnn_eml | FAILED | image_synthetic | cnn_eml | SyntheticShapeEnergyDataset | RuntimeError("Too many open files. Communication with the workers is no longer possible. Please increase the limit using `ulimit -n` in the shell or change the sharing strategy by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code") |
| ablation_image_pure_eml | FAILED | image_synthetic | pure_eml | SyntheticShapeEnergyDataset | RuntimeError("Too many open files. Communication with the workers is no longer possible. Please increase the limit using `ulimit -n` in the shell or change the sharing strategy by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code") |
| ablation_image_pure_eml_v2 | COMPLETED | image_synthetic | pure_eml_v2 | SyntheticShapeEnergyDataset |  |
| ablation_image_eff_attr4 | COMPLETED | image_synthetic | EfficientEMLImageClassifier_attr4 | SyntheticShapeEnergyDataset |  |
| ablation_image_eff_attr8 | COMPLETED | image_synthetic | EfficientEMLImageClassifier_attr8 | SyntheticShapeEnergyDataset |  |
| ablation_image_eff_window5 | COMPLETED | image_synthetic | EfficientEMLImageClassifier_window5 | SyntheticShapeEnergyDataset |  |
| ablation_image_eff_no_warmup | COMPLETED | image_synthetic | EfficientEMLImageClassifier_no_warmup | SyntheticShapeEnergyDataset |  |
| ablation_text_eff_window8 | FAILED | text_synthetic | EfficientEMLTextEncoder_window8 | SyntheticTextEnergyDataset | RuntimeError("Too many open files. Communication with the workers is no longer possible. Please increase the limit using `ulimit -n` in the shell or change the sharing strategy by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code") |
| ablation_text_eff_window16 | FAILED | text_synthetic | EfficientEMLTextEncoder_window16 | SyntheticTextEnergyDataset | RuntimeError("Too many open files. Communication with the workers is no longer possible. Please increase the limit using `ulimit -n` in the shell or change the sharing strategy by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code") |
| ablation_text_eff_window32 | FAILED | text_synthetic | EfficientEMLTextEncoder_window32 | SyntheticTextEnergyDataset | RuntimeError("Too many open files. Communication with the workers is no longer possible. Please increase the limit using `ulimit -n` in the shell or change the sharing strategy by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code") |
| ablation_no_composition | NOT RUN | image_synthetic | EfficientEMLImageClassifier_no_composition | SyntheticShapeEnergyDataset | model switch is not standardized |
| ablation_no_attractor | NOT RUN | image_synthetic | EfficientEMLImageClassifier_no_attractor | SyntheticShapeEnergyDataset | model switch is not standardized |
| ablation_head_without_ambiguity | NOT RUN | image_synthetic | prototype_head_without_ambiguity | SyntheticShapeEnergyDataset | head switch is not standardized |
| ablation_sigmoid_gate_mean | NOT RUN | mechanism_probe | sigmoid_gate_mean | synthetic_probe_tensors | graph mode is not implemented |
| ablation_thresholded_null | NOT RUN | mechanism_probe | responsibility_thresholded_null | synthetic_probe_tensors | threshold mode is not implemented |
| cifar_cnn_eml | COMPLETED | image_cifar | cnn_eml | CIFAR10 |  |
| cifar_pure_eml | COMPLETED | image_cifar | pure_eml | CIFAR10 |  |
| cifar_pure_eml_v2 | COMPLETED | image_cifar | pure_eml_v2 | CIFAR10 |  |
| cifar_efficient_eml_image | COMPLETED | image_cifar | EfficientEMLImageClassifier | CIFAR10 |  |
| ablation_text_eff_window8_workers0 | COMPLETED | text_synthetic | EfficientEMLTextEncoder_window8_workers0 | SyntheticTextEnergyDataset |  |
| ablation_text_eff_window16_workers0 | COMPLETED | text_synthetic | EfficientEMLTextEncoder_window16_workers0 | SyntheticTextEnergyDataset |  |
| ablation_text_eff_window32_workers0 | COMPLETED | text_synthetic | EfficientEMLTextEncoder_window32_workers0 | SyntheticTextEnergyDataset |  |
| ablation_image_cnn_eml_workers0 | COMPLETED | image_synthetic | cnn_eml_workers0 | SyntheticShapeEnergyDataset |  |
| ablation_image_pure_eml_workers0 | COMPLETED | image_synthetic | pure_eml_workers0 | SyntheticShapeEnergyDataset |  |

Failed runs: 5
Not-run entries: 5

## 4. Datasets

| dataset | synthetic/real | notes |
| --- | --- | --- |
| CIFAR10 | real/optional | requires local data/dependency |
| SyntheticShapeEnergyDataset | synthetic | offline |
| SyntheticTextEnergyDataset | synthetic | offline |
| synthetic_probe_tensors | synthetic | offline |

## 5. Models Compared

| model | parameter count | task names | key mechanisms |
| --- | ---: | --- | --- |
| EfficientEMLImageClassifier | 170179 | image_cifar | see config artifacts |
| EfficientEMLImageClassifier_attr4 | 117076 | image_synthetic | see config artifacts |
| EfficientEMLImageClassifier_attr8 | 117204 | image_synthetic | see config artifacts |
| EfficientEMLImageClassifier_no_attractor | 0 | image_synthetic | see config artifacts |
| EfficientEMLImageClassifier_no_composition | 0 | image_synthetic | see config artifacts |
| EfficientEMLImageClassifier_no_warmup | 117076 | image_synthetic | see config artifacts |
| EfficientEMLImageClassifier_window5 | 117076 | image_synthetic | see config artifacts |
| EfficientEMLTextEncoder_window16 | 0 | text_synthetic | see config artifacts |
| EfficientEMLTextEncoder_window16_workers0 | 92950 | text_synthetic | see config artifacts |
| EfficientEMLTextEncoder_window32 | 0 | text_synthetic | see config artifacts |
| EfficientEMLTextEncoder_window32_workers0 | 92950 | text_synthetic | see config artifacts |
| EfficientEMLTextEncoder_window8 | 0 | text_synthetic | see config artifacts |
| EfficientEMLTextEncoder_window8_workers0 | 92950 | text_synthetic | see config artifacts |
| ablation_gate_sigmoid_seed0 | 14151 | mechanism_probe | see config artifacts |
| ablation_gate_sigmoid_seed1 | 14151 | mechanism_probe | see config artifacts |
| ablation_gate_sigmoid_seed2 | 14151 | mechanism_probe | see config artifacts |
| ablation_resp_no_null_seed0 | 14151 | mechanism_probe | see config artifacts |
| ablation_resp_no_null_seed1 | 14151 | mechanism_probe | see config artifacts |
| ablation_resp_no_null_seed2 | 14151 | mechanism_probe | see config artifacts |
| ablation_resp_null_seed0 | 14151 | mechanism_probe | see config artifacts |
| ablation_resp_null_seed1 | 14151 | mechanism_probe | see config artifacts |
| ablation_resp_null_seed2 | 14151 | mechanism_probe | see config artifacts |
| cnn_eml | 0 | image_cifar, image_synthetic | see config artifacts |
| cnn_eml_workers0 | 162644 | image_synthetic | see config artifacts |
| prototype_head_without_ambiguity | 0 | image_synthetic | see config artifacts |
| pure_eml | 0 | image_cifar, image_synthetic | see config artifacts |
| pure_eml_v2 | 71189 | image_cifar, image_synthetic | see config artifacts |
| pure_eml_workers0 | 47638 | image_synthetic | see config artifacts |
| responsibility_thresholded_null | 0 | mechanism_probe | see config artifacts |
| sigmoid_gate_mean | 0 | mechanism_probe | see config artifacts |

## 6. Main Results

### Image
| run_id | model | dataset | final metric | best metric | loss | accuracy | time sec | params |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ablation_image_pure_eml_v2 | pure_eml_v2 | SyntheticShapeEnergyDataset | 0.109375 | 0.328125 | 1.6359 | 0.1094 | 1.7702131271362305 | 71189 |
| ablation_image_eff_attr4 | EfficientEMLImageClassifier_attr4 | SyntheticShapeEnergyDataset | 0.15625 | 0.34375 | 1.6222 | 0.1562 | 1.520871639251709 | 117076 |
| ablation_image_eff_attr8 | EfficientEMLImageClassifier_attr8 | SyntheticShapeEnergyDataset | 0.25 | 0.40625 | 1.5897 | 0.2500 | 2.6269795894622803 | 117204 |
| ablation_image_eff_window5 | EfficientEMLImageClassifier_window5 | SyntheticShapeEnergyDataset | 0.109375 | 0.375 | 1.6142 | 0.1094 | 1.149688482284546 | 117076 |
| ablation_image_eff_no_warmup | EfficientEMLImageClassifier_no_warmup | SyntheticShapeEnergyDataset | 0.265625 | 0.296875 | 1.6085 | 0.2656 | 1.674051284790039 | 117076 |
| cifar_cnn_eml | cnn_eml | CIFAR10 | 0.6966796875 | 0.6966796875 | 0.8466 | 0.7148 | 53.06012463569641 | 246117 |
| cifar_pure_eml | pure_eml | CIFAR10 | 0.3857421875 | 0.3857421875 | 1.6892 | 0.3516 | 48.45666170120239 | 117797 |
| cifar_pure_eml_v2 | pure_eml_v2 | CIFAR10 | 0.3431640625 | 0.3833984375 | 1.6665 | 0.3750 | 102.29234170913696 | 220228 |
| cifar_efficient_eml_image | EfficientEMLImageClassifier | CIFAR10 | 0.2310546875 | 0.2537109375 | 1.9718 | 0.2461 | 25.30465316772461 | 170179 |
| ablation_image_cnn_eml_workers0 | cnn_eml_workers0 | SyntheticShapeEnergyDataset | 0.953125 | 1.0 | 0.3554 | 0.9531 | 16.250306606292725 | 162644 |
| ablation_image_pure_eml_workers0 | pure_eml_workers0 | SyntheticShapeEnergyDataset | 0.328125 | 0.5 | 1.4465 | 0.3281 | 5.7693750858306885 | 47638 |

### Text
| run_id | model | dataset | final metric | best metric | loss | accuracy | time sec | params |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ablation_text_eff_window8_workers0 | EfficientEMLTextEncoder_window8_workers0 | SyntheticTextEnergyDataset | 0.44842565059661865 | 0.4983498454093933 | 2.1090 | 0.4484 | 25.918776035308838 | 92950 |
| ablation_text_eff_window16_workers0 | EfficientEMLTextEncoder_window16_workers0 | SyntheticTextEnergyDataset | 0.4079952836036682 | 0.428720623254776 | 2.0972 | 0.4080 | 25.262678384780884 | 92950 |
| ablation_text_eff_window32_workers0 | EfficientEMLTextEncoder_window32_workers0 | SyntheticTextEnergyDataset | 0.449638307094574 | 0.47152942419052124 | 2.1149 | 0.4496 | 26.068734169006348 | 92950 |

### Efficiency
| run_id | model | task | examples/sec | tokens/sec | step time | peak memory MB | params |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ablation_gate_sigmoid_seed0 | ablation_gate_sigmoid_seed0 | mechanism_probe |  |  | 0.2591054439544678 | 9.3701171875 | 14151 |
| ablation_resp_no_null_seed0 | ablation_resp_no_null_seed0 | mechanism_probe |  |  | 0.01846599578857422 | 9.3935546875 | 14151 |
| ablation_resp_null_seed0 | ablation_resp_null_seed0 | mechanism_probe |  |  | 0.0038118362426757812 | 9.39404296875 | 14151 |
| ablation_gate_sigmoid_seed1 | ablation_gate_sigmoid_seed1 | mechanism_probe |  |  | 0.0025920867919921875 | 9.39404296875 | 14151 |
| ablation_resp_no_null_seed1 | ablation_resp_no_null_seed1 | mechanism_probe |  |  | 0.0037975311279296875 | 9.39404296875 | 14151 |
| ablation_resp_null_seed1 | ablation_resp_null_seed1 | mechanism_probe |  |  | 0.0031583309173583984 | 9.39404296875 | 14151 |
| ablation_gate_sigmoid_seed2 | ablation_gate_sigmoid_seed2 | mechanism_probe |  |  | 0.0028247833251953125 | 9.39404296875 | 14151 |
| ablation_resp_no_null_seed2 | ablation_resp_no_null_seed2 | mechanism_probe |  |  | 0.0034537315368652344 | 9.39404296875 | 14151 |
| ablation_resp_null_seed2 | ablation_resp_null_seed2 | mechanism_probe |  |  | 0.0038557052612304688 | 9.39404296875 | 14151 |
| ablation_image_pure_eml_v2 | pure_eml_v2 | image_synthetic | 3107.143588022178 |  | 0.02059769630432129 | 634.10888671875 | 71189 |
| ablation_image_eff_attr4 | EfficientEMLImageClassifier_attr4 | image_synthetic | 1933.5690381692586 |  | 0.03309941291809082 | 634.10888671875 | 117076 |
| ablation_image_eff_attr8 | EfficientEMLImageClassifier_attr8 | image_synthetic | 1934.4050616492157 |  | 0.03308510780334473 | 634.10888671875 | 117204 |
| ablation_image_eff_window5 | EfficientEMLImageClassifier_window5 | image_synthetic | 2350.3467792069064 |  | 0.027230024337768555 | 634.10888671875 | 117076 |
| ablation_image_eff_no_warmup | EfficientEMLImageClassifier_no_warmup | image_synthetic | 1892.0160701447724 |  | 0.033826351165771484 | 634.10888671875 | 117076 |
| cifar_cnn_eml | cnn_eml | image_cifar | 16892.29475803914 |  | 0.015152931213378906 | 446.22802734375 | 246117 |
| cifar_pure_eml | pure_eml | image_cifar | 21433.683807090387 |  | 0.01194143295288086 | 477.9833984375 | 117797 |
| cifar_pure_eml_v2 | pure_eml_v2 | image_cifar | 3766.5107690580758 |  | 0.06796574592590332 | 4817.4462890625 | 220228 |
| cifar_efficient_eml_image | EfficientEMLImageClassifier | image_cifar | 6019.305785865245 |  | 0.04252791404724121 | 4817.4462890625 | 170179 |
| ablation_text_eff_window8_workers0 | EfficientEMLTextEncoder_window8_workers0 | text_synthetic |  | 120739.95074076389 | 0.015255928039550781 | 168.27099609375 | 92950 |
| ablation_text_eff_window16_workers0 | EfficientEMLTextEncoder_window16_workers0 | text_synthetic |  | 99013.4215610081 | 0.017179489135742188 | 247.95849609375 | 92950 |
| ablation_text_eff_window32_workers0 | EfficientEMLTextEncoder_window32_workers0 | text_synthetic |  | 91963.74103809269 | 0.01954030990600586 | 415.27099609375 | 92950 |
| ablation_image_cnn_eml_workers0 | cnn_eml_workers0 | image_synthetic | 5674.568354296586 |  | 0.011278390884399414 | 96.1640625 | 162644 |
| ablation_image_pure_eml_workers0 | pure_eml_workers0 | image_synthetic | 3913.0532944606416 |  | 0.016355514526367188 | 96.1640625 | 47638 |

### Stability
NaN/Inf counts are recorded when runners emit `nan_inf_count`; otherwise MISSING.

## 7. Ablation Results

### Responsibility / Null / Update Probes
| run_id | model | dataset | final metric | best metric | loss | accuracy | time sec | params |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ablation_gate_sigmoid_seed0 | ablation_gate_sigmoid_seed0 | synthetic_probe_tensors | 1.0 | 1.0 | 0.0000 |  | 0.25910401344299316 | 14151 |
| ablation_resp_no_null_seed0 | ablation_resp_no_null_seed0 | synthetic_probe_tensors | 1.0 | 1.0 | 0.0000 |  | 0.018465518951416016 | 14151 |
| ablation_resp_null_seed0 | ablation_resp_null_seed0 | synthetic_probe_tensors | 1.0 | 1.0 | 0.0000 |  | 0.003811359405517578 | 14151 |
| ablation_gate_sigmoid_seed1 | ablation_gate_sigmoid_seed1 | synthetic_probe_tensors | 1.0 | 1.0 | 0.0000 |  | 0.0025916099548339844 | 14151 |
| ablation_resp_no_null_seed1 | ablation_resp_no_null_seed1 | synthetic_probe_tensors | 1.0 | 1.0 | 0.0000 |  | 0.0037970542907714844 | 14151 |
| ablation_resp_null_seed1 | ablation_resp_null_seed1 | synthetic_probe_tensors | 1.0 | 1.0 | 0.0000 |  | 0.0031576156616210938 | 14151 |
| ablation_gate_sigmoid_seed2 | ablation_gate_sigmoid_seed2 | synthetic_probe_tensors | 1.0 | 1.0 | 0.0000 |  | 0.0028243064880371094 | 14151 |
| ablation_resp_no_null_seed2 | ablation_resp_no_null_seed2 | synthetic_probe_tensors | 1.0 | 1.0 | 0.0000 |  | 0.0034532546997070312 | 14151 |
| ablation_resp_null_seed2 | ablation_resp_null_seed2 | synthetic_probe_tensors | 1.0 | 1.0 | 0.0000 |  | 0.0038552284240722656 | 14151 |

### All Ablation Cells
| run_id | status | task | model | key settings | best | final | loss | reason |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| ablation_gate_sigmoid_seed0 | COMPLETED | mechanism_probe | ablation_gate_sigmoid_seed0 | responsibility_mode=False, use_null=False, update_mode=sigmoid | 1.0 | 1.0 | 0.0000 |  |
| ablation_resp_no_null_seed0 | COMPLETED | mechanism_probe | ablation_resp_no_null_seed0 | responsibility_mode=True, use_null=False, update_mode=precision | 1.0 | 1.0 | 0.0000 |  |
| ablation_resp_null_seed0 | COMPLETED | mechanism_probe | ablation_resp_null_seed0 | responsibility_mode=True, use_null=True, update_mode=precision | 1.0 | 1.0 | 0.0000 |  |
| ablation_gate_sigmoid_seed1 | COMPLETED | mechanism_probe | ablation_gate_sigmoid_seed1 | responsibility_mode=False, use_null=False, update_mode=sigmoid | 1.0 | 1.0 | 0.0000 |  |
| ablation_resp_no_null_seed1 | COMPLETED | mechanism_probe | ablation_resp_no_null_seed1 | responsibility_mode=True, use_null=False, update_mode=precision | 1.0 | 1.0 | 0.0000 |  |
| ablation_resp_null_seed1 | COMPLETED | mechanism_probe | ablation_resp_null_seed1 | responsibility_mode=True, use_null=True, update_mode=precision | 1.0 | 1.0 | 0.0000 |  |
| ablation_gate_sigmoid_seed2 | COMPLETED | mechanism_probe | ablation_gate_sigmoid_seed2 | responsibility_mode=False, use_null=False, update_mode=sigmoid | 1.0 | 1.0 | 0.0000 |  |
| ablation_resp_no_null_seed2 | COMPLETED | mechanism_probe | ablation_resp_no_null_seed2 | responsibility_mode=True, use_null=False, update_mode=precision | 1.0 | 1.0 | 0.0000 |  |
| ablation_resp_null_seed2 | COMPLETED | mechanism_probe | ablation_resp_null_seed2 | responsibility_mode=True, use_null=True, update_mode=precision | 1.0 | 1.0 | 0.0000 |  |
| ablation_image_cnn_eml | FAILED | image_synthetic | cnn_eml | see config |  |  |  | RuntimeError("Too many open files. Communication with the workers is no longer possible. Please increase the limit using `ulimit -n` in the shell or change the sharing strategy by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code") |
| ablation_image_pure_eml | FAILED | image_synthetic | pure_eml | see config |  |  |  | RuntimeError("Too many open files. Communication with the workers is no longer possible. Please increase the limit using `ulimit -n` in the shell or change the sharing strategy by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code") |
| ablation_image_pure_eml_v2 | COMPLETED | image_synthetic | pure_eml_v2 | warmup_enabled=True, early_stop=True, patience=25 | 0.328125 | 0.109375 | 1.6359 |  |
| ablation_image_eff_attr4 | COMPLETED | image_synthetic | EfficientEMLImageClassifier_attr4 | warmup_enabled=True, early_stop=True, patience=25 | 0.34375 | 0.15625 | 1.6222 |  |
| ablation_image_eff_attr8 | COMPLETED | image_synthetic | EfficientEMLImageClassifier_attr8 | warmup_enabled=True, early_stop=True, patience=25 | 0.40625 | 0.25 | 1.5897 |  |
| ablation_image_eff_window5 | COMPLETED | image_synthetic | EfficientEMLImageClassifier_window5 | warmup_enabled=True, early_stop=True, patience=25 | 0.375 | 0.109375 | 1.6142 |  |
| ablation_image_eff_no_warmup | COMPLETED | image_synthetic | EfficientEMLImageClassifier_no_warmup | warmup_enabled=False, early_stop=True, patience=25 | 0.296875 | 0.265625 | 1.6085 |  |
| ablation_text_eff_window8 | FAILED | text_synthetic | EfficientEMLTextEncoder_window8 | see config |  |  |  | RuntimeError("Too many open files. Communication with the workers is no longer possible. Please increase the limit using `ulimit -n` in the shell or change the sharing strategy by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code") |
| ablation_text_eff_window16 | FAILED | text_synthetic | EfficientEMLTextEncoder_window16 | see config |  |  |  | RuntimeError("Too many open files. Communication with the workers is no longer possible. Please increase the limit using `ulimit -n` in the shell or change the sharing strategy by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code") |
| ablation_text_eff_window32 | FAILED | text_synthetic | EfficientEMLTextEncoder_window32 | see config |  |  |  | RuntimeError("Too many open files. Communication with the workers is no longer possible. Please increase the limit using `ulimit -n` in the shell or change the sharing strategy by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code") |
| ablation_no_composition | NOT RUN | image_synthetic | EfficientEMLImageClassifier_no_composition | see config |  |  |  | model switch is not standardized |
| ablation_no_attractor | NOT RUN | image_synthetic | EfficientEMLImageClassifier_no_attractor | see config |  |  |  | model switch is not standardized |
| ablation_head_without_ambiguity | NOT RUN | image_synthetic | prototype_head_without_ambiguity | see config |  |  |  | head switch is not standardized |
| ablation_sigmoid_gate_mean | NOT RUN | mechanism_probe | sigmoid_gate_mean | see config |  |  |  | graph mode is not implemented |
| ablation_thresholded_null | NOT RUN | mechanism_probe | responsibility_thresholded_null | see config |  |  |  | threshold mode is not implemented |
| ablation_text_eff_window8_workers0 | COMPLETED | text_synthetic | EfficientEMLTextEncoder_window8_workers0 | early_stop=True, patience=25, seq_len=64 | 0.4983498454093933 | 0.44842565059661865 | 2.1090 |  |
| ablation_text_eff_window16_workers0 | COMPLETED | text_synthetic | EfficientEMLTextEncoder_window16_workers0 | early_stop=True, patience=25, seq_len=64 | 0.428720623254776 | 0.4079952836036682 | 2.0972 |  |
| ablation_text_eff_window32_workers0 | COMPLETED | text_synthetic | EfficientEMLTextEncoder_window32_workers0 | early_stop=True, patience=25, seq_len=64 | 0.47152942419052124 | 0.449638307094574 | 2.1149 |  |
| ablation_image_cnn_eml_workers0 | COMPLETED | image_synthetic | cnn_eml_workers0 | warmup_enabled=True, early_stop=True, patience=25 | 1.0 | 0.953125 | 0.3554 |  |
| ablation_image_pure_eml_workers0 | COMPLETED | image_synthetic | pure_eml_workers0 | warmup_enabled=True, early_stop=True, patience=25 | 0.5 | 0.328125 | 1.4465 |  |

Other ablation axes remain `NOT RUN` unless listed in the status table.

## 8. EML Diagnostics

| run_id | model | drive_mean | drive_std | resistance_mean | resistance_std | energy_mean | energy_std | null_weight_mean | responsibility_entropy_mean | update_strength_mean | update_gate_mean | attractor_diversity | ambiguity_mean | sample_uncertainty_mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ablation_gate_sigmoid_seed0 | ablation_gate_sigmoid_seed0 | 0.0005 | 0.0023 | -0.0001 | 0.0020 | -0.8809 | 0.2130 |  | 1.4710 | 1.0000 | 0.2689 |  |  |  |
| ablation_resp_no_null_seed0 | ablation_resp_no_null_seed0 | -0.0000 | 0.0019 | 0.0000 | 0.0019 | -0.8810 | 0.2129 |  | 1.3863 | 1.0000 | 0.3113 |  |  |  |
| ablation_resp_null_seed0 | ablation_resp_null_seed0 | 0.0002 | 0.0020 | -0.0002 | 0.0029 | -0.8809 | 0.2130 | 0.2919 | 1.5855 | 0.7081 | 0.3113 |  |  |  |
| ablation_gate_sigmoid_seed1 | ablation_gate_sigmoid_seed1 | 0.0002 | 0.0017 | -0.0001 | 0.0022 | -0.8809 | 0.2129 |  | 1.4710 | 1.0000 | 0.2689 |  |  |  |
| ablation_resp_no_null_seed1 | ablation_resp_no_null_seed1 | -0.0001 | 0.0022 | -0.0001 | 0.0019 | -0.8810 | 0.2130 |  | 1.3863 | 1.0000 | 0.3113 |  |  |  |
| ablation_resp_null_seed1 | ablation_resp_null_seed1 | 0.0006 | 0.0017 | -0.0001 | 0.0022 | -0.8809 | 0.2130 | 0.2918 | 1.5855 | 0.7082 | 0.3113 |  |  |  |
| ablation_gate_sigmoid_seed2 | ablation_gate_sigmoid_seed2 | -0.0001 | 0.0018 | -0.0000 | 0.0020 | -0.8810 | 0.2129 |  | 1.4710 | 1.0000 | 0.2689 |  |  |  |
| ablation_resp_no_null_seed2 | ablation_resp_no_null_seed2 | 0.0002 | 0.0021 | 0.0001 | 0.0020 | -0.8809 | 0.2130 |  | 1.3863 | 1.0000 | 0.3113 |  |  |  |
| ablation_resp_null_seed2 | ablation_resp_null_seed2 | -0.0003 | 0.0020 | 0.0000 | 0.0021 | -0.8810 | 0.2129 | 0.2919 | 1.5855 | 0.7081 | 0.3113 |  |  |  |
| ablation_image_pure_eml_v2 | pure_eml_v2 | -0.0004 | 0.0174 | 0.0007 | 0.0388 | -0.1452 | 0.0824 |  |  |  |  |  |  | 0.7170 |
| ablation_image_eff_attr4 | EfficientEMLImageClassifier_attr4 | 0.0107 | 0.0657 | 1.3580 | 0.1051 | -0.2326 | 0.0181 | 0.1585 |  | 0.8551 | 0.3977 | 0.9673 | 1.1854 | 0.7195 |
| ablation_image_eff_attr8 | EfficientEMLImageClassifier_attr8 | -0.0467 | 0.1209 | 1.3662 | 0.1986 | -0.2491 | 0.0414 | 0.1572 |  | 0.8570 | 0.3949 | 0.9438 | 2.6147 | 0.6917 |
| ablation_image_eff_window5 | EfficientEMLImageClassifier_window5 | -0.0361 | 0.0131 | 1.3934 | 0.0755 | -0.2507 | 0.0152 | 0.0996 |  | 0.9286 | 0.4228 | 0.9581 | 1.3098 | 0.7069 |
| ablation_image_eff_no_warmup | EfficientEMLImageClassifier_no_warmup | 0.0315 | 0.0949 | 1.3478 | 0.1274 | -0.2323 | 0.0269 | 0.1591 |  | 0.8552 | 0.3997 | 0.9653 | -0.3555 | 0.7138 |
| cifar_cnn_eml | cnn_eml | -1.0910 | 2.7358 | 2.3739 | 2.8476 | -0.9118 | 0.6656 |  |  |  |  |  |  | 0.0018 |
| cifar_pure_eml | pure_eml | -0.1067 | 3.0001 | 1.3461 | 4.3795 | -0.7632 | 0.6492 |  |  |  |  |  |  | 0.0010 |
| cifar_pure_eml_v2 | pure_eml_v2 | -1.0610 | 3.7475 | 1.5547 | 4.7240 | -0.1694 | 0.6892 |  |  |  |  |  |  | 0.0061 |
| cifar_efficient_eml_image | EfficientEMLImageClassifier | -3.5754 | 5.0805 | 9.2558 | 6.9924 | -0.4755 | 0.4269 | 0.1899 |  | 0.8181 | 0.3455 | 0.8940 | 4.0432 | 0.0988 |
| ablation_text_eff_window8_workers0 | EfficientEMLTextEncoder_window8_workers0 | 0.2794 | 3.7634 | 1.9276 | 3.9778 | 0.3610 | 0.9552 | 0.5736 |  | 0.4265 | 0.3141 | 0.2535 |  |  |
| ablation_text_eff_window16_workers0 | EfficientEMLTextEncoder_window16_workers0 | -1.0595 | 3.2276 | 6.0782 | 7.7438 | -0.1240 | 0.6091 | 0.6071 |  | 0.3925 | 0.2508 | 0.1732 |  |  |
| ablation_text_eff_window32_workers0 | EfficientEMLTextEncoder_window32_workers0 | -3.4756 | 4.7590 | 8.1639 | 8.1348 | -0.2437 | 0.6587 | 0.5810 |  | 0.4197 | 0.2796 | 0.2059 |  |  |
| ablation_image_cnn_eml_workers0 | cnn_eml_workers0 | 0.1018 | 2.1181 | 0.4715 | 1.6635 | -0.6823 | 0.6971 |  |  |  |  |  |  | 0.0693 |
| ablation_image_pure_eml_workers0 | pure_eml_workers0 | -0.0269 | 0.1404 | 0.0268 | 0.1936 | -1.0115 | 0.0515 |  |  |  |  |  |  | 0.6878 |

Resistance-noise, resistance-occlusion, and resistance-corruption correlations are included when emitted by a run; otherwise MISSING.

## 9. Training Curves

### ablation_gate_sigmoid_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.25910401344299316 |

### ablation_resp_no_null_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.018465518951416016 |

### ablation_resp_null_seed0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.003811359405517578 |

### ablation_gate_sigmoid_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0025916099548339844 |

### ablation_resp_no_null_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0037970542907714844 |

### ablation_resp_null_seed1

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0031576156616210938 |

### ablation_gate_sigmoid_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0028243064880371094 |

### ablation_resp_no_null_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0034532546997070312 |

### ablation_resp_null_seed2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 |  |  | 0.0038552284240722656 |

### ablation_image_pure_eml_v2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 30 | 1.6159491539001465 | 0.265625 |  | 1.6339781284332275 |
| 31 | 1.619240403175354 | 0.21875 |  | 1.6689074039459229 |
| 32 | 1.6301709413528442 | 0.125 |  | 1.6960256099700928 |
| 33 | 1.6126338243484497 | 0.1875 |  | 1.7340342998504639 |
| 34 | 1.6359431743621826 | 0.109375 |  | 1.7642998695373535 |

### ablation_image_eff_attr4

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 26 | 1.6115859746932983 | 0.1875 |  | 1.3410422801971436 |
| 27 | 1.604670763015747 | 0.3125 |  | 1.3773071765899658 |
| 28 | 1.6115132570266724 | 0.265625 |  | 1.4240844249725342 |
| 29 | 1.6033588647842407 | 0.1875 |  | 1.473499059677124 |
| 30 | 1.622200608253479 | 0.15625 |  | 1.5171103477478027 |

### ablation_image_eff_attr8

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 49 | 1.5922528505325317 | 0.28125 |  | 2.3874404430389404 |
| 50 | 1.5890614986419678 | 0.265625 |  | 2.428870677947998 |
| 51 | 1.602734088897705 | 0.21875 |  | 2.473320960998535 |
| 52 | 1.6385481357574463 | 0.1875 |  | 2.5275213718414307 |
| 53 | 1.5897338390350342 | 0.25 |  | 2.6187245845794678 |

### ablation_image_eff_window5

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 1.6170053482055664 | 0.203125 |  | 0.9840061664581299 |
| 25 | 1.6159917116165161 | 0.109375 |  | 1.0291917324066162 |
| 26 | 1.6025605201721191 | 0.375 |  | 1.0651741027832031 |
| 27 | 1.6092787981033325 | 0.21875 |  | 1.1098556518554688 |
| 28 | 1.6142197847366333 | 0.109375 |  | 1.146094799041748 |

### ablation_image_eff_no_warmup

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 36 | 1.6095479726791382 | 0.1875 |  | 1.4835431575775146 |
| 37 | 1.610482096672058 | 0.125 |  | 1.528212308883667 |
| 38 | 1.6090998649597168 | 0.265625 |  | 1.5771963596343994 |
| 39 | 1.6105878353118896 | 0.171875 |  | 1.6219983100891113 |
| 40 | 1.6084572076797485 | 0.265625 |  | 1.6692121028900146 |

### cifar_cnn_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 796 | 0.8272292613983154 | 0.7109375 |  | 52.48865270614624 |
| 797 | 0.9099549055099487 | 0.7109375 |  | 52.57173681259155 |
| 798 | 0.7624297142028809 | 0.8125 |  | 52.653830766677856 |
| 799 | 0.8007527589797974 | 0.72265625 |  | 52.73429727554321 |
| 800 | 0.8466355204582214 | 0.71484375 |  | 52.815383434295654 |

### cifar_pure_eml

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 796 | 1.6704798936843872 | 0.3515625 |  | 47.886285066604614 |
| 797 | 1.5824766159057617 | 0.3671875 |  | 47.964985847473145 |
| 798 | 1.6160889863967896 | 0.36328125 |  | 48.04187726974487 |
| 799 | 1.6966311931610107 | 0.3671875 |  | 48.12495160102844 |
| 800 | 1.689180612564087 | 0.3515625 |  | 48.203890323638916 |

### cifar_pure_eml_v2

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 796 | 1.6344274282455444 | 0.38671875 |  | 101.1712532043457 |
| 797 | 1.6924251317977905 | 0.359375 |  | 101.29965472221375 |
| 798 | 1.6730531454086304 | 0.35546875 |  | 101.42768979072571 |
| 799 | 1.7040042877197266 | 0.33984375 |  | 101.5576388835907 |
| 800 | 1.6664942502975464 | 0.375 |  | 101.6867048740387 |

### cifar_efficient_eml_image

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 296 | 2.0088248252868652 | 0.2578125 |  | 24.566585779190063 |
| 297 | 1.9135112762451172 | 0.296875 |  | 24.63639259338379 |
| 298 | 1.9240082502365112 | 0.265625 |  | 24.714362144470215 |
| 299 | 1.907035231590271 | 0.31640625 |  | 24.78417992591858 |
| 300 | 1.9718323945999146 | 0.24609375 |  | 24.860454559326172 |

### ablation_text_eff_window8_workers0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 296 | 2.175420045852661 |  | 0.43369564414024353 | 25.501705408096313 |
| 297 | 2.108765125274658 |  | 0.4476280212402344 | 25.59810781478882 |
| 298 | 2.120192289352417 |  | 0.43926700949668884 | 25.694424867630005 |
| 299 | 2.065798044204712 |  | 0.4701283872127533 | 25.802366495132446 |
| 300 | 2.108999013900757 |  | 0.44842565059661865 | 25.8888578414917 |

### ablation_text_eff_window16_workers0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 296 | 2.1682653427124023 |  | 0.3682890236377716 | 24.82964587211609 |
| 297 | 2.1048617362976074 |  | 0.40528348088264465 | 24.928677797317505 |
| 298 | 2.170574426651001 |  | 0.3920265734195709 | 25.036067724227905 |
| 299 | 2.103318929672241 |  | 0.4048672616481781 | 25.139123678207397 |
| 300 | 2.0972280502319336 |  | 0.4079952836036682 | 25.231537342071533 |

### ablation_text_eff_window32_workers0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 296 | 2.1357030868530273 |  | 0.4213973879814148 | 25.632746934890747 |
| 297 | 2.163752555847168 |  | 0.4136807918548584 | 25.73889470100403 |
| 298 | 2.114790201187134 |  | 0.42065009474754333 | 25.832507848739624 |
| 299 | 2.1483993530273438 |  | 0.42385411262512207 | 25.940321445465088 |
| 300 | 2.114884376525879 |  | 0.449638307094574 | 26.037705183029175 |

### ablation_image_cnn_eml_workers0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 296 | 0.25721704959869385 | 0.96875 |  | 15.980318784713745 |
| 297 | 0.5591554045677185 | 0.828125 |  | 16.04701042175293 |
| 298 | 0.5771976709365845 | 0.84375 |  | 16.1061532497406 |
| 299 | 0.4314151108264923 | 0.875 |  | 16.16427969932556 |
| 300 | 0.35539549589157104 | 0.953125 |  | 16.2234148979187 |

### ablation_image_pure_eml_workers0

| step | train loss | train acc | token acc | wall time |
| ---: | ---: | ---: | ---: | ---: |
| 102 | 1.497443675994873 | 0.34375 |  | 5.535935401916504 |
| 103 | 1.4366295337677002 | 0.28125 |  | 5.59243106842041 |
| 104 | 1.5169588327407837 | 0.40625 |  | 5.644259691238403 |
| 105 | 1.5075563192367554 | 0.328125 |  | 5.7000648975372314 |
| 106 | 1.4464988708496094 | 0.328125 |  | 5.756926536560059 |


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

- Current evidence is preliminary until ablation and medium modes are run with repeat seeds.
- The validation framework now records enough data to answer whether EML is acting as representation trunk or only as head/gate.
- The exact next experiment is GPU ablation mode with at least two seeds, followed by CIFAR medium for `cnn_eml`, `pure_eml_v2`, and `EfficientEMLImageClassifier`.

## 13. Raw Artifacts

- ablation_gate_sigmoid_seed0: reports/runs_server_ablation_20260424_150157/20260424_070159_ablation_gate_sigmoid_seed0
  - history: reports/runs_server_ablation_20260424_150157/20260424_070159_ablation_gate_sigmoid_seed0/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070159_ablation_gate_sigmoid_seed0/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070159_ablation_gate_sigmoid_seed0/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070159_ablation_gate_sigmoid_seed0/summary.json
- ablation_resp_no_null_seed0: reports/runs_server_ablation_20260424_150157/20260424_070200_ablation_resp_no_null_seed0
  - history: reports/runs_server_ablation_20260424_150157/20260424_070200_ablation_resp_no_null_seed0/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070200_ablation_resp_no_null_seed0/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070200_ablation_resp_no_null_seed0/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070200_ablation_resp_no_null_seed0/summary.json
- ablation_resp_null_seed0: reports/runs_server_ablation_20260424_150157/20260424_070200_ablation_resp_null_seed0
  - history: reports/runs_server_ablation_20260424_150157/20260424_070200_ablation_resp_null_seed0/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070200_ablation_resp_null_seed0/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070200_ablation_resp_null_seed0/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070200_ablation_resp_null_seed0/summary.json
- ablation_gate_sigmoid_seed1: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_gate_sigmoid_seed1
  - history: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_gate_sigmoid_seed1/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_gate_sigmoid_seed1/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_gate_sigmoid_seed1/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_gate_sigmoid_seed1/summary.json
- ablation_resp_no_null_seed1: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_no_null_seed1
  - history: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_no_null_seed1/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_no_null_seed1/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_no_null_seed1/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_no_null_seed1/summary.json
- ablation_resp_null_seed1: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_null_seed1
  - history: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_null_seed1/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_null_seed1/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_null_seed1/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_null_seed1/summary.json
- ablation_gate_sigmoid_seed2: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_gate_sigmoid_seed2
  - history: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_gate_sigmoid_seed2/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_gate_sigmoid_seed2/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_gate_sigmoid_seed2/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_gate_sigmoid_seed2/summary.json
- ablation_resp_no_null_seed2: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_no_null_seed2
  - history: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_no_null_seed2/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_no_null_seed2/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_no_null_seed2/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_no_null_seed2/summary.json
- ablation_resp_null_seed2: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_null_seed2
  - history: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_null_seed2/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_null_seed2/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_null_seed2/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070201_ablation_resp_null_seed2/summary.json
- ablation_image_cnn_eml: reports/runs_server_ablation_20260424_150157/20260424_070203_ablation_image_cnn_eml
  - history: reports/runs_server_ablation_20260424_150157/20260424_070203_ablation_image_cnn_eml/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070203_ablation_image_cnn_eml/summary.json
- ablation_image_pure_eml: reports/runs_server_ablation_20260424_150157/20260424_070206_ablation_image_pure_eml
  - history: reports/runs_server_ablation_20260424_150157/20260424_070206_ablation_image_pure_eml/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070206_ablation_image_pure_eml/summary.json
- ablation_image_pure_eml_v2: reports/runs_server_ablation_20260424_150157/20260424_070206_ablation_image_pure_eml_v2
  - history: reports/runs_server_ablation_20260424_150157/20260424_070206_ablation_image_pure_eml_v2/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070206_ablation_image_pure_eml_v2/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070206_ablation_image_pure_eml_v2/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070206_ablation_image_pure_eml_v2/summary.json
- ablation_image_eff_attr4: reports/runs_server_ablation_20260424_150157/20260424_070207_ablation_image_eff_attr4
  - history: reports/runs_server_ablation_20260424_150157/20260424_070207_ablation_image_eff_attr4/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070207_ablation_image_eff_attr4/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070207_ablation_image_eff_attr4/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070207_ablation_image_eff_attr4/summary.json
- ablation_image_eff_attr8: reports/runs_server_ablation_20260424_150157/20260424_070209_ablation_image_eff_attr8
  - history: reports/runs_server_ablation_20260424_150157/20260424_070209_ablation_image_eff_attr8/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070209_ablation_image_eff_attr8/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070209_ablation_image_eff_attr8/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070209_ablation_image_eff_attr8/summary.json
- ablation_image_eff_window5: reports/runs_server_ablation_20260424_150157/20260424_070212_ablation_image_eff_window5
  - history: reports/runs_server_ablation_20260424_150157/20260424_070212_ablation_image_eff_window5/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070212_ablation_image_eff_window5/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070212_ablation_image_eff_window5/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070212_ablation_image_eff_window5/summary.json
- ablation_image_eff_no_warmup: reports/runs_server_ablation_20260424_150157/20260424_070213_ablation_image_eff_no_warmup
  - history: reports/runs_server_ablation_20260424_150157/20260424_070213_ablation_image_eff_no_warmup/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070213_ablation_image_eff_no_warmup/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070213_ablation_image_eff_no_warmup/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070213_ablation_image_eff_no_warmup/summary.json
- ablation_text_eff_window8: reports/runs_server_ablation_20260424_150157/20260424_070219_ablation_text_eff_window8
  - history: reports/runs_server_ablation_20260424_150157/20260424_070219_ablation_text_eff_window8/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070219_ablation_text_eff_window8/summary.json
- ablation_text_eff_window16: reports/runs_server_ablation_20260424_150157/20260424_070223_ablation_text_eff_window16
  - history: reports/runs_server_ablation_20260424_150157/20260424_070223_ablation_text_eff_window16/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070223_ablation_text_eff_window16/summary.json
- ablation_text_eff_window32: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_text_eff_window32
  - history: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_text_eff_window32/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_text_eff_window32/summary.json
- ablation_no_composition: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_no_composition
  - history: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_no_composition/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_no_composition/summary.json
- ablation_no_attractor: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_no_attractor
  - history: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_no_attractor/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_no_attractor/summary.json
- ablation_head_without_ambiguity: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_head_without_ambiguity
  - history: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_head_without_ambiguity/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_head_without_ambiguity/summary.json
- ablation_sigmoid_gate_mean: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_sigmoid_gate_mean
  - history: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_sigmoid_gate_mean/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_sigmoid_gate_mean/summary.json
- ablation_thresholded_null: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_thresholded_null
  - history: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_thresholded_null/history.json
  - metrics: MISSING
  - diagnostics: MISSING
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070227_ablation_thresholded_null/summary.json
- cifar_cnn_eml: reports/runs_server_ablation_20260424_150157/20260424_070351_cifar_cnn_eml
  - history: reports/runs_server_ablation_20260424_150157/20260424_070351_cifar_cnn_eml/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070351_cifar_cnn_eml/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070351_cifar_cnn_eml/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070351_cifar_cnn_eml/summary.json
- cifar_pure_eml: reports/runs_server_ablation_20260424_150157/20260424_070447_cifar_pure_eml
  - history: reports/runs_server_ablation_20260424_150157/20260424_070447_cifar_pure_eml/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070447_cifar_pure_eml/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070447_cifar_pure_eml/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070447_cifar_pure_eml/summary.json
- cifar_pure_eml_v2: reports/runs_server_ablation_20260424_150157/20260424_070537_cifar_pure_eml_v2
  - history: reports/runs_server_ablation_20260424_150157/20260424_070537_cifar_pure_eml_v2/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070537_cifar_pure_eml_v2/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070537_cifar_pure_eml_v2/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070537_cifar_pure_eml_v2/summary.json
- cifar_efficient_eml_image: reports/runs_server_ablation_20260424_150157/20260424_070721_cifar_efficient_eml_image
  - history: reports/runs_server_ablation_20260424_150157/20260424_070721_cifar_efficient_eml_image/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070721_cifar_efficient_eml_image/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070721_cifar_efficient_eml_image/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070721_cifar_efficient_eml_image/summary.json
- ablation_text_eff_window8_workers0: reports/runs_server_ablation_20260424_150157/20260424_070846_ablation_text_eff_window8_workers0
  - history: reports/runs_server_ablation_20260424_150157/20260424_070846_ablation_text_eff_window8_workers0/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070846_ablation_text_eff_window8_workers0/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070846_ablation_text_eff_window8_workers0/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070846_ablation_text_eff_window8_workers0/summary.json
- ablation_text_eff_window16_workers0: reports/runs_server_ablation_20260424_150157/20260424_070914_ablation_text_eff_window16_workers0
  - history: reports/runs_server_ablation_20260424_150157/20260424_070914_ablation_text_eff_window16_workers0/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070914_ablation_text_eff_window16_workers0/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070914_ablation_text_eff_window16_workers0/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070914_ablation_text_eff_window16_workers0/summary.json
- ablation_text_eff_window32_workers0: reports/runs_server_ablation_20260424_150157/20260424_070939_ablation_text_eff_window32_workers0
  - history: reports/runs_server_ablation_20260424_150157/20260424_070939_ablation_text_eff_window32_workers0/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_070939_ablation_text_eff_window32_workers0/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_070939_ablation_text_eff_window32_workers0/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_070939_ablation_text_eff_window32_workers0/summary.json
- ablation_image_cnn_eml_workers0: reports/runs_server_ablation_20260424_150157/20260424_071204_ablation_image_cnn_eml_workers0
  - history: reports/runs_server_ablation_20260424_150157/20260424_071204_ablation_image_cnn_eml_workers0/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_071204_ablation_image_cnn_eml_workers0/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_071204_ablation_image_cnn_eml_workers0/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_071204_ablation_image_cnn_eml_workers0/summary.json
- ablation_image_pure_eml_workers0: reports/runs_server_ablation_20260424_150157/20260424_071222_ablation_image_pure_eml_workers0
  - history: reports/runs_server_ablation_20260424_150157/20260424_071222_ablation_image_pure_eml_workers0/history.json
  - metrics: reports/runs_server_ablation_20260424_150157/20260424_071222_ablation_image_pure_eml_workers0/metrics.csv
  - diagnostics: reports/runs_server_ablation_20260424_150157/20260424_071222_ablation_image_pure_eml_workers0/diagnostics.csv
  - summary: reports/runs_server_ablation_20260424_150157/20260424_071222_ablation_image_pure_eml_workers0/summary.json

## 14. Appendix: Commands

```bash
pytest
python scripts/run_eml_validation_suite.py --mode smoke --device cpu
python scripts/generate_eml_report.py
python scripts/run_eml_validation_suite.py --mode ablation --device cuda
python scripts/run_eml_validation_suite.py --mode cifar-medium --device cuda
python scripts/run_eml_validation_suite.py --mode text-medium --device cuda
```
