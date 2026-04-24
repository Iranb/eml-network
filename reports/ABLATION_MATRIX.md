# EML Ablation Matrix

## Matrix Policy

The full cross-product is intentionally not run by default. The validation suite uses one-factor-at-a-time ablations, baseline runs, and best-combined runs. Missing or unsupported cells must be marked `NOT RUN` or `NOT AVAILABLE`.

## Image Model Families

| id | model | status | primary question |
| --- | --- | --- | --- |
| IMG-A | `cnn_eml` | AVAILABLE | Strong practical baseline; is EML mostly useful after CNN features? |
| IMG-B | `pure_eml` | AVAILABLE | How weak is older pure patch EML? |
| IMG-C | `pure_eml_v2` | AVAILABLE | Does local EML image processing improve over older pure path? |
| IMG-D | `EfficientEMLImageClassifier` | AVAILABLE | Does local responsibility/precision/attractor representation help? |
| IMG-E | `LocalConvBaseline` | NOT AVAILABLE | Needed to separate local convolution benefit from EML benefit. |

## Text Model Families

| id | model | status | primary question |
| --- | --- | --- | --- |
| TXT-A | `LocalTextCodec + simple head` | NOT AVAILABLE AS STANDARD RUN | Needed as non-EML local text baseline. |
| TXT-B | `EMLTextBackbone + LocalTextGenerationHead` | AVAILABLE | Does residual-bank text EML help over codec? |
| TXT-C | `EfficientEMLTextEncoder + EfficientEMLTextGenerationHead` | AVAILABLE | Does EML representation path support causal text modeling? |
| TXT-D | `small recurrent/conv sanity baseline` | NOT AVAILABLE | Useful future baseline. |

## Mechanism Ablations

| axis | values | current support |
| --- | --- | --- |
| Responsibility propagation | `sigmoid_gate_sum`, `sigmoid_gate_mean`, `normalized_responsibility_no_null`, `normalized_responsibility_with_null`, `normalized_responsibility_with_null_and_threshold` | partial: graph supports responsibility on/off and null on/off; threshold not implemented. |
| Null behavior | `no_null`, `null_logit_fixed_0`, `null_logit_with_margin`, `evidence_threshold_learned`, `evidence_threshold_fixed` | partial: no-null and fixed null logit exist. |
| State update | `sigmoid_update`, `precision_update`, `precision_update_identity_init`, `precision_update_no_old_confidence` | partial: sigmoid and precision exist. |
| Composition | `no_composition`, `mean_pool_composition`, `EML_composition` | partial: EML composition exists; no-composition/mean-pool toggles need wrappers. |
| Attractor memory | `no_attractor`, `attractor_4`, `attractor_8`, `attractor_16` | partial: attractor count configurable; no-attractor wrapper not standardized. |
| Classification head | `linear_head`, `cosine_prototype_head`, `EMLPrototypeClassifier_without_ambiguity`, `EMLPrototypeClassifier_with_ambiguity` | partial: ambiguity-aware head exists; simpler heads need wrappers. |
| Loss terms | `CE only`, `CE + pairwise`, `CE + resistance`, `CE + energy clipping`, `CE + responsibility entropy`, `full loss` | partial: several loss helpers exist; suite starts with CE-only. |
| Warmup | `disabled`, `linear`, `cosine`, `exp branch always on` | partial: disabled/linear supported by runners. |
| Local window | image `3`, image `5`, text `8`, text `16`, text `32` | supported through efficient/field configs. |

## Minimal Factorial Plan

| run group | planned runs | required status behavior |
| --- | --- | --- |
| smoke | one synthetic image old baseline, one synthetic image efficient model, one synthetic text efficient model, graph responsibility probes, update probes | must run on CPU if dependencies work. |
| ablation | responsibility off/on/no-null/null, sigmoid vs precision update, warmup off/on, attractor count 4 vs 8 where supported | run on GPU if available; otherwise `NOT RUN`. |
| cifar-medium | `cnn_eml`, `pure_eml_v2`, `EfficientEMLImageClassifier`, optional `cnn_eml_stage` | run only when CIFAR-10 and torchvision are available. |
| text-medium | `EMLTextBackbone`, `EfficientEMLTextEncoder`, optional local baseline | run on synthetic text by default; local corpus optional. |

## Recommended Default Seeds

| mode | seeds |
| --- | --- |
| smoke | `0` |
| ablation | `0`, `1` minimum; `0`, `1`, `2` preferred |
| medium | `0` unless resources allow repeat |

## Metrics Required Per Group

| group | primary metrics | required diagnostics |
| --- | --- | --- |
| image | CE loss, accuracy, examples/sec | drive/resistance/energy, ambiguity, sample uncertainty, attractor diversity if present |
| text | next-token CE, token accuracy, perplexity | drive/resistance/energy, null weight, responsibility entropy, update strength |
| graph probes | message norm, weight mass, null weight, finite checks | responsibility entropy, update strength, update norm |
| efficiency | wall time, step time, parameter count, peak memory if available | local positions, window size, attractor count if exposed |
