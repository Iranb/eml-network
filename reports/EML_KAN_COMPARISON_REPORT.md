# EML KAN-Style Comparison Report

## Scope
- Local experiments run only EML-native models on existing synthetic image/text validation tasks.
- KAN is paper-only per request: https://arxiv.org/abs/2404.19756
- EML operator reference: https://arxiv.org/html/2603.21852v2
- This is not evidence that EML beats KAN on KAN's original function-fitting/PDE tasks; task families differ.
- Early-stop completeness: 7/7 local comparison rows early-stopped.

## Architecture Translation
- KAN paper structure used here: layers are matrices of learnable univariate edge functions and destination nodes sum incoming edge outputs.
- Local EML implementation: each edge function is `base(silu(x)) + scale * sEML(drive(x), resistance(x))`.
- Direct `exp(x) - log(y)` is not stacked raw; the repository's stable sEML primitive keeps fp32 islands and bounded drive for training stability.

## Paper-Only KAN Comparator
| source | comparator status | relevant result | local action |
| --- | --- | --- | --- |
| KAN arXiv 2404.19756 | NOT RUN | Reports smaller KANs can match or beat larger MLPs on small AI+Science function-fitting tasks, while Feynman KAN/MLP behavior is comparable on average. | Marked `NOT RUN`; no local KAN experiment was run. |
| EML arXiv 2603.21852v2 | Reference primitive | Defines EML as a universal binary expression-tree operator and reports symbolic-regression proof-of-concept, with harder blind recovery at deeper trees. | Implemented stable sEML edge functions, not raw complex EML trees. |

## Aggregated Image Results
| model | n | best metric | mean final metric | mean loss | mean params |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMLEdgeImageClassifier_kan_style | 1 | 0.3750 | 0.1250 | 1.6120 | 10889.0000 |
| EfficientEMLImageClassifier | 1 | 0.3750 | 0.2500 | 1.6057 | 115573.0000 |
| cnn_eml | 1 | 0.6250 | 0.2500 | 1.5949 | 162644.0000 |

## Aggregated Text Results
| model | n | best metric | mean final metric | mean loss | mean params |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMLEdgeTextLM_kan_style | 1 | 0.4608 | 0.2995 | 3.9456 | 53736.0000 |
| EfficientEMLTextEncoder | 1 | 0.4294 | 0.3430 | 3.1884 | 92950.0000 |
| LocalCausalConvLM | 1 | 0.4505 | 0.3909 | 2.3376 | 11668.0000 |
| SmallGRULM | 1 | 0.2227 | 0.1053 | 3.7472 | 11796.0000 |

## Image Runs
| run_id | status | model | best | final | loss | steps | early stop | params | time sec | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| kan_compare_image_cnn_eml | COMPLETED | cnn_eml | 0.6250 | 0.2500 | 1.5949 | 20 | True | 162644 | 0.2468 |  |
| kan_compare_image_efficient_eml | COMPLETED | EfficientEMLImageClassifier | 0.3750 | 0.2500 | 1.6057 | 21 | True | 115573 | 0.5140 |  |
| kan_compare_image_eml_edge | COMPLETED | EMLEdgeImageClassifier_kan_style | 0.3750 | 0.1250 | 1.6120 | 11 | True | 10889 | 0.0741 |  |

## Text Runs
| run_id | status | model | best | final | loss | steps | early stop | params | time sec | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| kan_compare_text_local_conv | COMPLETED | LocalCausalConvLM | 0.4505 | 0.3909 | 2.3376 | 94 | True | 11668 | 0.3432 |  |
| kan_compare_text_small_gru | COMPLETED | SmallGRULM | 0.2227 | 0.1053 | 3.7472 | 68 | True | 11796 | 0.4306 |  |
| kan_compare_text_efficient_eml | COMPLETED | EfficientEMLTextEncoder | 0.4294 | 0.3430 | 3.1884 | 177 | True | 92950 | 2.8431 |  |
| kan_compare_text_eml_edge | COMPLETED | EMLEdgeTextLM_kan_style | 0.4608 | 0.2995 | 3.9456 | 70 | True | 53736 | 3.3604 |  |

## Missing Or Failed
| run_id | status | model | reason |
| --- | --- | --- | --- |
| kan_paper_reference | NOT RUN | KAN_arxiv_2404_19756 | User requested no local KAN experiment; comparing against reported paper results only. |

## Raw Artifacts
- `kan_paper_reference`: `reports/runs/20260505_123237_kan_paper_reference`
- `kan_compare_image_cnn_eml`: `reports/runs/20260505_123238_kan_compare_image_cnn_eml`
- `kan_compare_image_efficient_eml`: `reports/runs/20260505_123238_kan_compare_image_efficient_eml`
- `kan_compare_image_eml_edge`: `reports/runs/20260505_123239_kan_compare_image_eml_edge`
- `kan_compare_text_local_conv`: `reports/runs/20260505_123239_kan_compare_text_local_conv`
- `kan_compare_text_small_gru`: `reports/runs/20260505_123239_kan_compare_text_small_gru`
- `kan_compare_text_efficient_eml`: `reports/runs/20260505_123240_kan_compare_text_efficient_eml`
- `kan_compare_text_eml_edge`: `reports/runs/20260505_123242_kan_compare_text_eml_edge`
