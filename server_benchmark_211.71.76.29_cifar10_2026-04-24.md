# Server Benchmark: 211.71.76.29

Date: 2026-04-24

## Setup

- Server: `211.71.76.29`
- Remote repo path: `~/code/EML`
- Dataset root: `~/dataset/data`
- Dataset used: `cifar10`
- Python: `3.10.20`
- Torch: `2.7.1+cu118`
- Torchvision: `0.22.1+cu118`
- GPU used: `NVIDIA TITAN Xp`
- Benchmark type: fixed-budget short training benchmark

## Budget

- Batch size: `32`
- Train batches per model: `50`
- Eval batches per model: `10`
- Models compared:
  - `cnn_eml`
  - `cnn_eml_stage`
  - `pure_eml`
  - `pure_eml_v2`

## Results

| Model | Train Loss | Train Acc | Eval Loss | Eval Acc | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cnn_eml` | 2.4076 | 0.2169 | 2.2707 | 0.3188 | 5.24 |
| `cnn_eml_stage` | 2.4104 | 0.1838 | 2.4150 | 0.1844 | 8.24 |
| `pure_eml` | 2.4496 | 0.1625 | 2.4881 | 0.1344 | 1.71 |
| `pure_eml_v2` | 2.2096 | 0.1594 | 2.1807 | 0.2313 | 5.92 |

## Ranking By Eval Accuracy

1. `cnn_eml` — `0.3188`
2. `pure_eml_v2` — `0.2313`
3. `cnn_eml_stage` — `0.1844`
4. `pure_eml` — `0.1344`

## Additional Eval Metrics

| Model | Eval CE | Eval Pairwise | Eval Resistance | Eval Prototype Diversity | Eval Entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cnn_eml` | 2.0286 | 0.9482 | 1.0431 | 0.0384 | 2.2193 |
| `cnn_eml_stage` | 2.1510 | 1.0414 | 1.1066 | 0.0374 | 2.2199 |
| `pure_eml` | 2.2248 | 1.0314 | 1.1375 | 0.0108 | 2.2488 |
| `pure_eml_v2` | 2.1802 | 1.1783 | 1.1396 | 0.0466 | 2.1737 |

## Notes

- This was not a full convergence run. It was a controlled short-budget comparison to get same-server, same-dataset, same-environment numbers across the existing image models.
- An earlier attempt using a larger shared GPU ran into out-of-memory due concurrent users on that device. The final numbers above are from the stable rerun on the TITAN Xp with a reduced fixed budget.
- The strongest early result in this run was `cnn_eml`.
- `pure_eml_v2` was the best pure EML-style image model in this short CIFAR-10 benchmark.
