# Remote Real-Data CIFAR Head Ablation Report

## Summary

This report summarizes the real-data run performed on server `211.71.76.29`.
The run used CIFAR-10 from the server-local dataset path and did not download data.

Main result: this run does **not** prove that the EML prototype head is better than ordinary CNN heads on the same CIFAR-10 setup. The generated full report records mixed paired evidence: centered EML wins `5/21` paired comparisons.

## Environment

| field | value |
| --- | --- |
| server | `211.71.76.29` |
| hostname | `user-NF5468M6` |
| remote user | `hyq` |
| remote workspace | `/data16T/hyq/eml-network-realtest-20260424_162523` |
| Python | `3.10.20` |
| Torch | `2.7.1+cu118` |
| Torchvision | `0.22.1+cu118` |
| CUDA visible to job | `True` |
| GPU used | `NVIDIA RTX 5880 Ada Generation` |
| Titan XP usage | Not used; job ran with `CUDA_VISIBLE_DEVICES=1` |
| num_workers | `0` |

## Dataset

| field | value |
| --- | --- |
| dataset | CIFAR-10 |
| server path | `/data16T/hyq/dataset/data/cifar-10-batches-py` |
| root passed to torchvision | `/data16T/hyq/dataset/data` |
| download | `False` |
| seeds | `0, 1, 2` |
| train subset per seed | `2048` |
| validation subset per seed | `512` |
| test subset per seed | `512` |

## Commands Run

```bash
CUDA_VISIBLE_DEVICES=1 /data16T/hyq/miniconda3/envs/simgcd/bin/python \
  scripts/run_head_ablation.py \
  --dataset cifar10 --mode medium --seeds 0 1 2 --device cuda \
  --data-dir /data16T/hyq/dataset/data --num-workers 0 \
  --batch-size 64 --steps 200 --feature-steps 100 --force-features

CUDA_VISIBLE_DEVICES=1 /data16T/hyq/miniconda3/envs/simgcd/bin/python \
  scripts/run_cnn_head_end_to_end_ablation.py \
  --dataset cifar10 --mode medium --seeds 0 1 2 --device cuda \
  --data-dir /data16T/hyq/dataset/data --num-workers 0 \
  --batch-size 64 --steps 200

CUDA_VISIBLE_DEVICES=1 /data16T/hyq/miniconda3/envs/simgcd/bin/python \
  scripts/generate_head_ablation_report.py \
  --runs-root reports/head_ablation/runs \
  --output reports/HEAD_ABLATION_REPORT.md
```

## Run Status

| status | count |
| --- | ---: |
| COMPLETED | 48 |
| NOT RUN | 6 |
| FAILED | 0 |

The `NOT RUN` entries are expected: CE + prototype-pairwise loss is not applicable to the linear and MLP heads.

## Mean Results Across Seeds

| experiment | model | loss mode | n | mean test acc | mean test loss | mean train sec |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| end_to_end | cosine_prototype | ce | 3 | 0.4570 | 1.5469 | 4.6696 |
| end_to_end | cosine_prototype | ce_pairwise | 3 | 0.4538 | 1.5502 | 4.6189 |
| end_to_end | eml_bank_centered_ambiguity | ce | 3 | 0.4238 | 1.6442 | 7.2876 |
| end_to_end | eml_bank_centered_ambiguity | ce_pairwise | 3 | 0.4251 | 1.6662 | 7.2701 |
| end_to_end | eml_centered_ambiguity | ce | 3 | 0.4186 | 1.6414 | 6.3565 |
| end_to_end | eml_centered_ambiguity | ce_pairwise | 3 | 0.3906 | 1.6665 | 6.2948 |
| end_to_end | eml_no_ambiguity | ce | 3 | 0.3906 | 1.6791 | 6.1553 |
| end_to_end | eml_no_ambiguity | ce_pairwise | 3 | 0.4245 | 1.6157 | 6.0321 |
| end_to_end | linear | ce | 3 | 0.4714 | 1.4435 | 3.9977 |
| end_to_end | mlp | ce | 3 | 0.4212 | 1.6126 | 4.0132 |
| frozen_features | cosine_prototype | ce/frozen | 3 | 0.4382 | 1.5888 | 1.4541 |
| frozen_features | eml_centered_ambiguity | ce/frozen | 3 | 0.4219 | 1.6595 | 2.9840 |
| frozen_features | eml_no_ambiguity | ce/frozen | 3 | 0.4180 | 1.6619 | 2.9281 |
| frozen_features | eml_raw_ambiguity | ce/frozen | 3 | 0.4225 | 1.6601 | 2.9442 |
| frozen_features | linear | ce/frozen | 3 | 0.4368 | 1.5852 | 0.9692 |
| frozen_features | mlp | ce/frozen | 3 | 0.4310 | 1.5494 | 0.9398 |

## Best Single Runs

| experiment | best model | seed | test accuracy | test loss |
| --- | --- | ---: | ---: | ---: |
| frozen_features | cosine_prototype | 2 | 0.4746 | 1.5104 |
| end_to_end | linear, CE | 0 | 0.4824 | 1.4377 |
| end_to_end | EML bank centered ambiguity, CE + pairwise | 2 | 0.4824 | 1.5528 |

## Interpretation

- Frozen CNN features: cosine prototype and linear heads beat the EML heads on mean test accuracy.
- End-to-end CE-only: the linear head has the best mean accuracy and lowest mean loss.
- End-to-end CE + pairwise: cosine prototype remains the best mean prototype-family result.
- EML bank + centered ambiguity has one tied best single-seed accuracy, but its mean accuracy is below the linear CE baseline and its loss is higher.
- On this real CIFAR-10 medium run, the EML head contribution is **not proven** better than ordinary linear, MLP, or cosine heads.

## Artifact Paths

- Full generated report: `reports/remote_realtest_20260424_162523/HEAD_ABLATION_REPORT.md`
- Raw run summaries: `reports/remote_realtest_20260424_162523/runs/summary.csv`
- Per-run metrics/configs/diagnostics: `reports/remote_realtest_20260424_162523/runs/`
- Remote source workspace: `/data16T/hyq/eml-network-realtest-20260424_162523`

## Known Limitations

- This was a medium run on CIFAR-10 subsets, not a full CIFAR-10 training protocol.
- Only the head-ablation real-data path was run here; representation-trunk ablations were not rerun in this remote job.
- Results are limited to seeds `0, 1, 2`.
- No download was performed; only server-local CIFAR-10 was used.
