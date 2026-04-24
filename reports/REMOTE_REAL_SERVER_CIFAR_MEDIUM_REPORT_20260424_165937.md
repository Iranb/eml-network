# Remote Real Server CIFAR-10 Medium Report

## Executive Summary
- Server: `211.71.76.29`
- Remote repo copy: `/data16T/hyq/eml-network-realtest-20260424_165620`
- Remote run root: `reports/real_cifar_20260424_165937`
- Dataset: CIFAR-10 from `/data16T/hyq/dataset/data` (`50000` train, `10000` test available through Torchvision).
- Run scale: script `medium` mode, using `2048` train / `512` val / `512` test examples per seed.
- GPU constraint: Titan Xp was not exposed to training commands; `CUDA_VISIBLE_DEVICES=1` made only `NVIDIA RTX 5880 Ada Generation` visible.
- Completed runs: `48`; NOT RUN: `6`; failed: `0`.
- Best single run: `e2e_cifar10_cosine_prototype_ce_pairwise_seed2` with test accuracy `0.5645` and test loss `1.3460`.
- Claim status: EML heads are not proven better on this run; the best end-to-end results are ordinary cosine-prototype heads.

## Environment Check
```text
cuda_available True
visible_device_count 1
visible_device_0 NVIDIA RTX 5880 Ada Generation
```

## Frozen CNN Feature Head Isolation
| model | n | mean test acc | best test acc | mean test loss | mean final train acc | mean final train loss | mean ECE | mean Brier | mean time sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cosine_prototype | 3 | 0.5078 | 0.5176 | 1.3834 | 0.5938 | 1.2024 | 0.0795 | 0.6306 | 3.0840 |
| eml_centered_ambiguity | 3 | 0.5117 | 0.5312 | 1.4684 | 0.5938 | 1.2585 | 0.0918 | 0.6445 | 9.9945 |
| eml_no_ambiguity | 3 | 0.5111 | 0.5273 | 1.4717 | 0.5990 | 1.2628 | 0.0886 | 0.6457 | 6.2493 |
| eml_raw_ambiguity | 3 | 0.5111 | 0.5273 | 1.4693 | 0.5938 | 1.2598 | 0.0903 | 0.6448 | 6.3172 |
| linear | 3 | 0.5130 | 0.5312 | 1.3422 | 0.6094 | 1.1295 | 0.0574 | 0.6191 | 1.8231 |
| mlp | 3 | 0.5280 | 0.5391 | 1.3836 | 0.6354 | 0.9843 | 0.1086 | 0.6276 | 1.8499 |

## End-To-End CNN + Head, CE Only
| model | n | mean test acc | best test acc | mean test loss | mean final train acc | mean final train loss | mean ECE | mean Brier | mean time sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cosine_prototype | 3 | 0.5508 | 0.5605 | 1.3416 | 0.7708 | 0.8934 | 0.0618 | 0.6012 | 7.6565 |
| eml_bank_centered_ambiguity | 3 | 0.4323 | 0.4570 | 1.6120 | 0.5469 | 1.3244 | 0.0690 | 0.7074 | 12.8654 |
| eml_centered_ambiguity | 3 | 0.4349 | 0.4570 | 1.5986 | 0.4740 | 1.4912 | 0.0506 | 0.7094 | 11.2900 |
| eml_no_ambiguity | 3 | 0.4336 | 0.4883 | 1.6000 | 0.4479 | 1.4281 | 0.0822 | 0.7097 | 14.0689 |
| linear | 3 | 0.5013 | 0.5352 | 1.4124 | 0.6615 | 0.9231 | 0.0778 | 0.6391 | 7.3140 |
| mlp | 3 | 0.4492 | 0.4629 | 1.5275 | 0.6094 | 1.0892 | 0.0883 | 0.6773 | 6.4515 |

## End-To-End CNN + Head, CE + Prototype Pairwise
| model | n | mean test acc | best test acc | mean test loss | mean final train acc | mean final train loss | mean ECE | mean Brier | mean time sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cosine_prototype | 3 | 0.5456 | 0.5645 | 1.3364 | 0.7656 | 0.8931 | 0.0601 | 0.6000 | 8.5326 |
| eml_bank_centered_ambiguity | 3 | 0.4355 | 0.4551 | 1.6508 | 0.6146 | 1.2959 | 0.0684 | 0.7202 | 12.9150 |
| eml_centered_ambiguity | 3 | 0.4277 | 0.4688 | 1.6263 | 0.4792 | 1.4603 | 0.0723 | 0.7149 | 11.4821 |
| eml_no_ambiguity | 3 | 0.4134 | 0.4375 | 1.6516 | 0.4427 | 1.4791 | 0.0594 | 0.7279 | 11.0807 |

## NOT RUN Entries
| run_id | model | reason |
| --- | --- | --- |
| e2e_cifar10_linear_ce_pairwise_seed0 | linear | pairwise prototype margin is not applicable |
| e2e_cifar10_mlp_ce_pairwise_seed0 | mlp | pairwise prototype margin is not applicable |
| e2e_cifar10_linear_ce_pairwise_seed1 | linear | pairwise prototype margin is not applicable |
| e2e_cifar10_mlp_ce_pairwise_seed1 | mlp | pairwise prototype margin is not applicable |
| e2e_cifar10_linear_ce_pairwise_seed2 | linear | pairwise prototype margin is not applicable |
| e2e_cifar10_mlp_ce_pairwise_seed2 | mlp | pairwise prototype margin is not applicable |

## Raw Artifacts
- Full generated report: `reports/remote_realtest_real_cifar_20260424_165937/HEAD_ABLATION_REPORT.md`
- Summary CSV: `reports/remote_realtest_real_cifar_20260424_165937/runs/summary.csv`
- Run directories: `reports/remote_realtest_real_cifar_20260424_165937/runs`
- Remote logs: `reports/remote_realtest_real_cifar_20260424_165937/logs`

## Commands Used
```bash
CUDA_VISIBLE_DEVICES=1 /data16T/hyq/miniconda3/envs/simgcd/bin/python scripts/run_head_ablation.py --dataset cifar10 --mode medium --seeds 0 1 2 --device cuda --data-dir /data16T/hyq/dataset/data --num-workers 0 --batch-size 64 --runs-root reports/real_cifar_20260424_165937/runs --features-root reports/real_cifar_20260424_165937/features
CUDA_VISIBLE_DEVICES=1 /data16T/hyq/miniconda3/envs/simgcd/bin/python scripts/run_cnn_head_end_to_end_ablation.py --dataset cifar10 --mode medium --seeds 0 1 2 --device cuda --data-dir /data16T/hyq/dataset/data --num-workers 0 --batch-size 64 --runs-root reports/real_cifar_20260424_165937/runs
CUDA_VISIBLE_DEVICES=1 /data16T/hyq/miniconda3/envs/simgcd/bin/python scripts/generate_head_ablation_report.py --runs-root reports/real_cifar_20260424_165937/runs --output reports/real_cifar_20260424_165937/HEAD_ABLATION_REPORT.md
```
