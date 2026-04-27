# Early-Stop Audit

Date: 2026-04-27

## Scope

This audit checks whether the current report artifacts under `reports/` were run until the configured early-stop condition, with special focus on the latest true uncertainty/resistance benchmark:

- `reports/EML_UNCERTAINTY_RESISTANCE_TRUE_REPORT.md`
- `reports/uncertainty_resistance_true_20260427/runs/summary.csv`

The repository contains older smoke, ablation, and archived run folders whose summaries either predate early-stop logging or intentionally used fixed-step smoke settings. Those are not treated as proof that the latest benchmark completed early-stop.

## Repository-Wide Report Scan

A scan of `reports/**/summary.csv` and `reports/**/summary.json` found:

| category | count |
| --- | ---: |
| total records scanned | 1864 |
| `early_stop_triggered=true` | 238 |
| `early_stop_triggered=false` | 540 |
| missing early-stop field | 1086 |

Interpretation:

- Many older artifacts do not record early-stop metadata.
- Many smoke/probe runs are fixed-budget by design.
- The latest true benchmark did record early-stop metadata and had rows that hit the configured step ceiling.

## Latest True Benchmark Before Rerun

`reports/uncertainty_resistance_true_20260427/runs/summary.csv` had 48 completed rows:

| status | count |
| --- | ---: |
| early-stopped | 34 |
| hit 250-step ceiling | 14 |
| missing early-stop metadata | 0 |

Rows that hit the 250-step ceiling:

| model | seed | old steps | old best step | old clean acc |
| --- | ---: | ---: | ---: | ---: |
| linear | 0 | 250 | 250 | 0.4596 |
| cosine_prototype | 0 | 250 | 175 | 0.4609 |
| eml_no_ambiguity | 0 | 250 | 200 | 0.4583 |
| eml_centered_ambiguity | 0 | 250 | 225 | 0.4635 |
| eml_supervised_resistance | 0 | 250 | 200 | 0.4544 |
| merc_linear | 0 | 250 | 200 | 0.4388 |
| merc_energy | 0 | 250 | 250 | 0.4401 |
| eml_no_ambiguity | 1 | 250 | 200 | 0.4753 |
| eml_supervised_resistance | 1 | 250 | 250 | 0.5091 |
| merc_linear | 1 | 250 | 200 | 0.4609 |
| merc_energy | 1 | 250 | 250 | 0.4792 |
| mlp | 2 | 250 | 200 | 0.5156 |
| eml_no_ambiguity | 2 | 250 | 250 | 0.4766 |
| merc_energy | 2 | 250 | 250 | 0.4987 |

Conclusion: the latest true benchmark did **not** fully run until early stop before this audit.

## Rerun Configuration

Server:

- host: `211.71.76.29`
- worktree: `/data16T/hyq/eml-network-true-20260427-worktree`
- dataset: `/data16T/hyq/dataset/data`
- Python: `/data16T/hyq/miniconda3/envs/simgcd/bin/python`
- device request: `cuda`
- GPU selection: `CUDA_VISIBLE_DEVICES=2`
- observed CUDA device: RTX 5880 Ada Generation
- Titan Xp was not selected.
- dataloader workers: `0`

The benchmark runner was updated to support targeted reruns:

- `--heads`
- `--backbone-steps`
- `--head-steps`
- `--eval-interval`

Rerun output:

- report: `reports/EML_UNCERTAINTY_RESISTANCE_EARLYSTOP_RERUN_REPORT.md`
- raw runs: `reports/uncertainty_resistance_true_earlystop_rerun_20260427/runs/`

Commands run:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/run_uncertainty_resistance_benchmark.py \
  --dataset cifar10 --mode medium --device cuda \
  --data-dir /data16T/hyq/dataset/data --num-workers 0 --batch-size 64 \
  --runs-root reports/uncertainty_resistance_true_earlystop_rerun_20260427/runs \
  --report reports/EML_UNCERTAINTY_RESISTANCE_EARLYSTOP_RERUN_REPORT.md \
  --seeds 0 \
  --heads linear cosine_prototype eml_no_ambiguity eml_centered_ambiguity eml_supervised_resistance merc_linear merc_energy \
  --backbone-steps 300 --head-steps 800 --eval-interval 25 \
  --early-stop-patience 4 --early-stop-min-evals 2

CUDA_VISIBLE_DEVICES=2 python scripts/run_uncertainty_resistance_benchmark.py \
  --dataset cifar10 --mode medium --device cuda \
  --data-dir /data16T/hyq/dataset/data --num-workers 0 --batch-size 64 \
  --runs-root reports/uncertainty_resistance_true_earlystop_rerun_20260427/runs \
  --report reports/EML_UNCERTAINTY_RESISTANCE_EARLYSTOP_RERUN_REPORT.md \
  --seeds 1 \
  --heads eml_no_ambiguity eml_supervised_resistance merc_linear merc_energy \
  --backbone-steps 300 --head-steps 1200 --eval-interval 25 \
  --early-stop-patience 4 --early-stop-min-evals 2

CUDA_VISIBLE_DEVICES=2 python scripts/run_uncertainty_resistance_benchmark.py \
  --dataset cifar10 --mode medium --device cuda \
  --data-dir /data16T/hyq/dataset/data --num-workers 0 --batch-size 64 \
  --runs-root reports/uncertainty_resistance_true_earlystop_rerun_20260427/runs \
  --report reports/EML_UNCERTAINTY_RESISTANCE_EARLYSTOP_RERUN_REPORT.md \
  --seeds 2 \
  --heads mlp eml_no_ambiguity merc_energy \
  --backbone-steps 300 --head-steps 1200 --eval-interval 25 \
  --early-stop-patience 4 --early-stop-min-evals 2

CUDA_VISIBLE_DEVICES=2 python scripts/run_uncertainty_resistance_benchmark.py \
  --dataset cifar10 --mode medium --device cuda \
  --data-dir /data16T/hyq/dataset/data --num-workers 0 --batch-size 64 \
  --runs-root reports/uncertainty_resistance_true_earlystop_rerun_20260427/runs \
  --report reports/EML_UNCERTAINTY_RESISTANCE_EARLYSTOP_RERUN_REPORT.md \
  --seeds 0 --heads merc_energy \
  --backbone-steps 300 --head-steps 1600 --eval-interval 25 \
  --early-stop-patience 4 --early-stop-min-evals 2
```

## Rerun Results

All 14 previously capped model/seed combinations now have a successful rerun row with `early_stop_triggered=true`.

| model | seed | early stop | rerun steps | best step | clean acc |
| --- | ---: | --- | ---: | ---: | ---: |
| linear | 0 | true | 225 | 125 | 0.4531 |
| cosine_prototype | 0 | true | 275 | 175 | 0.4596 |
| eml_no_ambiguity | 0 | true | 250 | 150 | 0.4102 |
| eml_centered_ambiguity | 0 | true | 250 | 150 | 0.4180 |
| eml_supervised_resistance | 0 | true | 250 | 150 | 0.4206 |
| merc_linear | 0 | true | 300 | 200 | 0.4388 |
| merc_energy | 0 | true | 675 | 575 | 0.4609 |
| eml_no_ambiguity | 1 | true | 150 | 50 | 0.3945 |
| eml_supervised_resistance | 1 | true | 400 | 300 | 0.4857 |
| merc_linear | 1 | true | 325 | 225 | 0.4779 |
| merc_energy | 1 | true | 375 | 275 | 0.4753 |
| mlp | 2 | true | 300 | 200 | 0.5130 |
| eml_no_ambiguity | 2 | true | 150 | 50 | 0.3867 |
| merc_energy | 2 | true | 125 | 25 | 0.1797 |

Notes:

- The first seed-0 `merc_energy` rerun with an 800-step ceiling still hit the cap at `steps_run=800`, `best_step=775`.
- It was superseded by the seed-0 `merc_energy` 1600-step rerun, which early-stopped at `steps_run=675`, `best_step=575`.
- The reruns retrain the frozen feature extractor per seed, so the accuracy values are replacement verification runs rather than exact continuations of the original capped rows.

## Final Status

For the latest true uncertainty/resistance benchmark:

- Original complete rows: 48
- Original early-stopped rows: 34
- Original capped rows requiring rerun: 14
- Capped rows with successful early-stop rerun: 14
- Remaining latest true benchmark rows without an early-stop or replacement: 0

Older reports still contain fixed-budget, missing-metadata, or non-early-stopped runs. They are retained as historical artifacts and should not be described as fully early-stopped unless their own focused rerun is performed.

