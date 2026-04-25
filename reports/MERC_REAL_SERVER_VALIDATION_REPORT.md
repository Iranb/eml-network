# MERC Real Server Validation Report

## 1. Environment

- Server: `211.71.76.29`
- Remote workspace: `/data16T/hyq/eml-network-merc-20260425_105808`
- Dataset root: `/data16T/hyq/dataset/data`
- Python: `/data16T/hyq/miniconda3/envs/simgcd/bin/python`
- Device policy: `CUDA_VISIBLE_DEVICES=1`
- Visible training GPU: `NVIDIA RTX 5880 Ada Generation`
- Titan Xp usage: not used
- Seeds: `0 1 2`
- Batch size: `64`
- DataLoader workers: `0`
- Early stop settings: `patience=4`, `min_evals=3`

## 2. Commands Run

```bash
CUDA_VISIBLE_DEVICES=1 /data16T/hyq/miniconda3/envs/simgcd/bin/python scripts/run_merc_toy_experiments.py --mode medium --device cuda --seeds 0 1 2
CUDA_VISIBLE_DEVICES=1 /data16T/hyq/miniconda3/envs/simgcd/bin/python scripts/run_head_ablation.py --dataset cifar10 --mode medium --seeds 0 1 2 --device cuda --data-dir /data16T/hyq/dataset/data --num-workers 0 --batch-size 64 --include-merc --early-stop-patience 4 --early-stop-min-evals 3 --runs-root reports/merc_head_ablation/runs
CUDA_VISIBLE_DEVICES=1 /data16T/hyq/miniconda3/envs/simgcd/bin/python scripts/run_cnn_head_end_to_end_ablation.py --dataset cifar10 --mode medium --seeds 0 1 2 --device cuda --data-dir /data16T/hyq/dataset/data --num-workers 0 --batch-size 64 --include-merc --early-stop-patience 4 --early-stop-min-evals 3 --runs-root reports/merc_end_to_end_rerun/runs
CUDA_VISIBLE_DEVICES=1 /data16T/hyq/miniconda3/envs/simgcd/bin/python scripts/run_merc_synthetic_evidence.py --mode medium --device cuda --seeds 0 1 2 --num-workers 0 --data-dir /data16T/hyq/dataset/data
```

## 3. Toy Nonlinear Tasks

The toy runs support the MERC hypothesis only partially.

- Conjunctive evidence:
  - MERC reached `>= 0.9` accuracy much faster than linear and MLP in all three seeds.
  - Example: `merc_toy_conjunctive_merc_seed0` reached `0.9766` best accuracy and hit `0.9` by step `3`.
- Conflict suppression:
  - MERC and `merc_energy` were competitive with MLP and old EML gate.
  - Example bests: `merc` `0.9336`, `merc_energy` `0.9414`, `mlp` `0.9414`.
- XOR:
  - MERC did not establish an advantage.
  - Best MERC result was `0.6641` for `merc` on seed `0`, but other seeds were weak and inconsistent.

Conclusion from toy tasks:

- MERC shows useful multiplicative behavior on conjunctive evidence.
- MERC does not yet show robust superiority on harder interaction structure.
- The toy report conclusion remains valid: **MERC neuron design is not yet justified**.

## 4. Frozen CNN Feature Head Isolation on CIFAR-10

All heads used the same frozen CNN features.

### Best observed frozen-feature accuracies

| head | best test acc | mean test acc |
| --- | ---: | ---: |
| `mlp` | `0.5371` | `0.5273` |
| `merc_linear` | `0.5254` | `0.5169` |
| `eml_centered_ambiguity` | `0.5293` | `0.5137` |
| `cosine_prototype` | `0.5215` | `0.5091` |
| `linear` | `0.5293` | `0.5098` |
| `merc_energy` | `0.5059` | `0.4954` |

Observations:

- MERC did **not** beat MLP on frozen CNN features.
- `merc_linear` was slightly above cosine on mean accuracy, but below MLP and below the best linear/old-EML runs.
- `merc_energy` underperformed `merc_linear`.
- Frozen-feature claim is therefore **not strong enough to support a MERC head advantage**.

## 5. End-to-End CNN + Head on CIFAR-10

The end-to-end rerun includes the fixed `merc_block_*` variants.

### Best observed end-to-end accuracies

| model | best test acc | mean test acc |
| --- | ---: | ---: |
| `cosine_prototype` | `0.5605` | `0.5449` |
| `linear` | `0.5469` | `0.4974` |
| `eml_no_ambiguity` | `0.5020` | `0.4544` |
| `eml_centered_ambiguity` | `0.4785` | `0.4505` |
| `merc_linear` | `0.3848` | `0.3555` |
| `merc_energy` | `0.3730` | `0.3229` |
| `merc_block_energy` | `0.2520` | `0.2096` |
| `merc_block_linear` | `0.2070` | `0.1934` |

Observations:

- The strongest end-to-end baseline remains `cosine_prototype`.
- MERC heads underperform cosine, linear, and the better old EML variants.
- Adding a MERC residual block before the MERC head made performance materially worse.
- Pairwise runs were correctly marked `NOT RUN` for heads where prototype pairwise margin is not applicable.

End-to-end conclusion:

- **No-go** for MERC as the current CIFAR-10 head replacement.
- **No-go** for MERC residual block exploration in this form.

## 6. Synthetic Evidence / Conflict Diagnostics

Synthetic evidence runs were near-saturated in raw classification accuracy, so the useful signal is in the evidence/conflict alignment metrics.

### Mean synthetic evidence diagnostics

| model | mean test acc | support-evidence corr | conflict-resistance corr |
| --- | ---: | ---: | ---: |
| `merc_linear` | `0.9941` | `0.6066` | `-0.0802` |
| `merc_energy` | `0.9941` | `0.2176` | `-0.0676` |
| `merc_linear_small` | `0.9941` | `0.3200` | `-0.0865` |
| `merc_energy_small` | `0.9935` | `0.3997` | `-0.0722` |

Observations:

- MERC does expose support factors that correlate with synthetic evidence labels, especially `merc_linear`.
- Conflict/resistance alignment is weak and slightly negative in this setup.
- That means the support side is promising, but the conflict/resistance factorization is not working yet.

## 7. Calibration and Efficiency

- Frozen-feature MERC heads were slower than `mlp` and usually slower than cosine.
- End-to-end MERC heads were materially slower than cosine while also performing worse.
- Example:
  - frozen `mlp` best run: `0.5371` test accuracy in `2.17s`
  - frozen `merc_linear` best run: `0.5254` in `6.63s`
  - end-to-end cosine best run: `0.5605` in `8.23s`
  - end-to-end `merc_linear` best run: `0.3848` in `13.72s`

Calibration:

- MERC did not show a consistent calibration win.
- On frozen features, `merc_linear` ECE was generally worse than `linear` and `cosine`.
- On end-to-end CIFAR, MERC block variants had low ECE in some runs but at unusably low accuracy, which is not a useful trade.

## 8. Go / No-Go Conclusion

### Claim status

- Does MERC beat linear? **No on end-to-end CIFAR; mixed on frozen features.**
- Does MERC beat MLP? **No.**
- Does MERC beat cosine prototype? **No.**
- Does MERC beat old EML head? **No clear advantage.**
- Does MERC show support-factor alignment? **Yes, on synthetic evidence.**
- Does MERC show conflict/resistance alignment? **Not yet.**
- Is MERC worth using as a head right now? **No-go.**
- Is MERC worth exploring as a representation block right now? **No-go in the current implementation.**

### Recommended next step

Do not expand MERC into larger backbone usage yet. If this line continues, the work should narrow to:

1. support/conflict supervision quality on controlled synthetic tasks
2. conflict/resistance redesign before any larger CIFAR reruns
3. matched-parameter toy tasks where multiplicative support is the primary hypothesis under test

## 9. Raw Artifact Paths

- Detailed toy report: [reports/MERC_TOY_REPORT.md](/Users/iranb/Library/Mobile%20Documents/com~apple~CloudDocs/OpenClawThings/EML/reports/MERC_TOY_REPORT.md)
- Frozen-feature report: [reports/MERC_HEAD_ABLATION_REPORT.md](/Users/iranb/Library/Mobile%20Documents/com~apple~CloudDocs/OpenClawThings/EML/reports/MERC_HEAD_ABLATION_REPORT.md)
- End-to-end report: [reports/MERC_END_TO_END_REPORT.md](/Users/iranb/Library/Mobile%20Documents/com~apple~CloudDocs/OpenClawThings/EML/reports/MERC_END_TO_END_REPORT.md)
- Synthetic evidence report: [reports/MERC_SYNTHETIC_EVIDENCE_REPORT.md](/Users/iranb/Library/Mobile%20Documents/com~apple~CloudDocs/OpenClawThings/EML/reports/MERC_SYNTHETIC_EVIDENCE_REPORT.md)
- Remote summary CSVs:
  - [merc_toy_summary.csv](/Users/iranb/Library/Mobile%20Documents/com~apple~CloudDocs/OpenClawThings/EML/reports/remote_merc_validation_20260425_105808/summaries/merc_toy_summary.csv)
  - [merc_head_ablation_summary.csv](/Users/iranb/Library/Mobile%20Documents/com~apple~CloudDocs/OpenClawThings/EML/reports/remote_merc_validation_20260425_105808/summaries/merc_head_ablation_summary.csv)
  - [merc_end_to_end_summary.csv](/Users/iranb/Library/Mobile%20Documents/com~apple~CloudDocs/OpenClawThings/EML/reports/remote_merc_validation_20260425_105808/summaries/merc_end_to_end_summary.csv)
  - [merc_synthetic_evidence_summary.csv](/Users/iranb/Library/Mobile%20Documents/com~apple~CloudDocs/OpenClawThings/EML/reports/remote_merc_validation_20260425_105808/summaries/merc_synthetic_evidence_summary.csv)
