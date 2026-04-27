# EML Claim Status

Current status is based on completed local and real-server reports in `reports/`. Do not upgrade a claim unless a new measured report supports it.

| Claim | Status | Current evidence | Next validation target |
| --- | --- | --- | --- |
| EML as clean classifier head | Not proven | Frozen CNN feature and end-to-end head ablations do not clearly beat linear, MLP, or cosine prototype heads under matched conditions. | Keep linear/MLP/cosine as primary baselines and test uncertainty/calibration instead of claiming clean top-1. |
| EML as backbone | No-go currently | Efficient and field image backbones have not matched the stable CNN baselines in available reports. | Treat field/efficient modules as experimental branches until synthetic and CIFAR ablations improve. |
| MERC head | No-go currently | Real-server CIFAR frozen-feature results show MERC heads below the best MLP/cosine baselines. | Redesign only after toy and synthetic evidence tasks show robust support/conflict alignment. |
| MERC block | No-go currently | End-to-end CIFAR runs show MERC residual block variants made performance worse. | Do not insert MERC blocks into main baselines without a new positive ablation. |
| Support factorization | Preliminary positive | MERC support factors show positive synthetic evidence alignment in limited evidence-label runs. | Measure support-evidence correlation across seeds and harder synthetic tasks. |
| Resistance/conflict factorization | Not working yet | Conflict/resistance alignment is weak or negative in current synthetic evidence measurements. | Focus on resistance supervision, corruption labels, and conflict-specific probes. |
| Uncertainty/selective classification | Next target | Existing uncertainty benchmark suggests selective prediction may be a more plausible target than clean top-1, but evidence is incomplete. | Run `scripts/run_uncertainty_resistance_benchmark.py` on synthetic and corrupted CIFAR, record calibration, selective risk, AUROC, and resistance correlations. |

## Claim Rules

- Do not claim EML beats cosine on clean accuracy unless a matched report shows it.
- Do not claim MERC works as a head or block from toy-only results.
- Mark CIFAR/corruption experiments as `NOT RUN` when `torchvision` or local data is unavailable.
- Treat field, efficient, and MERC modules as experimental until their reports justify promotion.
