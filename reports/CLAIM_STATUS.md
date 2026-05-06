# EML Claim Status

Current status is based on completed local and real-server reports in `reports/`. Do not upgrade a claim unless a new measured report supports it.

| Claim | Status | Current evidence | Next validation target |
| --- | --- | --- | --- |
| EML as clean classifier head | Not proven | Frozen CNN feature and end-to-end head ablations do not clearly beat linear, MLP, or cosine prototype heads under matched conditions. | Keep linear/MLP/cosine as primary baselines and test uncertainty/calibration instead of claiming clean top-1. |
| EML as backbone | No-go currently | Efficient and field image backbones have not matched the stable CNN baselines in available reports. | Treat field/efficient modules as experimental branches until synthetic and CIFAR ablations improve. |
| KAN-style sEML edge-function network | Experimental, mixed | `reports/EML_KAN_COMPARISON_REPORT.md` shows the edge-function branch is trainable, but image accuracy remains below `cnn_eml`; text best token accuracy is competitive in one synthetic run but weaker than local conv/efficient EML on final loss/accuracy. `reports/EML_KAN_MLP_FAIR_COMPARISON_REPORT.md` adds a KANbeFair-style 41-server CUDA comparison against MLP baselines with 27/27 early-stopped rows: EML-KAN strongly wins symbolic regression vs same-width and parameter-matched MLP, loses localized regression to same-width MLP after sufficient training, and is mixed on shifted classification. | Treat EML-KAN as task-dependent. Add real KAN/spline and B-spline MLP references under matched parameter/FLOP settings before claiming operator superiority. |
| sEML as a KAN edge-operator replacement | Promising on long function-fitting run, not final | `reports/KAN_OPERATOR_REPLACEMENT_REAL_REPORT.md` runs a non-smoke 41-server CUDA comparison with larger data, wider topology, 3 tasks, and 3 seeds. sEML replacement has lower mean best validation MSE than the degree-1 spline KAN on all three tasks, despite fewer parameters. Some spline and local-bump rows still hit the configured step cap, so this is not yet a fully early-stopped final comparison. | Run an early-stop-complete rerun or raise the step cap for capped rows; then test parameter-matched spline/EML variants before making a durable replacement claim. |
| EML thresholded null mechanism | Mechanistically supported | Responsibility probes show high null weight under weak/all-noise evidence and high neighbor weight under strong evidence. | Test as a plug-in refinement under corruption before claiming downstream benefit. |
| EML precision update | Mechanistically supported | Precision-update probes show preserve/update behavior based on old confidence and new evidence. | Test as a plug-in update under noisy evidence before claiming representation benefit. |
| MERC head | No-go currently | Real-server CIFAR frozen-feature results show MERC heads below the best MLP/cosine baselines. | Redesign only after toy and synthetic evidence tasks show robust support/conflict alignment. |
| MERC block | No-go currently | End-to-end CIFAR runs show MERC residual block variants made performance worse. | Do not insert MERC blocks into main baselines without a new positive ablation. |
| Support factorization | Preliminary positive | MERC support factors show positive synthetic evidence alignment in limited evidence-label runs. | Measure support-evidence correlation across seeds and harder synthetic tasks. |
| Resistance/conflict factorization | Not working yet | Conflict/resistance alignment is weak or negative in current synthetic evidence measurements. | Focus on resistance supervision, corruption labels, and conflict-specific probes. |
| Uncertainty/selective classification | Next target | Existing uncertainty benchmark suggests selective prediction may be a more plausible target than clean top-1, but evidence is incomplete. | Run `scripts/run_uncertainty_frozen_feature_benchmark.py`, `scripts/run_uncertainty_end_to_end_benchmark.py`, and `scripts/run_responsibility_plugin_benchmark.py`, then regenerate `reports/EML_PLUGGABLE_PRIMITIVE_REPORT.md`. |
| EML agent/risk scorer | Plausible but untested | No durable report currently validates agent-style utility-vs-risk scoring. | Run a separate risk-decision benchmark and mark missing runs as NOT RUN. |

## Claim Rules

- Do not claim EML beats cosine on clean accuracy unless a matched report shows it.
- Do not claim MERC works as a head or block from toy-only results.
- Mark CIFAR/corruption experiments as `NOT RUN` when `torchvision` or local data is unavailable.
- Treat field, efficient, and MERC modules as experimental until their reports justify promotion.
- Treat model comparisons that hit a step cap as incomplete until an early-stopped replacement row is recorded.
