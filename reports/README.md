# Reports and generated outputs

This directory contains generated metrics, configurations, diagnostics and candidate submissions.
It is organised by experiment family:

| Directory | Contents |
|---|---|
| `atd_experiments/` | ATD and Smart Tagging ablations |
| `v1_no_training/` | first complete non-trained pipeline |
| `clustering/` | cluster assignments, quality tables and smoothing sweeps |
| `temporal_control/` | Viterbi and score-based temporal model outputs |
| `adaptive_temporal/` | adaptive-window metrics and Pareto frontier |
| `video_heuristics/` | tactical-video motion/template/flow diagnostics |
| `video_ball_recovery/` | sampled ball-recovery diagnostics |
| `atd_experiments_video/` | older video experiment outputs |

Metrics are exploratory until they have been rerun with the validation protocol in
[`docs/ROADMAP.md`](../docs/ROADMAP.md). A report should be interpreted together with its JSON
configuration and the command that generated it.

Candidate CSVs are intermediate research artefacts. Before using one as a submission, run:

```powershell
uv run python evaluation/validate_submission.py path/to/candidate.csv
```
