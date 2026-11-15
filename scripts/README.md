# Research scripts

The scripts are command-line entry points. They are intentionally kept in one directory because
several experiments import shared helpers from neighbouring scripts. Run them from the repository
root with `uv run python ...`.

## Dependency order

```text
raw assets in data/
  -> build_feature_store.py
  -> run_feature_clustering.py
  -> evaluate_cluster_quality.py
  -> run_cluster_heuristics.py / sweep_smoothing_windows.py
  -> run_temporal_control_model.py
  -> run_adaptive_temporal_model.py
```

The ATD and video branches can be run independently after the raw assets are available.

## Script index

| Script | Purpose | Main output |
|---|---|---|
| `build_feature_store.py` | Build one row per frame from ATD, tags, homography and visual detections | `features/` |
| `run_atd_experiments.py` | Compare ATD and Smart Tagging ablations | `reports/atd_experiments/` |
| `run_v1_no_training.py` | Reproduce the first complete non-trained pipeline | `reports/v1_no_training/` |
| `run_feature_clustering.py` | Fit KMeans and GMM context regimes | `reports/clustering/` |
| `evaluate_cluster_quality.py` | Score clusters as diagnostics only | `reports/clustering/` |
| `run_cluster_heuristics.py` | Use cluster confidence and quality gates | `reports/clustering/` |
| `sweep_smoothing_windows.py` | Compare temporal windows from 25 to 125 frames | `reports/clustering/` |
| `run_temporal_control_model.py` | Continuous H/A/D emissions plus Viterbi decoding | `reports/temporal_control/` |
| `run_adaptive_temporal_model.py` | Boundary- and uncertainty-aware smoothing | `reports/adaptive_temporal/` |
| `video_heuristics_experiment.py` | Motion, template and optical-flow candidates | `reports/video_heuristics/` |
| `video_ball_recovery_experiment.py` | Bright-ball recovery on sampled gaps | `reports/video_ball_recovery/` |

## Common options

Most scripts accept `--features`, `--clusters`, `--ground-truth`, `--data-dir` or `--out-dir`.
Use `--help` before changing defaults. Video scripts additionally accept a sample count because a
full sequential scan is expensive.

## Rules for new experiments

- Write metrics and configuration to a new report directory or a clearly named existing one.
- Never overwrite the official baseline implementation to test a hypothesis.
- Keep public labels out of feature construction and model fitting.
- Add the command and interpretation to `docs/experiments/benchmark.md`.
