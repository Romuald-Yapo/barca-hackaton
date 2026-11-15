# Architecture and data flow

## Goal

Produce a frame-level possession label (`H`, `A`, `D` or `X`) and a temporally coherent sequence
of possession spells. The pipeline is deliberately split into feature construction, candidate
prediction, temporal decoding and evaluation so that every improvement can be compared with the
official baseline.

## Pipeline

```text
challenge assets in data/
        |
        +--> starter loaders and official baseline
        |          |
        |          +--> baseline submissions
        |
        +--> scripts/build_feature_store.py
                    |
                    +--> features/BAR-ATH_frame_features.parquet
                    |       +--> scripts/run_feature_clustering.py
                    |       |       +--> reports/clustering/
                    |       |
                    |       +--> ATD / cluster / temporal experiments
                    |               +--> reports/temporal_control/
                    |               +--> reports/adaptive_temporal/
                    |
                    +--> scripts/run_atd_experiments.py
                    +--> scripts/video_*_experiment.py
                            |
                            +--> reports/<experiment>/

candidate CSVs --> evaluation/validate_submission.py
                --> evaluation/run_evaluation.py
                --> public metrics and reports
```

## Responsibilities by directory

### `evaluation/`

The scoring contract. `possession_metrics.py` defines the metrics,
`run_evaluation.py` scores submissions and `validate_submission.py` checks the CSV contract before
submission. This directory should remain independent from research heuristics.

### `starter/`

Official data loaders, the reference baseline and introductory notebooks. The baseline is the
control condition and should not be silently modified by an experiment.

### `scripts/`

Research entry points. Scripts import shared evaluation and starter utilities, write their outputs
to `reports/`, and expose paths and sample sizes through command-line arguments. See
[`scripts/README.md`](../scripts/README.md) for the dependency order.

### `features/`

Generated frame-level data. The feature store is the interface between raw challenge assets and
the modelling experiments. It is reproducible from `scripts/build_feature_store.py` and should not
be edited manually.

### `reports/`

Generated experiment outputs: metrics, JSON configuration, diagnostic tables and candidate
submissions. Reports are evidence for decisions; they are not model code.

## Data contracts

- Frame indexing follows the challenge timeline; the public evaluation window is
  `11185:18685`.
- All features must be computed without using private labels.
- Public labels may be read only for diagnostics and final benchmark tables.
- Every candidate submission must be passed through `evaluation/validate_submission.py`.
- A report must identify the input assets, command, configuration and evaluation split.

## Current modelling layers

1. **Local evidence:** nearest player, team-level proximity, ball visibility, Smart Tagging and
   optional visual candidates.
2. **Context:** feature-store quality indicators and unsupervised cluster confidence.
3. **Temporal decoding:** majority smoothing, stateful rules, Viterbi and adaptive windows.
4. **Evaluation:** frame metrics, alive/dead metrics, team accuracy and possession-spell metrics.

The next architectural improvement is to extract common feature and decoding primitives from the
large experiment scripts into a small importable package. That should happen only after the data
contracts and validation split are fixed, so the refactor does not hide experimental differences.
