# Barça Hackathon — Ball Possession Modelling

This repository contains our work on the **More than a Hack 2027** ball-possession challenge.
The objective is to predict, for every video frame, whether the ball is controlled by the home
team (`H`), the away team (`A`), is dead (`D`), or is outside the evaluated interval (`X`).

The repository now contains both the official challenge material and our reproducible research
pipeline. The short version is: we built a frame-level feature store, tested ATD and Smart
Tagging signals, evaluated visual ball-recovery heuristics, added unsupervised context features,
and tested temporal decoders that produce coherent possession spells.

## Current status

The work is exploratory and is evaluated on the public window only: **7,500 frames and 17
possessions**. The public labels are used for benchmarking, not for frame-by-frame manual rules or
supervised training. Results must therefore be validated on a larger independent split before
selecting a final submission.

| Track | Current result | Status |
|---|---:|---|
| Official baseline, smooth 50 frames | `macroF1 0.621`, `spellF1 0.326` | Reference |
| Score-based temporal model + Viterbi | `macroF1 0.597`, `spellF1 0.520` | Best public `spellF1`, exploratory |
| Adaptive temporal smoothing | `macroF1 0.636`, `spellF1 0.409` | Best balanced candidate |
| Unsupervised clustering | GMM-8 purity `75.4%` | Useful context, not a classifier |

The main conclusion so far is that **temporal regularisation matters more than adding isolated
local features**. A long global window improves possession-spell recovery but can absorb real
transitions. Adaptive smoothing is a better compromise between frame accuracy and spell quality.

## Quick start

Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/) are recommended.

```powershell
uv sync
```

The evaluator can run against the public ground truth and the checked-in baseline CSVs without
the private match assets:

```powershell
uv run python evaluation/run_evaluation.py
```

Rebuilding the official baseline requires the downloaded ATD assets in `data/`:

```powershell
uv run python starter/baseline.py
```

For the research pipeline, install or unpack the challenge assets in `data/`. The expected
formats and frame alignment are documented in [DATA.md](DATA.md). Then run the stages in order:

```powershell
# 1. Build frame-level features from ATD, Smart Tagging and homography data
uv run python scripts/build_feature_store.py

# 2. Discover unsupervised context regimes
uv run python scripts/run_feature_clustering.py
uv run python scripts/evaluate_cluster_quality.py

# 3. Evaluate cluster-aware heuristics and temporal models
uv run python scripts/run_cluster_heuristics.py
uv run python scripts/run_temporal_control_model.py
uv run python scripts/run_adaptive_temporal_model.py

# 4. Validate any candidate submission
uv run python evaluation/validate_submission.py path/to/submission.csv
```

All experiment scripts are deterministic unless a sampling parameter is explicitly changed. They
write metrics, configurations and candidate CSVs under `reports/`; generated feature files are
written under `features/`.

## What has been done

### 1. Data and alignment QA

- Checked the tactical ATD timeline over 199,319 frames.
- Checked Smart Tagging intervals and homography alignment.
- Measured ball visibility, player gaps and tracking quality.
- Kept public ground truth restricted to final scoring and diagnostics.

### 2. Feature store

[`scripts/build_feature_store.py`](scripts/build_feature_store.py) creates one row per frame with
54 features, including:

- nearest-player distances and estimated arrival times;
- ball visibility, gaps, short interpolation, speed and acceleration;
- team centroids, dispersion and nearby-player counts;
- Smart Tagging, set-piece and transition indicators;
- homography and tracking quality;
- outputs from the bright-blob, template and optical-flow video experiments.

### 3. Baselines and ablations

[`scripts/run_atd_experiments.py`](scripts/run_atd_experiments.py) compares the official baseline
with distance, collective, Smart Tagging, control-time, interpolation, stateful and video-based
variants. The experiments show that most local additions do not improve the headline metric after
temporal smoothing.

### 4. Unsupervised context

[`scripts/run_feature_clustering.py`](scripts/run_feature_clustering.py) applies standardisation,
PCA, MiniBatch KMeans and diagonal GMMs. The clusters mostly describe visibility and tracking
regimes. They are useful for confidence and quality gating, but they do not provide a stable
home/away possession identity.

### 5. Temporal decoding

[`scripts/run_temporal_control_model.py`](scripts/run_temporal_control_model.py) builds continuous
`H/A/D` emissions and decodes them with Viterbi, including switch penalties and minimum alive
durations. [`scripts/run_adaptive_temporal_model.py`](scripts/run_adaptive_temporal_model.py)
changes the smoothing window near boundaries and uncertain frames.

### 6. Visual experiments

The tactical video experiments test motion, bright-ball detection, template matching and optical
flow. They can recover useful ball candidates, but their current spatial error is not yet low or
stable enough to replace the ATD baseline. Broadcast video and audio have not yet been integrated.

## Repository map

```text
README.md                         project overview and reproducible commands
docs/
  README.md                       documentation index
  ARCHITECTURE.md                 data flow and responsibilities
  ROADMAP.md                      next work, priorities and acceptance criteria
  experiments/                    protocols and experiment reports
  strategy/                       longer modelling plan
evaluation/                       official scorer, validator and public labels
starter/                          official loaders, baseline and notebooks
scripts/                          research and experiment entry points
features/                         generated frame-level feature store
reports/                          generated metrics, submissions and summaries
data/                             local challenge assets (not committed)
```

Start with [docs/README.md](docs/README.md) if you are joining the project. Read
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) to understand how the pipeline connects, then
[docs/ROADMAP.md](docs/ROADMAP.md) for the next tasks.

## Official challenge documentation

The following files describe the challenge contract and remain the source of truth for data,
metrics and submission rules:

- [CHALLENGE.md](CHALLENGE.md) — task and label definitions;
- [DATA.md](DATA.md) — assets and frame alignment;
- [EVALUATION.md](EVALUATION.md) — metrics and scoring;
- [SUBMISSION.md](SUBMISSION.md) — output contract and validation;
- [RULES.md](RULES.md) — competition rules;
- [FAQ.md](FAQ.md) — common questions.

## Reproducibility and data policy

The private match data and derived full-match predictions must not be committed. The `.gitignore`
keeps local environments, private assets and generated submission files out of Git. Every report
should record the command, input data assumptions, public-window usage and output configuration.

Do not tune a final model on the 17 public possessions alone. Use them to catch pipeline errors,
then confirm decisions with an independent validation protocol before submission.

## Licence

Code and documentation follow the repository licence in [LICENSE](LICENSE). Match assets are
subject to the challenge terms and are not redistributed by this repository.
