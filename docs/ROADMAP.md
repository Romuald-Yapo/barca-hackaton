# Roadmap

This roadmap separates completed work from hypotheses that still need evidence. The public window
contains only 17 possessions, so a public score alone is not sufficient to promote a model to a
final submission.

## Done

- [x] Flattened the repository and separated project documentation from challenge documentation.
- [x] Audited ATD, Smart Tagging, homography and frame alignment.
- [x] Built a 199,319-row, 54-column frame-level feature store.
- [x] Reproduced the official baselines and measured ATD/Smart Tagging ablations.
- [x] Tested video motion, bright-ball, template and optical-flow heuristics.
- [x] Tested KMeans/GMM context features without supervised training.
- [x] Tested score-based Viterbi decoding and adaptive temporal smoothing.

## Priority 0 — make the result trustworthy

### 1. Freeze a validation protocol

- Define a validation split by contiguous possessions, not random frames.
- Keep the public ground truth as a diagnostic set, not as a hyperparameter search loop.
- Record all selected parameters and compare the same candidates on every split.
- Add regression checks for frame offsets, label codes, missing-ball handling and submission
  length.

Acceptance criterion: a candidate can be reproduced from one command and its score is accompanied
by the split, configuration and number of predicted spells.

### 2. Consolidate the current candidates

Re-run the following with the frozen protocol:

- official `baseline_smooth50` and `baseline_smooth120`;
- Viterbi control model;
- adaptive smoothing selected for macro-F1;
- adaptive smoothing selected for spell-F1.

Acceptance criterion: no candidate is called an improvement unless it wins on an independent split
without a large regression in the secondary metric.

## Priority 1 — improve the most informative missing signal

### 3. Integrate broadcast video and audio

The current visual experiments use tactical video only. The broadcast feed and audio should be
used mainly for event boundaries: whistles, restarts, cuts, replays and dead-ball transitions.
They are unlikely to identify the possessing team directly, but can improve the temporal decoder.

Work items:

- obtain and document the broadcast asset;
- synchronise broadcast frames with the tactical timeline;
- detect cuts and replay segments;
- extract audio boundary candidates;
- measure whether boundary precision improves spell-F1 without increasing false transitions.

Acceptance criterion: an ablation shows the effect of audio/broadcast signals independently of the
existing ATD features.

### 4. Improve ball-absence handling

The ball is not visible in a substantial fraction of scored frames. Compare explicit missing-ball
states, short-gap interpolation and uncertainty-aware emissions. Avoid filling long gaps with a
team label unless the temporal evidence supports it.

Acceptance criterion: fewer fragmented spells and better team accuracy on frames where both teams
are alive, measured on the frozen validation split.

## Priority 2 — make experiments maintainable

### 5. Extract reusable modelling code

The scripts currently contain useful shared functions but also import one another directly. Move
stable primitives into a small package, for example:

```text
src/possession/
  io.py             data loading and frame alignment
  features.py       feature-store construction
  emissions.py      H/A/D evidence scores
  decoding.py       smoothing, stateful rules and Viterbi
  submissions.py    label conversion and CSV writing
```

Keep the current scripts as thin experiment entry points. Add unit tests before changing numerical
behaviour.

Acceptance criterion: two experiment scripts use the same tested primitive and produce unchanged
baseline outputs.

### 6. Add experiment manifests

Each report directory should contain the command, Git revision, input paths, public-window policy,
random seed and selected parameters. This makes old results auditable when new feature versions
are generated.

## Priority 3 — supervised modelling, only after validation is ready

Potential next models are a calibrated HMM/CRF, a temporal convolutional model and a two-stage
dead/alive then home/away model. They should be evaluated only after a valid training/validation
protocol exists. The public labels must not be treated as a training set for a final model.

## Decision rules

- Optimise the official headline metric, but always report macro-F1, alive/dead F1, team accuracy,
  predicted spell count and mean spell duration.
- Prefer a simpler model when the gain is not stable across validation splits.
- Keep every rejected experiment documented when it answers a useful question.
- Do not delete raw metrics or candidate submissions that support a published conclusion.
