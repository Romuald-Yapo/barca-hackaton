# experiments/ — supervised possession model (exp09) and video ball detector (exp10)

Brought over from [lucgmensah/more-than-a-hack-2027](https://github.com/lucgmensah/more-than-a-hack-2027/tree/lucg/experiments)
(branch `lucg`), where experiments 01-07 also live (HMM, Metrica validation, abstention,
semi-Markov). Only what experiments 9 and 10 need is copied here. The shared files they rely on
(`games.py`, `evaluation/possession_metrics.py`, `starter/*`) are byte-identical in both
repositories.

| folder | what it is |
|---|---|
| [exp09_supervised](exp09_supervised/README.md) | **Current best submission.** Gradient-boosted trees on 159 per-frame features (touches, context ±3 s, team shape). Trained on 7 Bundesliga matches of official tracking (DFL open data, possession and ball status on every frame), damaged to look like the BAR-ATH tracking. A separate dead/alive model ignores ball motion. The broadcast tracking is used as a second ball sensor. Penalty Viterbi on top. |
| [exp10_video_ball](exp10_video_ball/README.md) | **Optional, needs a CUDA GPU, never run in full.** A ball detector trained on the tactical video with algorithmic pseudo-labels, to fill the frames where the tracking has no ball. Switched on with `exp09_supervised/predict.py --video-ball`. |
| [exp08_friend_learned](exp08_friend_learned/README.md) | The score-based Viterbi of `scripts/run_temporal_control_model.py`, ported line by line, with its weights learned on Metrica. exp09 reuses its Viterbi (`friend.py`). |
| `exp02_metrica_validation/` | Helper modules only: Metrica download, event-based labels, damage. Used to build the Metrica validation copies. |
| `lib/` | Shared loaders and scorer helpers (`common.py`), older features and HMM used by exp08. |

## Results (no BAR-ATH label used to fit or to choose anything)

| method | Metrica, other game, spellF1 | DFL held-out, spellF1 | public window spellF1 / macroF1 (check only) |
|---|---|---|---|
| official baseline, vote ±50 | 0.38 | | 0.33 / — |
| `run_temporal_control_model.py`, published config | 0.35 | | 0.52 / 0.60 (chosen on this window) |
| same model, weights learned on Metrica (exp08) | 0.55 | | 0.37 / 0.65 |
| **exp09** | **0.70** | **0.65-0.66** | **0.67 / 0.84** |

## Running

```bash
uv sync --group modeling                                   # numpy, pandas, opencv, torch (+ notebook group)
uv pip install -r experiments/requirements-exp09.txt       # lightgbm (not in pyproject on purpose)

uv run python experiments/exp09_supervised/predict.py      # regenerate the exp09 submission, ~2 min
```

Expected output of `predict.py`:
* `experiments/out/exp09/submission.csv`;
* validator line "285 possessions of mean 12.8 s";
* public check spellF1 0.671, macroF1 0.839;
* on the machine it was produced on, sha256 `147033f58934bdb3e0bda5055156177de25b6428440eb55623a831fc59ca2bce`.

Retraining, video detector and the order of the scripts: see the two READMEs.

Data. The match assets go in `data/` as usual; `MTAH_DATA_DIR` also works. The scripts download
the external data themselves, into `data/external/` (already gitignored by `data/`):
* DFL: 2.4 GB, CC-BY 4.0;
* Metrica sample games.

Generated files go to `experiments/cache/` and `experiments/out/`; both are gitignored. They
contain frame-level derived tracking and full-match labels, which must not be published (RULES.md).
