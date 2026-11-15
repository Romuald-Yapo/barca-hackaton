# exp02 helpers (Metrica sample games)

Only the modules that experiments 8 and 9 import. The full experiment 2 lives in
[lucgmensah/more-than-a-hack-2027](https://github.com/lucgmensah/more-than-a-hack-2027/tree/lucg/experiments/exp02_metrica_validation).

| module | role |
|---|---|
| `download.py` | fetches Metrica sample games 1 and 2 into `data/external/metrica/` |
| `metrica_io.py` | loads a game as the same dict as the BAR-ATH loader (metres, speeds) |
| `ground_truth.py` | H/A/D per frame from the Metrica event data. A proxy: event-coded, not corrected tracking. |
| `degrade.py` | makes clean Metrica look like BAR-ATH. `degrade_faithful` is the one used; `degrade` is the older, far too harsh one, kept for reference. |
