# Feature store

`BAR-ATH_frame_features.parquet` is a generated, frame-level table with 199,319 rows and 54
columns. It is built by:

```powershell
uv run python scripts/build_feature_store.py
```

The table combines ATD, Smart Tagging, homography, tracking quality and optional video detector
outputs. `feature_store_qa.json` records basic quality checks, including frame count and ball
visibility.

The feature store is an intermediate artefact, not a hand-edited dataset. Rebuild it when the
upstream assets or feature definitions change, and keep the generated file out of commits unless
there is a deliberate reason to version it.
