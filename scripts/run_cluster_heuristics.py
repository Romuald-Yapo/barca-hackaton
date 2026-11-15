"""Evaluate deterministic cluster-aware possession heuristics.

No public labels are used to construct predictions. Clusters are used only as
unsupervised context: GMM confidence, cluster stability, and cluster-level
tracking quality. Public labels are read at the very end for scoring.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as metrics  # noqa: E402
from starter.baseline import smooth as official_smooth  # noqa: E402
from starter.baseline import write_submission  # noqa: E402

HOME, AWAY, NOBODY = 0, 1, -1
PUBLIC_START, PUBLIC_END = 11185, 18685


def to_labels(codes):
    return np.asarray(np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D")))


def majority_codes(codes, window):
    return metrics.label_codes(official_smooth(codes, window))


def adaptive_smooth(codes, windows):
    """Centered majority vote where the half-width can vary per frame."""
    codes = np.asarray(codes)
    windows = np.asarray(windows, dtype=np.int32)
    home_cumulative = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cumulative = np.concatenate(([0], np.cumsum(codes == AWAY)))
    indexes = np.arange(len(codes))
    low = np.maximum(0, indexes - windows)
    high = np.minimum(len(codes), indexes + windows + 1)
    home = home_cumulative[high] - home_cumulative[low]
    away = away_cumulative[high] - away_cumulative[low]
    labels = np.full(len(codes), NOBODY, dtype=np.int8)
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], HOME, AWAY)
    return labels


def nearest_codes(data):
    codes = data["nearest_codes"].to_numpy(dtype=np.int8)
    return np.where(codes < 0, NOBODY, codes)


def centroid_codes(data, nearest):
    home = data["ball_home_centroid_distance_m"].to_numpy(dtype=np.float32)
    away = data["ball_away_centroid_distance_m"].to_numpy(dtype=np.float32)
    valid = np.isfinite(home) & np.isfinite(away) & (data.ball_observed.to_numpy() == 1)
    codes = nearest.copy()
    codes[valid] = np.where(home[valid] <= away[valid], HOME, AWAY)
    return codes


def blended_codes(data, nearest, alpha, use_centroid_mask=None):
    """Blend nearest-player and team-centroid evidence in robust standardized units."""
    nearest_advantage = -data["difference"].to_numpy(dtype=np.float32)
    centroid_advantage = (
        data["ball_away_centroid_distance_m"].to_numpy(dtype=np.float32)
        - data["ball_home_centroid_distance_m"].to_numpy(dtype=np.float32)
    )
    valid_nearest = np.isfinite(nearest_advantage)
    valid_centroid = np.isfinite(centroid_advantage)
    nearest_scale = np.nanmedian(np.abs(nearest_advantage[valid_nearest]))
    centroid_scale = np.nanmedian(np.abs(centroid_advantage[valid_centroid]))
    nearest_scale = max(float(nearest_scale), 1e-3)
    centroid_scale = max(float(centroid_scale), 1e-3)
    score = nearest_advantage / nearest_scale + alpha * centroid_advantage / centroid_scale
    valid = valid_nearest & valid_centroid & (data.ball_observed.to_numpy() == 1)
    if use_centroid_mask is not None:
        valid &= use_centroid_mask
    codes = nearest.copy()
    codes[valid] = np.where(score[valid] >= 0, HOME, AWAY)
    return codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=ROOT / "features/BAR-ATH_frame_features.parquet")
    parser.add_argument("--clusters", type=Path, default=ROOT / "reports/clustering/clustered_features.parquet")
    parser.add_argument("--ground-truth", type=Path, default=ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/clustering")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "submissions").mkdir(exist_ok=True)

    data = pd.read_parquet(args.features)
    clusters = pd.read_parquet(args.clusters)
    nearest = nearest_codes(data)
    gmm = clusters["gmm_8"].to_numpy(dtype=np.int16)
    confidence = clusters["gmm_8_confidence"].to_numpy(dtype=np.float32)
    changed = np.r_[False, gmm[1:] != gmm[:-1]]
    change_count = pd.Series(changed.astype(np.float32)).rolling(25, min_periods=1, center=True).mean().to_numpy()

    # This quality score is entirely unsupervised and uses only observable tracking properties.
    margin = np.abs(data["difference"].to_numpy(dtype=np.float32))
    margin_quality = np.clip(margin / 3.0, 0.0, 1.0)
    quality = (
        0.45 * data["ball_observed"].to_numpy(dtype=np.float32)
        + 0.30 * data["tracking_complete_enough"].to_numpy(dtype=np.float32)
        + 0.25 * np.nan_to_num(margin_quality, nan=0.0)
    )
    cluster_quality = pd.Series(quality).groupby(gmm, observed=True).mean()
    quality_by_frame = np.asarray([cluster_quality.get(cluster, 0.0) for cluster in gmm], dtype=np.float32)
    quality_cut = float(np.nanmedian(cluster_quality.to_numpy()))

    predictions = {
        "cluster_baseline_raw": nearest,
        "cluster_baseline_smooth25": majority_codes(nearest, 25),
        "cluster_baseline_smooth50": majority_codes(nearest, 50),
        "cluster_baseline_smooth125": majority_codes(nearest, 125),
        "cluster_centroid_raw": centroid_codes(data, nearest),
    }

    # Context-aware abstention: only suppress evidence in low-quality regimes.
    for threshold in (0.55, 0.70):
        reliable = (confidence >= threshold) & (quality_by_frame >= quality_cut)
        gated = nearest.copy()
        gated[~reliable] = NOBODY
        predictions[f"cluster_gate{int(threshold * 100)}_smooth50"] = majority_codes(gated, 50)

    # Stable high-confidence regimes use a short window; uncertain or changing regimes
    # use a longer vote to reduce one-frame ownership flips.
    for threshold, long_window in ((0.55, 75), (0.70, 75), (0.70, 125)):
        stable = (confidence >= threshold) & (quality_by_frame >= quality_cut) & (change_count < 0.20)
        windows = np.where(stable, 25, long_window)
        predictions[f"cluster_adaptive{int(threshold * 100)}_{long_window}"] = adaptive_smooth(nearest, windows)

    # Add team-centroid evidence only where the GMM says the frame belongs clearly
    # to a stable, high-quality regime; otherwise retain the nearest-player decision.
    for alpha in (0.25, 0.50, 1.00):
        use_centroid = (confidence >= 0.55) & (quality_by_frame >= quality_cut) & (change_count < 0.20)
        blended = blended_codes(data, nearest, alpha, use_centroid_mask=use_centroid)
        predictions[f"cluster_centroid_blend{str(alpha).replace('.', '')}_smooth50"] = majority_codes(blended, 50)

    ground_truth = metrics.label_codes(metrics.read_label_csv(args.ground_truth))
    rows = []
    for name, codes in predictions.items():
        labels = to_labels(codes)
        output_path = args.out_dir / "submissions" / f"{name}.csv"
        write_submission(str(output_path), labels)
        encoded = metrics.label_codes(labels)
        full_result = metrics.evaluate(ground_truth, encoded)
        public_result = metrics.evaluate(ground_truth[PUBLIC_START:PUBLIC_END], encoded[PUBLIC_START:PUBLIC_END])
        rows.append({
            "experiment": name,
            "macro_f1": full_result["macro_f1"],
            "spell_f1_mean": full_result["spell_f1_mean"],
            "alive_dead_f1": full_result["alive_dead_f1"],
            "team_acc_gt_alive": full_result["team_acc_gt_alive"],
            "team_acc_both_alive": full_result["team_acc_both_alive"],
            "public_macro_f1": public_result["macro_f1"],
            "public_spell_f1_mean": public_result["spell_f1_mean"],
            "public_team_acc_gt_alive": public_result["team_acc_gt_alive"],
            "public_team_acc_both_alive": public_result["team_acc_both_alive"],
            "public_accuracy": public_result["accuracy"],
            "n_pred_spells": full_result["n_pred_spells"],
        })
    table = pd.DataFrame(rows).sort_values(["public_spell_f1_mean", "public_macro_f1"], ascending=False)
    table.to_csv(args.out_dir / "cluster_heuristics_metrics.csv", index=False)
    config = {
        "gmm_method": "gmm_8",
        "quality_weights": {"ball_observed": 0.45, "tracking_complete_enough": 0.30, "nearest_margin": 0.25},
        "quality_cut": quality_cut,
        "public_ground_truth_used_for_scoring_only": True,
    }
    (args.out_dir / "cluster_heuristics_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nOutputs: {args.out_dir}")


if __name__ == "__main__":
    main()
