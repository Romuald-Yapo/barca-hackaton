"""Sweep majority-vote windows for baseline and cluster-aware raw calls."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as metrics  # noqa: E402
from starter.baseline import write_submission  # noqa: E402
from run_cluster_heuristics import (  # noqa: E402
    blended_codes,
    majority_codes,
    nearest_codes,
    to_labels,
)

HOME, AWAY, NOBODY = 0, 1, -1
PUBLIC_START, PUBLIC_END = 11185, 18685


def make_cluster_gates(data, clusters):
    gmm = clusters["gmm_8"].to_numpy(dtype=np.int16)
    confidence = clusters["gmm_8_confidence"].to_numpy(dtype=np.float32)
    margin = np.abs(data["difference"].to_numpy(dtype=np.float32))
    quality = (
        0.45 * data["ball_observed"].to_numpy(dtype=np.float32)
        + 0.30 * data["tracking_complete_enough"].to_numpy(dtype=np.float32)
        + 0.25 * np.clip(np.nan_to_num(margin, nan=0.0) / 3.0, 0.0, 1.0)
    )
    cluster_quality = pd.Series(quality).groupby(gmm, observed=True).mean()
    quality_by_frame = np.asarray([cluster_quality.get(cluster, 0.0) for cluster in gmm], dtype=np.float32)
    quality_cut = float(np.nanmedian(cluster_quality.to_numpy()))
    base = nearest_codes(data)
    calls = {"baseline": base}
    for threshold in (0.55, 0.70):
        reliable = (confidence >= threshold) & (quality_by_frame >= quality_cut)
        gated = base.copy()
        gated[~reliable] = NOBODY
        calls[f"cluster_gate{int(threshold * 100)}"] = gated
    stable = (confidence >= 0.55) & (quality_by_frame >= quality_cut)
    for alpha in (0.25, 0.50, 1.00):
        calls[f"cluster_centroid_blend{str(alpha).replace('.', '')}"] = blended_codes(
            data, base, alpha, use_centroid_mask=stable
        )
    return calls, quality_cut


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=ROOT / "features/BAR-ATH_frame_features.parquet")
    parser.add_argument("--clusters", type=Path, default=ROOT / "reports/clustering/clustered_features.parquet")
    parser.add_argument("--ground-truth", type=Path, default=ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/clustering")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "smoothing_submissions").mkdir(exist_ok=True)

    data = pd.read_parquet(args.features)
    clusters = pd.read_parquet(args.clusters)
    raw_calls, quality_cut = make_cluster_gates(data, clusters)
    ground_truth = metrics.label_codes(metrics.read_label_csv(args.ground_truth))
    rows = []
    for method, codes in raw_calls.items():
        for window in range(25, 126, 5):
            encoded = majority_codes(codes, window)
            labels = to_labels(encoded)
            full_result = metrics.evaluate(ground_truth, encoded)
            public_result = metrics.evaluate(
                ground_truth[PUBLIC_START:PUBLIC_END], encoded[PUBLIC_START:PUBLIC_END]
            )
            name = f"{method}_smooth{window}"
            write_submission(str(args.out_dir / "smoothing_submissions" / f"{name}.csv"), labels)
            rows.append(
                {
                    "method": method,
                    "window": window,
                    "macro_f1": full_result["macro_f1"],
                    "spell_f1_mean": full_result["spell_f1_mean"],
                    "alive_dead_f1": full_result["alive_dead_f1"],
                    "team_acc_gt_alive": full_result["team_acc_gt_alive"],
                    "team_acc_both_alive": full_result["team_acc_both_alive"],
                    "n_pred_spells": full_result["n_pred_spells"],
                    "public_macro_f1": public_result["macro_f1"],
                    "public_spell_f1_mean": public_result["spell_f1_mean"],
                    "public_team_acc_gt_alive": public_result["team_acc_gt_alive"],
                    "public_team_acc_both_alive": public_result["team_acc_both_alive"],
                    "public_accuracy": public_result["accuracy"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "smoothing_sweep_metrics.csv", index=False)
    best_public_spell = table.loc[table.public_spell_f1_mean.idxmax()]
    best_public_macro = table.loc[table.public_macro_f1.idxmax()]
    best_full_spell = table.loc[table.spell_f1_mean.idxmax()]
    summary = {
        "windows": list(range(25, 126, 5)),
        "quality_cut": quality_cut,
        "best_public_spell_f1": best_public_spell.to_dict(),
        "best_public_macro_f1": best_public_macro.to_dict(),
        "best_full_file_spell_f1": best_full_spell.to_dict(),
        "public_ground_truth_used_for_scoring_only": True,
    }
    (args.out_dir / "smoothing_sweep_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print("Best public Spell-F1:")
    print(best_public_spell.to_string())
    print("\nBest public macro-F1:")
    print(best_public_macro.to_string())
    print("\nBest full-file Spell-F1:")
    print(best_full_spell.to_string())
    print(f"\nMetrics: {args.out_dir / 'smoothing_sweep_metrics.csv'}")


if __name__ == "__main__":
    main()
