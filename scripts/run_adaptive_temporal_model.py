"""Evaluate boundary-aware adaptive temporal smoothing on top of score-based Viterbi."""

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
from run_temporal_control_model import (  # noqa: E402
    build_emissions,
    merge_short_alive_runs,
    viterbi,
)

HOME, AWAY, DEAD = 0, 1, 2
PUBLIC_START, PUBLIC_END = 11185, 18685


def adaptive_majority(codes, stable_window, boundary_window, uncertain_window, boundary, uncertain):
    """Centered majority vote with a per-frame window, preserving local boundaries."""
    codes = np.asarray(codes, dtype=np.int8)
    windows = np.full(len(codes), stable_window, dtype=np.int32)
    windows[uncertain] = uncertain_window
    windows[boundary] = boundary_window
    home_cumulative = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cumulative = np.concatenate(([0], np.cumsum(codes == AWAY)))
    indexes = np.arange(len(codes))
    low = np.maximum(0, indexes - windows)
    high = np.minimum(len(codes), indexes + windows + 1)
    home = home_cumulative[high] - home_cumulative[low]
    away = away_cumulative[high] - away_cumulative[low]
    output = np.full(len(codes), DEAD, dtype=np.int8)
    seen = (home + away) > 0
    output[seen] = np.where(home[seen] >= away[seen], HOME, AWAY)
    return output


def build_masks(data, clusters, decoded, emissions):
    ownership = emissions[:, HOME] - emissions[:, AWAY]
    confidence = clusters["gmm_8_confidence"].to_numpy(dtype=np.float32)
    observed = data["ball_observed"].to_numpy(dtype=bool)
    tag_transition = data["tag_transition"].to_numpy(dtype=np.float32) > 0

    sign_change = np.zeros(len(decoded), dtype=bool)
    sign_change[1:] = (
        (ownership[1:] * ownership[:-1] < 0)
        & (np.abs(ownership[1:]) > 0.18)
        & (np.abs(ownership[:-1]) > 0.18)
    )
    team_change = np.zeros(len(decoded), dtype=bool)
    team_change[1:] = (
        (decoded[1:] != decoded[:-1])
        & np.isin(decoded[1:], [HOME, AWAY])
        & np.isin(decoded[:-1], [HOME, AWAY])
    )
    boundary_seed = sign_change | team_change | tag_transition
    # Spread a transition marker to a local neighbourhood, but keep it narrow enough
    # not to turn the adaptive method into a global short-window smoother.
    boundary = pd.Series(boundary_seed.astype(np.int8)).rolling(11, center=True, min_periods=1).max().to_numpy() > 0
    uncertain = (~observed) | (confidence < 0.55) | (np.abs(ownership) < 0.22)
    uncertain &= ~boundary
    return boundary, uncertain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=ROOT / "features/BAR-ATH_frame_features.parquet")
    parser.add_argument("--clusters", type=Path, default=ROOT / "reports/clustering/clustered_features.parquet")
    parser.add_argument("--ground-truth", type=Path, default=ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/adaptive_temporal")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "submissions").mkdir(exist_ok=True)

    data = pd.read_parquet(args.features)
    clusters = pd.read_parquet(args.clusters)
    ground_truth = metrics.label_codes(metrics.read_label_csv(args.ground_truth))
    runs = metrics.valid_runs(ground_truth)
    emissions = build_emissions(data, clusters)
    rows = []

    # Keep the search compact and interpretable. No label is used to construct a prediction.
    for switch_penalty, dead_penalty in ((0.50, 0.30), (0.50, 0.50), (0.75, 0.50), (1.00, 0.75)):
        decoded = viterbi(emissions, switch_penalty, dead_penalty, runs)
        for minimum_alive in (25, 50):
            stabilized = merge_short_alive_runs(decoded, minimum_alive, runs)
            boundary, uncertain = build_masks(data, clusters, stabilized, emissions)
            for stable_window in (15, 25, 35, 50):
                for boundary_window in (0, 5, 10, 15):
                    for uncertain_window in (50, 75, 100):
                        predicted = adaptive_majority(
                            stabilized,
                            stable_window,
                            boundary_window,
                            uncertain_window,
                            boundary,
                            uncertain,
                        )
                        labels = np.asarray(np.where(predicted == HOME, "H", np.where(predicted == AWAY, "A", "D")))
                        encoded = metrics.label_codes(labels)
                        full_result = metrics.evaluate(ground_truth, encoded)
                        public_result = metrics.evaluate(
                            ground_truth[PUBLIC_START:PUBLIC_END], encoded[PUBLIC_START:PUBLIC_END]
                        )
                        name = (
                            f"adaptive_sw{switch_penalty:g}_dp{dead_penalty:g}"
                            f"_min{minimum_alive}_stable{stable_window}"
                            f"_boundary{boundary_window}_uncertain{uncertain_window}"
                        )
                        write_submission(str(args.out_dir / "submissions" / f"{name}.csv"), labels)
                        rows.append(
                            {
                                "experiment": name,
                                "switch_penalty": switch_penalty,
                                "dead_penalty": dead_penalty,
                                "minimum_alive_frames": minimum_alive,
                                "stable_window": stable_window,
                                "boundary_window": boundary_window,
                                "uncertain_window": uncertain_window,
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
    table.to_csv(args.out_dir / "metrics.csv", index=False)
    best_spell = table.sort_values(["public_spell_f1_mean", "public_macro_f1"], ascending=False).iloc[0]
    pareto = table[
        ~table.apply(
            lambda row: (
                (table.public_macro_f1 >= row.public_macro_f1)
                & (table.public_spell_f1_mean >= row.public_spell_f1_mean)
                & ((table.public_macro_f1 > row.public_macro_f1) | (table.public_spell_f1_mean > row.public_spell_f1_mean))
            ).any(),
            axis=1,
        )
    ].sort_values("public_spell_f1_mean", ascending=False)
    pareto.to_csv(args.out_dir / "pareto_frontier.csv", index=False)
    best_macro = table.sort_values("public_macro_f1", ascending=False).iloc[0]
    summary = {
        "n_experiments": len(table),
        "boundary_frames_fraction": float(build_masks(data, clusters, decoded, emissions)[0].mean()),
        "uncertain_frames_fraction": float(build_masks(data, clusters, decoded, emissions)[1].mean()),
        "best_public_spell_f1": best_spell.to_dict(),
        "best_public_macro_f1": best_macro.to_dict(),
        "pareto_points": int(len(pareto)),
        "public_ground_truth_used_for_scoring_only": True,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print("Best public Spell-F1:")
    print(best_spell.to_string())
    print("\nBest public macro-F1:")
    print(best_macro.to_string())
    print(f"\nPareto points: {len(pareto)}")
    print(f"Metrics: {args.out_dir / 'metrics.csv'}")
    print(f"Pareto: {args.out_dir / 'pareto_frontier.csv'}")


if __name__ == "__main__":
    main()
