"""Evaluate a score-based temporal possession model without supervised training.

The model builds continuous home/away/dead emissions from the ATD feature store,
then decodes a globally coherent H/A/D sequence with a 3-state Viterbi pass.
Public labels are used only for the final benchmark.
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
from starter.baseline import write_submission  # noqa: E402
from run_cluster_heuristics import majority_codes  # noqa: E402

HOME, AWAY, DEAD = 0, 1, 2
PUBLIC_START, PUBLIC_END = 11185, 18685


def signed_tanh(value, scale):
    return np.tanh(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) / scale)


def build_emissions(data, clusters):
    home_best = data.home_best.to_numpy(dtype=np.float32)
    away_best = data.away_best.to_numpy(dtype=np.float32)
    nearest_advantage = away_best - home_best
    eta_advantage = data.away_eta.to_numpy(dtype=np.float32) - data.home_eta.to_numpy(dtype=np.float32)

    home_collective = 1.0 / (1.0 + np.nan_to_num(home_best, nan=50.0)) + 0.05 * data.home_nearby.to_numpy()
    away_collective = 1.0 / (1.0 + np.nan_to_num(away_best, nan=50.0)) + 0.05 * data.away_nearby.to_numpy()
    collective_advantage = home_collective - away_collective

    centroid_advantage = (
        data.ball_away_centroid_distance_m.to_numpy(dtype=np.float32)
        - data.ball_home_centroid_distance_m.to_numpy(dtype=np.float32)
    )
    tag_advantage = data.tag_team_score.to_numpy(dtype=np.float32)
    observed = data.ball_observed.to_numpy(dtype=np.float32)
    complete = data.tracking_complete_enough.to_numpy(dtype=np.float32)
    set_piece = np.clip(data.tag_set_piece.to_numpy(dtype=np.float32), 0.0, 2.0)
    transition = np.clip(data.tag_transition.to_numpy(dtype=np.float32), 0.0, 2.0)

    # Symmetric ownership evidence. Keeping it continuous is the key difference from
    # the nearest-player baseline.
    ownership = (
        1.25 * signed_tanh(nearest_advantage, 2.0)
        + 0.35 * signed_tanh(eta_advantage, 1.5)
        + 0.45 * signed_tanh(collective_advantage, 0.12)
        + 0.20 * signed_tanh(centroid_advantage, 8.0)
        + 0.15 * np.clip(tag_advantage, -2.0, 2.0)
    )
    ownership *= 0.25 + 0.75 * observed

    confidence = clusters["gmm_8_confidence"].to_numpy(dtype=np.float32)
    # Dead-ball evidence is strongest when the ball is missing, a set-piece is active,
    # tracking is incomplete, or the ownership evidence is intrinsically ambiguous.
    ambiguity = np.clip(1.0 - np.abs(ownership), 0.0, 1.0)
    dead = (
        1.35 * (1.0 - observed)
        + 0.45 * set_piece
        + 0.25 * (1.0 - complete)
        + 0.20 * (1.0 - confidence)
        + 0.20 * ambiguity * observed
        - 0.20 * transition
    )
    return np.column_stack((ownership, -ownership, dead)).astype(np.float32)


def viterbi(emissions, switch_penalty, dead_penalty, runs):
    """Decode H/A/D with separate team-switch and alive/dead penalties."""
    n_frames = len(emissions)
    labels = np.full(n_frames, DEAD, dtype=np.int8)
    transition = np.array(
        [
            [0.0, -switch_penalty, -dead_penalty],
            [-switch_penalty, 0.0, -dead_penalty],
            [-dead_penalty, -dead_penalty, 0.0],
        ],
        dtype=np.float32,
    )
    for start, stop in runs:
        if stop <= start:
            continue
        dp = emissions[start].astype(np.float32).copy()
        back = np.empty((stop - start, 3), dtype=np.int8)
        back[0] = np.arange(3, dtype=np.int8)
        for offset in range(1, stop - start):
            candidates = dp[:, None] + transition
            back[offset] = np.argmax(candidates, axis=0).astype(np.int8)
            dp = emissions[start + offset] + np.max(candidates, axis=0)
        current = int(np.argmax(dp))
        labels[stop - 1] = current
        for offset in range(stop - start - 1, 0, -1):
            current = int(back[offset, current])
            labels[start + offset - 1] = current
    return labels


def merge_short_alive_runs(labels, minimum_length, runs):
    output = labels.copy()
    for start, stop in runs:
        index = start
        while index < stop:
            end = index + 1
            while end < stop and output[end] == output[index]:
                end += 1
            length = end - index
            if output[index] in (HOME, AWAY) and length < minimum_length:
                before = output[index - 1] if index > start else DEAD
                after = output[end] if end < stop else DEAD
                if before == after and before in (HOME, AWAY, DEAD):
                    output[index:end] = before
            index = end
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=ROOT / "features/BAR-ATH_frame_features.parquet")
    parser.add_argument("--clusters", type=Path, default=ROOT / "reports/clustering/clustered_features.parquet")
    parser.add_argument("--ground-truth", type=Path, default=ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/temporal_control")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "submissions").mkdir(exist_ok=True)

    data = pd.read_parquet(args.features)
    clusters = pd.read_parquet(args.clusters)
    ground_truth = metrics.label_codes(metrics.read_label_csv(args.ground_truth))
    runs = metrics.valid_runs(ground_truth)
    emissions = build_emissions(data, clusters)
    rows = []

    # The sweep is deliberately small and interpretable: no parameter is learned from labels.
    for switch_penalty in (0.25, 0.50, 0.75, 1.00, 1.50):
        for dead_penalty in (0.15, 0.30, 0.50, 0.75):
            decoded = viterbi(emissions, switch_penalty, dead_penalty, runs)
            for minimum_length in (0, 25, 50):
                stabilized = merge_short_alive_runs(decoded, minimum_length, runs)
                for window in (0, 25, 50, 75, 100, 120):
                    if window:
                        predicted_codes = majority_codes(stabilized, window)
                    else:
                        predicted_codes = stabilized
                    labels = np.asarray(np.where(predicted_codes == HOME, "H", np.where(predicted_codes == AWAY, "A", "D")))
                    encoded = metrics.label_codes(labels)
                    full_result = metrics.evaluate(ground_truth, encoded)
                    public_result = metrics.evaluate(
                        ground_truth[PUBLIC_START:PUBLIC_END], encoded[PUBLIC_START:PUBLIC_END]
                    )
                    name = f"viterbi_sw{switch_penalty:g}_dp{dead_penalty:g}_min{minimum_length}_smooth{window}"
                    write_submission(str(args.out_dir / "submissions" / f"{name}.csv"), labels)
                    rows.append(
                        {
                            "experiment": name,
                            "switch_penalty": switch_penalty,
                            "dead_penalty": dead_penalty,
                            "minimum_alive_frames": minimum_length,
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

    table = pd.DataFrame(rows).sort_values(["public_spell_f1_mean", "public_macro_f1"], ascending=False)
    table.to_csv(args.out_dir / "metrics.csv", index=False)
    best_spell = table.iloc[0]
    best_macro = table.sort_values("public_macro_f1", ascending=False).iloc[0]
    summary = {
        "n_experiments": len(table),
        "best_public_spell_f1": best_spell.to_dict(),
        "best_public_macro_f1": best_macro.to_dict(),
        "public_ground_truth_used_for_scoring_only": True,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print("Best public Spell-F1:")
    print(best_spell.to_string())
    print("\nBest public macro-F1:")
    print(best_macro.to_string())
    print(f"\nExperiments: {len(table)}")
    print(f"Metrics: {args.out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
