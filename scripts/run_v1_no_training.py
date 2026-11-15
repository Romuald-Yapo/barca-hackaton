"""Evaluate version 1: ATD evidence, short-gap interpolation, then majority voting."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import possession_metrics  # noqa: E402
from starter.baseline import build, smooth, write_submission  # noqa: E402
from run_atd_experiments import (  # noqa: E402
    N_FRAMES,
    PUBLIC_END,
    PUBLIC_START,
    interpolate_ball,
    load_slot_teams,
    load_tracking,
    nearest_features,
    evidence_codes,
)


def absent_run_lengths(ball_visible):
    lengths = np.zeros(len(ball_visible), dtype=np.int32)
    index = 0
    while index < len(ball_visible):
        if ball_visible[index]:
            index += 1
            continue
        end = index
        while end < len(ball_visible) and not ball_visible[end]:
            end += 1
        lengths[index:end] = end - index
        index = end
    return lengths


def to_labels(codes):
    return np.asarray(np.where(codes == 0, "H", np.where(codes == 1, "A", "D")))


def evaluate(gt_codes, labels):
    full = possession_metrics.evaluate(gt_codes, possession_metrics.label_codes(labels))
    public = possession_metrics.evaluate(
        gt_codes[PUBLIC_START:PUBLIC_END],
        possession_metrics.label_codes(labels)[PUBLIC_START:PUBLIC_END],
    )
    return full, public


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/v1_no_training")
    args = parser.parse_args()
    tactical = args.data_dir / "tactical"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "submissions").mkdir(exist_ok=True)

    slot_teams = load_slot_teams(tactical / "BAR-ATH_tactical_atd.xml")
    players, balls = load_tracking(tactical / "BAR-ATH_tactical_atd.txt", len(slot_teams))
    observed_features = nearest_features(players, balls, slot_teams)
    interpolated_features = nearest_features(players, interpolate_ball(balls, 5), slot_teams)
    official_codes = build(data_dir=str(args.data_dir))
    ball_visible = observed_features["ball_visible"]
    short_gap = absent_run_lengths(ball_visible) <= 5
    interpolated_codes = evidence_codes(interpolated_features)

    v1_codes = official_codes.copy()
    replacement = (~ball_visible) & short_gap & (interpolated_codes != -1)
    v1_codes[replacement] = interpolated_codes[replacement]
    predictions = {"v1_raw": to_labels(v1_codes)}
    for window in (25, 50, 125):
        predictions[f"v1_smooth{window}"] = smooth(v1_codes, window)

    gt_path = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
    gt_codes = possession_metrics.label_codes(possession_metrics.read_label_csv(gt_path))
    rows = []
    full_results = {}
    for name, labels in predictions.items():
        write_submission(str(args.out_dir / "submissions" / f"{name}.csv"), labels)
        full, public = evaluate(gt_codes, labels)
        full_results[name] = {"full_public_file": full, "public_window": public}
        rows.append({
            "experiment": name,
            "stage": "raw" if name.endswith("raw") else "majority_voting",
            "public_macro_f1": public["macro_f1"],
            "public_spell_f1": public["spell_f1_mean"],
            "public_team_acc_gt_alive": public["team_acc_gt_alive"],
            "public_team_acc_both_alive": public["team_acc_both_alive"],
            "public_accuracy": public["accuracy"],
            "full_file_macro_f1": full["macro_f1"],
            "full_file_spell_f1": full["spell_f1_mean"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "metrics.csv", index=False)
    (args.out_dir / "metrics.json").write_text(json.dumps(full_results, indent=2), encoding="utf-8")
    (args.out_dir / "config.json").write_text(json.dumps({
        "version": "v1_no_training",
        "data_sources": ["tactical_atd", "public_ground_truth"],
        "short_gap_max_frames": 5,
        "majority_windows": [25, 50, 125],
        "public_window": [PUBLIC_START, PUBLIC_END],
        "n_frames": N_FRAMES,
    }, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nOutputs: {args.out_dir}")


if __name__ == "__main__":
    main()
