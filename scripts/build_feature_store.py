"""Build a local frame-level feature store from the supplied match data."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_atd_experiments import (  # noqa: E402
    N_FRAMES,
    PUBLIC_END,
    PUBLIC_START,
    load_slot_teams,
    load_tag_prior,
    load_tracking,
    nearest_features,
    interpolate_ball,
)


def run_lengths(mask):
    lengths = np.zeros(len(mask), dtype=np.int32)
    index = 0
    while index < len(mask):
        if mask[index]:
            index += 1
            continue
        end = index
        while end < len(mask) and not mask[end]:
            end += 1
        lengths[index:end] = end - index
        index = end
    return lengths


def interval_counts(events, key):
    starts = np.zeros(N_FRAMES + 1, dtype=np.int32)
    ends = np.zeros(N_FRAMES + 1, dtype=np.int32)
    for event in events:
        if key(event):
            start, end = max(0, event["start"]), min(N_FRAMES, event["end"])
            if end > start:
                starts[start] += 1
                ends[end] += 1
    return np.cumsum(starts[:-1] - ends[:-1])


def add_team_features(data, player, slot_teams):
    player_xy = player[:, :, :2].astype(np.float32).copy()
    player_xy[:, :, 0] *= 105.3
    player_xy[:, :, 1] *= 68.0
    visible = np.isfinite(player_xy).all(axis=2)
    for code, name in ((0, "home"), (1, "away")):
        positions = player_xy[:, slot_teams == code]
        team_visible = visible[:, slot_teams == code]
        count = team_visible.sum(axis=1).astype(np.int16)
        safe_positions = np.nan_to_num(positions, nan=0.0)
        centroid = safe_positions.sum(axis=1) / np.maximum(count[:, None], 1)
        centroid[count == 0] = np.nan
        data[f"{name}_players_visible"] = count
        data[f"{name}_centroid_x_m"] = centroid[:, 0]
        data[f"{name}_centroid_y_m"] = centroid[:, 1]
        data[f"{name}_spread_x_m"] = np.nanstd(positions[:, :, 0], axis=1)
        data[f"{name}_spread_y_m"] = np.nanstd(positions[:, :, 1], axis=1)
        speeds = player[:, slot_teams == code, 2]
        data[f"{name}_speed_mean"] = np.nanmean(speeds, axis=1)
        data[f"{name}_speed_median"] = np.nanmedian(speeds, axis=1)
    data["team_centroid_distance_m"] = np.hypot(
        data.home_centroid_x_m - data.away_centroid_x_m,
        data.home_centroid_y_m - data.away_centroid_y_m,
    )
    data["team_speed_difference"] = data.home_speed_mean - data.away_speed_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "features")
    args = parser.parse_args()
    tactical = args.data_dir / "tactical"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    slot_teams = load_slot_teams(tactical / "BAR-ATH_tactical_atd.xml")
    player, balls = load_tracking(tactical / "BAR-ATH_tactical_atd.txt", len(slot_teams))
    observed = np.isfinite(balls[:, :2]).all(axis=1)
    short_ball = interpolate_ball(balls, 5)
    features = nearest_features(player, balls, slot_teams)
    data = pd.DataFrame(features)
    data["frame"] = np.arange(N_FRAMES, dtype=np.int32)
    data["is_graded_frame"] = 0
    data.loc[PUBLIC_START:90651, "is_graded_frame"] = 1
    data.loc[110808:190197, "is_graded_frame"] = 1
    data["ball_observed"] = observed.astype(np.int8)
    data["ball_gap_length_frames"] = run_lengths(observed)
    data["ball_gap_length_s"] = data.ball_gap_length_frames / 25.0
    data["ball_short_interpolated"] = (~observed & np.isfinite(short_ball[:, :2]).all(axis=1)).astype(np.int8)

    ball_m = balls[:, :2].astype(np.float32).copy()
    ball_m[:, 0] *= 105.3
    ball_m[:, 1] *= 68.0
    ball_for_derivatives = short_ball[:, :2].astype(np.float32).copy()
    ball_for_derivatives[:, 0] *= 105.3
    ball_for_derivatives[:, 1] *= 68.0
    ball_velocity = np.gradient(ball_for_derivatives, axis=0) * 25.0
    ball_acceleration = np.gradient(ball_velocity, axis=0) * 25.0
    data["ball_speed_derived_mps"] = np.linalg.norm(ball_velocity, axis=1)
    data["ball_acceleration_derived_mps2"] = np.linalg.norm(ball_acceleration, axis=1)
    data["ball_motion_observed"] = observed.astype(np.int8)
    data["ball_x_m"] = ball_m[:, 0]
    data["ball_y_m"] = ball_m[:, 1]
    add_team_features(data, player, slot_teams)
    data["ball_home_centroid_distance_m"] = np.hypot(data.ball_x_m - data.home_centroid_x_m, data.ball_y_m - data.home_centroid_y_m)
    data["ball_away_centroid_distance_m"] = np.hypot(data.ball_x_m - data.away_centroid_x_m, data.ball_y_m - data.away_centroid_y_m)
    data["centroid_distance_difference_m"] = data.ball_home_centroid_distance_m - data.ball_away_centroid_distance_m

    tag_prior, events = load_tag_prior(tactical / "BAR-ATH_tactical_smart_tagging.xml")
    data["tag_team_score"] = tag_prior
    data["tag_active_count"] = interval_counts(events, lambda event: True)
    data["tag_set_piece"] = interval_counts(events, lambda event: str(event["code"]).upper() == "SET PIECES")
    data["tag_transition"] = interval_counts(events, lambda event: "TRANSITION" in str(event["code"]).upper())
    data["tag_absent"] = (data.tag_active_count == 0).astype(np.int8)

    homography = pd.read_csv(tactical / "BAR-ATH_tactical_homography.csv")
    data["homography_valid"] = homography["valid"].astype(np.int8)
    data["homography_reprojection_error"] = homography["reprojection_error"].astype(np.float32)
    data["tracking_complete_enough"] = ((data.home_players_visible >= 8) & (data.away_players_visible >= 8)).astype(np.int8)

    video_detections = ROOT / "reports/video_heuristics/detections.csv"
    if video_detections.exists():
        detections = pd.read_csv(video_detections)
        for method in ("bright", "template", "flow"):
            found = detections[detections.method == method].groupby("frame")["found"].max()
            data[f"video_{method}_sampled"] = data.frame.isin(found.index).astype(np.int8)
            data[f"video_{method}_found"] = data.frame.map(found).fillna(False).astype(np.int8)
        data["video_sampled_any"] = data.filter(regex="video_.*_sampled").max(axis=1)
    else:
        data["video_sampled_any"] = 0

    path = args.out_dir / "BAR-ATH_frame_features.parquet"
    data.to_parquet(path, index=False)
    qa = {
        "n_frames": len(data),
        "n_features": len(data.columns),
        "ball_observed_rate": float(observed.mean()),
        "graded_ball_observed_rate": float(observed[data.is_graded_frame.astype(bool)].mean()),
        "smart_tagging_events": len(events),
        "homography_valid_rate": float(data.homography_valid.mean()),
        "video_detections_included": video_detections.exists(),
        "missing_rate_mean": float(data.isna().mean().mean()),
    }
    (args.out_dir / "feature_store_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2))
    print(f"Features: {path}")


if __name__ == "__main__":
    main()
