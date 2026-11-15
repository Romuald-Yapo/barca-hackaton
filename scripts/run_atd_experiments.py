"""Run official and deterministic ATD/Smart Tagging ablations."""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import possession_metrics as metrics  # noqa: E402
from starter.baseline import build as official_build  # noqa: E402
from starter.baseline import smooth as official_smooth  # noqa: E402
from starter.baseline import write_submission  # noqa: E402

FPS = 25
N_FRAMES = 199319
PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0
HOME, AWAY, NOBODY = 0, 1, -1
PUBLIC_START = 11185
PUBLIC_END = 18685


def parse_triplet(value):
    if not value or value.startswith("N"):
        return np.nan, np.nan, np.nan
    values = []
    for item in value.split(",")[:3]:
        try:
            values.append(float(item))
        except ValueError:
            values.append(np.nan)
    return tuple((values + [np.nan] * 3)[:3])


def load_slot_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"Expected two teams in {meta_path}")
    return np.asarray(
        [HOME if player.get("teamId") == team_ids[0] else AWAY
         for player in root.iter("Player")],
        dtype=np.int8,
    )


def load_tracking(raw_path, n_slots):
    frame_ids = []
    players = []
    balls = []
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip().split(":")
            if len(fields) != 3:
                continue
            frame_ids.append(int(fields[0]))
            frame_players = np.full((n_slots, 3), np.nan, dtype=np.float32)
            for slot, value in enumerate(fields[1].split(";")[:n_slots]):
                frame_players[slot] = parse_triplet(value)
            players.append(frame_players)
            balls.append(parse_triplet(fields[2]))
    frame_ids = np.asarray(frame_ids, dtype=np.int32)
    players = np.asarray(players, dtype=np.float32)
    balls = np.asarray(balls, dtype=np.float32)
    if len(frame_ids) != N_FRAMES or not np.array_equal(frame_ids, np.arange(1, N_FRAMES + 1)):
        raise ValueError("ATD is not the expected contiguous tactical timeline")
    return players, balls


def load_tag_prior(path):
    prior = np.zeros(N_FRAMES, dtype=np.float32)
    events = []
    for instance in ET.parse(path).iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        start = max(0, int(round(start_s * FPS)))
        end = min(N_FRAMES, int(round(end_s * FPS)))
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        team = str(tags.get("Team", "")).strip().lower()
        value = 1.0 if team == "barcelona" else -1.0 if team in {"athletic club", "atletic club"} else 0.0
        if end > start and value:
            prior[start:end] += value
        events.append({"code": instance.findtext("code"), "start": start, "end": end})
    return np.clip(prior, -2.0, 2.0), events


def interpolate_ball(balls, max_gap):
    output = balls.copy()
    for coordinate in range(2):
        series = pd.Series(output[:, coordinate])
        output[:, coordinate] = series.interpolate(limit=max_gap, limit_area="inside").to_numpy()
    return output


def nearest_features(players, balls, slot_teams):
    player_xy = players[:, :, :2].copy()
    player_xy[:, :, 0] *= PITCH_LENGTH
    player_xy[:, :, 1] *= PITCH_WIDTH
    ball_xy = balls[:, :2].copy()
    ball_xy[:, 0] *= PITCH_LENGTH
    ball_xy[:, 1] *= PITCH_WIDTH
    player_visible = np.isfinite(player_xy).all(axis=2)
    ball_visible = np.isfinite(ball_xy).all(axis=1)
    distances = np.sqrt(((player_xy - ball_xy[:, None, :]) ** 2).sum(axis=2))
    distances[~player_visible | ~ball_visible[:, None]] = np.nan
    speeds = players[:, :, 2].copy()
    speeds[~np.isfinite(speeds)] = 0.0
    eta = distances / np.maximum(speeds, 0.5)
    best = {}
    nearby = {}
    best_eta = {}
    for team, name in ((HOME, "home"), (AWAY, "away")):
        team_distances = distances[:, slot_teams == team]
        team_eta = eta[:, slot_teams == team]
        best_values = np.min(np.where(np.isfinite(team_distances), team_distances, np.inf), axis=1)
        best_values[best_values == np.inf] = np.nan
        best[name] = best_values
        nearby[name] = np.nansum(team_distances <= 5.0, axis=1)
        eta_values = np.min(np.where(np.isfinite(team_eta), team_eta, np.inf), axis=1)
        eta_values[eta_values == np.inf] = np.nan
        best_eta[name] = eta_values
    best_home = best["home"]
    best_away = best["away"]
    difference = best_home - best_away
    nearest_codes = np.full(len(ball_visible), NOBODY, dtype=np.int8)
    nearest_codes[ball_visible & (difference < 0)] = HOME
    nearest_codes[ball_visible & (difference > 0)] = AWAY
    return {
        "ball_visible": ball_visible,
        "home_best": best_home,
        "away_best": best_away,
        "difference": difference,
        "home_nearby": nearby["home"],
        "away_nearby": nearby["away"],
        "home_eta": best_eta["home"],
        "away_eta": best_eta["away"],
        "nearest_codes": nearest_codes,
    }


def smooth_codes(codes, window):
    return official_smooth(codes, window)


def evidence_codes(features, margin=None, tag_prior=None, collective=False, control=False):
    advantage = -features["difference"].copy()
    valid = features["ball_visible"].copy()
    if margin is not None:
        valid &= np.abs(advantage) >= margin
    if collective:
        home_score = np.nan_to_num(1.0 / (1.0 + features["home_best"])) + 0.05 * features["home_nearby"]
        away_score = np.nan_to_num(1.0 / (1.0 + features["away_best"])) + 0.05 * features["away_nearby"]
        advantage = home_score - away_score
    if control:
        advantage = features["away_eta"] - features["home_eta"]
    if tag_prior is not None:
        advantage = advantage + 0.08 * tag_prior
    codes = np.full(len(advantage), NOBODY, dtype=np.int8)
    codes[valid & (advantage > 0)] = HOME
    codes[valid & (advantage < 0)] = AWAY
    return codes


def bridge_short_dead_runs(labels, max_length):
    output = labels.copy()
    index = 0
    while index < len(output):
        if output[index] != "D":
            index += 1
            continue
        end = index
        while end < len(output) and output[end] == "D":
            end += 1
        before = output[index - 1] if index else ""
        after = output[end] if end < len(output) else ""
        if end - index <= max_length and before in {"H", "A"} and before == after:
            output[index:end] = before
        index = end
    return output


def stateful_labels(codes, switch_confirmation=5, hold_length=50):
    output = np.full(len(codes), "D", dtype="<U1")
    for start, end in ((11185, 90652), (110808, 190198)):
        current = NOBODY
        pending = NOBODY
        pending_count = 0
        frames_since_evidence = hold_length + 1
        for index in range(start, end):
            code = int(codes[index])
            if code == NOBODY:
                frames_since_evidence += 1
                output[index] = "H" if current == HOME and frames_since_evidence <= hold_length else "A" if current == AWAY and frames_since_evidence <= hold_length else "D"
                continue
            frames_since_evidence = 0
            if current == NOBODY:
                current = code
                pending = NOBODY
                pending_count = 0
            elif code == current:
                pending = NOBODY
                pending_count = 0
            elif code == pending:
                pending_count += 1
                if pending_count >= switch_confirmation:
                    current = code
                    pending = NOBODY
                    pending_count = 0
            else:
                pending = code
                pending_count = 1
            output[index] = "H" if current == HOME else "A" if current == AWAY else "D"
    return output


def project_points(matrices, points):
    denominator = matrices[:, 2, 0] * points[:, 0] + matrices[:, 2, 1] * points[:, 1] + matrices[:, 2, 2]
    projected_x = (matrices[:, 0, 0] * points[:, 0] + matrices[:, 0, 1] * points[:, 1] + matrices[:, 0, 2]) / denominator
    projected_y = (matrices[:, 1, 0] * points[:, 0] + matrices[:, 1, 1] * points[:, 1] + matrices[:, 1, 2]) / denominator
    return np.column_stack((projected_x, projected_y))


def video_motion_scores(video_path, homography_path, balls, candidate_labels, max_samples=500):
    homography = pd.read_csv(homography_path)
    matrix_columns = [f"h{row}{column}" for row in range(3) for column in range(3)]
    matrices = homography[matrix_columns].to_numpy(dtype=np.float32).reshape(-1, 3, 3)
    valid_homography = homography["valid"].to_numpy(dtype=bool)
    ball_for_video = interpolate_ball(balls, 50)[:, :2]
    projected = np.full((N_FRAMES, 2), np.nan, dtype=np.float32)
    usable = valid_homography & np.isfinite(ball_for_video).all(axis=1)
    projected[usable] = project_points(matrices[usable], ball_for_video[usable])

    graded = np.zeros(N_FRAMES, dtype=bool)
    graded[11185:90652] = True
    graded[110808:190198] = True
    candidate_indices = np.flatnonzero((np.asarray(candidate_labels) == "D") & graded & np.isfinite(projected).all(axis=1))
    reference_indices = np.flatnonzero((np.asarray(candidate_labels) != "D") & graded & np.isfinite(projected).all(axis=1))

    def sample_indices(indices, limit):
        if len(indices) <= limit:
            return indices
        positions = np.linspace(0, len(indices) - 1, limit, dtype=np.int64)
        return indices[positions]

    candidate_indices = sample_indices(candidate_indices, max_samples // 2)
    reference_indices = sample_indices(reference_indices, max_samples - len(candidate_indices))
    sampled_indices = np.unique(np.concatenate((candidate_indices, reference_indices)))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open tactical video: {video_path}")
    global_motion = np.full(N_FRAMES, np.nan, dtype=np.float32)
    local_motion = np.full(N_FRAMES, np.nan, dtype=np.float32)
    scale_x, scale_y = 480.0 / 1920.0, 270.0 / 1080.0
    for frame_index in sampled_indices:
        if frame_index == 0:
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index - 1))
        previous_ok, previous_frame = capture.read()
        current_ok, current_frame = capture.read()
        if not previous_ok or not current_ok:
            continue
        previous = cv2.cvtColor(cv2.resize(previous_frame, (480, 270)), cv2.COLOR_BGR2GRAY)
        current = cv2.cvtColor(cv2.resize(current_frame, (480, 270)), cv2.COLOR_BGR2GRAY)
        difference = cv2.absdiff(current, previous)
        global_motion[frame_index] = float(difference.mean())
        point = projected[frame_index]
        if np.isfinite(point).all():
            center_x, center_y = int(point[0] * scale_x), int(point[1] * scale_y)
            radius = 24
            left, right = max(0, center_x - radius), min(current.shape[1], center_x + radius)
            top, bottom = max(0, center_y - radius), min(current.shape[0], center_y + radius)
            if right > left and bottom > top:
                local_motion[frame_index] = float(difference[top:bottom, left:right].mean())
    capture.release()
    observed = np.asarray(candidate_labels) != "D"
    reference = observed & np.isfinite(global_motion)
    global_threshold = float(np.nanpercentile(global_motion[reference], 50)) if reference.any() else np.nan
    local_reference = reference & np.isfinite(local_motion)
    local_threshold = float(np.nanpercentile(local_motion[local_reference], 35)) if local_reference.any() else np.nan
    corrected = np.asarray(candidate_labels).copy()
    if np.isfinite(global_threshold) and np.isfinite(local_threshold):
        active = (corrected == "D") & np.isfinite(global_motion) & np.isfinite(local_motion) & (global_motion >= global_threshold) & (local_motion >= local_threshold)
        for index in np.flatnonzero(active):
            start = max(0, index - 50)
            end = min(N_FRAMES, index + 51)
            previous_labels = corrected[start:index][::-1]
            next_labels = corrected[index + 1:end]
            previous_team = next((label for label in previous_labels if label in {"H", "A"}), None)
            next_team = next((label for label in next_labels if label in {"H", "A"}), None)
            if previous_team is not None and previous_team == next_team:
                corrected[index] = previous_team
    return corrected, {
        "video_available": True,
        "video_experiment": "video_motion_local_roi",
        "video_samples_requested": int(len(sampled_indices)),
        "video_samples_read": int(np.isfinite(global_motion).sum()),
        "video_global_motion_threshold": global_threshold,
        "video_local_motion_threshold": local_threshold,
        "video_projected_ball_frames": int(np.isfinite(projected).all(axis=1).sum()),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--ground-truth", type=Path, default=ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/atd_experiments")
    parser.add_argument("--video-samples", type=int, default=500)
    args = parser.parse_args()
    tactical = args.data_dir / "tactical"
    meta_path = tactical / "BAR-ATH_tactical_atd.xml"
    raw_path = tactical / "BAR-ATH_tactical_atd.txt"
    tags_path = tactical / "BAR-ATH_tactical_smart_tagging.xml"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "submissions").mkdir(exist_ok=True)

    official_codes = official_build(data_dir=str(args.data_dir))
    slot_teams = load_slot_teams(meta_path)
    players, balls = load_tracking(raw_path, len(slot_teams))
    features = nearest_features(players, balls, slot_teams)
    interpolated_features = nearest_features(players, interpolate_ball(balls, 5), slot_teams)
    tag_prior, events = load_tag_prior(tags_path)
    margin_codes = evidence_codes(features, margin=0.75)
    collective_codes = evidence_codes(features, collective=True)
    tags_codes = evidence_codes(features, tag_prior=tag_prior)
    control_codes = evidence_codes(features, control=True)
    interpolated_codes = evidence_codes(interpolated_features)

    predictions = {
        "baseline_raw": np.asarray(np.where(official_codes == HOME, "H", np.where(official_codes == AWAY, "A", "D"))),
        "baseline_smooth25": smooth_codes(official_codes, 25),
        "baseline_smooth50": smooth_codes(official_codes, 50),
        "baseline_smooth125": smooth_codes(official_codes, 125),
        "margin_raw": np.asarray(np.where(margin_codes == HOME, "H", np.where(margin_codes == AWAY, "A", "D"))),
        "collective_raw": np.asarray(np.where(collective_codes == HOME, "H", np.where(collective_codes == AWAY, "A", "D"))),
        "tags_raw": np.asarray(np.where(tags_codes == HOME, "H", np.where(tags_codes == AWAY, "A", "D"))),
        "control_raw": np.asarray(np.where(control_codes == HOME, "H", np.where(control_codes == AWAY, "A", "D"))),
        "ball_interp5_raw": np.asarray(np.where(interpolated_codes == HOME, "H", np.where(interpolated_codes == AWAY, "A", "D"))),
    }
    for name, codes in {
        "margin": margin_codes,
        "collective": collective_codes,
        "tags": tags_codes,
        "control": control_codes,
        "ball_interp5": interpolated_codes,
    }.items():
        for window in (25, 50, 125):
            predictions[f"{name}_smooth{window}"] = smooth_codes(codes, window)
    predictions["bridge_short_gaps50"] = bridge_short_dead_runs(predictions["baseline_smooth50"], 50)
    predictions["stateful_baseline"] = stateful_labels(official_codes)
    predictions["stateful_control"] = stateful_labels(control_codes)
    video_path = tactical / "BAR-ATH_tactical.mp4"
    video_info = {"video_available": video_path.exists(), "video_experiment": "skipped"}
    if video_path.exists():
        video_labels, video_info = video_motion_scores(
            video_path,
            tactical / "BAR-ATH_tactical_homography.csv",
            balls,
            predictions["baseline_smooth50"],
            max_samples=args.video_samples,
        )
        predictions["video_motion_smooth50"] = video_labels

    ground_truth = metrics.label_codes(metrics.read_label_csv(args.ground_truth))
    rows = []
    full_results = {}
    for name, labels in predictions.items():
        output_path = args.out_dir / "submissions" / f"{name}.csv"
        write_submission(str(output_path), labels)
        result = metrics.evaluate(ground_truth, metrics.label_codes(labels))
        public_result = metrics.evaluate(
            ground_truth[PUBLIC_START:PUBLIC_END],
            metrics.label_codes(labels)[PUBLIC_START:PUBLIC_END],
        )
        full_results[name] = {
            "full_public_file": json_safe(result),
            "public_window_17_spells": json_safe(public_result),
        }
        rows.append({
            "experiment": name,
            "stage": "raw" if "raw" in name else "temporal",
            "macro_f1": result["macro_f1"],
            "spell_f1_mean": result["spell_f1_mean"],
            "spell_f1_at_headline": result["spell_f1_at_headline"],
            "alive_dead_f1": result["alive_dead_f1"],
            "team_acc_gt_alive": result["team_acc_gt_alive"],
            "team_acc_both_alive": result["team_acc_both_alive"],
            "n_pred_spells": result["n_pred_spells"],
            "spell_count_error": result["spell_count_error"],
            "mean_spell_duration_s": result["mean_spell_duration_s"],
            "pred_home_share": result["pred_home_share"],
            "public_macro_f1": public_result["macro_f1"],
            "public_spell_f1_mean": public_result["spell_f1_mean"],
            "public_team_acc_gt_alive": public_result["team_acc_gt_alive"],
            "public_team_acc_both_alive": public_result["team_acc_both_alive"],
            "public_accuracy": public_result["accuracy"],
        })
    table = pd.DataFrame(rows).sort_values(["spell_f1_mean", "macro_f1"], ascending=False)
    table.to_csv(args.out_dir / "metrics.csv", index=False)
    table[[
        "experiment", "stage", "public_macro_f1", "public_spell_f1_mean",
        "public_team_acc_gt_alive", "public_team_acc_both_alive", "public_accuracy",
    ]].to_csv(args.out_dir / "public_window_metrics.csv", index=False)
    (args.out_dir / "metrics.json").write_text(json.dumps(full_results, indent=2), encoding="utf-8")
    config = {
        "data_dir": str(args.data_dir), "ground_truth": str(args.ground_truth),
        "n_frames": N_FRAMES, "n_smart_tagging_events": len(events),
        **video_info,
        "homography_available": (tactical / "BAR-ATH_tactical_homography.csv").exists(),
        "interpolation_max_gap_frames": 5,
        "state_switch_confirmation_frames": 5,
        "state_hold_length_frames": 50,
        "video_samples_limit": args.video_samples,
        "public_ground_truth_only": True,
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nOutputs: {args.out_dir}")


if __name__ == "__main__":
    main()
