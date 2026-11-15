"""Compare non-trained visual heuristics for recovering the ball from tactical video."""

import argparse
import json
import sys
from pathlib import Path

import cv2
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
    evidence_codes,
    load_slot_teams,
    load_tracking,
    nearest_features,
    interpolate_ball,
)
from video_ball_recovery_experiment import (  # noqa: E402
    detect_bright_ball,
    load_homography,
    project,
    sample_indices,
    unproject,
)


def detect_template_ball(previous_frame, current_frame, center):
    previous = cv2.cvtColor(cv2.resize(previous_frame, (960, 540)), cv2.COLOR_BGR2GRAY)
    current = cv2.cvtColor(cv2.resize(current_frame, (960, 540)), cv2.COLOR_BGR2GRAY)
    center = np.asarray(center, dtype=np.float32) / 2.0
    x, y = np.rint(center).astype(int)
    radius = 10
    if x < radius or y < radius or x + radius >= current.shape[1] or y + radius >= current.shape[0]:
        return None
    template = previous[y - radius:y + radius, x - radius:x + radius]
    search_radius = 65
    left, right = max(0, x - search_radius), min(current.shape[1], x + search_radius)
    top, bottom = max(0, y - search_radius), min(current.shape[0], y + search_radius)
    search = current[top:bottom, left:right]
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    candidate = np.array([left + location[0] + radius, top + location[1] + radius], dtype=np.float32) * 2.0
    distance = float(np.linalg.norm(candidate - np.asarray(center) * 2.0))
    if not np.isfinite(score) or score < 0.15:
        return None
    return candidate, float(score), distance


def detect_flow_ball(previous_frame, current_frame, center):
    previous = cv2.cvtColor(cv2.resize(previous_frame, (960, 540)), cv2.COLOR_BGR2GRAY)
    current = cv2.cvtColor(cv2.resize(current_frame, (960, 540)), cv2.COLOR_BGR2GRAY)
    center = np.asarray(center, dtype=np.float32) / 2.0
    x, y = np.rint(center).astype(int)
    radius = 35
    left, right = max(0, x - radius), min(previous.shape[1], x + radius)
    top, bottom = max(0, y - radius), min(previous.shape[0], y + radius)
    crop = previous[top:bottom, left:right]
    points = cv2.goodFeaturesToTrack(crop, maxCorners=20, qualityLevel=0.01, minDistance=3)
    if points is None or len(points) < 2:
        return None
    points[:, 0, 0] += left
    points[:, 0, 1] += top
    next_points, status, error = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if next_points is None:
        return None
    valid = status.ravel().astype(bool) & np.isfinite(next_points).all(axis=2).ravel()
    if valid.sum() < 2:
        return None
    displacement = np.median((next_points[valid] - points[valid]).reshape(-1, 2), axis=0)
    candidate = (center + displacement) * 2.0
    confidence = float(min(1.0, valid.sum() / 10.0))
    distance = float(np.linalg.norm(displacement) * 2.0)
    return candidate, confidence, distance


def evaluate(gt, labels):
    codes = possession_metrics.label_codes(labels)
    full = possession_metrics.evaluate(gt, codes)
    public = possession_metrics.evaluate(gt[PUBLIC_START:PUBLIC_END], codes[PUBLIC_START:PUBLIC_END])
    return full, public


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/video_heuristics")
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    tactical = args.data_dir / "tactical"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "submissions").mkdir(exist_ok=True)
    slot_teams = load_slot_teams(tactical / "BAR-ATH_tactical_atd.xml")
    players, balls = load_tracking(tactical / "BAR-ATH_tactical_atd.txt", len(slot_teams))
    matrices, valid_homography = load_homography(tactical / "BAR-ATH_tactical_homography.csv")
    official_codes = build(data_dir=str(args.data_dir))
    baseline = smooth(official_codes, 50)
    gt = possession_metrics.label_codes(possession_metrics.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"))
    observed_ball = np.isfinite(balls[:, :2]).all(axis=1)
    graded = np.zeros(N_FRAMES, dtype=bool)
    graded[PUBLIC_START:90652] = True
    graded[110808:190198] = True
    interpolated = interpolate_ball(balls, 50)[:, :2]
    candidate = np.isfinite(interpolated).all(axis=1) & valid_homography & graded
    observed = sample_indices(np.flatnonzero(observed_ball & candidate), args.samples)
    gaps = sample_indices(np.flatnonzero(~observed_ball & candidate), args.samples)
    modes = {"bright": detect_bright_ball, "template": detect_template_ball, "flow": detect_flow_ball}
    recovered = {name: balls.copy() for name in modes}
    records = []
    capture = cv2.VideoCapture(str(tactical / "BAR-ATH_tactical.mp4"))
    if not capture.isOpened():
        raise RuntimeError("Unable to open tactical video")
    for kind, indices in (("observed", observed), ("gap", gaps)):
        for index in indices:
            index = int(index)
            capture.set(cv2.CAP_PROP_POS_FRAMES, index - 1 if index else index)
            previous_ok, previous_frame = capture.read()
            current_ok, current_frame = capture.read() if index else (previous_ok, previous_frame)
            if not current_ok:
                continue
            center = project(matrices[index], balls[index, :2] if kind == "observed" else interpolated[index])
            for name, detector in modes.items():
                result = detector(current_frame, center) if name == "bright" else detector(previous_frame, current_frame, center)
                record = {"frame": index, "kind": kind, "method": name, "found": result is not None}
                if result is not None:
                    pixel, confidence, distance = result
                    record.update({"confidence": confidence, "pixel_distance": distance})
                    if kind == "gap":
                        recovered_position = unproject(matrices[index], pixel)
                        valid_position = np.isfinite(recovered_position).all() and np.all((recovered_position >= 0) & (recovered_position <= 1))
                        record["recovered"] = bool(valid_position)
                        if valid_position:
                            recovered[name][index, :2] = recovered_position
                records.append(record)
    capture.release()
    detections = pd.DataFrame(records)
    rows = []
    for name, ball_data in recovered.items():
        features = nearest_features(players, ball_data, slot_teams)
        codes = evidence_codes(features)
        predictions = {f"video_{name}_raw": np.where(codes == 0, "H", np.where(codes == 1, "A", "D"))}
        for window in (25, 50, 125):
            predictions[f"video_{name}_smooth{window}"] = smooth(codes, window)
        for experiment, labels in predictions.items():
            write_submission(str(args.out_dir / "submissions" / f"{experiment}.csv"), labels)
            full, public = evaluate(gt, labels)
            rows.append({"experiment": experiment, "public_macro_f1": public["macro_f1"], "public_spell_f1": public["spell_f1_mean"], "public_team_acc": public["team_acc_gt_alive"], "public_accuracy": public["accuracy"], "full_file_macro_f1": full["macro_f1"], "full_file_spell_f1": full["spell_f1_mean"]})
    write_submission(str(args.out_dir / "submissions/baseline_smooth50.csv"), baseline)
    baseline_full, baseline_public = evaluate(gt, baseline)
    rows.append({"experiment": "baseline_smooth50", "public_macro_f1": baseline_public["macro_f1"], "public_spell_f1": baseline_public["spell_f1_mean"], "public_team_acc": baseline_public["team_acc_gt_alive"], "public_accuracy": baseline_public["accuracy"], "full_file_macro_f1": baseline_full["macro_f1"], "full_file_spell_f1": baseline_full["spell_f1_mean"]})
    table = pd.DataFrame(rows).sort_values(["public_spell_f1", "public_macro_f1"], ascending=False)
    table.to_csv(args.out_dir / "metrics.csv", index=False)
    detections.to_csv(args.out_dir / "detections.csv", index=False)
    summary = {"samples_observed": int(len(observed)), "samples_gaps": int(len(gaps)), "methods": {}}
    for name in modes:
        subset = detections[detections.method == name]
        known = subset[subset.kind == "observed"]
        gaps_subset = subset[subset.kind == "gap"]
        summary["methods"][name] = {"observed_detection_rate": float(known.found.mean()), "gap_recovery_rate": float(gaps_subset.recovered.eq(True).mean()), "observed_median_pixel_error": float(known.loc[known.found, "pixel_distance"].median())}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
