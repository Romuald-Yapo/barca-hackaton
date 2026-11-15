"""Validate and apply a local, non-trained visual ball recovery experiment."""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
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
    load_slot_teams,
    load_tracking,
    nearest_features,
    evidence_codes,
)


def load_homography(path):
    table = pd.read_csv(path)
    columns = [f"h{row}{column}" for row in range(3) for column in range(3)]
    matrices = table[columns].to_numpy(dtype=np.float32).reshape(-1, 3, 3)
    return matrices, table["valid"].to_numpy(dtype=bool)


def project(matrix, point):
    denominator = matrix[2, 0] * point[0] + matrix[2, 1] * point[1] + matrix[2, 2]
    return np.array([
        (matrix[0, 0] * point[0] + matrix[0, 1] * point[1] + matrix[0, 2]) / denominator,
        (matrix[1, 0] * point[0] + matrix[1, 1] * point[1] + matrix[1, 2]) / denominator,
    ], dtype=np.float32)


def unproject(matrix, pixel):
    inverse = np.linalg.inv(matrix)
    point = inverse @ np.array([pixel[0], pixel[1], 1.0], dtype=np.float32)
    return point[:2] / point[2]


def sample_indices(indices, limit):
    indices = np.asarray(indices, dtype=np.int32)
    if len(indices) <= limit:
        return indices
    return indices[np.linspace(0, len(indices) - 1, limit, dtype=np.int64)]


def read_frame(capture, index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    success, frame = capture.read()
    return frame if success else None


def detect_bright_ball(frame, center, search_radius=80):
    height, width = frame.shape[:2]
    x, y = int(round(center[0])), int(round(center[1]))
    left, right = max(0, x - search_radius), min(width, x + search_radius)
    top, bottom = max(0, y - search_radius), min(height, y + search_radius)
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 145], dtype=np.uint8), np.array([180, 105, 255], dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 2 <= area <= 350:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        candidate = np.array([
            left + moments["m10"] / moments["m00"],
            top + moments["m01"] / moments["m00"],
        ])
        distance = float(np.linalg.norm(candidate - np.asarray(center)))
        candidates.append((distance, area, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    distance, area, candidate = candidates[0]
    confidence = max(0.0, 1.0 - distance / search_radius) * min(1.0, area / 20.0)
    return candidate, confidence, distance


def evaluate_predictions(gt, labels):
    full = possession_metrics.evaluate(gt, possession_metrics.label_codes(labels))
    public = possession_metrics.evaluate(
        gt[PUBLIC_START:PUBLIC_END],
        possession_metrics.label_codes(labels)[PUBLIC_START:PUBLIC_END],
    )
    return {
        "full_public_file": full,
        "public_window": public,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/video_ball_recovery")
    parser.add_argument("--samples", type=int, default=300)
    args = parser.parse_args()
    tactical = args.data_dir / "tactical"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "submissions").mkdir(exist_ok=True)

    slot_teams = load_slot_teams(tactical / "BAR-ATH_tactical_atd.xml")
    players, balls = load_tracking(tactical / "BAR-ATH_tactical_atd.txt", len(slot_teams))
    matrices, valid_homography = load_homography(tactical / "BAR-ATH_tactical_homography.csv")
    official_codes = build(data_dir=str(args.data_dir))
    baseline_labels = smooth(official_codes, 50)
    ground_truth = possession_metrics.label_codes(possession_metrics.read_label_csv(
        ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"))

    ball_visible = np.isfinite(balls[:, :2]).all(axis=1)
    graded = np.zeros(N_FRAMES, dtype=bool)
    graded[PUBLIC_START:90652] = True
    graded[110808:190198] = True
    observed = sample_indices(np.flatnonzero(ball_visible & graded & valid_homography), args.samples)
    interpolated = balls[:, :2].copy()
    for coordinate in range(2):
        interpolated[:, coordinate] = pd.Series(interpolated[:, coordinate]).interpolate(limit=50, limit_area="inside")
    gaps = sample_indices(np.flatnonzero(~ball_visible & graded & valid_homography & np.isfinite(interpolated).all(axis=1)), args.samples)

    capture = cv2.VideoCapture(str(tactical / "BAR-ATH_tactical.mp4"))
    if not capture.isOpened():
        raise RuntimeError("Tactical video cannot be opened")
    records = []
    recovered_balls = balls.copy()
    observed_errors = []
    for kind, indices in (("observed", observed), ("gap", gaps)):
        for index in indices:
            frame = read_frame(capture, int(index))
            if frame is None:
                continue
            if kind == "observed":
                center = project(matrices[index], balls[index, :2])
            else:
                center = project(matrices[index], interpolated[index])
            result = detect_bright_ball(frame, center)
            record = {"frame": int(index), "kind": kind, "found": result is not None}
            if result is not None:
                pixel, confidence, distance = result
                record.update({"confidence": confidence, "pixel_distance": distance})
                if kind == "observed":
                    observed_errors.append(distance)
                else:
                    recovered = unproject(matrices[index], pixel)
                    if np.isfinite(recovered).all() and np.all((recovered >= 0) & (recovered <= 1)):
                        recovered_balls[index, :2] = recovered
                        record["recovered"] = True
                    else:
                        record["recovered"] = False
            records.append(record)
    capture.release()
    detections = pd.DataFrame(records)

    recovered_features = nearest_features(players, recovered_balls, slot_teams)
    recovered_codes = evidence_codes(recovered_features)
    recovered_raw = np.where(recovered_codes == 0, "H", np.where(recovered_codes == 1, "A", "D"))
    recovered_smooth = smooth(recovered_codes, 50)
    predictions = {
        "baseline_smooth50": baseline_labels,
        "video_ball_recovery_raw": recovered_raw,
        "video_ball_recovery_smooth25": smooth(recovered_codes, 25),
        "video_ball_recovery_smooth50": recovered_smooth,
        "video_ball_recovery_smooth125": smooth(recovered_codes, 125),
    }
    rows = []
    for name, labels in predictions.items():
        write_submission(str(args.out_dir / "submissions" / f"{name}.csv"), labels)
        result = evaluate_predictions(ground_truth, labels)
        public = result["public_window"]
        rows.append({"experiment": name, "public_macro_f1": public["macro_f1"], "public_spell_f1": public["spell_f1_mean"], "public_team_acc": public["team_acc_gt_alive"], "public_accuracy": public["accuracy"]})
    pd.DataFrame(rows).to_csv(args.out_dir / "metrics.csv", index=False)
    detections.to_csv(args.out_dir / "detections.csv", index=False)
    summary = {
        "observed_samples": int(len(observed)),
        "gap_samples": int(len(gaps)),
        "observed_detection_rate": float(detections.loc[detections.kind == "observed", "found"].mean()),
        "observed_median_pixel_error": float(np.median(observed_errors)) if observed_errors else None,
        "gap_recovery_rate": float(detections.loc[detections.kind == "gap", "recovered"].eq(True).mean()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
