"""Run unsupervised KMeans and Gaussian mixture diagnostics on frame features."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics  # noqa: E402

PUBLIC_START, PUBLIC_END = 11185, 18685


def label_distribution(features, clusters, labels):
    table = pd.DataFrame({"cluster": clusters, "label": labels})
    return table.groupby(["cluster", "label"], observed=False).size().reset_index(name="count")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=ROOT / "features/BAR-ATH_frame_features.parquet")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/clustering")
    parser.add_argument("--sample-size", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--gmm-sample-size", type=int, default=8000)
    parser.add_argument("--gmm-max-iter", type=int, default=40)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(args.features)
    excluded = {"frame", "is_graded_frame"}
    numeric = features.select_dtypes(include=["number"]).columns.difference(excluded)
    numeric = [column for column in numeric if features[column].notna().any()]
    matrix = features[numeric].replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median()).fillna(0.0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix).astype(np.float32)
    components = min(12, scaled.shape[1])
    reduced = PCA(n_components=components, random_state=0).fit_transform(scaled).astype(np.float32)
    sample_size = min(args.sample_size, len(scaled))
    sample_indices = np.linspace(0, len(scaled) - 1, sample_size, dtype=np.int64)
    sample = reduced[sample_indices]
    output = features[["frame", "ball_observed", "tag_active_count", "homography_reprojection_error"]].copy()
    summaries = []
    gt = possession_metrics.label_codes(possession_metrics.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"))

    for n_clusters in (8, 12, 16):
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init=5, batch_size=2048).fit(sample)
        labels = np.concatenate([
            kmeans.predict(reduced[start:start + args.batch_size])
            for start in range(0, len(reduced), args.batch_size)
        ])
        name = f"kmeans_{n_clusters}"
        output[name] = labels
        counts = np.bincount(labels, minlength=n_clusters)
        summaries.extend({"method": name, "cluster": cluster, "count": int(counts[cluster]), "ball_observed_rate": float(output.loc[labels == cluster, "ball_observed"].mean()), "tag_active_rate": float((output.loc[labels == cluster, "tag_active_count"] > 0).mean())} for cluster in range(n_clusters))

        public = output.iloc[PUBLIC_START:PUBLIC_END].copy()
        public["gt_label"] = gt[PUBLIC_START:PUBLIC_END]
        label_distribution(public, labels[PUBLIC_START:PUBLIC_END], public.gt_label).to_csv(args.out_dir / f"{name}_public_label_distribution.csv", index=False)

    gmm_sample_size = min(args.gmm_sample_size, len(sample))
    gmm_sample = sample[:gmm_sample_size]
    for n_components in (8, 12):
        mixture = GaussianMixture(
            n_components=n_components,
            random_state=0,
            covariance_type="diag",
            max_iter=args.gmm_max_iter,
            reg_covar=1e-4,
            n_init=1,
        ).fit(gmm_sample)
        labels = np.concatenate([
            mixture.predict(reduced[start:start + args.batch_size])
            for start in range(0, len(reduced), args.batch_size)
        ])
        confidence = np.concatenate([
            mixture.predict_proba(reduced[start:start + args.batch_size]).max(axis=1)
            for start in range(0, len(reduced), args.batch_size)
        ]).astype(np.float32)
        name = f"gmm_{n_components}"
        output[name] = labels
        output[f"{name}_confidence"] = confidence
        counts = np.bincount(labels, minlength=n_components)
        summaries.extend({"method": name, "cluster": cluster, "count": int(counts[cluster]), "ball_observed_rate": float(output.loc[labels == cluster, "ball_observed"].mean()), "tag_active_rate": float((output.loc[labels == cluster, "tag_active_count"] > 0).mean())} for cluster in range(n_components))
        public = output.iloc[PUBLIC_START:PUBLIC_END].copy()
        public["gt_label"] = gt[PUBLIC_START:PUBLIC_END]
        label_distribution(public, labels[PUBLIC_START:PUBLIC_END], public.gt_label).to_csv(args.out_dir / f"{name}_public_label_distribution.csv", index=False)

    output.to_parquet(args.out_dir / "clustered_features.parquet", index=False)
    pd.DataFrame(summaries).to_csv(args.out_dir / "cluster_summary.csv", index=False)
    config = {"feature_count": len(numeric), "sample_size": sample_size, "gmm_sample_size": gmm_sample_size, "pca_components": components, "kmeans_clusters": [8, 12, 16], "gmm_components": [8, 12], "public_ground_truth_used_for_diagnostic_only": True}
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))
    print(f"Clusters: {args.out_dir / 'clustered_features.parquet'}")


if __name__ == "__main__":
    main()
