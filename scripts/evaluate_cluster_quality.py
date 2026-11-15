"""Evaluate whether unsupervised frame clusters carry possession-label signal.

The public labels are used only as an offline diagnostic. They are never used
to fit the scaler, PCA, or clustering models.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_START, PUBLIC_END = 11185, 18685

import sys

sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clusters",
        type=Path,
        default=ROOT / "reports/clustering/clustered_features.parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports/clustering/cluster_quality.csv",
    )
    args = parser.parse_args()

    data = pd.read_parquet(args.clusters)
    gt = possession_metrics.label_codes(
        possession_metrics.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    )
    public_gt = np.asarray(gt[PUBLIC_START:PUBLIC_END])
    rows = []
    for method in [column for column in data.columns if column.startswith(("kmeans_", "gmm_"))]:
        labels = data[method].to_numpy()[PUBLIC_START:PUBLIC_END]
        table = pd.crosstab(labels, public_gt)
        purity = float(table.max(axis=1).sum() / table.to_numpy().sum())
        majority_accuracy = purity
        rows.append(
            {
                "method": method,
                "public_frames": int(len(public_gt)),
                "n_clusters_used": int(np.unique(labels).size),
                "majority_label_accuracy": majority_accuracy,
                "purity": purity,
                "adjusted_rand_index": float(adjusted_rand_score(public_gt, labels)),
                "normalized_mutual_information": float(normalized_mutual_info_score(public_gt, labels)),
            }
        )

    result = pd.DataFrame(rows).sort_values("normalized_mutual_information", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
