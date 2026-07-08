#!/usr/bin/env python3
"""
Compute clustering quality metrics for different representations.

Metrics:
- Silhouette score: higher is better
- Davies-Bouldin index: lower is better
- Calinski-Harabasz score: higher is better

Outputs:
- results/metrics/clustering_metrics.csv
- results/figures/clustering_<metric>_<label_type>.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler


METRICS_DIR = Path("results/metrics")
FIG_DIR = Path("results/figures")

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "whole_embedding": {
        "x_path": Path("results/embeddings/X_whole_embedding_subset499.npy"),
        "meta_path": Path("results/embeddings/whole_embedding_subset499_metadata.csv"),
        "label": "Whole Emb.",
    },
    "precs_embedding": {
        "x_path": Path("results/embeddings/X_precs_embedding_subset499.npy"),
        "meta_path": Path("results/embeddings/precs_embedding_subset499_metadata.csv"),
        "label": "Pre-CS Emb.",
    },
    "original_cpp": {
        "x_path": Path("results/cpp_features/X_original_cpp_subset499.npy"),
        "meta_path": Path("results/cpp_features/original_cpp_subset499_metadata.csv"),
        "label": "Original CPP",
    },
    "embedding_cpp": {
        "x_path": Path("results/embedding_cpp/X_embedding_cpp_subset499.npy"),
        "meta_path": Path("results/embedding_cpp/embedding_cpp_subset499_metadata.csv"),
        "label": "Embedding-CPP",
    },
}


METRIC_INFO = {
    "silhouette": {
        "title": "Silhouette score",
        "ylabel": "Silhouette score",
        "higher_better": True,
    },
    "davies_bouldin": {
        "title": "Davies-Bouldin index",
        "ylabel": "Davies-Bouldin index",
        "higher_better": False,
    },
    "calinski_harabasz": {
        "title": "Calinski-Harabasz score",
        "ylabel": "Calinski-Harabasz score",
        "higher_better": True,
    },
}


def compute_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    X_scaled = StandardScaler().fit_transform(X)

    return {
        "silhouette": silhouette_score(X_scaled, labels),
        "davies_bouldin": davies_bouldin_score(X_scaled, labels),
        "calinski_harabasz": calinski_harabasz_score(X_scaled, labels),
    }

def plot_summary(df_results, label_type):
    sub = df_results[df_results["label_type"] == label_type].copy()

    order = list(DATASETS.keys())
    sub["representation"] = pd.Categorical(
        sub["representation"],
        categories=order,
        ordered=True,
    )
    sub = sub.sort_values("representation")

    labels = sub["display_name"]

    metrics = [
        ("silhouette", "Silhouette ↑"),
        ("davies_bouldin", "Davies-Bouldin ↓"),
        ("calinski_harabasz", "Calinski-Harabasz ↑"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (metric, title) in zip(axes, metrics):
        ax.bar(labels, sub[metric])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle(
        "Binary clustering metrics"
        if label_type == "binary"
        else "Multiclass clustering metrics",
        fontsize=14,
    )

    plt.tight_layout()

    out_path = FIG_DIR / f"clustering_metrics_{label_type}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)

def main() -> None:
    rows = []

    for name, paths in DATASETS.items():
        X = np.load(paths["x_path"])
        meta = pd.read_csv(paths["meta_path"])

        print(f"\n{name}")
        print("X:", X.shape)

        label_sets = {
            "binary": meta["label_binary"].astype(str).values,
            "sp_type": meta["sp_type"].astype(str).values,
        }

        for label_name, labels in label_sets.items():
            scores = compute_metrics(X, labels)

            row = {
                "representation": name,
                "display_name": paths["label"],
                "label_type": label_name,
                **scores,
            }
            rows.append(row)

            print(label_name, scores)

    df_results = pd.DataFrame(rows)

    out_path = METRICS_DIR / "clustering_metrics.csv"
    df_results.to_csv(out_path, index=False)

    print("\nSaved:", out_path)
    print(df_results)

    plot_summary(df_results, "binary")
    plot_summary(df_results, "sp_type")

if __name__ == "__main__":
    main()