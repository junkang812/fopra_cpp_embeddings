#!/usr/bin/env python3
"""
Plot UMAP comparison for three representations:
1. Pre-CS mean ESM2 embeddings
2. Original CPP features
3. Embedding-based CPP features
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "whole_embedding": {
    "x_path": Path("results/embeddings/X_whole_embedding_subset499.npy"),
    "meta_path": Path("results/embeddings/whole_embedding_subset499_metadata.csv"),
    "title": "UMAP of whole-sequence mean ESM2 embeddings",
    "out_path": FIG_DIR / "umap_whole_embedding_sp_type.png",
    "meta_out": Path("results/embeddings/umap_whole_embedding_metadata.csv"),    
    },
    "precs_embedding": {
        "x_path": Path("results/embeddings/X_precs_embedding_subset499.npy"),
        "meta_path": Path("results/embeddings/precs_embedding_subset499_metadata.csv"),
        "title": "UMAP of pre-CS mean ESM2 embeddings",
        "out_path": FIG_DIR / "umap_precs_embedding_sp_type.png",
        "meta_out": Path("results/embeddings/umap_precs_embedding_metadata.csv"),
    },
    "original_cpp": {
        "x_path": Path("results/cpp_features/X_original_cpp_subset499.npy"),
        "meta_path": Path("results/cpp_features/original_cpp_subset499_metadata.csv"),
        "title": "UMAP of original CPP features",
        "out_path": FIG_DIR / "umap_original_cpp_sp_type.png",
        "meta_out": Path("results/cpp_features/umap_original_cpp_metadata.csv"),
    },
    "embedding_cpp": {
        "x_path": Path("results/embedding_cpp/X_embedding_cpp_subset499.npy"),
        "meta_path": Path("results/embedding_cpp/embedding_cpp_subset499_metadata.csv"),
        "title": "UMAP of embedding-based CPP features",
        "out_path": FIG_DIR / "umap_embedding_cpp_sp_type.png",
        "meta_out": Path("results/embedding_cpp/umap_embedding_cpp_metadata.csv"),
    },
}

def plot_umap(name: str, config: dict) -> None:
    X = np.load(config["x_path"])
    meta = pd.read_csv(config["meta_path"])

    print(f"\n{name}")
    print("X:", X.shape)
    print(meta["sp_type"].value_counts())

    X_scaled = StandardScaler().fit_transform(X)

    # Stabilize UMAP by reducing very high-dimensional matrices first.
    n_pca = min(50, X_scaled.shape[1], X_scaled.shape[0] - 1)

    if n_pca >= 2:
        X_input = PCA(n_components=n_pca, random_state=42).fit_transform(X_scaled)
    else:
        X_input = X_scaled

    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42,
        n_jobs=1,
    )

    X_umap = reducer.fit_transform(X_input)

    meta["UMAP1"] = X_umap[:, 0]
    meta["UMAP2"] = X_umap[:, 1]

    plt.figure(figsize=(8, 6))

    for sp_type, sub in meta.groupby("sp_type"):
        plt.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            label=sp_type,
            alpha=0.75,
            s=30,
        )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(config["title"])
    plt.legend()
    plt.tight_layout()

    plt.savefig(config["out_path"], dpi=300)
    plt.close()

    config["meta_out"].parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(config["meta_out"], index=False)

    print("Saved figure:", config["out_path"])
    print("Saved metadata:", config["meta_out"])


def main() -> None:
    for name, config in DATASETS.items():
        plot_umap(name, config)


if __name__ == "__main__":
    main()