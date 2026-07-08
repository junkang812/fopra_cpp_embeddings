#!/usr/bin/env python3
"""
Plot PCA comparison for three representations:
1. Pre-CS mean ESM2 embeddings
2. Original CPP features
3. Embedding-based CPP features
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "whole_embedding": {
    "x_path": Path("results/embeddings/X_whole_embedding_subset499.npy"),
    "meta_path": Path("results/embeddings/whole_embedding_subset499_metadata.csv"),
    "title": "PCA of whole-sequence mean ESM2 embeddings",
    "out_path": FIG_DIR / "pca_whole_embedding_sp_type.png",
    "meta_out": Path("results/embeddings/pca_whole_embedding_metadata.csv"),
    },
    "precs_embedding": {
        "x_path": Path("results/embeddings/X_precs_embedding_subset499.npy"),
        "meta_path": Path("results/embeddings/precs_embedding_subset499_metadata.csv"),
        "title": "PCA of pre-CS mean ESM2 embeddings",
        "out_path": FIG_DIR / "pca_precs_embedding_sp_type.png",
        "meta_out": Path("results/embeddings/pca_precs_embedding_metadata.csv"),
    },
    "original_cpp": {
        "x_path": Path("results/cpp_features/X_original_cpp_subset499.npy"),
        "meta_path": Path("results/cpp_features/original_cpp_subset499_metadata.csv"),
        "title": "PCA of original CPP features",
        "out_path": FIG_DIR / "pca_original_cpp_sp_type.png",
        "meta_out": Path("results/cpp_features/pca_original_cpp_metadata.csv"),
    },
    "embedding_cpp": {
        "x_path": Path("results/embedding_cpp/X_embedding_cpp_subset499.npy"),
        "meta_path": Path("results/embedding_cpp/embedding_cpp_subset499_metadata.csv"),
        "title": "PCA of embedding-based CPP features",
        "out_path": FIG_DIR / "pca_embedding_cpp_sp_type.png",
        "meta_out": Path("results/embedding_cpp/pca_embedding_cpp_metadata.csv"),
    },
}


def plot_pca(name: str, config: dict) -> None:
    X = np.load(config["x_path"])
    meta = pd.read_csv(config["meta_path"])

    print(f"\n{name}")
    print("X:", X.shape)
    print(meta["sp_type"].value_counts())

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    meta["PC1"] = X_pca[:, 0]
    meta["PC2"] = X_pca[:, 1]

    print("Explained variance:", pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 6))

    for sp_type, sub in meta.groupby("sp_type"):
        plt.scatter(sub["PC1"], sub["PC2"], label=sp_type, alpha=0.75, s=30)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
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
        plot_pca(name, config)


if __name__ == "__main__":
    main()