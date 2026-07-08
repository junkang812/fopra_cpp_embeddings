#!/usr/bin/env python3
"""
Plot UMAP comparison for four representations.

For each representation, two plots are saved:
1. Colored by sp_type
2. Colored by binary label: NO_SP vs SP
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FIG_DIR = Path("results/figures/binary_feature")
BINARY_DIR = FIG_DIR / "binary"
MULTI_DIR = FIG_DIR / "multiclass"

FIG_DIR.mkdir(parents=True, exist_ok=True)
BINARY_DIR.mkdir(parents=True, exist_ok=True)
MULTI_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "whole_embedding": {
        "x_path": Path("results/embeddings/X_whole_embedding_subset499.npy"),
        "meta_path": Path("results/embeddings/whole_embedding_subset499_metadata.csv"),
        "title": "UMAP of whole-sequence mean ESM2 embeddings",
        "out_prefix": "umap_whole_embedding",
    },
    "precs_embedding": {
        "x_path": Path("results/embeddings/X_precs_embedding_subset499.npy"),
        "meta_path": Path("results/embeddings/precs_embedding_subset499_metadata.csv"),
        "title": "UMAP of pre-CS mean ESM2 embeddings",
        "out_prefix": "umap_precs_embedding",
    },
    "original_cpp": {
        "x_path": Path("results/cpp_features/binary_feature/X_original_cpp_subset499.npy"),
        "meta_path": Path("results/cpp_features/binary_feature/metadata.csv"),
        "title": "UMAP of original CPP features",
        "out_prefix": "umap_original_cpp",
    },
    "embedding_cpp": {
        "x_path": Path("results/embedding_cpp/binary_feature/X_embedding_cpp_subset499.npy"),
        "meta_path": Path("results/embedding_cpp/binary_feature/metadata.csv"),
        "title": "UMAP of embedding-based CPP features",
        "out_prefix": "umap_embedding_cpp",
    },
}


def binary_name(label: int) -> str:
    return "SP" if int(label) == 1 else "NO_SP"


def save_scatter(
    meta: pd.DataFrame,
    group_col: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))

    for label, sub in meta.groupby(group_col):
        plt.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            label=str(label),
            alpha=0.75,
            s=30,
        )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved figure:", out_path)


def plot_umap(name: str, config: dict) -> None:
    X = np.load(config["x_path"])
    meta = pd.read_csv(config["meta_path"])

    print(f"\n{name}")
    print("X:", X.shape)
    print(meta["sp_type"].value_counts())

    X_scaled = StandardScaler().fit_transform(X)

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
    meta["binary_label"] = meta["label_binary"].map(binary_name)

    # 1. sp_type plot
    save_scatter(
        meta=meta,
        group_col="sp_type",
        title=f"{config['title']} by SP type",
        out_path=MULTI_DIR / f"{config['out_prefix']}.png",
    )

    # 2. binary plot
    save_scatter(
        meta=meta,
        group_col="binary_label",
        title=f"{config['title']} by binary label",
        out_path=BINARY_DIR / f"{config['out_prefix']}.png",
    )

def main() -> None:
    for name, config in DATASETS.items():
        plot_umap(name, config)


if __name__ == "__main__":
    main()