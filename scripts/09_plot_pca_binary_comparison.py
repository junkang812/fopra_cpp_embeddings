"""
Plot PCA comparison for four representations.

For each representation, two plots are saved:
1. Colored by sp_type
2. Colored by binary label: NO_SP vs SP

For CPP and embedding-CPP, these plots currently use binary-selected features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FIG_DIR = Path("results/figures")
BINARY_DIR = FIG_DIR / "binary_feature" / "binary"
MULTICLASS_DIR = FIG_DIR / "binary_feature" / "multiclass"

BINARY_DIR.mkdir(parents=True, exist_ok=True)
MULTICLASS_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "whole_embedding": {
        "x_path": Path("results/embeddings/X_whole_embedding_subset499.npy"),
        "meta_path": Path("results/embeddings/whole_embedding_subset499_metadata.csv"),
        "title": "PCA of whole-sequence mean ESM2 embeddings",
        "out_prefix": "pca_whole_embedding",
    },
    "precs_embedding": {
        "x_path": Path("results/embeddings/X_precs_embedding_subset499.npy"),
        "meta_path": Path("results/embeddings/precs_embedding_subset499_metadata.csv"),
        "title": "PCA of pre-CS mean ESM2 embeddings",
        "out_prefix": "pca_precs_embedding",
    },
    "original_cpp": {
        "x_path": Path("results/cpp_features/binary_feature/X_original_cpp_subset499.npy"),
        "meta_path": Path("results/cpp_features/binary_feature/metadata.csv"),
        "title": "PCA of original CPP features",
        "out_prefix": "pca_original_cpp",
    },
    "embedding_cpp": {
        "x_path": Path("results/embedding_cpp/binary_feature/X_embedding_cpp_subset499.npy"),
        "meta_path": Path("results/embedding_cpp/binary_feature/metadata.csv"),
        "title": "PCA of embedding-based CPP features",
        "out_prefix": "pca_embedding_cpp",
    },
}

def binary_name(label: int) -> str:
    return "SP" if int(label) == 1 else "NO_SP"


def save_scatter(
    meta: pd.DataFrame,
    group_col: str,
    title: str,
    out_path: Path,
    x_col: str = "PC1",
    y_col: str = "PC2",
) -> None:
    plt.figure(figsize=(8, 6))

    for label, sub in meta.groupby(group_col):
        plt.scatter(
            sub[x_col],
            sub[y_col],
            label=str(label),
            alpha=0.75,
            s=30,
        )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved figure:", out_path)


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
    meta["binary_label"] = meta["label_binary"].map(binary_name)

    print("Explained variance:", pca.explained_variance_ratio_)

    pc_title = (
        f"PC1 {pca.explained_variance_ratio_[0] * 100:.1f}%, "
        f"PC2 {pca.explained_variance_ratio_[1] * 100:.1f}%"
    )

    save_scatter(
        meta=meta,
        group_col="sp_type",
        title=f"{config['title']} by SP type\n{pc_title}",
        out_path=MULTICLASS_DIR / f"{config['out_prefix']}.png",
    )

    save_scatter(
        meta=meta,
        group_col="binary_label",
        title=f"{config['title']} by binary label\n{pc_title}",
        out_path=BINARY_DIR / f"{config['out_prefix']}.png",
    )


def main() -> None:
    for name, config in DATASETS.items():
        plot_pca(name, config)


if __name__ == "__main__":
    main()