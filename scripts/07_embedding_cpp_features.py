"""
Run embedding-based CPP on a SignalP subset.

Pipeline:
1. Load full-sequence residue embeddings.
2. Normalize embeddings to [0, 1] using EmbeddingPreprocessor.encode.
3. Load precomputed embedding pseudo-scales and pseudo-categories.
4. Slice numerical embeddings into TMD and JMD_C parts.
5. Run CPP numerical mode.
6. Save selected features, feature matrix, metadata, and PCA plot.
"""

from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import aaanalysis as aa
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


CSV_PATH = Path("data/processed/subset_500_fullseq.csv")
EMB_PATH = Path("data/embeddings/esm2_65M/subset_499_fullseq_embeddings.pt")

OUT_DIR = Path("results/embedding_cpp")
FIG_DIR = Path("results/figures")

SCALES_PATH = OUT_DIR / "embedding_pseudo_scales.csv"
CAT_PATH = OUT_DIR / "embedding_pseudo_cat.csv"

POST_LEN = 15
TMD_LEN_REF = 24
RANDOM_STATE = 42


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    embeddings = torch.load(path, map_location="cpu")

    return {
        key.split("|")[0]: value.detach().cpu().numpy()
        for key, value in embeddings.items()
    }


def build_df_seq(df: pd.DataFrame) -> pd.DataFrame:
    df_seq = df[["entry", "full_sequence", "cs_position"]].rename(
        columns={"full_sequence": "sequence"}
    ).copy()

    # 1-based indexing expected by AAanalysis.
    # NO_SP already has pseudo cs_position=24 from preprocessing.
    df_seq["tmd_start"] = 1
    df_seq["tmd_stop"] = df_seq["cs_position"].astype(int)

    return df_seq


def plot_pca(X: np.ndarray, meta: pd.DataFrame, out_path: Path) -> None:
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    meta = meta.copy()
    meta["PC1"] = X_pca[:, 0]
    meta["PC2"] = X_pca[:, 1]

    print("Explained variance:", pca.explained_variance_ratio_, flush=True)

    plt.figure(figsize=(8, 6))

    for sp_type, sub in meta.groupby("sp_type"):
        plt.scatter(sub["PC1"], sub["PC2"], label=sp_type, alpha=0.75, s=30)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.title("PCA of embedding-based CPP features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved PCA plot:", out_path, flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    aa.options["verbose"] = False
    aa.options["random_state"] = RANDOM_STATE

    df = pd.read_csv(CSV_PATH)
    embeddings = load_embeddings(EMB_PATH)

    df = df[df["entry"].isin(embeddings.keys())].copy().reset_index(drop=True)

    print("Samples:", df.shape, flush=True)
    print(df["sp_type"].value_counts(), flush=True)

    df_seq = build_df_seq(df)

    ep = aa.EmbeddingPreprocessor(verbose=True)

    print("Encoding residue embeddings...", flush=True)
    dict_num = ep.encode(
        df_seq=df_seq,
        embeddings=embeddings,
        method="minmax",
    )
    print("Finished encoding.", flush=True)

    print("Loading pseudo-scales and pseudo-categories...", flush=True)
    df_scales = pd.read_csv(SCALES_PATH, index_col=0)
    df_cat = pd.read_csv(CAT_PATH)

    print("Pseudo-scales:", df_scales.shape, flush=True)
    print("Pseudo-categories:", df_cat.shape, flush=True)

    print("Slicing numerical embeddings into CPP parts...", flush=True)
    nf = aa.NumericalFeature()

    df_parts, dict_num_parts = nf.get_parts(
        df_seq=df_seq,
        dict_num=dict_num,
        list_parts=["tmd", "jmd_c"],
        jmd_n_len=0,
        jmd_c_len=POST_LEN,
        tmd_len=None,
    )

    print("df_parts shape:", df_parts.shape, flush=True)
    print(df_parts.head(), flush=True)

    y = df["label_binary"].astype(int).values

    split_kws = {
        "Segment": {
            "n_split_min": 1,
            "n_split_max": 5,
        },
    }

    print("Initializing CPP numerical mode...", flush=True)
    cpp = aa.CPP(
        df_scales=df_scales,
        df_cat=df_cat,
        df_parts=df_parts,
        split_kws=split_kws,
        accept_gaps=True,
    )

    print("Running CPP.run_num...", flush=True)
    df_feat = cpp.run_num(
        dict_num_parts=dict_num_parts,
        labels=y,
        tmd_len=TMD_LEN_REF,
        jmd_n_len=0,
        jmd_c_len=POST_LEN,
        max_cor=0.5,
        n_jobs=1,
        check_cat=True,
    )

    print("Selected embedding-CPP features:", df_feat.shape, flush=True)
    print(df_feat.head(), flush=True)

    feat_path = OUT_DIR / "embedding_cpp_selected_features.csv"
    df_feat.to_csv(feat_path, index=False)
    print("Saved selected features:", feat_path, flush=True)

    print("Creating embedding-CPP feature matrix...", flush=True)
    
    nf = aa.NumericalFeature()

    X_embed_cpp = nf.feature_matrix(
        features=df_feat["feature"],
        dict_num_parts=dict_num_parts,
        df_parts=df_parts,
        df_scales=df_scales,
        n_jobs=1,
    )
    print("Embedding-CPP matrix:", X_embed_cpp.shape, flush=True)

    x_path = OUT_DIR / "X_embedding_cpp_subset499.npy"
    np.save(x_path, X_embed_cpp)
    print("Saved feature matrix:", x_path, flush=True)

    meta = df[["entry", "sp_type", "label_binary", "kingdom"]].copy()

    meta_path = OUT_DIR / "embedding_cpp_subset499_metadata.csv"
    meta.to_csv(meta_path, index=False)
    print("Saved metadata:", meta_path, flush=True)

    pca_path = FIG_DIR / "pca_embedding_cpp_sp_type.png"
    plot_pca(X_embed_cpp, meta, pca_path)


if __name__ == "__main__":
    main()