#!/usr/bin/env python3

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

OUT_FEATURE_DIR = Path("results/cpp_features/binary_feature")

POST_CS_LEN = 15
MAX_COR = 0.5
RANDOM_STATE = 42


def main() -> None:
    OUT_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    aa.options["verbose"] = False
    aa.options["random_state"] = RANDOM_STATE

    df = pd.read_csv(CSV_PATH)

    # Keep exactly proteins for which embeddings exist
    embeddings = torch.load(EMB_PATH, map_location="cpu")
    embedding_entries = {k.split("|")[0] for k in embeddings.keys()}

    df = df[df["entry"].isin(embedding_entries)].copy().reset_index(drop=True)

    print("Samples:", df.shape)
    print(df["sp_type"].value_counts())

    # Use all AA scales for now to avoid AAclust multiprocessing issue on macOS.
    df_scales = aa.load_scales()
    print("Loaded scales:", df_scales.shape)

    split_kws = {
        "Segment": {
            "n_split_min": 1,
            "n_split_max": 5,
        },
        "Pattern": {
            "steps": [3, 4],
            "n_min": 1,
            "n_max": 2,
            "len_max": 5,
        },
        "PeriodicPattern": {
            "steps": [3, 4],
        },
    }

    df_parts = (
        df.rename(columns={"pre_cs": "tmd", "post_cs": "jmd_c"})
        [["tmd", "jmd_c"]]
        .copy()
    )

    y = df["label_binary"].astype(int).values

    cpp = aa.CPP(
        df_scales=df_scales,
        df_parts=df_parts,
        split_kws=split_kws,
        accept_gaps=True,
    )

    df_feat = cpp.run(
        labels=y,
        jmd_n_len=0,
        jmd_c_len=POST_CS_LEN,
        max_cor=MAX_COR,
    )

    print("Selected CPP features:", df_feat.shape)

    feature_path = OUT_FEATURE_DIR / "selected_features.csv"
    df_feat.to_csv(feature_path, index=False)
    print("Saved:", feature_path)

    sf = aa.SequenceFeature()

    X_cpp = sf.feature_matrix(
        df_parts=df_parts,
        features=df_feat["feature"],
        accept_gaps=True,
    )

    print("CPP matrix:", X_cpp.shape)

    np.save(OUT_FEATURE_DIR / "X_original_cpp_subset499.npy", X_cpp)

    meta_path = OUT_FEATURE_DIR / "metadata.csv"
    df[["entry", "sp_type", "label_binary", "kingdom"]].to_csv(meta_path, index=False)
    print("Saved:", meta_path)

if __name__ == "__main__":
    main()