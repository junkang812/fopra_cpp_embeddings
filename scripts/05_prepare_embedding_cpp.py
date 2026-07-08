"""
Prepare pseudo-scales for embedding-based CPP.

Pipeline:
1. Load full-sequence residue embeddings.
2. Normalize embeddings to [0, 1] using EmbeddingPreprocessor.encode.
3. Build embedding pseudo-scales.
4. Save embedding_pseudo_scales.csv.

The pseudo-categories are generated separately in:
    07_build_embedding_cat.py
"""

from pathlib import Path

import torch
import numpy as np
import pandas as pd
import aaanalysis as aa


CSV_PATH = Path("data/processed/subset_500_fullseq.csv")
EMB_PATH = Path("data/embeddings/esm2_65M/subset_499_fullseq_embeddings.pt")

OUT_DIR = Path("results/embedding_cpp")
SCALES_PATH = OUT_DIR / "embedding_pseudo_scales.csv"


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    aa.options["verbose"] = False

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

    print("Building embedding pseudo-scales...", flush=True)
    df_scales = ep.build_scales(
        df_seq=df_seq,
        dict_num=dict_num,
    )

    print("Pseudo-scales:", df_scales.shape, flush=True)
    df_scales.to_csv(SCALES_PATH)
    print("Saved pseudo-scales:", SCALES_PATH, flush=True)


if __name__ == "__main__":
    main()