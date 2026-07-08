"""
Build pseudo-categories for embedding dimensions.
"""

from pathlib import Path
import pandas as pd
import aaanalysis as aa

SCALES_PATH = Path("results/embedding_cpp/embedding_pseudo_scales.csv")
CAT_PATH = Path("results/embedding_cpp/embedding_pseudo_cat.csv")


def main() -> None:
    df_scales = pd.read_csv(SCALES_PATH, index_col=0)
    print("Loaded df_scales:", df_scales.shape)

    ep = aa.EmbeddingPreprocessor(verbose=True)

    df_cat = ep.build_cat(
        df_scales=df_scales,
        cat_min_th=0.5,
        subcat_min_th=0.7,
        metric="correlation",
        random_state=42,
    )

    print("Built df_cat:", df_cat.shape)
    print(df_cat.head())

    CAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_cat.to_csv(CAT_PATH, index=False)
    print("Saved:", CAT_PATH)


if __name__ == "__main__":
    main()