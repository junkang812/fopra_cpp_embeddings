from pathlib import Path
import torch
import pandas as pd
import numpy as np

CSV_PATH = Path("data/processed/subset_500_fullseq.csv")
EMB_PATH = Path("data/embeddings/esm2_65M/subset_499_fullseq_embeddings.pt")

OUT_DIR = Path("results/embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

embeddings = torch.load(EMB_PATH, map_location="cpu")
embeddings = {k.split("|")[0]: v for k, v in embeddings.items()}

rows = []
X_precs = []
X_whole = []

for _, row in df.iterrows():
    entry = row["entry"]

    if entry not in embeddings:
        continue

    emb = embeddings[entry]
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()

    cs = int(row["cs_position"])

    # pre-CS mean pooling
    pre_emb = emb[:cs].mean(axis=0)

    # whole-sequence mean pooling
    whole_emb = emb.mean(axis=0)

    X_precs.append(pre_emb)
    X_whole.append(whole_emb)
    rows.append(row)

meta = pd.DataFrame(rows).reset_index(drop=True)
X_precs = np.vstack(X_precs)
X_whole = np.vstack(X_whole)

print("Matched samples:", len(meta))
print("Pre-CS embedding matrix:", X_precs.shape)
print("Whole-sequence embedding matrix:", X_whole.shape)
print(meta["sp_type"].value_counts())

np.save(OUT_DIR / "X_precs_embedding_subset499.npy", X_precs)
np.save(OUT_DIR / "X_whole_embedding_subset499.npy", X_whole)

meta.to_csv(OUT_DIR / "precs_embedding_subset499_metadata.csv", index=False)
meta.to_csv(OUT_DIR / "whole_embedding_subset499_metadata.csv", index=False)

print("Saved:", OUT_DIR / "X_precs_embedding_subset499.npy")
print("Saved:", OUT_DIR / "X_whole_embedding_subset499.npy")
print("Saved metadata files")