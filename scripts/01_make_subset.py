import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/dataset_parsed.csv")
OUTPUT = Path("data/processed/subset_500.csv")

df = pd.read_csv(INPUT)

sample_plan = {
    "NO_SP": 250,
    "SP": 120,
    "LIPO": 70,
    "TAT": 35,
    "PILIN": 15,
    "TATLIPO": 10,
}

parts = []
for sp_type, n in sample_plan.items():
    sub = df[df["sp_type"] == sp_type]
    n_take = min(n, len(sub))
    parts.append(sub.sample(n=n_take, random_state=42))

subset = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
subset.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("Shape:", subset.shape)
print(subset["sp_type"].value_counts())
print(subset["label_binary"].value_counts())
