import time
from pathlib import Path

import pandas as pd
import requests

INPUT = Path("data/processed/subset_500.csv")
OUTPUT_CSV = Path("data/processed/subset_500_fullseq.csv")
OUTPUT_FASTA = Path("data/processed/subset_500_fullseq.fasta")

df = pd.read_csv(INPUT)

def fetch_uniprot_sequence(entry: str):
    url = f"https://rest.uniprot.org/uniprotkb/{entry}.fasta"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None

    lines = r.text.strip().splitlines()
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    return seq if seq else None

full_sequences = []

for i, entry in enumerate(df["entry"], start=1):
    seq = fetch_uniprot_sequence(entry)
    full_sequences.append(seq)
    print(f"{i}/{len(df)} {entry}: {'OK' if seq else 'FAILED'}", 
          f"len={len(seq) if seq else 'NA'}")
    time.sleep(0.2)

df["full_sequence"] = full_sequences
df.to_csv(OUTPUT_CSV, index=False)

with OUTPUT_FASTA.open("w") as f:
    for _, row in df.dropna(subset=["full_sequence"]).iterrows():
        f.write(f">{row['entry']}|{row['sp_type']}|cs={row['cs_position']}\n")
        f.write(f"{row['full_sequence']}\n")

print("Saved:", OUTPUT_CSV)
print("Saved:", OUTPUT_FASTA)
print("Missing:", df["full_sequence"].isna().sum())
