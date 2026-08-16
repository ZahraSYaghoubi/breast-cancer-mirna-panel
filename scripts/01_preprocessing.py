"""
Step 1: Preprocessing
---------------------
Reads the raw GEO file, identifies the real header row,
keeps only human (hsa-) miRNAs, and aggregates duplicated IDs.

Input : data/GSE41922_non-normalized.txt
Output: mirna_agg DataFrame (saved to results/mirna_agg.csv)
"""

import pandas as pd

# ===================== CONFIG =====================
FILE_PATH = "data/GSE41922_non-normalized.txt"  # relative path
# ==================================================

# 1) Read all lines to find the real header row
with open(FILE_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

# The header line is the one starting with "ID_REF"
header_line = next(
    i for i, line in enumerate(lines)
    if line.strip().startswith("ID_REF")
)

print(f"[INFO] Header found at line {header_line}")
print(f"[INFO] Header preview: {lines[header_line][:120]}")

# 2) Read the table starting from the detected header
df = pd.read_csv(FILE_PATH, sep="\t", header=header_line)

print(f"[INFO] Full table shape: {df.shape}")

# 3) Keep only human miRNAs (IDs starting with 'hsa-')
mirna_df = df[df["ID_REF"].str.startswith("hsa-", na=False)].copy()

print(f"[INFO] After keeping hsa-miRNAs: {mirna_df.shape}")

# 4) Check for duplicated miRNA IDs
n_duplicates = mirna_df["ID_REF"].duplicated().sum()
print(f"[INFO] Duplicated miRNA IDs: {n_duplicates}")

# 5) Average the rows of each duplicated miRNA
mirna_agg = mirna_df.groupby("ID_REF", as_index=True).mean(numeric_only=True)

print(f"[INFO] After aggregation: {mirna_agg.shape}")
print(f"[INFO] Duplicates after aggregation: "
      f"{mirna_agg.index.duplicated().sum()}")

# 6) Save to CSV for the next step
mirna_agg.to_csv("results/mirna_agg.csv", sep="\t")
print("[INFO] Saved to results/mirna_agg.csv")
