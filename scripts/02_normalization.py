"""
Step 2: Normalization (delta-Ct method)
---------------------------------------
Computes delta-Ct values using reference miRNAs (miR-103, miR-191),
then builds the machine-learning table (samples x miRNAs)
with a binary Target column (cancer = 1, healthy = 0).

Input : results/mirna_agg.csv
Output: results/data_ml.csv
"""

import numpy as np
import pandas as pd

# ===================== CONFIG =====================
REFERENCE_MIRNAS = ["hsa-miR-103", "hsa-miR-191"]
# ==================================================

# 1) Load aggregated miRNA data
mirna_agg = pd.read_csv(
    "results/mirna_agg.csv", sep="\t", index_col=0
)

# 2) Check that reference miRNAs exist
for mirna in REFERENCE_MIRNAS:
    print(f"[INFO] Reference {mirna} present: {mirna in mirna_agg.index}")

# 3) Compute reference Ct = mean of the two reference miRNAs
reference_ct = mirna_agg.loc[REFERENCE_MIRNAS].mean(axis=0)

# 4) delta-Ct for all miRNAs (subtract reference per sample)
delta_ct = mirna_agg.subtract(reference_ct, axis=1)

# 5) Remove the reference miRNAs from the features
features_ct = delta_ct.drop(index=REFERENCE_MIRNAS, errors="ignore")
print(f"[INFO] Feature miRNA count: {features_ct.shape[0]}")

# 6) Transpose -> samples are rows, miRNAs are columns
data_ml = features_ct.T
print(f"[INFO] ML table shape (samples x features): {data_ml.shape}")

# 7) Build Target: samples starting with 's' are cancer
#    (adjust this rule to match YOUR sample naming!)
data_ml["Target"] = (
    data_ml.index.to_series().str.startswith("s").astype(int)
)

# 8) Inspect class distribution
print("\n[INFO] Class distribution:")
print(data_ml["Target"].value_counts())

print("\n[INFO] Healthy (Target=0):",
      data_ml[data_ml["Target"] == 0].index.tolist())
print("[INFO] Cancer  (Target=1):",
      data_ml[data_ml["Target"] == 1].index.tolist())

# 9) Save
data_ml.to_csv("results/data_ml.csv", sep="\t")
print("\n[INFO] Saved to results/data_ml.csv")
