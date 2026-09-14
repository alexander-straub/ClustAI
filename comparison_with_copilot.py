"""
comparison_with_copilot.py

Script to compare performance of ClustAI with Microsoft Copilot.
Note: Performance of an old version of ClustAI is analyzed here.
Still better performance than Microsoft Copilot.

Author: Alexander Straub
"""

import pandas as pd
from sklearn.metrics import adjusted_rand_score
 
# ── 1. Load files ──────────────────────────────────────────────────────────────
ground_truth    = pd.read_excel(r"C:\Users\q645302.MCBMW1\Desktop\26-11-480_ATS3_NA5_5BB_CA_OR_WA.xlsx", sheet_name='Triggerliste')
my_results      = pd.read_excel(r"C:\Users\q645302.MCBMW1\Desktop\26-11-480_ATS3_NA5_5BB_CA_OR_WA - Copy_alpha_1.0.xlsx", sheet_name='Triggerliste')
# copilot_results = pd.read_excel(r"C:\Users\q645302.MCBMW1\Desktop\Test .xlsx files for ClustAI\copilot_results_raw.xlsx")
 
# ── 2. Column names ────────────────────────────────────────────────────────────
NOTE_COL    = "Note"
CLUSTER_COL = "Cluster"
 
# ── 3. Filter out noisy rows (based on my_results having xyz_noise) ────────────
noise_mask = my_results[CLUSTER_COL].astype(str).str.endswith("_noise")
clean_idx  = my_results.index[~noise_mask]
 
gt_clean      = ground_truth.loc[clean_idx, CLUSTER_COL]
my_clean      = my_results.loc[clean_idx, CLUSTER_COL]
# copilot_clean = copilot_results.loc[clean_idx, CLUSTER_COL]

# ── 4. Drop rows where either file has NaN in the Cluster column ───────────────
combined = pd.DataFrame({"gt": gt_clean, "my": my_clean})

nan_before = combined.isna().any(axis=1).sum()
combined.dropna(inplace=True)

if nan_before > 0:
    print(f"Dropped {nan_before} rows with NaN in Cluster column")

# ── 5. Compute ARI ─────────────────────────────────────────────────────────────
ari_mine    = adjusted_rand_score(combined["gt"], combined["my"])
# ari_copilot = adjusted_rand_score(gt_clean, copilot_clean)
 
# ── 5. Result ──────────────────────────────────────────────────────────────────
total_rows = len(ground_truth)
clean_rows = len(clean_idx)
noise_rows = total_rows - clean_rows - nan_before
 
print(f"Total rows       : {total_rows}")
print(f"Noisy rows removed: {noise_rows}")
print(f"NaN rows dropped  : {nan_before}")
print(f"Rows evaluated   : {clean_rows}")
print()
print(f"ARI — my_results     : {ari_mine:.4f}")
# print(f"ARI — copilot_results: {ari_copilot:.4f}")