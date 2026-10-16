"""
compute_ari.py

Single-run Adjusted Rand Index (ARI) evaluation for ClustAI.
Compares a ground-truth clustering (.xlsx) against a predicted
clustering (.xlsx) produced by the pipeline -- no cross-validation,
just one straight comparison.

Author: Alexander Straub
"""

import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

# ── CONFIG ──────────────────────
GT_PATH           = r"filepath_to_groundt_truth_excel"
PRED_PATH         = r"filepath_to_predicted_excel"

ID_COL            = "Note"        # column used to align rows between files
GT_CLUSTER_COL    = "Cluster"        # ground-truth cluster label column
PRED_CLUSTER_COL  = "Cluster"  # predicted cluster label column
SHEET_NAME        = "Triggerliste"
# ─────────────────────────────────────────────────────────────────────────


def run_ari_eval(
    gt_path:          str,
    pred_path:        str,
    id_col:            str = ID_COL,
    gt_cluster_col:    str = GT_CLUSTER_COL,
    pred_cluster_col:  str = PRED_CLUSTER_COL,
) -> float:
    """
    Load ground truth and predicted clusterings, align rows on `id_col`,
    and compute the Adjusted Rand Index between them.

    Returns the ARI score.
    """

    # 1. Load both files
    gt_df   = pd.read_excel(gt_path, sheet_name=SHEET_NAME)[[id_col, gt_cluster_col]]
    pred_df = pd.read_excel(pred_path, sheet_name=SHEET_NAME)[[id_col, pred_cluster_col]]

    gt_df   = gt_df.rename(columns={gt_cluster_col: "gt_cluster"})
    pred_df = pred_df.rename(columns={pred_cluster_col: "pred_cluster"})

    # 2. Align rows on shared ID
    merged = gt_df.merge(pred_df, on=id_col, how="inner")

    if len(merged) < len(gt_df) or len(merged) < len(pred_df):
        print(
            f"Warning: only {len(merged)} rows matched by '{id_col}' "
            f"(ground truth had {len(gt_df)}, predicted had {len(pred_df)}). "
            "Unmatched rows are excluded from the ARI calculation."
        )

    if len(merged) == 0:
        raise ValueError(
            f"No rows matched between the two files on '{id_col}'. "
            "Check that this column contains the same identifiers in both files."
        )

    true_labels = merged["gt_cluster"].astype(str).tolist()
    pred_labels = merged["pred_cluster"].astype(str).tolist()

    # 3. Encode both label sets together
    all_labels   = true_labels + pred_labels
    combined_enc = LabelEncoder().fit_transform(all_labels)
    true_enc     = combined_enc[: len(true_labels)]
    pred_enc     = combined_enc[len(true_labels) :]

    # 4. Compute ARI
    ari = adjusted_rand_score(true_enc, pred_enc)

    print(f"Rows compared: {len(merged)}")
    print(f"ARI Score: {ari:.4f}")

    return ari


if __name__ == "__main__":
    run_ari_eval(GT_PATH, PRED_PATH)
