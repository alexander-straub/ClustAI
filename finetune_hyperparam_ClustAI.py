"""
finetune_hyperparam_ClustAI.py

Script to finetune the hyperparameter 'similarity_threshold'.
Used for the distance in order to cluster events.

Author: Alexander Straub
"""


import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from itertools import combinations
from sklearn.metrics import adjusted_rand_score
from ClustAI_only_note import build_reference_indices_hdbscan, assign_clusters_hdbscan
from get_labeled_data import get_labeled_data
from helper_functions import extract_events_and_clusters_from_single_xlsx


def evaluate_threshold(
        labeled_data:      list[tuple[str, str]],
        model:             SentenceTransformer,
        thresholds:        list[float]
    ) -> pd.DataFrame:
    """
    Function to evaluate the optimal threshold based on ARI.

    Input:
        - labeled_data: (list[tuple[str, str]])
            List of labeled data
        - model: (SentenceTransformer)
            Pretrained sentence embedding model
        - thresholds: (list(float))
            List of floats to compare

    Output:
        pd.Dataframe with results
    """

    xlsx_file = r"C:\Users\q645302.MCBMW1\Desktop\26-07-500_ATS2_NA05_V142804_5BB_SD_PHX_FINAL_threshold_ana.xlsx"
    new_events, ref_clusters = extract_events_and_clusters_from_single_xlsx(xlsx_file)

    print("Building reference index from training split …")
    ref_indices = build_reference_indices_hdbscan(labeled_data, model)

    records = []

    for threshold in thresholds:
        result_df = assign_clusters_hdbscan(
            new_events            = new_events,
            reference_indices     = ref_indices,
            model                    = model,
            similarity_threshold     = threshold,
            hdbscan_min_cluster_size = 2,
            hdbscan_min_samples      = 1
    )

        # -- Filter out noise events before evaluation -------------------------
        noise_mask     = result_df["assigned_cluster"].str.endswith("_noise")
        clean_df       = result_df[~noise_mask]
        clean_true     = [ref_clusters[i] for i in clean_df.index]
        clean_assigned = clean_df["assigned_cluster"].tolist()
        ari = adjusted_rand_score(clean_true, clean_assigned) if len(clean_true) > 1 else float("nan")

        n_noise    = int(noise_mask.sum())
        noise_rate = float(noise_mask.mean())

        # -- Pair-based TP/TN/FP/FN on non-noise events -----------------------
        TP = TN = FP = FN = 0
        for i, j in combinations(range(len(clean_true)), 2):
            same_in_gt   = clean_true[i]     == clean_true[j]
            same_in_pred = clean_assigned[i]  == clean_assigned[j]

            if same_in_gt and same_in_pred:         TP += 1
            elif not same_in_gt and not same_in_pred: TN += 1
            elif not same_in_gt and same_in_pred:   FP += 1
            elif same_in_gt and not same_in_pred:   FN += 1

        precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
        recall    = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else float("nan")
        rand_index = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else float("nan")

        # -- Coverage: fraction of non-noise events matched to a known cluster -
        matched_mask = clean_df["nn_distance"] <= threshold
        coverage  = float(matched_mask.mean()) if len(clean_df) > 0 else float("nan")

        records.append({
            "threshold":   threshold,
            "ari":         round(ari, 4) if not np.isnan(ari) else None,
            "recall":      round(recall,     4) if not np.isnan(recall)     else None,
            "precision":   round(precision,  4) if not np.isnan(precision)  else None,
            "f1":          round(f1,         4) if not np.isnan(f1)         else None,
            "rand_index":  round(rand_index, 4) if not np.isnan(rand_index) else None,
            "coverage":    round(coverage,   4) if not np.isnan(coverage)   else None,
            "noise_rate":  round(noise_rate, 4),
            "n_matched":   int(matched_mask.sum()),
            "n_unmatched": int((~matched_mask).sum()),
            "n_noise":     n_noise,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        })

        print(
            f"  threshold={threshold:.2f} | "
            f"ARI={ari:.3f} | "
            f"recall={recall:.3f} | "
            f"precision={precision:.3f} | "
            f"F1={f1:.3f} | "
            f"coverage={coverage:.1%} | "
            f"noise_rate={noise_rate:.1%} ({n_noise} events)"
        )

    # Sort by recall descending so the best threshold is immediately visible
    return pd.DataFrame(records).sort_values("recall", ascending=False)


if __name__ == "__main__":
    # -------------- Adjustables ------------------------------
    model_path   = r"C:\Users\q645302.MCBMW1\Documents\ML_models\all-MiniLM-L6-v2"
    root_folder  = r"C:\Users\q645302.MCBMW1\BMW Group\ADAS EL-5-US-1 - General\21_Tools\ATP_Tool\external_files\Clustering_Database"
    # -----------------------------------------------------

    print("Loading data …")
    labeled_data = get_labeled_data(root_folder)

    print("Loading model …")
    model = SentenceTransformer(model_path)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    print("Evaluating thresholds …")
    results_df = evaluate_threshold(
            labeled_data = labeled_data,
            model        = model,
            thresholds   = thresholds,
        )

    print("\n── Summary ──")
    print(results_df.to_string(index=False))
    # --------- Save results in output file ----------------
    # param_results_path = r"C:\Users\q645302.MCBMW1\Desktop\Test .xlsx files for ClustAI\threshold_evaluation.csv"
    # if os.path.exists(param_results_path):
    #     os.remove(param_results_path)
    # results_df.to_csv(param_results_path, index=False)
    # print("Saved -> threshold_evaluation.csv")
    # ------------------------------------------------------