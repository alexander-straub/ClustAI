"""
kfold_cv_clustai.py

K-Fold Cross-Validation for ClustAI using Adjusted Rand Index (ARI).

Author: Alexander Straub
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from ClustAI import (
    build_reference_indices_hdbscan,
    assign_clusters_hdbscan
)
from get_labeled_data import get_labeled_data_with_comb_answer


def run_kfold_cv(
    labeled_data:               list[tuple[str, str, str]],
    model:                      SentenceTransformer,
    n_splits:                   int   = 5,
    similarity_threshold:       float = 0.15,
    hdbscan_min_cluster_size:   int =  2,
    hdbscan_min_samples:        int = 1,
    alpha:                      float = 0.5,    
    random_state:               int   = 42,
) -> pd.DataFrame:
    """
    Stratified K-Fold CV over labeled_data.
    Each fold:
      - Train split  -> build_reference_indices_hdbscan()
      - Test split   -> assign_clusters_hdbscan()
      - Score        -> ARI(true_labels, predicted_labels)

    Input:
        - labeled_data: (list[tuple[str, str, str]])
            Labeled data
        - model: (SentenceTransformer)
            Pretrained sentence embedding model
        - n_splits: (int)
            Number of splits for the k-fold CV
        - similarity_threshold: (float)
            Distance threshold for assigning existing cluster with NN or creating new cluster with HDBSCAN
        - hdbscan_min_cluster_size: (int)
            Minimum cluster size to create new cluster
        - hdbscan_min_samples: (int)
            Minimum samples for each cluster
        - alpha: (float)
            Weighting parameter of combined_answers
        - random_state: (int)
            Random seed for reproducability

    Output:
        pd.Dataframe with per-fold ARI scores and a summary row
    """

    # 1. Unpack labeled data
    clusters = [c for c, _, __ in labeled_data] # true cluster labels
    events   = [e for _, e, __ in labeled_data] # true notes
    answers  = [a for _, __, a in labeled_data] # true combined answers

    # Encode cluster strings -> integers (required by StratifiedKFold)
    le = LabelEncoder()
    clusters_enc = le.fit_transform(clusters)

    # 2. Stratified K-Fold split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    ari_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(events, clusters_enc)):
        print(f"\n── Fold {fold_idx + 1}/{n_splits} ──────────────────────────")

        # Build train / test splits
        train_data = [(clusters[i], events[i], answers[i]) for i in train_idx]
        test_events  = [events[i]   for i in test_idx]
        test_answers = [answers[i] for i in test_idx]
        test_labels  = [clusters[i] for i in test_idx]   # ground truth

        print(f"  Train size: {len(train_data)} | Test size: {len(test_events)}")

        # 3. Build reference index from train fold
        ref_indices = build_reference_indices_hdbscan(train_data, model, alpha=alpha)

        # 4. Predict clusters for test fold
        result_df = assign_clusters_hdbscan(
            new_events               = test_events,
            new_answers              = test_answers,
            reference_indices        = ref_indices,
            model                    = model,
            alpha                    = alpha,
            similarity_threshold     = similarity_threshold,
            hdbscan_min_cluster_size = hdbscan_min_cluster_size,
            hdbscan_min_samples      = hdbscan_min_samples
        )

        predicted_labels = result_df["assigned_cluster"].tolist()

        # 5. Compute ARI
        # ARI needs integer-encoded labels; encode both arrays together
        # so unseen predicted cluster names (e.g. 'KL_new_0') get their own int
        all_labels   = test_labels + predicted_labels
        combined_enc = LabelEncoder().fit_transform(all_labels)
        true_enc     = combined_enc[:len(test_labels)]
        pred_enc     = combined_enc[len(test_labels):]

        # Compute ARI
        ari = adjusted_rand_score(true_enc, pred_enc)
        ari_scores.append(ari)

        print(f"ARI Score: {ari:.4f}")


    # 6. Mean ARI
    mean_ari = np.mean(ari_scores)

    return mean_ari, ari_scores


if __name__ == "__main__":

    MODEL_PATH   = r"C:\Users\q645302.MCBMW1\Documents\ML_models\all-MiniLM-L6-v2"
    ROOT_FOLDER  = r"C:\Users\q645302.MCBMW1\BMW Group\ADAS EL-5-US-1 - General\21_Tools\ATP_Tool\external_files\Clustering_Database"
    n_splits = 5

    print("Loading labeled data …")
    labeled_data = get_labeled_data_with_comb_answer(ROOT_FOLDER)
    print(f"  {len(labeled_data)} labeled events loaded")

    print("Loading model …")
    model = SentenceTransformer(MODEL_PATH)

    mean_ari, ari_scores = run_kfold_cv(
        labeled_data             = labeled_data,
        model                    = model,
        n_splits                 = n_splits,
        similarity_threshold     = 0.15,
        hdbscan_min_cluster_size = 2,
        hdbscan_min_samples      = 1,  
        alpha                    = 1,
    )

    print("\n══ K-Fold CV individual Results ══════════════════════════════")
    for i, score in enumerate(ari_scores, start=1):
        print(f"Run {i}: {score:.4f}")

    print("\n════════════════════════════════════")
    print(f"Mean ARI across {n_splits} folds: {mean_ari:.4f}")
    print("════════════════════════════════════")