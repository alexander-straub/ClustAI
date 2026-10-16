"""
get_result.py

Script to evaluate clustering performance.
This script checks pairwise (row by row) if the ground truth (gt) cluster
is the same as the cluster i have generated regarding a specific note.

Author: Alexander Straub
"""

import re
from itertools import combinations
import pandas as pd

def compute_pairwise_metrics(notes, gt_clusters, pred_clusters, label=""):
    """
    Compute pairwise clustering accuracy, precision, recall, F1 and rand index
    for a given set of notes/gt/pred cluster lists.

    Input:
        - notes: (list of str)
        - gt_clusters: (list of str)
        - pred_clusters: (list of str)
        - label: (str)
            Optional label used in the printed output (e.g. "ALL", "EXISTING", "NEW")

    Output:
        - metrics: (dict)
            Dictionary containing accuracy, precision, recall, f1, rand_index
    """
    total = 0
    correct = 0
    TP = TN = FP = FN = 0

    for i, j in combinations(range(len(notes)), 2):
        same_in_gt = gt_clusters[i] == gt_clusters[j]
        same_in_pred = pred_clusters[i] == pred_clusters[j]

        if same_in_gt == same_in_pred:
            correct += 1
        total += 1

        if same_in_gt and same_in_pred:
            TP += 1
        elif not same_in_gt and not same_in_pred:
            TN += 1
        elif not same_in_gt and same_in_pred:
            FP += 1
        elif same_in_gt and not same_in_pred:
            FN += 1

    print(f"\n===== Results: {label} =====")
    print(f"Notes compared: {len(notes)}")

    if total == 0:
        print("Not enough notes to compare (need at least 2). Skipping.")
        return None

    accuracy = correct / total * 100
    print(f"Pairs evaluated: {total}")
    print(f"Correctly grouped pairs: {correct}")
    print(f"Clustering accuracy: {accuracy:.2f}%")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

    precision = TP / (TP + FP) if (TP + FP) > 0 else float('nan')   # Of pairs I grouped together, how many were correct
    recall = TP / (TP + FN) if (TP + FN) > 0 else float('nan')      # Of pairs that should be together, how many did I find
    f1 = (2 * (precision * recall) / (precision + recall)
          if (precision + recall) > 0 else float('nan'))
    rand_index = (TP + TN) / (TP + TN + FP + FN)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Rand Index: {rand_index:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rand_index": rand_index,
    }

def analyze_existing_cluster_errors(df, new_cluster_pattern, label="EXISTING CLUSTERS - ERROR BREAKDOWN"):
    """
    For notes whose ground-truth cluster is an already-existing cluster,
    break down the "wrong" predictions into two sub-cases.
 
    Input:
        - df: (pd.DataFrame)
            Merged dataframe containing 'Note', 'GT_Cluster', 'Pred_Cluster'
        - new_cluster_pattern: (re.Pattern or str)
            Regex used to identify "new" HDBSCAN-generated cluster names.
            Applied to both GT_Cluster (to find the existing-cluster subset)
            and Pred_Cluster (to classify the wrong predictions).
        - label: (str)
            Optional label used in the printed output
 
    Output:
        - result: (dict)
            Counts, percentages, and the misassigned rows themselves
            (for manual review), or None if there's nothing to compare.
    """
    # ── 1. Restrict to notes whose ground truth is an already-existing cluster
    gt_is_new = df['GT_Cluster'].astype(str).str.contains(new_cluster_pattern)
    df_gt_existing = df[~gt_is_new].copy()
 
    n_total = len(df_gt_existing)
 
    print(f"\n===== {label} =====")
    if n_total == 0:
        print("No rows found where the ground truth is an existing cluster. Skipping.")
        return None
 
    # ── 2. Split into correct vs wrong ────────────────────────────────────
    correct_mask = df_gt_existing['Pred_Cluster'] == df_gt_existing['GT_Cluster']
    df_wrong = df_gt_existing[~correct_mask]
 
    n_correct = int(correct_mask.sum())
    n_wrong = len(df_wrong)
 
    print(f"Notes with GT = existing cluster: {n_total}")
    print(f"Correctly assigned:               {n_correct} ({n_correct / n_total * 100:.2f}%)")
    print(f"Wrongly assigned (total):         {n_wrong} ({n_wrong / n_total * 100:.2f}%)")
 
    if n_wrong == 0:
        print("No wrong assignments found.")
        return {
            "n_total": n_total,
            "n_correct": n_correct,
            "n_wrong": 0,
            "n_wrong_existing": 0,
            "n_wrong_new": 0,
        }
 
    # ── 3. Within the wrong ones, split by where they actually landed ─────
    pred_is_new_mask = df_wrong['Pred_Cluster'].astype(str).str.contains(new_cluster_pattern)
 
    df_wrong_existing = df_wrong[~pred_is_new_mask]  # case (a)
    df_wrong_new = df_wrong[pred_is_new_mask]        # case (b)
 
    n_wrong_existing = len(df_wrong_existing)
    n_wrong_new = len(df_wrong_new)
 
    print(f"  -> Wrong existing cluster:       {n_wrong_existing} "
          f"({n_wrong_existing / n_wrong * 100:.2f}% of wrong, "
          f"{n_wrong_existing / n_total * 100:.2f}% of total)")
    print(f"  -> Wrongly sent to new cluster:  {n_wrong_new} "
          f"({n_wrong_new / n_wrong * 100:.2f}% of wrong, "
          f"{n_wrong_new / n_total * 100:.2f}% of total)")
 
    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_wrong_existing": n_wrong_existing,
        "n_wrong_new": n_wrong_new,
        "pct_correct": n_correct / n_total * 100,
        "pct_wrong_existing_of_wrong": n_wrong_existing / n_wrong * 100,
        "pct_wrong_new_of_wrong": n_wrong_new / n_wrong * 100,
        "misassigned_existing_examples": df_wrong_existing[['Note', 'GT_Cluster', 'Pred_Cluster']],
        "misassigned_new_examples": df_wrong_new[['Note', 'GT_Cluster', 'Pred_Cluster']],
    }


def clustering_accuracy(groundtruth_file, result_file, new_cluster_pattern, noise_pattern=r"_noise$"):
    """
    Calculate accuracy of the clustering.
    Compares pair wise (row by row) if both cluster values 
    (from GT and file from ClustAI) are the same.

    Computes metrics for:
        - all rows
        - rows where the predicted cluster is an already-existing cluster
        - rows where the predicted cluster is a newly generated HDBSCAN cluster

    Input:
        - groundtruth_file: (str)
            Ground truth excel file
        - result_file: (str)
            Generated clustered excel file
        - new_cluster_pattern: (str)
            Regex pattern to filter out newly generated clusters

    Output:
        - results: (dict)
            Dictionary with metrics for "all", "existing" and "new" subsets
    """
    # Load files and filter out empty rows
    gt_filtered, res_filtered = get_valid_indices(groundtruth_file, result_file)

    df = pd.DataFrame({
        'Note': gt_filtered['Note'].reset_index(drop=True),
        'GT_Cluster': gt_filtered['Cluster'].reset_index(drop=True),
        'Pred_Cluster': res_filtered['Cluster'].reset_index(drop=True),
    })

    if df.empty:
        raise ValueError("No matching notes found between the two files.")
    
    # ── Exclude notes predicted into a "noise" cluster ────────────────────
    is_noise = df['Pred_Cluster'].astype(str).str.contains(noise_pattern)
    n_noise = int(is_noise.sum())
    if n_noise > 0:
        print(f"\nExcluding {n_noise} notes predicted into a noise cluster "
              f"({n_noise / len(df) * 100:.2f}% of {len(df)} total).")
    df = df[~is_noise].copy()

    # Split rows into "existing" clusters vs "new" (HDBSCAN-generated) clusters
    is_new_cluster = df['Pred_Cluster'].astype(str).str.contains(new_cluster_pattern)
    df_existing = df[~is_new_cluster]
    df_new = df[is_new_cluster]

    results = {}

    # All rows combined
    results["all"] = compute_pairwise_metrics(
        df['Note'].tolist(),
        df['GT_Cluster'].tolist(),
        df['Pred_Cluster'].tolist(),
        label="ALL"
    )

    # Only rows matched to already-existing clusters
    results["existing"] = compute_pairwise_metrics(
        df_existing['Note'].tolist(),
        df_existing['GT_Cluster'].tolist(),
        df_existing['Pred_Cluster'].tolist(),
        label="EXISTING CLUSTERS"
    )

    # Only rows matched to newly generated HDBSCAN clusters
    results["new"] = compute_pairwise_metrics(
        df_new['Note'].tolist(),
        df_new['GT_Cluster'].tolist(),
        df_new['Pred_Cluster'].tolist(),
        label="NEW CLUSTERS (HDBSCAN)"
    )

    # Row-level error breakdown for the existing-cluster population
    error_breakdown = analyze_existing_cluster_errors(df, new_cluster_pattern)

    # Name-agnostic rename-vs-scatter breakdown of the "wrongly sent to
    # new cluster" notes, using df_new as the full context for majority voting
    rename_vs_scatter = None
    if error_breakdown is not None and error_breakdown.get("n_wrong_new", 0) > 0:
        rename_vs_scatter = analyze_new_cluster_rename_vs_scatter(
            df_new, error_breakdown["misassigned_new_examples"]
        )
 
    return results, error_breakdown, rename_vs_scatter


def get_valid_indices(groundtruth_file, result_file):
    """
    Extract the non empty indices of the reference file.
    -> Filter out empty rows.

    Input:
        - groundtruth_file: (str)
            Ground truth excel file
        - result_file: (str)
            Generated clustered excel file

    Output:
        - gt_filtered: (pd dataframe)
            Pandas dataframe of filtered GT excel file
        - res_filtered: (pd dataframe)
            Pandas dataframe of filtered excel file clustered with ClustAI
    """
    gt = pd.read_excel(groundtruth_file, sheet_name='Triggerliste')
    res = pd.read_excel(result_file, sheet_name='Triggerliste')

    # Get indices where both columns are non-empty in file 1
    valid_idx = gt[gt['Cluster'].notna() & gt['Note'].notna()].index.tolist()

    # Apply same indices to both files
    gt_filtered = gt.loc[valid_idx]
    res_filtered = res.loc[valid_idx]

    return gt_filtered, res_filtered

def top_confused_cluster_pairs(df_wrong_existing, top_n=15):
    """
    Count how often each (GT_Cluster, Pred_Cluster) pair occurs among
    wrongly-matched-to-existing-cluster notes, sorted by frequency.

    Input:
        - df_wrong_existing: (pandas df)
            Pandas dataframe of rows with wrong existing cluster
        - top_n: (int)
            number of top cluster pairs which are wrong (for analysis only)

    Output:
        - confusion_counts: (pd dataframe)
            Pandas dataframe of the confusion counts
    """
    confusion_counts = (
        df_wrong_existing
        .groupby(['GT_Cluster', 'Pred_Cluster'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .reset_index(drop=True)
    )

    total_errors = confusion_counts['count'].sum()
    top = confusion_counts.head(top_n)

    print(f"\n===== TOP {top_n} CONFUSED CLUSTER PAIRS =====")
    print(f"(out of {total_errors} wrong-existing-cluster errors)\n")
    for _, row in top.iterrows():
        pct = row['count'] / total_errors * 100
        print(f"  True: {row['GT_Cluster']:<20} -> Predicted: {row['Pred_Cluster']:<20} "
              f"| {row['count']} notes ({pct:.1f}% of all errors)")

    top5_share = top.head(5)['count'].sum() / total_errors * 100
    print(f"\nTop 5 pairs account for {top5_share:.1f}% of all wrong-existing-cluster errors.")

    return confusion_counts

def analyze_new_cluster_rename_vs_scatter(df_new, df_wrong_new):
    """
    Function that distinguishes between correctly predicted new clusters and actual wrong predicted new clusters.
    It solves the problem that pure string matching of the newly generated clusters (xyz_new_0, xyz_new_1, ...)
    from HDBSCAN always results in wrong predictions ('CLR missing offer' doesnt match 'CLR_new_0').
    Idea:
    1) Take a newly generated cluster (e.g. "CLR_new_0" with 10 notes in it):
        Note A: pred cluster: CLR_new_0 -> GT: "CLR missing offer"
        Note B: pred cluster: CLR_new_0 -> GT: "CLR missing offer"
        Note C: pred cluster: CLR_new_0 -> GT: "CLR missing offer"
        Note D: pred cluster: CLR_new_0 -> GT: "CLR missing offer"
        Note E: pred cluster: CLR_new_0 -> GT: "CLR missing offer"
        Note F: pred cluster: CLR_new_0 -> GT: "CLR missing offer"
        Note G: pred cluster: CLR_new_0 -> GT: "CLR missing offer"
        Note H: pred cluster: CLR_new_0 -> GT: "CLR missing offer"
        Note I: pred cluster: CLR_new_0 -> GT: "CLR missing offer"
        Note J: pred cluster: CLR_new_0 -> GT: "CLR gap search"   <- the odd one out
    2) Find the majority of the GT cluster -> "CLR missing offer"
    3) Distinguish notes: 
        - Note A to Note I are clustered correctly
        - Note J is clustered incorrectly
    4) Take the next newly generated cluster (e.g. "CLR_new_1")
    => same process
 
    Input:
        - df_new: (pd.DataFrame)
            All notes predicted into any new HDBSCAN cluster (columns
            'Note', 'GT_Cluster', 'Pred_Cluster'). Must include notes beyond
            just the wrong ones, so a single outlier can't define the majority.
        - df_wrong_new: (pd.DataFrame)
            Subset of df_new where GT_Cluster is an existing cluster and
            Pred_Cluster != GT_Cluster (i.e. error_breakdown["misassigned_new_examples"])
 
    Output:
        - result: (dict)
            Counts, percentages, and example rows for both sub-cases,
            or None if there's nothing to compare.
    """
    print(f"\n===== NEW-CLUSTER ASSIGNMENT: CORRECTLY CLUSTERED vs INCORRECTLY CLUSTERED =====")
 
    if df_wrong_new is None or len(df_wrong_new) == 0:
        print("No notes were wrongly sent to a new cluster. Skipping.")
        return None
 
    # Majority (mode) GT_Cluster per predicted new-cluster group, computed
    # over all notes routed there so one outlier can't define the "identity"
    majority_map = (
        df_new.groupby('Pred_Cluster')['GT_Cluster']
        .agg(lambda s: s.value_counts().idxmax())
    )
    group_sizes = df_new.groupby('Pred_Cluster').size()
 
    df_wrong_new = df_wrong_new.copy()
    df_wrong_new['Majority_GT_For_PredCluster'] = df_wrong_new['Pred_Cluster'].map(majority_map)
    df_wrong_new['Group_Size'] = df_wrong_new['Pred_Cluster'].map(group_sizes)
 
    is_clean_rename = df_wrong_new['GT_Cluster'] == df_wrong_new['Majority_GT_For_PredCluster']
 
    df_clean = df_wrong_new[is_clean_rename]
    df_scattered = df_wrong_new[~is_clean_rename]
 
    n_total = len(df_wrong_new)
    n_clean = len(df_clean)
    n_scattered = len(df_scattered)
 
    print(f"Total notes wrongly sent to new cluster: {n_total}")
    print(f"  -> Actually correct (grouped correctly, just needs renaming): "
          f"{n_clean} ({n_clean / n_total * 100:.2f}%)")
    print(f"  -> Actually wrong (mixed with unrelated notes):              "
          f"{n_scattered} ({n_scattered / n_total * 100:.2f}%)")
 
    return {
        "n_total": n_total,
        "n_clean_rename": n_clean,
        "n_scattered": n_scattered,
        "pct_clean_rename": n_clean / n_total * 100,
        "pct_scattered": n_scattered / n_total * 100,
        "clean_rename_examples": df_clean[['Note', 'GT_Cluster', 'Pred_Cluster', 'Group_Size']],
        "scattered_examples": df_scattered[
            ['Note', 'GT_Cluster', 'Pred_Cluster', 'Majority_GT_For_PredCluster', 'Group_Size']
        ],
    }


if __name__ == "__main__":
    # Regex pattern that identifies a "new" HDBSCAN-generated cluster name
    new_cluster_pattern = re.compile(r"_new_\d+$")

    groundtruth_xlsx = r"filepath_to_excel"
    clustai_xlsx = r"filepath_to_excel"
    results, error_breakdown, correct_vs_wrong_new_cluster = clustering_accuracy(groundtruth_xlsx, clustai_xlsx, new_cluster_pattern)
    top_confused_cluster_pairs(error_breakdown["misassigned_existing_examples"])

    if correct_vs_wrong_new_cluster is not None:
        print("\n--- Sample of real reclustering errors ---")
        print(correct_vs_wrong_new_cluster["scattered_examples"].head(20).to_string(index=False))
