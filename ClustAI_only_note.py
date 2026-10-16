"""
ClustAI.py

Script to cluster new events with an AI-based approach.
Sentence Embedding via pretrained all-MiniLM-L6-v2 (Download at HuggingFace).
DBSCAN as clustering algorithm.

Author: Alexander Straub
"""

import pandas as pd
import numpy as np
import regex as re
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import hdbscan
from collections import defaultdict
from openpyxl import load_workbook
from helper_functions import extract_events_from_single_xlsx
from get_labeled_data import get_labeled_data, get_labeled_data_from_cleaned_dataset
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def extract_prefix(event):
    """
    Extract the leading abbreviation (e.g. 'KL', 'CLR') from an event string.

    Input:
        - event: (str)
            Event string

    Output:
        - Leading abbreviation ('KL' for 'Keep lane')
    """
    match = re.match(r'^([A-Za-z]+)', event.strip().upper())
    return match.group(1) if match else "UNKNOWN"

class PrefixReferenceIndex:
    """
    Holds labeled embeddings for a single prefix subclass and exposes
    a nearest-neighbor lookup.
    """

    def __init__(self, prefix: str, embeddings: np.ndarray, labels: list[str]):
        self.prefix     = prefix
        self.embeddings = embeddings          # shape (N, dim)
        self.labels     = labels              # cluster name for each embedding
        self._nn        = NearestNeighbors(n_neighbors=1, metric="cosine")
        self._nn.fit(embeddings)

    def query(self, vectors: np.ndarray) -> tuple[list[str], np.ndarray]:
        """
        For each row in *vectors* return
          - the nearest cluster label
          - the cosine distance to that neighbor  (0 = identical, 1 = orthogonal)
        """
        distances, indices = self._nn.kneighbors(vectors)
        nearest_labels = [self.labels[i[0]] for i in indices]
        nearest_dists  = distances[:, 0]
        return nearest_labels, nearest_dists

def assign_clusters_hdbscan(
        new_events:       list[str],
        reference_indices: dict[str, PrefixReferenceIndex],
        model:            SentenceTransformer,
        similarity_threshold: float = 0.05,   # cosine distance | finetuned with corresponding .py script
        hdbscan_min_cluster_size: int = 2,
        hdbscan_min_samples: int = 1
    ) -> pd.DataFrame:
    """
    Assign a cluster label to every event in *new_events*.

    Strategy per prefix subclass:
      1. Embed all events.
      2. If a reference index exists for that prefix -> nearest-neighbor lookup.
         - distance <= threshold  -> assign the known cluster label.
         - distance >  threshold  -> mark as 'unmatched', collect for HDBSCAN.
      3. Run HDBSCAN only on the unmatched events to discover new clusters.
      4. New clusters are named  '<PREFIX>_new_<n>';
         HDBSCAN noise points get  '<PREFIX>_noise'. (when no neighbor is found)

    Input:
        - new_events: (list[str])
            List of new event strings
        - reference_indices: (dict[str, PrefixReferenceIndex])
            Dictionary of prefix, embeddings and labels
        - model: (Sentence embedding model)
            Sentence embedding model
        - similarity_threshold: (float)
            Distance threshold for assigning existing cluster with NN or creating new cluster with HDBSCAN
        - hdbscan_min_cluster_size: (int)
            Minimum cluster size to create new cluster
        - hdbscan_min_samples: (int)
            Minimum samples for each cluster

    Output:
        - pd.dataframe with columns [event, prefix, assigned_cluster, nn_distance]
    """
    # Group new events by prefix
    by_prefix: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for idx, event in enumerate(new_events):
        by_prefix[extract_prefix(event)].append((idx, event))

    results = [None] * len(new_events)   # preallocate so we keep original order

    for prefix, idx_event_pairs in by_prefix.items():
        idxs, events = zip(*idx_event_pairs)
        embeddings = model.encode(list(events), show_progress_bar=False)

        assigned   = [""] * len(events)
        nn_dists   = [1.0] * len(events)

        if prefix in reference_indices:
            ref = reference_indices[prefix]
            nn_labels, dists = ref.query(np.array(embeddings))

            unmatched_local_idxs = []

            for local_i, (label, dist) in enumerate(zip(nn_labels, dists)):
                nn_dists[local_i] = float(dist)
                if dist <= similarity_threshold:
                    assigned[local_i] = label          # known cluster
                else:
                    unmatched_local_idxs.append(local_i)

            # HDBSCAN only on truly unmatched events
            if unmatched_local_idxs:
                unmatched_embs = np.array([embeddings[i] for i in unmatched_local_idxs])
                db_labels = run_hdbscan(unmatched_embs, min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples)

                for local_i, db_label in zip(unmatched_local_idxs, db_labels):
                    if db_label == -1:
                        assigned[local_i] = f"{prefix}_noise"
                    else:
                        assigned[local_i] = f"{prefix}_new_{db_label}"

        else:
            # No labeled data for this prefix -> full HDBSCAN
            print(f"  [{prefix}] no reference index found — running full HDBSCAN")
            db_labels = run_hdbscan(np.array(embeddings), min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples)
            for local_i, db_label in enumerate(db_labels):
                assigned[local_i] = f"{prefix}_new_{db_label}" if db_label != -1 else f"{prefix}_noise"

        # Write back in original event order
        for local_i, (global_idx, event) in enumerate(zip(idxs, events)):
            results[global_idx] = {
                "event":            event,
                "prefix":           prefix,
                "assigned_cluster": assigned[local_i],
                "nn_distance":      nn_dists[local_i],
            }

    return pd.DataFrame(results)

def build_reference_indices_hdbscan(
        labeled_data: list[tuple[str, str]],
        model: SentenceTransformer,
    ) -> dict[str, PrefixReferenceIndex]:
    """
    Encode all labeled events and build one PrefixReferenceIndex per prefix.

    Input:
        - labeled_data: (list[tuple[str, str]])
            Ground truth labeled dataset
        - model: (Sentence embedding model)
            Sentence embedding model

    Output:
        - indices: (dict)
            Dictionary of prefix, embeddings and labels
    """
    # Group by prefix
    by_prefix: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for cluster, event, filepath in labeled_data:
        prefix = extract_prefix(event)
        by_prefix[prefix].append((cluster, event))

    indices: dict[str, PrefixReferenceIndex] = {}

    for prefix, pairs in by_prefix.items():
        clusters, events = zip(*pairs)           # unzip
        embeddings = model.encode(
            list(events),
            show_progress_bar=False,
            normalize_embeddings=True
        )
        indices[prefix] = PrefixReferenceIndex(
            prefix     = prefix,
            embeddings = np.array(embeddings),
            labels     = list(clusters),
        )
        print(f"  [{prefix}] reference index built — {len(events)} labeled events")

    return indices

def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: int = 1
) -> np.ndarray:
    """
    Safe HDBSCAN wrapper.
    Returns -1 noise labels if there are too few samples to form a cluster.

    Input:
        - embeddings: (np array)
            Numpy array of sentence embeddings
        - min_cluster_size: (int)
            Minimum cluster size to create new cluster
        - min_samples: (int)
            Minimum samples for each cluster
    """
    n_samples = len(embeddings)

    # HDBSCAN cannot cluster fewer points than min_cluster_size
    if n_samples < min_cluster_size:
        return np.full(n_samples, -1)

    # min_samples must be smaller than number of samples
    safe_min_samples = min(min_samples, n_samples - 1)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=safe_min_samples,
        metric="euclidean"
    )

    return clusterer.fit_predict(embeddings)


if __name__ == "__main__":
    # -------------- Adjustables ------------------------------
    model_path   = r"filepath_to_model"
    root_folder  = r"filepath_to_rootfolder"
    cleaned_database_filepath = r"filepath_to_database"
    excluded_file = r""    # files to exclude (empty string if no files should be excluded)
    # -----------------------------------------------------

    # Load the already labeled data from previous .xlsx files
    print("Loading labeled data …")
    labeled_data_sorted = get_labeled_data_from_cleaned_dataset(cleaned_database_filepath, excluded_file)
    print(f"  {len(labeled_data_sorted)} labeled events loaded")

    # Load the pretrained model
    print("Loading sentence-transformer model …")
    model = SentenceTransformer(model_path)

    # Build a lookup structure to cluster new events
    # 1. Group labeled data by prefix
    # 2. Encode each group
    # 3. Build a PrefixReferenceIndex per prefix
    print("Building reference indices …")
    ref_indices = build_reference_indices_hdbscan(labeled_data_sorted, model)

    # --------- Insert path of .xlsx file with new events that have to be clustered -------------
    xlsx_file = r"filepath_to_excel"
    new_events, new_comb_answers = extract_events_from_single_xlsx(xlsx_file)
    # -------------------------------------------------------------------------------------------

    # Cluster the new events with HDBSCAN
    print("Assigning clusters to new events …")
    result_df = assign_clusters_hdbscan(
        new_events            = new_events,
        reference_indices     = ref_indices,
        model                    = model,
        similarity_threshold     = 0.05,   # low, to get high probability of clustering to existing clusters correctly
        hdbscan_min_cluster_size = 2,
        hdbscan_min_samples      = 1
    )

    print(result_df.to_string(index=False))

    # --------------- Writing the actual .xlsx file -----------------
    wb = load_workbook(xlsx_file)
    ws = wb["Triggerliste"]

    header = {cell.value: cell.column for cell in ws[1]}
    cluster_col = header.get("Cluster")
    note_col    = header.get("Note")

    if cluster_col is None or note_col is None:
        raise ValueError("Could not find 'Cluster' or 'Note' columns in the sheet.")

    cluster_values = iter(result_df["assigned_cluster"].values)

    for row in ws.iter_rows(min_row=2):
        note_cell    = row[note_col - 1]
        cluster_cell = row[cluster_col - 1]
        if note_cell.value is not None and str(note_cell.value).strip() != "":
            cluster_cell.value = next(cluster_values)

    wb.save(xlsx_file)
    print(f"Saved (formatting preserved): {xlsx_file}")
    # ---------------------------------------------------------------
