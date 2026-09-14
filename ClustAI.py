"""
ClustAI.py

Script to cluster new events with an AI-based approach.
Sentence Embedding via pretrained all-MiniLM-L6-v2 (Download at HuggingFace).
DBSCAN or HDBSCAN as clustering algorithms.

Difference to ClustAI_only)note.py: Combined answers are used as additional information.
DBSCAN can be used as clustering algorithm.
Both differences result in worse performance.

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
from get_labeled_data import get_labeled_data_with_comb_answer
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

def fuse_embeddings(
    note_embeddings: np.ndarray,
    answer_embeddings: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Concatenate note and combined answer embeddings into a single vector.
    alpha controls the relative weight of the answer embedding.
    alpha=1.0 means equal weight; alpha=0.5 down-weights answers.
    """
    return np.concatenate(
        [note_embeddings, alpha * answer_embeddings], axis=1
    )

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


def build_reference_indices(
        labeled_data: list[tuple[str, str]],    # (cluster, event, combined answer)
        model: SentenceTransformer,
        alpha: float = 0.5,
    ) -> dict[str, PrefixReferenceIndex]:
    """
    Encode all labeled events and build one PrefixReferenceIndex per prefix.

    Input:
        - labeled_data: (list[tuple[str, str]])
            Ground truth labeled dataset
        - model: (Sentence embedding model)
            Sentence embedding model
        - alpha: (float)
            Weighting parameter for combined answers

    Output:
        - indices: (dict)
            Dictionary of prefix, embeddings and labels
    """
    # Group by prefix
    by_prefix: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for cluster, event , answer in labeled_data:
        prefix = extract_prefix(event)
        by_prefix[prefix].append((cluster, event, answer))

    indices: dict[str, PrefixReferenceIndex] = {}

    for prefix, triples in by_prefix.items():
        clusters, events, answers = zip(*triples)           # unzip
        note_embs   = model.encode(list(events),   show_progress_bar=False)
        answer_embs = model.encode(list(answers),  show_progress_bar=False)
        fused       = fuse_embeddings(
            np.array(note_embs), np.array(answer_embs), alpha=alpha
        )
        indices[prefix] = PrefixReferenceIndex(
            prefix     = prefix,
            embeddings = fused,
            labels     = list(clusters),
        )
        print(f"  [{prefix}] reference index built — {len(events)} labeled events")

    return indices

def assign_clusters(
        new_events:       list[str],
        new_answers:           list[str],
        reference_indices: dict[str, PrefixReferenceIndex],
        model:            SentenceTransformer,
        alpha:                 float = 0.5,
        similarity_threshold: float = 0.05,   # cosine distance | finetuned with corresponding .py script
        dbscan_eps:       float = 0.30,
        dbscan_min_samples: int = 2,
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
        - alpha: (float)
            Weighting parameter for combined answers
        - similarity_threshold: (float)
            Distance threshold for assigning existing cluster with NN or creating new cluster with HDBSCAN
        - dbscan_eps: (int)
            Minimum cluster size to create new cluster
        - dbscan_min_samples: (int)
            Minimum samples for each cluster

    Output:
        - pd.dataframe with columns [event, prefix, assigned_cluster, nn_distance]
    """
    # Group new events by prefix
    by_prefix: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for idx, (event, answer) in enumerate(zip(new_events, new_answers)):
        by_prefix[extract_prefix(event)].append((idx, event, answer))

    results = [None] * len(new_events)   # preallocate so we keep original order

    for prefix, idx_event_answer_triples in by_prefix.items():
        idxs, events, answers = zip(*idx_event_answer_triples)
        note_embs   = model.encode(list(events),  show_progress_bar=False)
        answer_embs = model.encode(list(answers), show_progress_bar=False)
        embeddings  = fuse_embeddings(
            np.array(note_embs), np.array(answer_embs), alpha=alpha
        )

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

            # DBSCAN only on truly unmatched events
            if unmatched_local_idxs:
                unmatched_embs = np.array([embeddings[i] for i in unmatched_local_idxs])
                db_labels = run_dbscan(unmatched_embs, dbscan_eps, dbscan_min_samples)

                for local_i, db_label in zip(unmatched_local_idxs, db_labels):
                    if db_label == -1:
                        assigned[local_i] = f"{prefix}_noise"
                    else:
                        assigned[local_i] = f"{prefix}_new_{db_label}"

        else:
            # No labeled data for this prefix -> full DBSCAN
            print(f"  [{prefix}] no reference index found — running full DBSCAN")
            db_labels = run_dbscan(np.array(embeddings), dbscan_eps, dbscan_min_samples)
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

def assign_clusters_hdbscan(
        new_events:       list[str],
        new_answers:           list[str],
        reference_indices: dict[str, PrefixReferenceIndex],
        model:            SentenceTransformer,
        alpha:                 float = 0.5,
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
        - alpha: (float)
            Weighting parameter for combined answers
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
    by_prefix: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for idx, (event, answer) in enumerate(zip(new_events, new_answers)):
        by_prefix[extract_prefix(event)].append((idx, event, answer))

    results = [None] * len(new_events)   # preallocate so we keep original order

    for prefix, idx_event_answer_triples in by_prefix.items():
        idxs, events, answers = zip(*idx_event_answer_triples)
        note_embs   = model.encode(list(events),  show_progress_bar=False)
        answer_embs = model.encode(list(answers), show_progress_bar=False)
        embeddings  = fuse_embeddings(
            np.array(note_embs), np.array(answer_embs), alpha=alpha
        )

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
        labeled_data: list[tuple[str, str, str]],
        model: SentenceTransformer,
        alpha: float = 0.5,
    ) -> dict[str, PrefixReferenceIndex]:
    """
    Encode all labeled events and build one PrefixReferenceIndex per prefix.

    Input:
        - labeled_data: (list[tuple[str, str]])
            Ground truth labeled dataset
        - model: (Sentence embedding model)
            Sentence embedding model
        - alpha: (float)
            Weighting parameter for combined answers

    Output:
        - indices: (dict)
            Dictionary of prefix, embeddings and labels
    """
    # Group by prefix
    by_prefix: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for cluster, event, answer in labeled_data:
        prefix = extract_prefix(event)
        by_prefix[prefix].append((cluster, event, answer))

    indices: dict[str, PrefixReferenceIndex] = {}

    for prefix, triples in by_prefix.items():
        clusters, events, answers = zip(*triples)           # unzip
        note_embs = model.encode(
            list(events),
            show_progress_bar=False,
            normalize_embeddings=True
        )
        answer_embs = model.encode(
            list(answers),
            show_progress_bar=False,
            normalize_embeddings=True
        )
        fused = fuse_embeddings(np.array(note_embs), np.array(answer_embs), alpha=alpha)

        indices[prefix] = PrefixReferenceIndex(
            prefix     = prefix,
            embeddings = fused,
            labels     = list(clusters),
        )
        print(f"  [{prefix}] reference index built — {len(events)} labeled events")

    return indices

def run_dbscan(embeddings: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Wrapper function so DBSCAN hyperparams are in one place.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    return db.fit_predict(embeddings)

def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: int = 1
) -> np.ndarray:
    """
    Safe HDBSCAN wrapper.
    Returns -1 noise labels if there are too few samples to form a cluster.
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
    model_path   = r"C:\Users\q645302.MCBMW1\Documents\ML_models\all-MiniLM-L6-v2"
    root_folder  = r"C:\Users\q645302.MCBMW1\BMW Group\ADAS EL-5-US-1 - General\21_Tools\ATP_Tool\external_files\Clustering_Database"
    alpha = 0.05
    # -----------------------------------------------------

    # Load the already labeled data from previous .xlsx files
    print("Loading labeled data …")
    labeled_data_sorted = get_labeled_data_with_comb_answer(root_folder)
    print(f"  {len(labeled_data_sorted)} labeled events loaded")

    # Load the pretrained model
    print("Loading sentence-transformer model …")
    model = SentenceTransformer(model_path)

    # Build a lookup structure to cluster new events
    # 1. Group labeled data by prefix
    # 2. Encode each group
    # 3. Build a PrefixReferenceIndex per prefix
    print("Building reference indices …")
    ref_indices = build_reference_indices_hdbscan(labeled_data_sorted, model, alpha=alpha)

    # --------- Insert path of .xlsx file with new events that have to be clustered -------------
    xlsx_file = r"C:\Users\q645302.MCBMW1\Desktop\26-11-480_ATS3_NA5_5BB_CA_OR_WA - Copy_alpha_0.05.xlsx"
    new_events, new_comb_ans = extract_events_from_single_xlsx(xlsx_file)
    # -------------------------------------------------------------------------------------------

    # # Cluster the new events wwith DBSCAN
    # print("Assigning clusters to new events …")
    # result_df = assign_clusters_hdbscan(
    #     new_events            = new_events,
    #     reference_indices     = ref_indices,
    #     model                 = model,
    #     similarity_threshold  = 0.15,   # key parameter to tune, maybe not ideal yet -> performance could be improved here
    #     dbscan_eps            = 0.30,   # maximum distance between two samples for one to be considered as in the neighborhood of the other
    #     dbscan_min_samples    = 2,      # min samples needed to create a new cluster
    # )

    # print(result_df.to_string(index=False))

    # Cluster the new events with HDBSCAN
    print("Assigning clusters to new events …")
    result_df = assign_clusters_hdbscan(
        new_events               = new_events,
        new_answers              = new_comb_ans,
        reference_indices        = ref_indices,
        model                    = model,
        alpha                    = alpha,
        similarity_threshold     = 0.15,   # key parameter to tune, maybe not ideal yet -> performance could be improved here
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
