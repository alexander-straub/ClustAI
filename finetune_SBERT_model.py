"""
finetune_SBERT_model.py

Script to finetune your pretrained sentence embedding model according to your database.

Author: Alexander Straub
"""

import pandas as pd
import random
from itertools import combinations
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import torch
import numpy as np

# =========================================================
# CONFIG
# =========================================================
EXCEL_PATH = r"filepath_to_excel"
MODEL_PATH = r"filepath_to_model"
OUTPUT_PATH = r"filepath_to_finetuned_model"
MAX_PAIRS_PER_CLUSTER = 30
VAL_SIZE = 0.2
BATCH_SIZE = 16
MAX_EPOCHS = 5
LEARNING_RATE = 2e-5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =========================================================
# 1. Load and clean ground truth data
# =========================================================
df = pd.read_excel(EXCEL_PATH)  # columns: Cluster, Notes, Filepath
df = df.dropna(subset=["Cluster", "Note"])
df["Cluster"] = df["Cluster"].astype(str)  # ensures plain Python strings, not Arrow-backed
df["Note"] = df["Note"].astype(str)  # adjust to your actual column name
df["Note"] = df["Note"].astype(str).str.strip()
df = df[df["Note"] != ""]

print(f"Loaded {len(df)} events across {df['Cluster'].nunique()} clusters")

cluster_counts = df["Cluster"].value_counts()
singletons = cluster_counts[cluster_counts == 1]
if len(singletons) > 0:
    print(f"Note: {len(singletons)} clusters have only 1 event and will be "
          f"skipped for training (no pair possible).")

# =========================================================
# 2. Split clusters into train/val (cluster-level split to avoid leakage)
# =========================================================
clusters = df["Cluster"].unique()
clusters = np.asarray(clusters)
train_clusters, val_clusters = train_test_split(
    clusters, test_size=VAL_SIZE, random_state=RANDOM_SEED
)

train_df = df[df["Cluster"].isin(train_clusters)]
val_df = df[df["Cluster"].isin(val_clusters)]

print(f"Train clusters: {len(train_clusters)} | Val clusters: {len(val_clusters)}")

# =========================================================
# 3. Build training pairs (positives only, capped per cluster)
# =========================================================
train_examples = []

for cluster_id, group in train_df.groupby("Cluster"):
    notes = group["Note"].tolist()
    if len(notes) < 2:
        continue
    pairs = list(combinations(notes, 2))
    if len(pairs) > MAX_PAIRS_PER_CLUSTER:
        pairs = random.sample(pairs, MAX_PAIRS_PER_CLUSTER)
    for text_a, text_b in pairs:
        train_examples.append(InputExample(texts=[text_a, text_b]))

print(f"Built {len(train_examples)} training pairs")

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

# =========================================================
# 4. Build validation pairs (with same/different-cluster labels)
# =========================================================
val_notes = val_df["Note"].tolist()
val_labels = val_df["Cluster"].tolist()

# Separate positive pairs (same cluster) from negative pairs (different cluster)
positive_pairs = []
negative_pairs = []

for i, j in combinations(range(len(val_notes)), 2):
    if val_labels[i] == val_labels[j]:
        positive_pairs.append((i, j))
    else:
        negative_pairs.append((i, j))

print(f"Available positive pairs: {len(positive_pairs)}")
print(f"Available negative pairs: {len(negative_pairs)}")

# Keep all positives, sample an equal (or small multiple) number of negatives
NEG_TO_POS_RATIO = 3 # balance neg to pos samples 
n_neg_to_sample = min(len(negative_pairs), len(positive_pairs) * NEG_TO_POS_RATIO)
sampled_negatives = random.sample(negative_pairs, n_neg_to_sample)

balanced_pairs = positive_pairs + sampled_negatives
random.shuffle(balanced_pairs)

val_texts_a = [val_notes[i] for i, j in balanced_pairs]
val_texts_b = [val_notes[j] for i, j in balanced_pairs]
val_scores = [1.0 if val_labels[i] == val_labels[j] else 0.0 for i, j in balanced_pairs]

print(f"Balanced val set — positives: {sum(val_scores):.0f}, negatives: {len(val_scores) - sum(val_scores):.0f}")

evaluator = EmbeddingSimilarityEvaluator(val_texts_a, val_texts_b, val_scores)

base_model = SentenceTransformer(MODEL_PATH, device=device)

baseline_results = evaluator(
    base_model,
    output_path=None,
    epoch=-1,   # just a label, doesn't affect anything
)

if isinstance(baseline_results, dict):
    baseline_score = baseline_results[evaluator.primary_metric]
else:
    baseline_score = baseline_results

baseline_score = float(baseline_score)

print(f"Baseline (non-finetuned) val score: {baseline_score:.4f}")
print(f"Positive val pairs (same cluster): {sum(val_scores)}")
print(f"Negative val pairs (diff cluster): {len(val_scores) - sum(val_scores)}")

# =========================================================
# 5. Manual training loop (bypasses model.fit() / datasets dependency)
# =========================================================
model = SentenceTransformer(MODEL_PATH, device=device)
train_loss = losses.MultipleNegativesRankingLoss(model).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# collate function to turn InputExamples into model-ready batches
def collate_fn(batch):
    texts = [[] for _ in range(len(batch[0].texts))]
    for example in batch:
        for i, text in enumerate(example.texts):
            texts[i].append(text)
    tokenized = [model.tokenize(t) for t in texts]
    # keep only tensor fields
    tokenized = [
        {k: v for k, v in feat.items() if torch.is_tensor(v)}
        for feat in tokenized
    ]
    return tokenized

train_dataloader.collate_fn = collate_fn

best_score = -1
for epoch in range(MAX_EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        # move batch to device
        batch = [{k: v.to(device) for k, v in features.items()} for features in batch]

        optimizer.zero_grad()
        loss_value = train_loss(batch, labels=None)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_value.item()

    avg_loss = total_loss / len(train_dataloader)

    # evaluate on validation set
    model.eval()

    eval_results = evaluator(
        model,
        output_path=None,
        epoch=epoch,
    )

    # Recent Sentence Transformers versions return a metric dictionary
    if isinstance(eval_results, dict):
        score = eval_results[evaluator.primary_metric]
    else:
        # Compatibility with older versions that returned a float
        score = eval_results

    score = float(score)

    print(
        f"Epoch {epoch + 1}/{MAX_EPOCHS} "
        f"- train loss: {avg_loss:.4f} "
        f"- val score: {score:.4f}"
    )

    if score > best_score:
        best_score = score
        model.save(OUTPUT_PATH)
        print(f"  -> New best model saved (val score: {score:.4f})")

print(f"Training complete. Best model saved to: {OUTPUT_PATH} (val score: {best_score:.4f})")
