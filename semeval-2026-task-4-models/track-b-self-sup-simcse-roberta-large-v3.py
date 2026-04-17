"""
Baseline system for Track B.

Notice that we embed the texts from the Track B file but perform the actual evaluation using labels from Track A.
"""

import sys
import pandas as pd
import sentence_transformers
import torch
from openai import models
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer
import numpy as np

def evaluate(labeled_data_path, embedding_lookup):
    df = pd.read_json(labeled_data_path, lines=True)

    # Map texts to embeddings
    df["anchor_embedding"] = df["anchor_text"].map(embedding_lookup)
    df["a_embedding"] = df["text_a"].map(embedding_lookup)
    df["b_embedding"] = df["text_b"].map(embedding_lookup)

    # Look up cosine similarities
    df["sim_a"] = df.apply(
        lambda row: cos_sim(row["anchor_embedding"], row["a_embedding"]), axis=1
    )
    df["sim_b"] = df.apply(
        lambda row: cos_sim(row["anchor_embedding"], row["b_embedding"]), axis=1
    )

    # Predict and calculate accuracy
    df["predicted_text_a_is_closer"] = df["sim_a"] > df["sim_b"]
    accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
    return accuracy


# Select baseline method
data = pd.read_json("../semeval-2026-task-4-datasets/semeval-2026-task-4-dev-v1/dev_track_a.jsonl", lines=True)

all_texts = list(set(data["anchor_text"]) |
                 set(data["text_a"]) |
                 set(data["text_b"]))

model = SentenceTransformer(
    # "track-b/track-b-simcse/checkpoints/princeton-nlp-sup-simcse-roberta-large",
    "YuxinJiang/sup-promcse-roberta-large"
)

embeddings = model.encode(
    all_texts,
    batch_size=32,
    show_progress_bar=True)

embedding_lookup = dict(zip(all_texts, embeddings))
accuracy = evaluate("../semeval-2026-task-4-datasets/semeval-2026-task-4-dev-v1/dev_track_a.jsonl", embedding_lookup)
print(f"Accuracy: {accuracy:.3f}")

np.save(f"track-b/experiments/self-simcse-roberta-large-v3/output/self-simcse-roberta-large-v3.npy", embeddings)
