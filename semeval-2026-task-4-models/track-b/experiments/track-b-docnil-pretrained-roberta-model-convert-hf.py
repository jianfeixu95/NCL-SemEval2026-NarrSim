"""
Baseline system for Track B.

Notice that we embed the texts from the Track B file but perform the actual evaluation using labels from Track A.
"""

import sys
import pandas as pd
import sentence_transformers
import torch
from openai import models
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModel


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

# tokenizer = AutoTokenizer.from_pretrained("../track-b-simcse/checkpoints/facebookai-roberta-large")
# backbone = AutoModel.from_pretrained("../track-b-simcse/checkpoints/facebookai-roberta-large")

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
backbone = AutoModel.from_pretrained("roberta-large")

state = torch.load(
    "./docnil-pretrained-roberta-model/DocNLI.pretrained.RoBERTA.model.pt",
    map_location="cpu"
)

if isinstance(state, dict) and any(k in state for k in ["state_dict", "model_state_dict"]):
    state = state.get("state_dict", state.get("model_state_dict"))

if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

# ✅ 关键：DocNLI 权重前缀是 roberta_single.
state = {
    (k[len("roberta_single."):]) if k.startswith("roberta_single.") else k: v
    for k, v in state.items()
}

missing, unexpected = backbone.load_state_dict(state, strict=False)
print("missing keys:", len(missing), "unexpected keys:", len(unexpected))

export_dir = "../track-b-simcse/checkpoints/docnil-pretrained-roberta-model/hf"
backbone.save_pretrained(export_dir)
tokenizer.save_pretrained(export_dir)