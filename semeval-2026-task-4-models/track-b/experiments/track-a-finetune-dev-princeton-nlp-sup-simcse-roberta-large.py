import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm

# ====== Paths ======
track_a_path = "../../../semeval-2026-task-4-datasets/semeval-2026-Task-4-test-v1/test_track_a.jsonl"
output_path = "track-b-finetune-dev-princeton-nlp-sup-simcse-roberta-large/output/track_a.jsonl"

model_path = "../track-b-simcse/runs/finetune-princeton-nlp-sup-simcse-roberta-large_20260125_145526"

df = pd.read_json(track_a_path, lines=True)

# 必须包含这些字段
# anchor_text, text_a, text_b
assert all(c in df.columns for c in ["anchor_text", "text_a", "text_b"])

# ====== Load model ======
model = SentenceTransformer(model_path)

# ====== Collect all unique texts to embed once ======
all_texts = pd.concat([
    df["anchor_text"],
    df["text_a"],
    df["text_b"]
]).unique().tolist()

print("Encoding texts...")
embeddings = model.encode(
    all_texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_tensor=True,
    normalize_embeddings=True
)

embedding_lookup = dict(zip(all_texts, embeddings))

# ====== Compute similarities and labels ======
records = []

print("Computing similarities and labels...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    a = embedding_lookup[row["anchor_text"]]
    b1 = embedding_lookup[row["text_a"]]
    b2 = embedding_lookup[row["text_b"]]

    sim_a = cos_sim(a, b1).item()
    sim_b = cos_sim(a, b2).item()

    text_a_is_closer = sim_a > sim_b

    records.append({
        "anchor_text": row["anchor_text"],
        "text_a": row["text_a"],
        "text_b": row["text_b"],
        "text_a_is_closer": bool(text_a_is_closer)
    })

# ====== Save to jsonl ======
out_df = pd.DataFrame(records)
out_df.to_json(output_path, orient="records", lines=True, force_ascii=False)

print("Saved to:", output_path)
