"""
Baseline system for Track B.

Notice that we embed the texts from the Track B file but perform the actual evaluation using labels from Track A.
"""

import sys
sys.path.append("SentEval")
import senteval
import pandas as pd
import torch
# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer, AutoModel
import numpy as np
import evaluation

def evaluate_accuracy(labeled_data_path, embedding_lookup):
    # 1️⃣ 读取数据
    df = pd.read_json(labeled_data_path, lines=True)

    # 2️⃣ 映射文本到 embeddings
    df["anchor_embedding"] = df["anchor_text"].map(embedding_lookup)
    df["a_embedding"] = df["text_a"].map(embedding_lookup)
    df["b_embedding"] = df["text_b"].map(embedding_lookup)

    # 3️⃣ 将 embedding 列堆成矩阵
    anchor = np.stack(df["anchor_embedding"].values)
    a_emb = np.stack(df["a_embedding"].values)
    b_emb = np.stack(df["b_embedding"].values)

    # 4️⃣ 定义批量余弦相似度函数
    def cosine_similarity(a, b):
        dot = np.sum(a * b, axis=1)
        norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        return dot / norm

    # 5️⃣ 计算余弦相似度
    sim_a = cosine_similarity(anchor, a_emb)
    sim_b = cosine_similarity(anchor, b_emb)

    # 6️⃣ 保存到 DataFrame
    df["sim_a"] = sim_a
    df["sim_b"] = sim_b

    # 7️⃣ 预测并计算准确率
    df["predicted_text_a_is_closer"] = df["sim_a"] > df["sim_b"]
    accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()

    return accuracy

def evaluate_semeval(model_name_or_path):
    sys.argv = [
        "run.py",
        "--model_name_or_path", f"{model_name_or_path}",
        "--pooler", "cls",
        "--mode", "test",
    ]
    evaluation.main()
    return None

# def encode(sentences, model):
#     # 将句子变成 token id
#     inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     # mean pooling
#     embeddings = outputs.last_hidden_state.mean(dim=1)
#     return embeddings

def encode(sentences, model, tokenizer, device, batch_size=32):
    """
    sentences: list of str
    model: AutoModel
    tokenizer: AutoTokenizer
    device: 'cuda' or 'cpu'
    batch_size: 每次处理的句子数量
    """
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch_texts = sentences[i:i+batch_size]

        # 编码到 token ids
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # 移回 CPU，释放 GPU 显存
            all_embeddings.append(embeddings.cpu())

        # 清理缓存
        torch.cuda.empty_cache()

    # 拼接所有 batch
    return torch.cat(all_embeddings, dim=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select baseline method
# model_name_or_path = "../../semeval-2026-task-4-baselines/checkpoints/all-minilm-l6-v2"
# model_name_or_path = "../../semeval-2026-task-4-baselines/checkpoints/princeton-nlp-sup-simcse-roberta-base"
model_name_or_path = "../../semeval-2026-task-4-baselines/checkpoints/princeton-nlp-sup-simcse-roberta-large"
data = pd.read_json("data/dev_track_b.jsonl", lines=True)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModel.from_pretrained(model_name_or_path).to(device)
embeddings = encode(list(data["text"]), model, tokenizer, device)
embedding_lookup = dict(zip(data["text"], embeddings))
accuracy = evaluate_accuracy("data/dev_track_a.jsonl", embedding_lookup)
semeval = evaluate_semeval(model_name_or_path)
print(f"Accuracy: {accuracy:.3f}")
