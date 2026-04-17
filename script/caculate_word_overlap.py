from statistics import mean

import pandas as pd
import re
from typing import Set


def tokenize(text: str) -> Set[str]:
    """
    简单英文 tokenizer：
    - 小写
    - 只保留字母数字
    """
    if not isinstance(text, str):
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return set(tokens)


def jaccard_overlap(a: Set[str], b: Set[str]) -> float:
    """
    Jaccard overlap: |A ∩ B| / |A ∪ B|
    """
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_word_overlap(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # tokenize
    df["anchor_tokens"] = df["sent0"].apply(tokenize)
    df["a_tokens"] = df["sent1"].apply(tokenize)
    df["b_tokens"] = df["hard_neg"].apply(tokenize)

    # overlap
    df["overlap_sent0"] = df.apply(
        lambda row: jaccard_overlap(row["anchor_tokens"], row["a_tokens"]),
        axis=1
    )

    df["overlap_hard_neg"] = df.apply(
        lambda row: jaccard_overlap(row["anchor_tokens"], row["b_tokens"]),
        axis=1
    )

    return df


if __name__ == "__main__":
    csv_path = "../semeval-2026-task-4-datasets/semeval-2026-task-4-train-v1/train_track_a.csv"
    df = compute_word_overlap(csv_path)

    print(f"dataset path: {csv_path}")
    print(f'sent0_sent1_overlap: {df["overlap_sent0"].mean()}')
    print(f'sent0_hard_neg_overlap: {df["overlap_hard_neg"].mean()}')


    csv_path = "../semeval-2026-task-4-datasets/semeval-2026-task-4-dev-v1/dev_track_a.csv"
    df = compute_word_overlap(csv_path)

    print(f"dataset path: {csv_path}")
    print(f'sent0_sent1_overlap: {df["overlap_sent0"].mean()}')
    print(f'sent0_hard_neg_overlap: {df["overlap_hard_neg"].mean()}')
