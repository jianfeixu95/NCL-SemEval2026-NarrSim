#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from transformers import AutoTokenizer

# -----------------------------
# Config: tokenizer & sentence split
# -----------------------------
TOKEN_RE = re.compile(r"\b\w+\b")  # 和你 compute_word_overlap 一致（\w 含 _，不含 '）
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
ROBERTA_NAME = "roberta-large"
roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_NAME, use_fast=True)

def tokenize_words(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    return TOKEN_RE.findall(text.lower())

def count_roberta_tokens(text: str) -> int:
    """
    Count tokens using roberta-large tokenizer (byte-level BPE).
    """
    if not isinstance(text, str) or not text:
        return 0
    # 不加 special tokens，纯文本 token 数
    return len(roberta_tokenizer.tokenize(text))

def count_sentences(text: str) -> int:
    """
    轻量句子切分：按 .!? 后的空格切；对长叙事足够用于统计。
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    s = text.strip()
    # 先粗略按换行合并
    s = re.sub(r"\s+", " ", s)
    parts = [p for p in SENT_SPLIT_RE.split(s) if p.strip()]
    return len(parts) if parts else 1


# -----------------------------
# IO
# -----------------------------
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} invalid JSON at line {line_no}: {e}")
            if isinstance(obj, dict):
                yield obj
            else:
                # 若不是 dict，转 dict 方便处理
                yield {"_value": obj}


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)


# -----------------------------
# Stats
# -----------------------------
def quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    w = pos - lo
    return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w


def summarize_int(values: List[int]) -> Dict[str, float]:
    if not values:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    vals = sorted(values)
    mean = sum(vals) / len(vals)
    median = vals[len(vals) // 2] if len(vals) % 2 == 1 else (vals[len(vals)//2 - 1] + vals[len(vals)//2]) / 2
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        "n": len(vals),
        "mean": mean,
        "median": median,
        "std": std,
        "p10": quantile(vals, 0.10),
        "p90": quantile(vals, 0.90),
        "min": float(vals[0]),
        "max": float(vals[-1]),
    }


# -----------------------------
# Label detection
# -----------------------------
LABEL_CANDIDATES = [
    "text_a_is_closer",  # SemEval track A/B 常见
    "label",
    "gold_label",
    "y",
    "target",
]


def normalize_label(v: Any) -> str:
    """
    把 label 规范化为字符串类别：
    - bool: True/False
    - 0/1: "0"/"1"
    - entailment/neutral/contradiction 等：原样小写
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        # int/float
        if int(v) == v:
            return str(int(v))
        return str(v)
    s = safe_str(v).strip().lower()
    return s if s else "missing"


def get_label(obj: Dict[str, Any]) -> Tuple[str, str]:
    """
    返回 (label_key, label_value_str)
    若找不到则 ("missing", "missing")
    """
    for k in LABEL_CANDIDATES:
        if k in obj:
            return k, normalize_label(obj.get(k))
    return "missing", "missing"


# -----------------------------
# Core per-file processing
# -----------------------------
TEXT_FIELDS = ["anchor_text", "text_a", "text_b"]


@dataclass
class FileStats:
    path: Path
    label_key: str
    label_counts: Counter
    length_stats: Dict[Tuple[str, str], Dict[str, float]]  # (field, metric) -> summary


def process_file(path: Path) -> FileStats:
    label_counts: Counter = Counter()
    label_key_seen: Counter = Counter()

    # values for distributions
    # For each field: sentence_count / word_count / token_count
    sent_counts = {f: [] for f in TEXT_FIELDS}
    word_counts = {f: [] for f in TEXT_FIELDS}
    token_counts = {f: [] for f in TEXT_FIELDS}
    roberta_token_counts = {f: [] for f in TEXT_FIELDS}

    for obj in iter_jsonl(path):
        k, v = get_label(obj)
        label_key_seen[k] += 1
        label_counts[v] += 1

        for f in TEXT_FIELDS:
            text = safe_str(obj.get(f, ""))
            # sentence
            sc = count_sentences(text)
            sent_counts[f].append(sc)
            # word count (基于简单空格切分；更贴近“词数”直觉)
            wc = len(text.split()) if text.strip() else 0
            word_counts[f].append(wc)
            # token count（按 TOKEN_RE）
            tc = len(tokenize_words(text))
            token_counts[f].append(tc)

            rtc = count_roberta_tokens(text)
            roberta_token_counts[f].append(rtc)

    # 哪个 label_key 被用得最多，就认为是该文件的 label 字段
    label_key = label_key_seen.most_common(1)[0][0] if label_key_seen else "missing"

    length_stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    for f in TEXT_FIELDS:
        length_stats[(f, "sentences")] = summarize_int(sent_counts[f])
        length_stats[(f, "words")] = summarize_int(word_counts[f])
        length_stats[(f, "tokens")] = summarize_int(token_counts[f])
        length_stats[(f, "roberta_tokens")] = summarize_int(roberta_token_counts[f])

    return FileStats(
        path=path,
        label_key=label_key,
        label_counts=label_counts,
        length_stats=length_stats,
    )


# -----------------------------
# Output writers
# -----------------------------
def write_label_stats(all_stats: List[FileStats], out_path: Path) -> None:
    """
    输出每个文件的 label 分布（count + ratio）
    """
    # collect all label values across files for consistent columns
    all_labels = set()
    for st in all_stats:
        all_labels.update(st.label_counts.keys())
    all_labels = sorted(all_labels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        # header
        cols = ["dataset_file", "label_key", "total"] + [f"count::{lab}" for lab in all_labels] + [f"ratio::{lab}" for lab in all_labels]
        f.write(",".join(cols) + "\n")

        for st in all_stats:
            total = sum(st.label_counts.values())
            row = [st.path.name, st.label_key, str(total)]
            # counts
            for lab in all_labels:
                row.append(str(st.label_counts.get(lab, 0)))
            # ratios
            for lab in all_labels:
                c = st.label_counts.get(lab, 0)
                r = (c / total) if total else 0.0
                row.append(f"{r:.6f}")
            f.write(",".join(row) + "\n")


def write_length_stats(all_stats: List[FileStats], out_path: Path) -> None:
    """
    输出每个文件、每个字段、每种长度指标的分布统计。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("dataset_file,field,metric,n,mean,median,std,p10,p90,min,max\n")
        for st in all_stats:
            for (field, metric), s in st.length_stats.items():
                f.write(
                    f"{st.path.name},{field},{metric},"
                    f"{int(s['n'])},{s['mean']:.6f},{s['median']:.6f},{s['std']:.6f},"
                    f"{s['p10']:.6f},{s['p90']:.6f},{s['min']:.6f},{s['max']:.6f}\n"
                )


# -----------------------------
# Main (no cmd args needed)
# -----------------------------
def main(jsonl_files: List[str], out_dir: str) -> None:
    out_dir_p = Path(out_dir).expanduser().resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    stats = []
    for p in jsonl_files:
        path = Path(p).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        st = process_file(path)
        stats.append(st)
        print(f"[OK] processed: {path.name} (label_key={st.label_key}, n={sum(st.label_counts.values())})")

    write_label_stats(stats, out_dir_p / "label_stats.csv")
    write_length_stats(stats, out_dir_p / "length_stats.csv")
    print(f"[DONE] saved to: {out_dir_p}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    jsonl_files = [
        str(script_dir / "input/sample_track_a.jsonl"),
        str(script_dir / "input/train_track_a.jsonl"),
        str(script_dir / "input/dev_track_a.jsonl"),
        str(script_dir / "input/dev_track_a_valid.jsonl"),
        str(script_dir / "input/dev_track_a_valid.jsonl"),
        str(script_dir / "input/dev_track_a_train.jsonl"),
    ]

    out_dir = str(script_dir / "output")
    main(jsonl_files, out_dir)
