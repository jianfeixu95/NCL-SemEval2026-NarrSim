#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute lexical overlap metrics for SemEval-2026 Task 4 jsonl files.

Metrics (for anchor_text vs text_a AND anchor_text vs text_b):
- jaccard (unigram set)
- anchor_overlap (|A∩B| / |A|)
- candidate_overlap (|A∩B| / |B|)
- 2-gram jaccard
- 3-gram jaccard
- rouge-1 (F1)
- rouge-2 (F1)
- pos_overlap (Jaccard over POS-filtered lemmas; prefers spaCy, falls back to NLTK)

Outputs:
- per-instance metrics CSV
- per-file aggregated summary CSV (mean/median/std/p10/p90)

Usage:
  python compute_overlap_metrics.py \
    --inputs /mnt/data/sample_track_a.jsonl /mnt/data/train_track_a.jsonl /mnt/data/dev_track_a.jsonl \
    --out_dir ./overlap_stats

You can also pass Track B files similarly.
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

from sympy import false

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")   # keeps simple contractions

# ----------------------------
# Tokenization + ngrams
# ----------------------------
def tokenize(text: str) -> List[str]:
    """
    与 compute_word_overlap 一致：
    - 小写
    - \b\w+\b 提取 token（字母/数字/下划线）
    - 注意：不保留 apostrophe（don't -> don, t）
    返回 list（后续 jaccard_set/coverage 会 set() 去重）
    """
    if not isinstance(text, str) or not text:
        return []
    return WORD_RE.findall(text.lower())

# def tokenize_set(text: str) -> Set[str]:
#     if not isinstance(text, str) or not text:
#         return set()
#     return set(WORD_RE.findall(text.lower()))

def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0:
        return []
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ----------------------------
# Overlap metrics (set-based)
# ----------------------------
def jaccard_set(a: Sequence, b: Sequence) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def coverage(a: Sequence, b: Sequence) -> float:
    """
    |A∩B| / |A|   (A->B coverage), where A and B are sequences treated as sets.
    """
    sa, sb = set(a), set(b)
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


# ----------------------------
# ROUGE (token overlap, bag-of-ngrams)
# We'll compute ROUGE-N F1 with candidate as hypothesis, anchor as reference by default.
# ----------------------------
def rouge_n_f1(reference_tokens: List[str], candidate_tokens: List[str], n: int) -> float:
    ref_ngrams = ngrams(reference_tokens, n)
    cand_ngrams = ngrams(candidate_tokens, n)
    if not ref_ngrams and not cand_ngrams:
        return 0.0
    if not ref_ngrams or not cand_ngrams:
        return 0.0

    ref_counts = Counter(ref_ngrams)
    cand_counts = Counter(cand_ngrams)

    overlap = 0
    for g, c in cand_counts.items():
        overlap += min(c, ref_counts.get(g, 0))

    # precision/recall
    p = overlap / sum(cand_counts.values()) if cand_counts else 0.0
    r = overlap / sum(ref_counts.values()) if ref_counts else 0.0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ----------------------------
# POS overlap (spaCy preferred, NLTK fallback)
# We compute Jaccard overlap over lemmas restricted to content POS.
# ----------------------------
@dataclass
class POSHelper:
    mode: str  # "spacy", "nltk", or "none"
    nlp: object = None

    def content_lemmas(self, text: str) -> List[str]:
        if not text:
            return []
        if self.mode == "spacy":
            # Keep NOUN/PROPN/VERB/ADJ/ADV as "content"
            doc = self.nlp(text)
            out = []
            for tok in doc:
                if tok.is_space or tok.is_punct:
                    continue
                if tok.pos_ in {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}:
                    lemma = tok.lemma_.lower().strip()
                    if lemma:
                        out.append(lemma)
            return out

        if self.mode == "nltk":
            # NLTK pos_tag: NN*, VB*, JJ*, RB*
            # Lemmatize with WordNetLemmatizer (approx).
            from nltk import pos_tag
            from nltk.stem import WordNetLemmatizer
            from nltk.corpus import wordnet

            toks = tokenize(text)
            if not toks:
                return []
            tags = pos_tag(toks)
            wnl = WordNetLemmatizer()

            def wn_pos(tag: str):
                if tag.startswith("J"):
                    return wordnet.ADJ
                if tag.startswith("V"):
                    return wordnet.VERB
                if tag.startswith("N"):
                    return wordnet.NOUN
                if tag.startswith("R"):
                    return wordnet.ADV
                return None

            out = []
            for w, t in tags:
                if t.startswith(("N", "V", "J", "R")):
                    p = wn_pos(t)
                    if p is None:
                        out.append(w)
                    else:
                        out.append(wnl.lemmatize(w, p))
            return out

        # none
        return []


def init_pos_helper() -> POSHelper:
    # Try spaCy first
    try:
        import spacy  # type: ignore

        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except Exception:
            # fallback to blank English model (no tagger by default)
            nlp = spacy.blank("en")
            # If blank, we likely cannot do POS; treat as none.
            return POSHelper(mode="none")
        return POSHelper(mode="spacy", nlp=nlp)
    except Exception:
        pass

    # Try NLTK
    try:
        import nltk  # type: ignore

        # Ensure required resources exist; if not, try download (may fail offline)
        needed = [
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
            ("corpora/wordnet", "wordnet"),
        ]
        for path, pkg in needed:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(pkg)
                except Exception:
                    return POSHelper(mode="none")

        return POSHelper(mode="nltk")
    except Exception:
        return POSHelper(mode="none")


# ----------------------------
# Core computation per pair
# ----------------------------
def compute_pair_metrics(anchor: str, cand: str, pos_helper: POSHelper) -> Dict[str, float]:
    a_tok = tokenize(anchor)
    c_tok = tokenize(cand)

    # unigram overlaps
    jacc = jaccard_set(a_tok, c_tok)
    a_cov = coverage(a_tok, c_tok)  # anchor coverage
    c_cov = coverage(c_tok, a_tok)  # candidate coverage

    # n-gram overlaps (set Jaccard)
    a2, c2 = ngrams(a_tok, 2), ngrams(c_tok, 2)
    a3, c3 = ngrams(a_tok, 3), ngrams(c_tok, 3)
    j2 = jaccard_set(a2, c2)
    j3 = jaccard_set(a3, c3)

    # ROUGE (F1)
    r1 = rouge_n_f1(a_tok, c_tok, 1)
    r2 = rouge_n_f1(a_tok, c_tok, 2)

    # POS overlap (content lemmas)
    pos_a = pos_helper.content_lemmas(anchor)
    pos_c = pos_helper.content_lemmas(cand)
    pos_j = jaccard_set(pos_a, pos_c) if pos_a or pos_c else 0.0

    return {
        "jaccard": jacc,
        "anchor_overlap": a_cov,
        "candidate_overlap": c_cov,
        "2gram_jaccard": j2,
        "3gram_jaccard": j3,
        "rouge_1_f1": r1,
        "rouge_2_f1": r2,
        "pos_overlap": pos_j,
    }


# ----------------------------
# IO helpers
# ----------------------------
def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} invalid JSON at line {line_no}: {e}")


def safe_get_text(obj: Dict, key: str) -> str:
    v = obj.get(key, "")
    if v is None:
        return ""
    if not isinstance(v, str):
        return str(v)
    return v


def infer_dataset_tags(path: Path) -> Tuple[str, str]:
    """
    Infer split and track from filename like:
      sample_track_a.jsonl / train_track_b.jsonl / dev_track_a.jsonl
    """
    name = path.stem.lower()
    split = "unknown"
    track = "unknown"

    for s in ["sample", "train", "dev", "test"]:
        if name.startswith(s + "_") or f"_{s}_" in name or name.endswith("_" + s):
            split = s
            break
        if name.startswith(s):
            split = s
            break

    if "track_a" in name or "track-a" in name or name.endswith("a"):
        track = "track_a"
    if "track_b" in name or "track-b" in name or name.endswith("b"):
        track = "track_b"

    return split, track


# ----------------------------
# Aggregation
# ----------------------------
def quantile(sorted_vals: List[float], q: float) -> float:
    # simple linear interpolation quantile
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


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
        }
    vals = sorted(values)
    mean = sum(vals) / len(vals)
    median = vals[len(vals) // 2] if len(vals) % 2 == 1 else (vals[len(vals)//2 - 1] + vals[len(vals)//2]) / 2
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "p10": quantile(vals, 0.10),
        "p90": quantile(vals, 0.90),
    }


# ----------------------------
# Main pipeline
# ----------------------------
def process_file(path: Path, out_dir: Path, pos_helper: POSHelper) -> Tuple[Path, Dict[str, Dict[str, float]]]:
    split, track = infer_dataset_tags(path)

    per_instance_out = Path(f"{out_dir}/{path.stem}_pair_metrics.csv")
    summary_out = Path(f"{out_dir}/{path.stem}_summary.csv")

    # Collect metric lists for aggregation
    metrics_a: Dict[str, List[float]] = {}
    metrics_b: Dict[str, List[float]] = {}

    # Write per-instance CSV
    fieldnames = [
        "dataset_file",
        "split",
        "track",
        "idx",
        "pair",
        # metrics:
        "jaccard",
        "anchor_overlap",
        "candidate_overlap",
        "2gram_jaccard",
        "3gram_jaccard",
        "rouge_1_f1",
        "rouge_2_f1",
        "pos_overlap",
    ]

    with per_instance_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for i, obj in enumerate(iter_jsonl(path)):
            anchor = safe_get_text(obj, "anchor_text")
            closer_label = safe_get_text(obj, "text_a_is_closer")
            if closer_label=="True":
                text_a = safe_get_text(obj, "text_a")
                text_b = safe_get_text(obj, "text_b")
            else:
                text_a = safe_get_text(obj, "text_b")
                text_b = safe_get_text(obj, "text_a")


            m_a = compute_pair_metrics(anchor, text_a, pos_helper)
            m_b = compute_pair_metrics(anchor, text_b, pos_helper)

            # store for aggregation
            for k, v in m_a.items():
                metrics_a.setdefault(k, []).append(v)
            for k, v in m_b.items():
                metrics_b.setdefault(k, []).append(v)

            # write rows
            row_a = {
                "dataset_file": path.name,
                "split": split,
                "track": track,
                "idx": i,
                "pair": "anchor_text_vs_pos",
                **m_a,
            }
            row_b = {
                "dataset_file": path.name,
                "split": split,
                "track": track,
                "idx": i,
                "pair": "anchor_text_vs_neg",
                **m_b,
            }
            w.writerow(row_a)
            w.writerow(row_b)

    # Build summary
    # summary structure: metric -> (pair -> stats)
    summary: Dict[str, Dict[str, float]] = {}
    with summary_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset_file", "split", "track", "pair", "metric", "mean", "median", "std", "p10", "p90", "n"])
        for pair_name, md in [("anchor_text_vs_pos", metrics_a), ("anchor_text_vs_neg", metrics_b)]:
            for metric, vals in md.items():
                stats = summarize(vals)
                w.writerow([
                    path.name, split, track, pair_name, metric,
                    stats["mean"], stats["median"], stats["std"], stats["p10"], stats["p90"],
                    len(vals)
                ])
                summary[f"{pair_name}:{metric}"] = stats | {"n": len(vals)}

    return per_instance_out, summary


def main(input_dir, out_dir):
    Path(out_dir).expanduser().resolve().mkdir(parents=false, exist_ok=True)

    pos_helper = init_pos_helper()
    if pos_helper.mode == "none":
        print("[WARN] POS overlap unavailable (spaCy model or NLTK resources not found). pos_overlap will be 0.0.")
    else:
        print(f"[INFO] POS overlap enabled via: {pos_helper.mode}")

    for p in input_dir:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(path)
        per_inst_csv, _ = process_file(path, out_dir, pos_helper)
        print(f"[OK] {path.name} -> {per_inst_csv}")

    print(f"[DONE] Outputs saved to: {out_dir}")


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    input_dir = [
        f"{script_dir}/input/sample_track_a.jsonl",
        f"{script_dir}/input/dev_track_a.jsonl",
        f"{script_dir}/input/train_track_a.jsonl"]
    out_dir = f"{script_dir}/output"
    main(input_dir, out_dir)
