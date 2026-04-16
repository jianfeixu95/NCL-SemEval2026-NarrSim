# 📘 NCL-SemEval2026-NarrSim

**Narrative Similarity Modeling for SemEval-2026 Task 4**

This repository presents our system for **SemEval-2026 Task 4: Narrative Similarity**, addressing both:

- **Track A**: Aspect-based reasoning using LLM agents
- **Track B**: Narrative embedding similarity using contrastive learning

---

## 🧠 Overview

Narrative similarity requires modeling not only surface-level semantic overlap but also **deeper structural alignment** between stories. Two narratives may share similar topics but differ significantly in:

- **Theme** (what the story is about)
- **Actions** (what happens)
- **Outcome** (what results from events)

To address this, we propose a **hybrid framework** that combines:

- Structured reasoning with Large Language Models (LLMs)
- Robust embedding-based similarity modeling

---

## 🔹 Method

### 🧩 Track A: ABNS-Agents

We introduce **Aspect-Based Narrative Similarity Agents (ABNS-Agents)**, a multi-agent reasoning framework that decomposes narrative comparison into structured components.

#### Pipeline

1. **Aspect Extraction**
    - Extract:
        - Theme
        - Actions
        - Outcome

2. **Aspect-level Comparison**
    - Compare candidate narratives against the anchor narrative for each aspect

3. **Decision Module**
    - Aggregate aspect-level judgments into a final decision

#### Decision Strategies

- **Weighted Decision**
    - Theme: 0.35
    - Actions: 0.45
    - Outcome: 0.20

- **Non-weighted Decision**
    - Majority voting across aspects

---

### 🔍 Track B: Embedding-based Similarity

We model narrative similarity using **contrastive sentence embeddings**.

#### Model Configuration

- Backbone: **Sup-SimCSE-RoBERTa-large**
- Pooling: CLS
- Objective: Supervised contrastive learning

#### Training Enhancements

- Hard negative sampling
- Auxiliary MLM loss (optional)
- Careful hyperparameter tuning

## 🏗️ Project Structure

```bash
NCL-SemEval2026-NarrSim/
├── semeval-2026-task-4-baselines/   # Baseline implementations and reference models
├── semeval-2026-task-4-datasets/    # Datasets including train/dev/test splits and preprocessing outputs
├── semeval-2026-task-4-models/      # Models for Track A (LLM agents) and Track B (embeddings), including training and evaluation scripts
├── semeval-2026-task-4-submit/      # Final prediction files for leaderboard submission
├── scripts/                         # Data preprocessing, analysis, and quality evaluation scripts
└── README.md
```

---

## 📈 Evaluation

### Main Metric

- **Accuracy**

### Auxiliary Evaluation

We additionally evaluate embedding quality using:

- STS12–STS16  
- STS-B  
- SICK-R  

---

## 🧪 Experiments

We conduct extensive experiments on:

- Supervised vs unsupervised contrastive learning  
- Impact of hard negatives  
- Data quality and split strategies  
- Weighted vs non-weighted agent decisions  

---

## 🔍 Key Findings

- **ABNS-Agent** improves narrative-level reasoning via structured decomposition, achieving **70.25% accuracy** with enhanced interpretability.
- **NSConE** provides robust and scalable similarity estimation through contrastive learning, reaching **68.5% accuracy**.
- The two approaches are **complementary**, combining structure-aware reasoning with efficient representation learning.
- Results highlight the importance of modeling **narrative structure beyond superficial semantics**.

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{xu-etal-2026-narrsim,
  title = {NCL&HKU-NarrSim at SemEval-2026 Task 4: Aspect-Based Agents and Supervised Contrastive Embeddings for Narrative Similarity},
  author = {Xu, Jianfei and Zhu, Ting and Chen, Mingyang and Liang, Huizhi},
  booktitle = {Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)},
  year = {2026},
  publisher = {Association for Computational Linguistics}
}
```
