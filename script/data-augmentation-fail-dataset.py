import json
import pandas as pd
from pathlib import Path


def load_jsonl_anchors(
        jsonl_path: str,
        anchor_key: str = "anchor"
) -> set:
    """
    从 jsonl 文件中读取所有 anchor，返回 set
    """
    anchors = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}")
            if anchor_key in obj and isinstance(obj[anchor_key], str):
                anchors.add(obj[anchor_key])
    return anchors


def filter_csv_by_missing_anchor(
        csv_path: str,
        jsonl_path: str,
        csv_anchor_col: str = "sent0",
        jsonl_anchor_key: str = "anchor_prototype",
        output_csv_path: str = "filtered_missing_anchor.csv",
):
    # 1. 读 CSV
    df = pd.read_csv(csv_path)

    if csv_anchor_col not in df.columns:
        raise ValueError(f"CSV column '{csv_anchor_col}' not found")

    # 2. 读 JSONL anchors
    jsonl_anchors = load_jsonl_anchors(
        jsonl_path,
        anchor_key=jsonl_anchor_key
    )

    print(f"Loaded {len(jsonl_anchors)} anchors from JSONL")

    # 3. 过滤：CSV.anchor_text 不在 JSONL.anchor 中
    mask_missing = ~df[csv_anchor_col].isin(jsonl_anchors)
    df_missing = df[mask_missing]

    # 4. 保存
    df_missing.to_csv(output_csv_path, index=False)

    print(f"Original CSV rows: {len(df)}")
    print(f"Missing-anchor rows: {len(df_missing)}")
    print(f"Saved to: {output_csv_path}")

    return df_missing


if __name__ == "__main__":
    csv_path = "../semeval-2026-task-4-datasets/semeval-2026-task-4-dev-v1/dev_track_a_train.csv"
    jsonl_path = "../semeval-2026-task-4-datasets/semeval-2026-task-4-dev-v1/dev_track_a_train_augmented.jsonl"

    filter_csv_by_missing_anchor(
        csv_path=csv_path,
        jsonl_path=jsonl_path,
        csv_anchor_col="sent0",   # CSV 中 anchor 列名
        jsonl_anchor_key="anchor_prototype", # JSONL 中 anchor 的 key（如果叫 anchor 就改成 "anchor"）
        output_csv_path="../semeval-2026-task-4-datasets/semeval-2026-task-4-dev-v1/dev_track_a_train_buchong.csv"
    )
