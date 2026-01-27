import json
import numpy as np
import pandas as pd

jsonl_path = "../semeval-2026-task-4-datasets/semeval-2026-task-4-dev-v1/dev_track_a.jsonl"  # 改成你的路径

anchor_lens = []
a_lens = []
b_lens = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        anchor_lens.append(len(ex["anchor_text"]))
        a_lens.append(len(ex["text_a"]))
        b_lens.append(len(ex["text_b"]))

def stat(name, arr):
    list_arr = pd.Series(arr)
    arr = np.array(arr)
    print(f"{name}:")
    print(f"  mean = {arr.mean():.2f}")
    print(f"  min  = {arr.min()}")
    print(f"  max  = {arr.max()}")
    print(f"  std  = {arr.std():.2f}")
    print(f"  quantile  = {list_arr.quantile([0.25, 0.5, 0.75, 0.9, 0.95])}")
    print()

stat("anchor_text length", anchor_lens)
stat("text_a length", a_lens)
stat("text_b length", b_lens)
