import json
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-large")  # 或你用的模型

lengths = []

with open("../data/dev_track_a.jsonl", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        for k in ["anchor_text", "text_a", "text_b"]:
            if k in ex:
                if ex[k] is None:
                    continue
                ids = tokenizer(
                    ex[k],
                    add_special_tokens=True,
                    truncation=False
                )["input_ids"]
                lengths.append(len(ids))

for p in [50, 75, 90, 95, 99]:
    print(f"{p}th percentile:", np.percentile(lengths, p))

print("max length:", max(lengths))
