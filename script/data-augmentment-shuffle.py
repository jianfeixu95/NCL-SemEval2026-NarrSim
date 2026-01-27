import json
from pathlib import Path


def expand_triplet_jsonl(
        input_jsonl: str,
        output_jsonl: str,
):
    input_jsonl = Path(input_jsonl)
    output_jsonl = Path(output_jsonl)

    with input_jsonl.open("r", encoding="utf-8") as fin, \
            output_jsonl.open("w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}")

            # 必须字段检查
            required_fields = [
                "anchor_text", "text_a", "text_b",
                "anchor_prototype", "text_a_prototype", "text_b_prototype",
            ]
            for f in required_fields:
                if f not in obj:
                    raise ValueError(f"Missing field '{f}' at line {line_no}")

            # 把原始字段取出来
            anchor_text = obj["anchor_text"]
            text_a = obj["text_a"]
            text_b = obj["text_b"]

            anchor_proto = obj["anchor_prototype"]
            text_a_proto = obj["text_a_prototype"]
            text_b_proto = obj["text_b_prototype"]

            expanded_samples = [
                # 1
                {
                    "sent0": anchor_text,
                    "sent1": text_a,
                    "hard_neg": text_b,
                },
                # 2
                {
                    "sent0": anchor_proto,
                    "sent1": text_a,
                    "hard_neg": text_b,
                },
                # 3
                {
                    "sent0": anchor_text,
                    "sent1": text_a,
                    "hard_neg": text_b_proto,
                },
                # 4
                {
                    "sent0": anchor_text,
                    "sent1": text_a_proto,
                    "hard_neg": text_b,
                },
                # 5
                {
                    "sent0": anchor_proto,
                    "sent1": text_a_proto,
                    "hard_neg": text_b_proto,
                },
            ]

            # 写入 jsonl
            for sample in expanded_samples:
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Done. Expanded jsonl written to: {output_jsonl}")

if __name__ == '__main__':

    expand_triplet_jsonl(
        input_jsonl="../semeval-2026-task-4-datasets/semeval-2026-task-4-dev-v1/dev_track_a_train_augmented.jsonl",
        output_jsonl="../semeval-2026-task-4-datasets/semeval-2026-task-4-dev-v1/dev_track_a_train_augmented_shuffled.jsonl",
    )
