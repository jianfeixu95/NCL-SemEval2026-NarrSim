import json
import sys


def convert_track_a_data_to_track_b_data(input_path, output_path):

    texts = set()   # 用集合自动去重
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue

            obj = json.loads(line)

            texts.add(str(obj["anchor_text"]).strip())
            texts.add(str(obj["text_a"]).strip())
            texts.add(str(obj["text_b"]).strip())

    print("Unique texts:", len(texts))

    with open(output_path, "w", encoding="utf-8") as fout:
        for t in texts:
            fout.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    print("Saved to:", output_path)

if __name__ == '__main__':
    input_path = "../semeval-2026-task-4-datasets/semeval-2026-task-4-train-v1/train_track_a.jsonl"
    output_path = "../semeval-2026-task-4-datasets/semeval-2026-task-4-train-v1/train_track_b.jsonl"
    convert_track_a_data_to_track_b_data(input_path, output_path)