import json
import csv


def convert_jsonl_to_csv(input_jsonl_file, output_csv_file):
    with open(input_jsonl_file, "r", encoding="utf-8") as fin, \
         open(output_csv_file, "w", encoding="utf-8", newline="") as fout:

        writer = csv.writer(fout)
        # 表头
        writer.writerow(["sent0", "sent1", "hard_neg"])

        for line in fin:
            if not line.strip():
                continue

            obj = json.loads(line)

            anchor = obj["anchor_text"]
            text_a = obj["text_a"]
            text_b = obj["text_b"]

            if anchor is None or anchor == "":
                print(line)
                continue

            if obj["text_a_is_closer"]:
                closer = text_a
                farther = text_b
            else:
                closer = text_b
                farther = text_a

            writer.writerow([anchor, closer, farther])

        print("Done! Saved to:", output_csv_file)


if __name__ == '__main__':

    input_jsonl_file = "../../../../semeval-2026-task-4-datasets/semeval-2026-task-4-train-v1/train_track_a_processed.jsonl"  # 你的jsonl文件
    output_csv_file = "../../../../semeval-2026-task-4-datasets/semeval-2026-task-4-train-v1/train_track_a_processed.csv"  # 输出csv文件
    convert_jsonl_to_csv(input_jsonl_file, output_csv_file)
