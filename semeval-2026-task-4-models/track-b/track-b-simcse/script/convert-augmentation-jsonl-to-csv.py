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

            sent0 = obj["anchor_text"]
            sent1 = obj["text_a"]
            hard_neg = obj["text_b"]

            if sent0 is None or sent0 == "":
                print(line)
                continue

            writer.writerow([sent0, sent1, hard_neg])

        print("Done! Saved to:", output_csv_file)


if __name__ == '__main__':

    input_jsonl_file = "../data/train_track_a_augmented.jsonl"  # 你的jsonl文件
    output_csv_file = "../data/train_track_a_augmented.csv"  # 输出csv文件
    convert_jsonl_to_csv(input_jsonl_file, output_csv_file)
