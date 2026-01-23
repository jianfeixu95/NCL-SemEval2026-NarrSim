import json
import csv


def convert_jsonl_to_csv(input_jsonl_file, output_csv_file):
    with open(input_jsonl_file, "r", encoding="utf-8") as fin, \
         open(output_csv_file, "w", encoding="utf-8", newline="") as fout:

        writer = csv.writer(fout)
        # 表头
        writer.writerow(["anchor_text", "text_a", "text_b", "text_a_is_closer"])

        for line in fin:
            if not line.strip():
                continue

            obj = json.loads(line)

            anchor = obj["anchor_text"]
            text_a = obj["text_a"]
            text_b = obj["text_b"]
            text_a_is_closer = obj["text_a_is_closer"]

            if anchor is None or anchor == "":
                print(line)
                continue

            writer.writerow([anchor, text_a, text_b, text_a_is_closer])

        print("Done! Saved to:", output_csv_file)


if __name__ == '__main__':

    input_jsonl_file = "../data/dev_track_a.jsonl"  # 你的jsonl文件
    output_csv_file = "../data/dev_track_a_test.csv"  # 输出csv文件
    convert_jsonl_to_csv(input_jsonl_file, output_csv_file)
