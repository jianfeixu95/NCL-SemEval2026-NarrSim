import json

def convert_train_data_to_standard(input_path, output_path):

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            obj = json.loads(line)

            new_obj = {
                "anchor_text": obj["anchor_text"],
                "text_a": obj["text_a"],
                "text_b": obj["text_b"],
                "text_a_is_closer": obj["text_a_is_closer"]
            }

            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

    print("Done! Saved to:", output_path)

if __name__ == '__main__':
    input_path = "../semeval-2026-task-4-datasets/semeval-2026-task-4-train-v1/synthetic_data_for_classification.jsonl"
    output_path = "../semeval-2026-task-4-datasets/semeval-2026-task-4-train-v1/train_track_a.jsonl"
    convert_train_data_to_standard(input_path, output_path)