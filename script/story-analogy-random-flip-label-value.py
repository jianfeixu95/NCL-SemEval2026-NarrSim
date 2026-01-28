import json
import random

INPUT_FILE =  "../semeval-2026-task-4-datasets/story-analogy/story_analogy_flat.jsonl"
OUTPUT_FILE =  "../semeval-2026-task-4-datasets/story-analogy/story_analogy_flat_flipped.jsonl"

# 随机种子（保证可复现；不需要可删）
random.seed(42)

def main():
    total = 0
    flipped = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
            open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            ex = json.loads(line)
            total += 1

            # 随机决定标签
            new_label = random.choice([True, False])

            if new_label is False:
                # 交换 text_a 和 text_b
                ex["text_a"], ex["text_b"] = ex["text_b"], ex["text_a"]
                ex["text_a_is_closer"] = False
                flipped += 1
            else:
                ex["text_a_is_closer"] = True

            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Done.")
    print(f"Total records processed: {total}")
    print(f"Label flipped to False (text swapped): {flipped}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
