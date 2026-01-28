import json

INPUT_FILE = "../semeval-2026-task-4-datasets/story-analogy/story_analogy_flat.jsonl"      # 原 jsonl
OUTPUT_FILE = "../semeval-2026-task-4-datasets/story-analogy/story_analogy_noun.jsonl"  # 新 jsonl（只保留 noun）

def main():
    kept = 0
    total = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
            open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue

            ex = json.loads(line)
            if ex.get("type") == "noun":
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Done.")
    print(f"Total lines read: {total}")
    print(f"Lines kept (type == noun): {kept}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
