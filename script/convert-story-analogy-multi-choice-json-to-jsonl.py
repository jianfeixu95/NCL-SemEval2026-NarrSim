import json
from pathlib import Path

INPUT_PATH = Path("../semeval-2026-task-4-datasets/story-analogy/storyanalogy_multiple_choice.json")
OUTPUT_PATH = Path("../semeval-2026-task-4-datasets/story-analogy/story_analogy_flat.jsonl")

def main():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    out_lines = 0
    skipped = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for ex_id, ex in enumerate(data):
            source = ex.get("source")
            choices = ex.get("choices", [])
            answer_idx = ex.get("answer")
            types = ex.get("types", [])

            # basic sanity checks
            if (
                    source is None
                    or not isinstance(choices, list)
                    or not isinstance(types, list)
                    or answer_idx is None
                    or not (0 <= int(answer_idx) < len(choices))
                    or len(types) != len(choices)
            ):
                skipped += 1
                continue

            anchor_text = source
            text_a = choices[int(answer_idx)]

            # For each distractor of type random/noun => make one row
            for i, (ch, t) in enumerate(zip(choices, types)):
                if i == int(answer_idx):
                    continue
                if t not in ("random", "noun", "target"):
                    continue
                if t == "target":
                    # In MC setting, target should be the answer; skip if it appears elsewhere unexpectedly
                    continue

                record = {
                    "anchor_text": anchor_text,
                    "text_a": text_a,
                    "text_b": ch,
                    "text_a_is_closer": True,
                    "type": t,          # random / noun
                    "ex_id": ex_id,     # optional: keep traceability
                    "b_index": i        # optional: which choice became text_b
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_lines += 1

    print(f"Done. Wrote {out_lines} lines to: {OUTPUT_PATH}")
    if skipped:
        print(f"Skipped {skipped} malformed examples.")

if __name__ == "__main__":
    main()
