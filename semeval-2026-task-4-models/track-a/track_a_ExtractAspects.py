import os
import json
import hashlib
from typing import List, Dict

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
import openai
from tqdm import tqdm

# -----------------------------
# Schema
# -----------------------------
class StoryAspects(BaseModel):
    theme: str = Field(..., description="1 sentence abstract theme. No names/time/place.")
    action: List[str] = Field(..., description="3 key events in chronological order.")
    outcome: str = Field(..., description="1 sentence final outcome/end state only.")


SYSTEM_EXTRACT = (
    "You are an expert annotator for narrative similarity. "
    "Extract ONLY the three aspects:\n"
    "1) Abstract Theme: high-level theme, ignore names/time/place and writing style.\n"
    "2) Course of Action: key events and their order.\n"
    "3) Outcomes: final outcome/end state only.\n"
    "Return STRICT JSON that matches the provided schema."
)

MODEL_EXTRACT = "gpt-4o-mini" #"gpt-4o-mini"


# -----------------------------
# Cache utils
# -----------------------------
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def load_cache(path: str) -> Dict[str, dict]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache: Dict[str, dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# -----------------------------
# LLM extraction
# -----------------------------
def extract_aspects(client: OpenAI, story_text: str) -> StoryAspects:
    
    max_tokens=500
    # print("DEBUG max_tokens:", max_tokens, type(max_tokens))

    for _ in range(4):  # 500 -> 900 -> 1620 -> 2916
        try:
            completion = client.chat.completions.parse(
                model=MODEL_EXTRACT,
                max_tokens=max_tokens,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_EXTRACT},
                    {
                        "role": "user", "content": (
                            "Story:\n"
                            f"{story_text}\n\n"
                            "Output STRICT JSON with schema:\n"
                            '{"theme":"...","action":["..."],"outcome":"..."}\n\n'
                            "Rules: theme=1 sentence; action=EXACTLY 3 short events; outcome=1 sentence."
                              ),
                    },
                ],
                response_format=StoryAspects,
            )
            return completion.choices[0].message.parsed
        
        except openai.LengthFinishReasonError:
            max_tokens = int(max_tokens * 1.8)
            continue
    raise RuntimeError("Failed to extract aspects after multiple retries.") 


def get_aspects_cached(client: OpenAI, text: str, cache: Dict[str, dict]) -> StoryAspects:
    k = sha1(text.strip())
    if k in cache:
        return StoryAspects(**cache[k])
    asp = extract_aspects(client, text)
    cache[k] = asp.model_dump()
    return asp


# -----------------------------
# Main: read -> extract -> save CSV
# -----------------------------
if __name__ == "__main__":
    input_path = "data/test_track_a.jsonl"     # 需换成 train/test/dev
    out_csv = "outputs/test_track_a_aspects_4omini.csv"
    cache_path = "cache/test_story_aspects_cache_4omini.json"

    os.makedirs("outputs", exist_ok=True)
    # os.makedirs("cache", exist_ok=True)
    client = OpenAI()

    df = pd.read_json(input_path, lines=True)

    # cache = load_cache(cache_path)

    # 逐行抽取
    anchor_theme, anchor_action, anchor_outcome = [], [], []
    a_theme, a_action, a_outcome = [], [], []
    b_theme, b_action, b_outcome = [], [], []

    pbar = tqdm(total=len(df) * 3, desc="LLM calls (anchor/A/B)")

    for i, row in df.iterrows():
        try:
            anchor = extract_aspects(client, row["anchor_text"]); pbar.update(1)
            a = extract_aspects(client, row["text_a"]); pbar.update(1)
            b = extract_aspects(client, row["text_b"]); pbar.update(1)

            anchor_theme.append(anchor.theme)
            anchor_action.append(" | ".join(anchor.action))   # CSV里用分隔符存 list
            anchor_outcome.append(anchor.outcome)

            a_theme.append(a.theme)
            a_action.append(" | ".join(a.action))
            a_outcome.append(a.outcome)

            b_theme.append(b.theme)
            b_action.append(" | ".join(b.action))
            b_outcome.append(b.outcome)
        
            
        except Exception as e:
            print("FAILED at row:", i)
            print("lens:", len(row["anchor_text"]), len(row["text_a"]), len(row["text_b"]))
            print("error:", repr(e))
            raise
    pbar.close()

    # 追加到原 df
    df["anchor_theme"] = anchor_theme
    df["anchor_action"] = anchor_action
    df["anchor_outcome"] = anchor_outcome

    df["a_theme"] = a_theme
    df["a_action"] = a_action
    df["a_outcome"] = a_outcome

    df["b_theme"] = b_theme
    df["b_action"] = b_action
    df["b_outcome"] = b_outcome

    # 保存
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    save_cache(cache, cache_path)

    print(f"Saved: {out_csv}")
    print(f"Cache : {cache_path}")
