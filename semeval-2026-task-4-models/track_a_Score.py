import os, re
import json
from enum import Enum
from typing import List, Optional, Dict

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

# -----------------------------
# Schemas
# -----------------------------
class ResponseEnum(str, Enum):
    A = "A"
    B = "B"


class StoryAspects(BaseModel):
    theme: str
    action: List[str]
    outcome: str


class DecisionAspects(BaseModel):
    abstract_theme: bool
    course_of_action: bool
    outcomes: bool


class SimilarityDecision(BaseModel):
    closer: ResponseEnum
    aspects: DecisionAspects
    why: str


# -----------------------------
# Prompts
# -----------------------------
SYSTEM_DECIDE = (
    "You are an expert annotator for narrative similarity. "
    "You must decide whether Story A or Story B is narratively closer to the Anchor. "
    "You MUST base your decision ONLY on these aspects: "
    "Abstract Theme, Course of Action, Outcomes. "
    "Ignore names/time/place and writing style. "
    "Return STRICT JSON only."
)

MODEL_DECIDE = "gpt-4o"


# -----------------------------
# Helpers
# -----------------------------
def split_action(action_str: str) -> List[str]:
    """Convert 'e1 | e2 | e3' back to list."""
    if action_str is None or (isinstance(action_str, float) and pd.isna(action_str)):
        return []
    s = str(action_str).strip()
    if not s:
        return []
    return [x.strip() for x in s.split("|") if x.strip()]

def extract_json_object(text: str) -> str:
    """Extract the first JSON object from a string (handles ```json ... ``` blocks)."""
    if text is None:
        return ""
    t = text.strip()
    if not t:
        return ""

    # remove markdown fences if present
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s*```$", "", t).strip()

    # find first {...} span using a simple brace-matching scan
    start = t.find("{")
    if start == -1:
        return ""

    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return t[start:i+1].strip()

    return ""  # no balanced JSON found


def repair_to_json(client: OpenAI, bad_text: str) -> str:
    """Ask the model to output ONLY valid JSON for our schema."""
    resp = client.chat.completions.create(
        model=MODEL_DECIDE,
        messages=[
            {"role": "system", "content": "You convert text into STRICT valid JSON only. Output JSON only."},
            {
                "role": "user",
                "content": (
                    "Fix the following output into STRICT valid JSON with this exact schema:\n"
                    '{"closer":"A|B","aspects":{"abstract_theme":true/false,"course_of_action":true/false,"outcomes":true/false},"why":"short"}\n\n'
                    "Text to fix:\n"
                    f"{bad_text}"
                ),
            },
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

def normalize_decision_dict(d: dict) -> dict:
    """
    Aggressive normalization to ALWAYS produce:
    {
      "closer": "A" or "B",
      "aspects": {...},
      "why": "..."
    }
    """

    raw_text = json.dumps(d, ensure_ascii=False)

    # --------
    # 1) Extract closer (VERY aggressive)
    # --------
    closer = None

    # common direct keys
    for k in ["closer", "chosen_story", "answer", "choice", "winner"]:
        v = d.get(k)
        if isinstance(v, str):
            v = v.strip().upper()
            if v in ["A", "B"]:
                closer = v
                break
            if "A" in v and "B" not in v:
                closer = "A"
                break
            if "B" in v and "A" not in v:
                closer = "B"
                break

    # nested decision cases
    if closer is None:
        for v in d.values():
            if isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, str):
                        t = vv.upper()
                        if "A" in t and "B" not in t:
                            closer = "A"
                            break
                        if "B" in t and "A" not in t:
                            closer = "B"
                            break

    # regex fallback from full text
    if closer is None:
        if re.search(r"\bA\b", raw_text) and not re.search(r"\bB\b", raw_text):
            closer = "A"
        elif re.search(r"\bB\b", raw_text) and not re.search(r"\bA\b", raw_text):
            closer = "B"

    # ULTIMATE fallback (never crash)
    if closer is None:
        closer = "A"   # deterministic fallback (or random.choice(["A","B"]))

    # --------
    # 2) Normalize aspects
    # --------
    flags = {"abstract_theme": False, "course_of_action": False, "outcomes": False}

    aspects = d.get("aspects") or d.get("similarity_aspects")
    if isinstance(aspects, dict):
        for k in flags:
            if k in aspects:
                flags[k] = bool(aspects[k])
    elif isinstance(aspects, list):
        norm = {str(x).lower() for x in aspects}
        if "theme" in norm or "abstract_theme" in norm:
            flags["abstract_theme"] = True
        if "action" in norm or "course_of_action" in norm:
            flags["course_of_action"] = True
        if "outcome" in norm or "outcomes" in norm:
            flags["outcomes"] = True

    # --------
    # 3) Why
    # --------
    why = (
        d.get("why")
        or d.get("reason")
        or d.get("explanation")
        or "Decision based on narrative aspects."
    )

    if not isinstance(why, str):
        why = str(why)

    return {
        "closer": closer,
        "aspects": flags,
        "why": why
    }

def decide_closer_aspects_only(
    client: OpenAI,
    anchor: StoryAspects,
    a: StoryAspects,
    b: StoryAspects,
) -> SimilarityDecision:
    user_content = (
        "Anchor aspects:\n"
        f"{anchor.model_dump_json(ensure_ascii=False)}\n\n"
        "Story A aspects:\n"
        f"{a.model_dump_json(ensure_ascii=False)}\n\n"
        "Story B aspects:\n"
        f"{b.model_dump_json(ensure_ascii=False)}\n\n"
        "Task:\n"
        "1) Decide which story (A or B) is narratively closer to Anchor.\n"
        "2) Mark which aspects mainly drive the similarity between Anchor and the chosen story "
        "(abstract_theme/course_of_action/outcomes). Multiple can be true.\n\n"
        "Return STRICT JSON only (no markdown, no extra text)."
    )

    resp = client.chat.completions.create(
        model=MODEL_DECIDE,
        messages=[
            {"role": "system", "content": SYSTEM_DECIDE},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )

    content = (resp.choices[0].message.content or "").strip()
    if not content:
        # empty output -> force a repair round using the whole response object text
        content = repair_to_json(client, "EMPTY_OUTPUT")

    json_str = extract_json_object(content)
    if not json_str:
        # try repair once
        fixed = repair_to_json(client, content)
        json_str = extract_json_object(fixed)

    if not json_str:
        raise ValueError(f"Could not extract JSON.\nRaw content:\n{content}")

    raw = json.loads(json_str)
    data = normalize_decision_dict(raw)
    return SimilarityDecision(**data)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    input_csv = "outputs/test_track_a_aspects_4omini.csv"
    out_jsonl = "outputs/test_track_a_pred_4o.jsonl"
    out_debug_csv = "outputs/test_track_a_pred_with_debug_4o.csv"

    os.makedirs("outputs", exist_ok=True)
    client = OpenAI()

    df = pd.read_csv(input_csv)

    preds = []
    whys = []
    aspect_flags = []

    pbar = tqdm(total=len(df) * 3, desc="LLM calls (anchor/A/B)")

    for i, row in df.iterrows():
        anchor = StoryAspects(
            theme=str(row["anchor_theme"]), 
            action=split_action(row["anchor_action"]),
            outcome=str(row["anchor_outcome"]),
        )
        pbar.update(1)

        a = StoryAspects(
            theme=str(row["a_theme"]),
            action=split_action(row["a_action"]),
            outcome=str(row["a_outcome"]),
        )
        pbar.update(1)

        b = StoryAspects(
            theme=str(row["b_theme"]),
            action=split_action(row["b_action"]),
            outcome=str(row["b_outcome"]),
        )
        pbar.update(1)

        decision = decide_closer_aspects_only(client, anchor, a, b)

        # Convert to required bool: text_a_is_closer
        text_a_is_closer_pred = (decision.closer == ResponseEnum.A)

        preds.append(text_a_is_closer_pred)
        whys.append(decision.why)
        aspect_flags.append(decision.aspects.model_dump())

    df["predicted_text_a_is_closer"] = preds
    df["debug_why"] = whys
    df["debug_aspects"] = [json.dumps(x, ensure_ascii=False) for x in aspect_flags]

    # If dev has gold label, compute accuracy
    if "text_a_is_closer" in df.columns:
        acc = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
        print(f"Accuracy (aspects-only): {acc:.4f}")

    # Write JSONL submission-style: overwrite text_a_is_closer with predictions
    out_df = df.copy()
    out_df["text_a_is_closer"] = out_df["predicted_text_a_is_closer"]
    # drop helper columns if you want a cleaner submission file
    out_df = out_df.drop(columns=["predicted_text_a_is_closer"], errors="ignore")
    out_df.to_json(out_jsonl, orient="records", lines=True, force_ascii=False)
    print(f"Saved submission jsonl: {out_jsonl}")

    # Save debug CSV (optional)
    df.to_csv(out_debug_csv, index=False, encoding="utf-8-sig")
    print(f"Saved debug csv: {out_debug_csv}")
