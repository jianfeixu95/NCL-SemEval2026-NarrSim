import os
import csv
import json
import time
from pathlib import Path
from openai import OpenAI
from typing import Dict, Iterator

OPENAI_API_KEY="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"
client = OpenAI(api_key=OPENAI_API_KEY)

# PROMPT_TEMPLATE = r"""
# 你是一个专业 narrative 数据构造器。你将基于三个“极简narrative原型句”（anchor_text / text_a / text_b）分别生成三段长篇叙事文本，用于 narrative 三元组语义相似度判断任务。
#
# ### 核心原则（必须遵守）
# 1) 不改变每个原型句的“核心语义骨架”，只做优化、扩充、细化与叙事化；不得引入与原型句相矛盾的设定。
# 2) 每个 narrative 必须自然融合以下三方面内容为“一个连续段落”（不要分点、不要小标题）：
#    - Abstract Theme：故事的思想、隐喻、动机与内在逻辑
#    - Course of Action：关键事件序列、反复出现的模式、例如明显转折或临界时刻
#    - Outcomes：结果、影响、以及对主题的呼应/反思
# 3) 语义关系约束：
#    - anchor_text：作为参照中心的 narrative。
#    - text_a：在主题、语义上要与 anchor_text 高度相似。
#    - text_b：表层可共享部分表达，但在核心主题或结局上必须显著偏离 anchor_text。
# 4) 长度与输出格式：
#    - 每段约 750-950 words（允许合理浮动，但不要短于 700 words，也不要超过 1100 words）。
#    - You must output valid JSON only. Do NOT include explanations, comments, markdown, or extra text.
#    - The JSON object must contain exactly three keys, spelled exactly as follows:
# {{
#   "anchor_text": "...",
#   "text_a": "...",
#   "text_b": "..."
# }}
#    - Each value must be a single English paragraph enclosed in double quotes.
#    - Do NOT include newline characters inside the text values.
#    - Do NOT include any extra keys.
#
# ### 输入原型句（不要原样照抄，要扩写为长篇叙事）
# anchor narrative: [{anchor}]
# text_a narrative: [{text_a}]
# text_b narrative: [{text_b}]
#
# 现在开始生成，严格遵守“仅三行输出”的格式要求。
# """.strip()

PROMPT_TEMPLATE = r"""
You are a professional narrative data constructor. Based on three narrative prototype sentences (anchor_text / text_a / text_b), you will generate three long-form narratives for a narrative triplet semantic similarity task.

### Core Principles (Must Follow)
1) Do NOT alter the core semantic skeleton of each prototype sentence. You may only refine, expand, elaborate, and narrativize it. Do NOT introduce assumptions or developments that contradict the original prototype.
2) Each narrative must naturally integrate the following three aspects into one single continuous paragraph (do NOT use bullet points, section headers, or explicit labels):
   - Abstract Theme: the underlying ideas, metaphors, motivations, and internal logic of the story
   - Course of Action: the sequence of central developments, recurring patterns, and at least one clear turning point or threshold moment
   - Outcomes: the results, consequences, and how they reflect or reframe the abstract theme
3) Semantic relationship constraints:
   - anchor_text: serves as the semantic reference narrative.
   - text_a: must be highly similar to anchor_text in theme and overall meaning.
   - text_b: may share surface-level expressions with anchor_text, but must clearly diverge in core theme and/or final outcome.
4) Length and output format:
   - Each narrative should be approximately 750–950 words (reasonable variation allowed, but do NOT go below 700 words or above 1100 words).
   - You must output valid JSON only. Do NOT include explanations, comments, markdown, or any extra text. The JSON object must contain exactly three keys, spelled exactly as follows:
{{
  "anchor_text": "...",
  "text_a": "...",
  "text_b": "..."
}}
   - Each value must be a single English paragraph enclosed in double quotes.
   - Do NOT include newline characters inside the text values.
   - Do NOT include any extra keys.

### Input Prototype Sentences (Do NOT copy verbatim; expand them into long-form narratives)
anchor narrative: [{anchor}]
text_a narrative: [{text_a}]
text_b narrative: [{text_b}]

Now begin generation and strictly comply with the requirement to output only the three JSON fields.
""".strip()

def iter_csv(path: Path) -> Iterator[Dict[str, str]]:
    """逐行安全读取 CSV，每行返回一个 dict（column_name -> value）"""
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header row")

        for line_no, row in enumerate(reader, start=2):  # header 是第 1 行
            # 跳过全空行
            if all(v is None or str(v).strip() == "" for v in row.values()):
                continue

            # 清理字段
            cleaned = {
                k: (v.strip() if isinstance(v, str) else v)
                for k, v in row.items()
            }

            yield cleaned

def append_jsonl(path: Path, obj: dict):
    """逐行追加写入 jsonl（不中断已有内容）"""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ====== 配置 ======
INPUT_PATH = Path("../semeval-2026-task-4-datasets/semeval-2026-task-4-dev-v1/dev_track_a_train_buchong.csv")
OUTPUT_PATH = Path("../semeval-2026-task-4-datasets/semeval-2026-task-4-dev-v1/dev_track_a_train_augmented.jsonl")

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.8
MAX_TOKENS = 5000
SLEEP_SECONDS = 1.0

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

for example in iter_csv(INPUT_PATH):

    anchor = example["sent0"]
    text_a = example["sent1"]
    text_b = example["hard_neg"]

    # 构造 prompt
    prompt = PROMPT_TEMPLATE.format(
        anchor=anchor,
        text_a=text_a,
        text_b=text_b,
    )

    # 调用模型
    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
    )

    raw_output = response.output_text.strip()

    try:
        generated = json.loads(raw_output)
    except json.JSONDecodeError as e:
        print("[WARN] JSON parse failed")
        print(raw_output[:500])
        continue

    out_record = {
        "model_name": MODEL_NAME,
        "anchor_prototype": anchor,
        "text_a_prototype": text_a,
        "text_b_prototype": text_b,
        "anchor_text": generated["anchor_text"],
        "text_a": generated["text_a"],
        "text_b": generated["text_b"],
    }

    append_jsonl(OUTPUT_PATH, out_record)

    print(f"[OK] processed")
    time.sleep(SLEEP_SECONDS)

print("All done.")

