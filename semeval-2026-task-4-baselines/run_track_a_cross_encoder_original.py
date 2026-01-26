import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import os
import json

# ================= 配置区域 =================
# 1. 输入数据路径
DATA_DIR = "tell me again"  # 数据存放目录
INPUT_FILENAME = "dev_track_a_processed.jsonl"
INPUT_PATH = os.path.join(DATA_DIR, INPUT_FILENAME)

# 2. 输出结果路径
OUTPUT_DIR = "output"
OUTPUT_FILENAME = f"track_a_cross_encoder_results_{INPUT_FILENAME}"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# 3. 模型配置
# 推荐使用 stsb-distilroberta-base 或 ms-marco-MiniLM-L-6-v2
MODEL_NAME = "cross-encoder/stsb-distilroberta-base"

# 4. 字段配置 (根据需要切换 "text_a" 或 "text_a_anon")
# 如果想跑原始数据，保持现状；如果想跑匿名化数据，请在字段名后加 "_anon"
FIELD_ANCHOR = "anchor_text_anon" 
FIELD_A = "text_a_anon"
FIELD_B = "text_b_anon"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ===============================================

def load_data(file_path):
    """
    严谨加载 JSONL 数据并进行初步校验
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"【错误】无法在路径找到数据文件: {file_path}")
    
    print(f"[*] 正在加载数据集: {file_path}")
    df = pd.read_json(file_path, lines=True)
    print(f"[*] 数据加载完成，样本总数: {len(df)}")
    return df

def run_experiment():
    # 1. 设备检测
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 运行设备确认: {device.upper()}")

    # 2. 初始化 Cross-Encoder 模型
    # Cross-Encoder 直接输出相似度分数（通常在 0-1 之间）
    print(f"[*] 正在初始化 Cross-Encoder 模型: {MODEL_NAME}")
    try:
        model = CrossEncoder(MODEL_NAME, device=device)
    except Exception as e:
        print(f"【错误】模型加载失败，请检查网络或模型名称: {e}")
        return

    # 3. 加载数据
    df = load_data(INPUT_PATH)

    # 4. 实验推理
    predictions = []
    correct_count = 0
    total_count = 0

    print(f"[*] 开始进行深度交互推理 (Cross-Encoding)...")
    
    # 使用迭代器并开启进度条
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {INPUT_FILENAME}"):
        # 提取文本
        anchor = str(row.get(FIELD_ANCHOR, ""))
        text_a = str(row.get(FIELD_A, ""))
        text_b = str(row.get(FIELD_B, ""))

        # Cross-Encoder 核心逻辑：
        # 将 (Anchor, A) 和 (Anchor, B) 构造为输入对进行打分
        # 模型会同时处理这两个句子，捕捉细节层面的因果和情节关联
        try:
            # 获取分数的列表
            scores = model.predict([
                (anchor, text_a), 
                (anchor, text_b)
            ])
            
            score_a = float(scores[0])
            score_b = float(scores[1])
        except Exception as e:
            print(f"【跳过】第 {index} 行发生推理错误: {e}")
            continue

        # 判定逻辑
        predicted_a_is_closer = score_a > score_b

        # 准确率评估 (如果存在标签)
        if "text_a_is_closer" in row:
            ground_truth = bool(row["text_a_is_closer"])
            if predicted_a_is_closer == ground_truth:
                correct_count += 1
            total_count += 1
        
        # 构建严谨的结果记录
        result_entry = row.to_dict()
        result_entry.update({
            "experiment_model": MODEL_NAME,
            "experiment_source_file": INPUT_FILENAME,
            "score_a": score_a,
            "score_b": score_b,
            "predicted_text_a_is_closer": predicted_a_is_closer
        })
        predictions.append(result_entry)

    # 5. 打印实验报告
    print("\n" + "="*60)
    print(f"实验报告总结")
    print(f"模型名称: {MODEL_NAME}")
    print(f"数据集:   {INPUT_FILENAME}")
    if total_count > 0:
        accuracy = correct_count / total_count
        print(f"准确率 (Accuracy): {accuracy:.4f} ({correct_count}/{total_count})")
    print("="*60 + "\n")

    # 6. 持久化存储
    print(f"[*] 正在保存实验详细结果至: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("[*] 结果保存完成。")

if __name__ == "__main__":
    run_experiment()