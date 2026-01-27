import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import os
import json

# ================= 配置区域 =================
# 输入数据路径 (请确保这是你处理过的匿名化数据路径)
INPUT_FILENAME = "dev_track_a_processed.jsonl" 
DATA_DIR = "tell me again"  # 假设你的数据在 data 文件夹下
INPUT_PATH = os.path.join(DATA_DIR, INPUT_FILENAME)

# 输出结果路径
OUTPUT_FILE = "output/track_a_labse_prediction.jsonl"
# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# 模型名称 (Hugging Face Hub ID)
MODEL_NAME = "sentence-transformers/LaBSE"
# ===========================================

def load_data(file_path):
    """
    严谨地加载 JSONL 数据，使用 Pandas 处理便于分析
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误：找不到输入文件 {file_path}")
    
    print(f"正在加载数据: {file_path} ...")
    # 读取 jsonl 文件
    df = pd.read_json(file_path, lines=True)
    print(f"数据加载完成，共 {len(df)} 条样本。")
    return df

def run_experiment():
    # 1. 设置设备 (GPU 优先，无 GPU 则使用 CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在初始化环境，使用设备: {device.upper()}")

    # 2. 加载模型 (这一步会自动从 Hugging Face 下载 LaBSE 模型到本地缓存)
    print(f"正在加载模型 {MODEL_NAME} (首次运行需要下载)...")
    try:
        model = SentenceTransformer(MODEL_NAME, device=device)
    except Exception as e:
        print(f"模型加载失败，请检查网络连接或 Hugging Face 访问权限。\n错误信息: {e}")
        return

    # 3. 加载数据
    print(f"正在读取实验文件: 【{INPUT_FILENAME}】") # 这里会在控制台明确打印文件名
    df = load_data(INPUT_PATH)

    # 4. 准备预测结果列表
    predictions = []
    correct_count = 0
    total_count = 0

    print("开始进行推理计算...")
    
    # 使用 tqdm 显示进度条
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"正在处理 {INPUT_FILENAME}"):
        # 获取匿名化后的文本字段
        # 注意：使用 .get() 方法防止字段缺失报错，这是严谨性的体现
        anchor_text = row.get("anchor_text_anon", "")
        text_a = row.get("text_a_anon", "")
        text_b = row.get("text_b_anon", "")
        
        # 确保文本不为空，如果为空给一个占位符，防止模型报错
        if not anchor_text: anchor_text = " "
        if not text_a: text_a = " "
        if not text_b: text_b = " "

        # 将文本编码为向量 (Embeddings)
        # convert_to_tensor=True 返回 PyTorch tensor，便于在 GPU 上计算
        embeddings = model.encode([anchor_text, text_a, text_b], convert_to_tensor=True)
        
        anchor_emb = embeddings[0]
        text_a_emb = embeddings[1]
        text_b_emb = embeddings[2]

        # 计算余弦相似度 (Cosine Similarity)
        # util.pytorch_cos_sim 返回的是一个矩阵，我们取 [0][0] 标量值
        score_a = util.pytorch_cos_sim(anchor_emb, text_a_emb).item()
        score_b = util.pytorch_cos_sim(anchor_emb, text_b_emb).item()

        # 做出预测：比较分数高低
        # 如果 Score A > Score B，则预测 A 更接近 (True)，否则为 False
        predicted_a_is_closer = score_a > score_b

        # 获取真实标签 (Ground Truth)
        # 原始数据中包含 'text_a_is_closer' 字段 (True/False)
        if "text_a_is_closer" in row:
            ground_truth = row["text_a_is_closer"]
            if predicted_a_is_closer == ground_truth:
                correct_count += 1
            total_count += 1
        
        # 保存结果行时，额外加入文件名信息，方便溯源
        result_row = row.to_dict()
        result_row["experiment_source_file"] = INPUT_FILENAME  # <--- 注入文件名
        result_row["score_a"] = score_a
        result_row["score_b"] = score_b
        result_row["predicted_text_a_is_closer"] = predicted_a_is_closer
        
        predictions.append(result_row)

    # 5. 计算并打印准确率
    if total_count > 0:
        accuracy = correct_count / total_count
        print(f"\n{'='*50}")
        print(f"实验完成！")
        print(f"模型: {MODEL_NAME}")
        print(f"实验数据: {INPUT_FILENAME}")
        print(f"准确率 (Accuracy): {accuracy:.4f} ({correct_count}/{total_count})")
        print(f"{'='*50}\n")
    else:
        print("\n警告：未在数据中找到 'text_a_is_closer' 标签，无法计算准确率。仅保存预测结果。")

    # 6. 保存预测结果到文件
    print(f"正在保存结果到: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("保存完成。")

if __name__ == "__main__":
    run_experiment()