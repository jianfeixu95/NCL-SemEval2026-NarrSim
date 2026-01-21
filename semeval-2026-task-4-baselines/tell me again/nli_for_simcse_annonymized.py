from flair.data import Sentence
from flair.models import SequenceTagger
import os
from tqdm import tqdm  # 导入进度条库
import pandas as pd

# 1. 加载模型
tagger = SequenceTagger.load('flair/ner-english-large')

def anonymize_text(text):
    """
    根据论文 5.1 节逻辑，实现实体重命名
    """
    if not text:
        return ""
    sentence = Sentence(text)
    tagger.predict(sentence)
    processed_text = text
    # 替换 PER (人名)、LOC (地点) 和 ORG (机构)
    for entity in sentence.get_spans('ner'):
        if entity.tag == 'PER':
            processed_text = processed_text.replace(entity.text, "[CHARACTER]")
        elif entity.tag == 'LOC':
            processed_text = processed_text.replace(entity.text, "[LOCATION]")
        elif entity.tag == 'ORG':
            processed_text = processed_text.replace(entity.text, "[ORGANIZATION]")
    return processed_text

# 2. 定义待处理的文件列表（改为 xls）
data_dir = 'tell me again'  # <-- 改成你的文件夹名，比如 'tell me again'；若就在当前目录用 '.'
files_to_process = [
    'nli_for_simcse.xls'
]

print(f"{'='*20} 开始批量执行匿名化预处理 {'='*20}\n")

# 外层进度条：监控文件整体处理进度
for filename in tqdm(files_to_process, desc="整体文件进度"):
    input_path = os.path.join(data_dir, filename)
    output_path = os.path.join(data_dir, filename.replace('.xls', '_processed.xlsx'))

    if not os.path.exists(input_path):
        tqdm.write(f"跳过文件（未找到）: {filename}")
        continue

    # 读取 xls（需要 xlrd 支持）
    df = pd.read_excel(input_path, engine='xlrd')

    # 预先计算行数，用于内层进度条的 total 参数
    line_count = len(df)

    # 确保存在需要的列（与数据匹配）
    required_cols = ['sent0', 'sent1', 'hard_neg']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}。当前列为: {df.columns.tolist()}")

    # ==========================
    # 中间落盘/断点续跑逻辑（按你给的结构嵌入）
    # ==========================
    tmp_output_path = output_path.replace('.xlsx', '_tmp.pkl')  # 用pkl做中间缓存最稳（注意这里是 .xlsx）
    save_every = 500

    # 如果之前跑崩过，尝试从缓存恢复
    start_idx = 0
    sent0_anon_list, sent1_anon_list, hard_neg_anon_list = [], [], []
    if os.path.exists(tmp_output_path):
        cache = pd.read_pickle(tmp_output_path)
        start_idx = cache["done"]
        sent0_anon_list = cache["sent0_anon"]
        sent1_anon_list = cache["sent1_anon"]
        hard_neg_anon_list = cache["hard_neg_anon"]
        tqdm.write(f"检测到缓存，已恢复到第 {start_idx} 行继续跑。")

    # 逐行处理，保持与原脚本相同“逐条匿名化 + 首行对比打印”的结构
    for i in tqdm(
        range(start_idx, line_count),
        total=line_count,
        initial=start_idx,
        desc=f"处理 {filename}",
        leave=False
    ):
        row = df.iloc[i]

        # 兼容 Excel 里可能出现的 NaN
        sent0 = "" if pd.isna(row['sent0']) else str(row['sent0'])
        sent1 = "" if pd.isna(row['sent1']) else str(row['sent1'])
        hard_neg = "" if pd.isna(row['hard_neg']) else str(row['hard_neg'])

        sent0_anon = anonymize_text(sent0)
        sent1_anon = anonymize_text(sent1)
        hard_neg_anon = anonymize_text(hard_neg)

        sent0_anon_list.append(sent0_anon)
        sent1_anon_list.append(sent1_anon)
        hard_neg_anon_list.append(hard_neg_anon)

        # 首行完整对比打印（对应三列）
        if i == 0:
            tqdm.write(f"\n{'#'*20} {filename} 效果对比 {'#'*20}")
            tqdm.write(f"【sent0 - 原始】:\n{sent0}\n")
            tqdm.write(f"【sent0 - 匿名】:\n{sent0_anon}\n")
            tqdm.write(f"{'-'*40}")
            tqdm.write(f"【sent1 - 原始】:\n{sent1}\n")
            tqdm.write(f"【sent1 - 匿名】:\n{sent1_anon}\n")
            tqdm.write(f"{'-'*40}")
            tqdm.write(f"【hard_neg - 原始】:\n{hard_neg}\n")
            tqdm.write(f"【hard_neg - 匿名】:\n{hard_neg_anon}\n")
            tqdm.write(f"{'='*80}\n")

        # 定期落盘，避免白跑
        if (i + 1) % save_every == 0:
            pd.to_pickle(
                {
                    "done": i + 1,
                    "sent0_anon": sent0_anon_list,
                    "sent1_anon": sent1_anon_list,
                    "hard_neg_anon": hard_neg_anon_list
                },
                tmp_output_path
            )

    # 循环结束后，写回 df 并导出 xlsx
    df['sent0_anon'] = sent0_anon_list
    df['sent1_anon'] = sent1_anon_list
    df['hard_neg_anon'] = hard_neg_anon_list
    df.to_excel(output_path, index=False)  # 不写engine更稳

    # 成功后删缓存
    if os.path.exists(tmp_output_path):
        os.remove(tmp_output_path)

print(f"\n{'='*20} 所有数据集处理完成！ {'='*20}")