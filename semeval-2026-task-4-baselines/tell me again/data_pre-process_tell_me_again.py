from flair.data import Sentence
from flair.models import SequenceTagger
import json
import os
from tqdm import tqdm  # 导入进度条库

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

# 2. 定义待处理的文件列表
data_dir = 'tell me again'
files_to_process = [
    'train_track_a.jsonl', 'train_track_b.jsonl',
    'dev_track_a.jsonl', 'dev_track_b.jsonl',
    'test_track_a.jsonl', 'test_track_b.jsonl',
    'sample_track_a.jsonl', 'sample_track_b.jsonl'
]

print(f"{'='*20} 开始批量执行匿名化预处理 {'='*20}\n")

# 外层进度条：监控 8 个文件的整体处理进度
for filename in tqdm(files_to_process, desc="整体文件进度"):
    input_path = os.path.join(data_dir, filename)
    output_path = os.path.join(data_dir, filename.replace('.jsonl', '_processed.jsonl'))
    
    if not os.path.exists(input_path):
        # 使用 tqdm.write 打印消息，避免干扰进度条显示
        tqdm.write(f"跳过文件（未找到）: {filename}")
        continue

    # 预先计算文件行数，用于内层进度条的 total 参数
    line_count = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            # 内层进度条：监控单个文件内每一行的处理进度
            # leave=False 表示处理完该文件后自动清除其进度条
            for i, line in enumerate(tqdm(f_in, total=line_count, desc=f"处理 {filename}", leave=False)):
                data = json.loads(line)
                
                # 3. 处理字段并存储原始备份用于对比展示
                if 'track_a' in filename:
                    data['anchor_text_anon'] = anonymize_text(data.get('anchor_text', ""))
                    data['text_a_anon'] = anonymize_text(data.get('text_a', ""))
                    data['text_b_anon'] = anonymize_text(data.get('text_b', ""))
                    
                    # 针对 Track A 的完整对比打印
                    if i == 0:
                        tqdm.write(f"\n{'#'*20} {filename} 效果对比 {'#'*20}")
                        tqdm.write(f"【Anchor - 原始】:\n{data['anchor_text']}\n")
                        tqdm.write(f"【Anchor - 匿名】:\n{data['anchor_text_anon']}\n")
                        tqdm.write(f"{'-'*40}")
                        tqdm.write(f"【Text A - 原始】:\n{data['text_a']}\n")
                        tqdm.write(f"【Text A - 匿名】:\n{data['text_a_anon']}\n")
                        tqdm.write(f"{'-'*40}")
                        tqdm.write(f"【Text B - 原始】:\n{data['text_b']}\n")
                        tqdm.write(f"【Text B - 匿名】:\n{data['text_b_anon']}\n")
                
                else:
                    data['text_anon'] = anonymize_text(data.get('text', ""))
                    
                    # 针对 Track B 的完整对比打印
                    if i == 0:
                        tqdm.write(f"\n{'#'*20} {filename} 效果对比 {'#'*20}")
                        tqdm.write(f"【Text - 原始】:\n{data['text']}\n")
                        tqdm.write(f"【Text - 匿名】:\n{data['text_anon']}\n")
                
                if i == 0:
                    tqdm.write(f"{'='*80}\n")
                
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"\n{'='*20} 所有 8 个数据集处理完成！ {'='*20}")