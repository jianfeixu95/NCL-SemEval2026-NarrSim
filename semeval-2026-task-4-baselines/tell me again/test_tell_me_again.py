from flair.data import Sentence
from flair.models import SequenceTagger
import json

# 1. 加载论文中提到的 Flair 命名实体识别模型 [cite: 207]
# 建议使用 'flair/ner-english-large' 以获得更高准确率
tagger = SequenceTagger.load('flair/ner-english-large')

def anonymize_text(text):
    """
    参考论文 5.1 节实现实体替换 [cite: 203]
    将文本中的人名和地点替换为统一占位符，消除实体偏见 [cite: 12]
    """
    if not text:
        return ""
    
    sentence = Sentence(text)
    tagger.predict(sentence)
    processed_text = text
    
    # 获取所有实体并按照标签进行替换 
    # 论文中提到：人物名字替换建议保持性别一致 [cite: 206]，此处使用 [CHARACTER] 简化演示
    for entity in sentence.get_spans('ner'):
        if entity.tag == 'PER':
            processed_text = processed_text.replace(entity.text, "[CHARACTER]")
        elif entity.tag == 'LOC':
            processed_text = processed_text.replace(entity.text, "[LOCATION]")
        elif entity.tag == 'ORG':
            processed_text = processed_text.replace(entity.text, "[ORGANIZATION]")
            
    return processed_text

print(f"{'='*30} 开始处理 Track A 数据集 {'='*30}\n")

# 使用 'with open' 语法确保文件处理的安全性和完整性
with open('tell me again/sample_track_a_processed.jsonl', 'w', encoding='utf-8') as f_out:
    with open('tell me again/sample_track_a.jsonl', 'r', encoding='utf-8') as f_in:
        for i, line in enumerate(f_in):
            data = json.loads(line)
            
            # 2. 对三元组中的所有文本字段执行匿名化
            data['anchor_text_anon'] = anonymize_text(data['anchor_text'])
            data['text_a_anon'] = anonymize_text(data['text_a'])
            data['text_b_anon'] = anonymize_text(data['text_b'])
            
            # 3. 打印前 1 条数据的完整处理结果，以便你直观校验 [CHARACTER] 替换效果
            if i == 0:
                print(f"--- [样本 #1 完整对比] ---")
                print(f"【Anchor 原始文本】:\n{data['anchor_text']}\n")
                print(f"【Anchor 处理后文本】:\n{data['anchor_text_anon']}\n")
                print("-" * 40)
                print(f"【Text A 原始文本】:\n{data['text_a']}\n")
                print(f"【Text A 处理后文本】:\n{data['text_a_anon']}\n")
                print("-" * 40)
                print(f"【Text B 原始文本】:\n{data['text_b']}\n")
                print(f"【Text B 处理后文本】:\n{data['text_b_anon']}\n")
                print("=" * 70)
            
            # 将更新后的字典写回新文件
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"\n处理完成！所有字段已匿名化并存入 sample_track_a_processed.jsonl")