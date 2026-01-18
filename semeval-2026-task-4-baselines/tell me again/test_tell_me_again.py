from tell_me_again import StoryDataset, SimilarityDataset
import json
from flair.data import Sentence
from flair.models import SequenceTagger
# 需要安装 flair 和对应的共指消解模型 (如 Xu and Choi, 2020)

# 加载论文中提到的 Flair 命名实体识别模型 
tagger = SequenceTagger.load('ner-balanced')

def anonymize_text(text):
    """
    参考论文 5.1 节实现实体替换 [cite: 203, 204]
    """
    sentence = Sentence(text)
    tagger.predict(sentence)
    
    # 获取所有实体并按照论文逻辑替换 [cite: 204]
    # 例如：人名 -> Natalie/Edward, 地点 -> Location B, 组织 -> Organization C
    # 论文提到：人物名字会根据美国人口普查数据随机抽样以保持性别一致 [cite: 206]
    processed_text = text
    for entity in sentence.get_spans('ner'):
        if entity.tag == 'PER':
            processed_text = processed_text.replace(entity.text, "Character_A") # 示例占位符
        elif entity.tag == 'LOC':
            processed_text = processed_text.replace(entity.text, "Location_B")
    return processed_text

# 处理 Track A
with open('sample_track_a_processed.jsonl', 'w') as f_out:
    with open('sample_track_a.jsonl', 'r') as f_in:
        for line in f_in:
            data = json.loads(line)
            # 对三元组中的所有文本进行匿名化
            data['anchor_text_anon'] = anonymize_text(data['anchor_text'])
            data['text_a_anon'] = anonymize_text(data['text_a'])
            data['text_b_anon'] = anonymize_text(data['text_b'])
            f_out.write(json.dumps(data) + '\n')import json
from flair.data import Sentence
from flair.models import SequenceTagger

# 加载模型
tagger = SequenceTagger.load('ner-balanced')

def anonymize_text(text):
    sentence = Sentence(text)
    tagger.predict(sentence)
    processed_text = text
    # 论文提到：人物名字替换应尽可能保持性别一致，这里简单演示占位符替换 [cite: 206]
    for entity in sentence.get_spans('ner'):
        if entity.tag == 'PER':
            processed_text = processed_text.replace(entity.text, "[CHARACTER]")
        elif entity.tag == 'LOC':
            processed_text = processed_text.replace(entity.text, "[LOCATION]")
    return processed_text

# 增强处理逻辑：增加可视化对比
print(f"{'='*30} 处理效果直观对比 {'='*30}\n")

with open('sample_track_a_processed.jsonl', 'w') as f_out:
    with open('sample_track_a.jsonl', 'r') as f_in:
        for i, line in enumerate(f_in):
            data = json.loads(line)
            
            # 执行匿名化
            data['anchor_text_anon'] = anonymize_text(data['anchor_text'])
            
            # 仅打印前 2 条数据进行直观校验
            if i < 2:
                print(f"--- 样本 #{i+1} ---")
                print(f"【原始文本】: {data['anchor_text'][:150]}...") 
                print(f"【处理后文本】: {data['anchor_text_anon'][:150]}...")
                print("-" * 50)
            
            f_out.write(json.dumps(data) + '\n')

print(f"\n处理完成！所有数据已存入 sample_track_a_processed.jsonl")