from tell_me_again import StoryDataset, SimilarityDataset
import json
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