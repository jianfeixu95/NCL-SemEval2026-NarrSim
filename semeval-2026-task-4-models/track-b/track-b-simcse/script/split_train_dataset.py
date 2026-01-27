from sklearn.model_selection import train_test_split
import pandas as pd

# 读取 jsonl 文件
df = pd.read_json("../data/dev_track_a.jsonl", lines=True)
print(f"total dataset size: {len(df)}")

# 划分训练 / 验证集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"train dataset size: {len(train_df)}")
print(f"eval dataset size: {len(val_df)}")

# 保存为 jsonl 文件
train_df.to_json("../data/dev_track_a_train.jsonl", orient="records", lines=True, force_ascii=False)
val_df.to_json("../data/dev_track_a_valid.jsonl", orient="records", lines=True, force_ascii=False)
