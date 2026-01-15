from google import genai

client = genai.Client(api_key="AIzaSyBcs8eNS996ndYaqw8K4EK5GxKZ6Hotz8w")

# 列出所有模型
print("--- 当前可用模型列表 ---")
for model in client.models.list():
    # 过滤出支持生成内容（generateContent）的模型
    if 'generateContent' in model.supported_actions:
        print(f"模型名称: {model.name}")