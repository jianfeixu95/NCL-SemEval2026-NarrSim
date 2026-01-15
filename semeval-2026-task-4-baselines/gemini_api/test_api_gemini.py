from google import genai
from google.genai import types
import os

client = genai.Client(api_key="AIzaSyBcs8eNS996ndYaqw8K4EK5GxKZ6Hotz8w")

# 发起调用
response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents="简要介绍使用机器学习判断文本相似度的方法",
    config=types.GenerateContentConfig(
        temperature=0.7,      # 随机性：0.0最严谨，1.0最发散
        top_p=0.95,
        max_output_tokens=50000, # 限制最大输出长度
        stop_sequences=["STOP!"], # 遇到该字符停止
    )
)

# 不要直接 print(response.text)，先检查是否有结果
if response.candidates:
    print(response.text)
else:
    # 如果没有候选结果，打印整个 response 看看是不是被安全拦截了
    print("模型未生成内容，可能触发了安全拦截。完整响应：")
    print(response)