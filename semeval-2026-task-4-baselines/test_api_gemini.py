from google import genai
from google.genai import types
import os

client = genai.Client(api_key="AIzaSyBcs8eNS996ndYaqw8K4EK5GxKZ6Hotz8w")

# 发起调用
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="写一首关于秋天的诗",
    config=types.GenerateContentConfig(
        temperature=0.7,      # 随机性：0.0最严谨，1.0最发散
        top_p=0.95,
        max_output_tokens=500, # 限制最大输出长度
        stop_sequences=["STOP!"], # 遇到该字符停止
    )
)

print(response.text)