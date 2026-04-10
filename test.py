import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 注意: 不同地域的base_url不通用（下方示例使用北京地域的 base_url）
# - 华北2（北京）: https://dashscope.aliyuncs.com/compatible-mode/v1
# - 美国（弗吉尼亚）: https://dashscope-us.aliyuncs.com/compatible-mode/v1
# - 新加坡: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # base_url="http://localhost:11434/v1", 本地ollama模型
)
completion = client.chat.completions.create(
    model="qwen3.5-flash",
    # model="deepseek-r1:1.5b",
    messages=[{'role': 'user', 'content': '你是谁？'}],
    stream=True
)

# print(completion.choices[0].message.content)
for chunk in completion:
    print(chunk.choices[0].delta.content, end="", flush=True)