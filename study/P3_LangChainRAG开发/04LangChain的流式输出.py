# from langchain_community.llms.tongyi import Tongyi
#
# model = Tongyi(model="qwen-max")
#
# # 通过stream方法获得流式输出
# res = model.stream(input="你是谁呀能做什么？")
#
# for chunk in res:
#     print(chunk, end="", flush=True)

# from langchain_ollama import OllamaLLM

# model = OllamaLLM(model="deepseek-r1:1.5b")

# res = model.stream(input="你是谁呀能做什么？")

# for chunk in res:
#     print(chunk, end="", flush=True)

from langchain_community.llms.tongyi import Tongyi

model = Tongyi(model='qwen-plus')

res = model.stream(input="你是谁呀能做什么？")

for chunk in res:
    print(chunk, end = "", flush=True)
