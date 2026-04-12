# """
# 提示词：用户的提问 + 向量库中检索到的参考资料
# """
# from langchain_community.chat_models import ChatTongyi
# from langchain_core.documents import Document
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_community.embeddings import DashScopeEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser


# def print_prompt(prompt):
#     print(prompt.to_string())
#     print("=" * 20)
#     return prompt


# model = ChatTongyi(model="qwen3-max")
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料:{context}。"),
#         ("user", "用户提问：{input}")
#     ]
# )

# vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings(model="text-embedding-v4"))

# # 准备一下资料（向量库的数据）
# # add_texts 传入一个 list[str]
# vector_store.add_texts(
#     ["减肥就是要少吃多练", "在减脂期间吃东西很重要,清淡少油控制卡路里摄入并运动起来", "跑步是很好的运动哦"])

# input_text = "怎么减肥？"

# # langchain中向量存储对象，有一个方法：as_retriever，可以返回一个Runnable接口的子类实例对象
# retriever = vector_store.as_retriever(search_kwargs={"k": 2})


# def format_func(docs: list[Document]):
#     if not docs:
#         return "无相关参考资料"

#     formatted_str = "["
#     for doc in docs:
#         formatted_str += doc.page_content
#     formatted_str += "]"

#     return formatted_str

# # chain
# chain = (
#     {"input": RunnablePassthrough(), "context": retriever | format_func} | prompt | print_prompt | model | StrOutputParser()
# )

# res = chain.invoke(input_text)
# print(res)
# """
# retriever:
#     - 输入：用户的提问       str
#     - 输出：向量库的检索结果  list[Document]
# prompt:
#     - 输入：用户的提问 + 向量库的检索结果   dict
#     - 输出：完整的提示词                 PromptValue
# """

# """
# 提示词：用户的提问 + 向量库中检索到的参考资料
# """



# def print_prompt(prompt):
#     print(prompt.to_string())
#     print("=" * 20)
#     return prompt


# model = ChatTongyi(model="qwen3-max")   # type: ignore
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料:{context}。"),
#         ("user", "用户提问：{input}")
#     ]
# )

# vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings(model="text-embedding-v4"))

# # 准备一下资料（向量库的数据）
# # add_texts 传入一个 list[str]
# vector_store.add_texts(
#     ["减肥就是要少吃多练", "在减脂期间吃东西很重要,清淡少油控制卡路里摄入并运动起来", "跑步是很好的运动哦"])

# input_text = "怎么减肥？"

# # 检索向量库
# result = vector_store.similarity_search(input_text, 2)
# reference_text = "["
# for doc in result:
#     reference_text += doc.page_content
# reference_text += "]"

# chain = prompt | print_prompt | model | StrOutputParser()

# res = chain.invoke({"input": input_text, "context": reference_text})
# print(res)

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

model = ChatTongyi(model="qwen3-max")   # type: ignore
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料:{context}。"),
        ("user", "用户提问：{input}")
    ]
)

vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings(model="text-embedding-v4"))

# 准备向量库数据
vector_store.add_texts(
    ["减肥就是要少吃多练", "在减脂期间吃东西很重要,清淡少油控制卡路里摄入并运动起来", "跑步是很好的运动哦"]
)

input_text = "怎么减肥？"
# """
# retriever:
#     - 输入：用户的提问       str
#     - 输出：向量库的检索结果  list[Document]
# prompt:
#     - 输入：用户的提问 + 向量库的检索结果   dict
#     - 输出：完整的提示词                 PromptValue
# """

# """
# 提示词
# langchain 中向量进行存储对象，有一个方法：as_retriever，可以返回一个Runnable接口的子类实例对象

def print_prompt(full_prompt):
    print(full_prompt.to_string())
    print('=' * 20)
    return full_prompt

def format_context(docs: list[Document]) -> str:
    formatted_str = "["
    for doc in docs:
        formatted_str += doc.page_content
    formatted_str += "]"

    return formatted_str

retriver = vector_store.as_retriever(search_kwargs={"k": 2})    # 能返回一个Runnable接口的子类实例对象

chain = {"input": RunnablePassthrough(), "context": retriver | format_context} | prompt | print_prompt | model | StrOutputParser()

res = chain.invoke(input_text)

print(res)

