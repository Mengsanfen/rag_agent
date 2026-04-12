# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_community.embeddings import DashScopeEmbeddings
# from langchain_community.document_loaders import CSVLoader

# vector_store = InMemoryVectorStore(
#     embedding=DashScopeEmbeddings()
# )


# loader = CSVLoader(
#     file_path="./data/info.csv",
#     encoding="utf-8",
#     source_column="source",     # 指定本条数据的来源是哪里
# )

# documents = loader.load()
# # id1 id2 id3 id4 ...
# # 向量存储的 新增、删除、检索
# vector_store.add_documents(
#     documents=documents,        # 被添加的文档，类型：list[Document]
#     ids=["id"+str(i) for i in range(1, len(documents)+1)] # 给添加的文档提供id（字符串）  list[str]
# )

# # 删除  传入[id, id...]
# vector_store.delete(["id1", "id2"])

# # 检索 返回类型list[Document]
# result = vector_store.similarity_search(
#     "瑞达法",
#     3       # 检索的结果要几个
# )

# print(result)
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader

vector_stores = InMemoryVectorStore(
    embedding=DashScopeEmbeddings(),

)

loader = CSVLoader(
    file_path='study\P3_LangChainRAG开发\data\info.csv',
    encoding='utf-8',
    source_column='source', # 指定本条数据的来源是哪里
)

documents = loader.load()   # 全量加载
# print(documents[0])

# id1 id2 id3 id4
# 向量存储的 新增、删除、检索
vector_stores.add_documents(
    documents=documents,
    ids=["id"+str(i) for i in range(1, len(documents) + 1)]
)

# 删除  传入[id, id]
vector_stores.delete(["id1", "id2"])

# 检索
result = vector_stores.similarity_search(
    query="python是不简单易学啊",
    k=3   
)

print(result)