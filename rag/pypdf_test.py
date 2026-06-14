from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path=r"D:\workspace\rag_agent\The implicit dynamics of in-context learning.pdf",
)

docs = loader.load()

print(f'docs的数量为: {len(docs)}')
print(docs[0].metadata)
print(docs[0].page_content)