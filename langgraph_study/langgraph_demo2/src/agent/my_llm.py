from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi


llm = ChatTongyi(model="qwen3-max") # type: ignore
