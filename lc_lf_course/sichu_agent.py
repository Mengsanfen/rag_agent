from langchain.chat_models import init_chat_model
import os

model = init_chat_model(
    model="qwen3.5-plus",
    model_provider="openai",
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

from langchain_tavily import TavilySearch

web_search = TavilySearch(
    max_results=5,
    topic='general',
)