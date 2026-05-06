from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from tavily import TavilyClient
import os
from typing import Type

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


class SearchArgs(BaseModel):
    query: str = Field(description="需要进行网络搜索的信息")

# 网络搜索的工具
class MySearchTool(BaseTool):
    # 工具名字
    name: str = "search_tool"
    description: str  = "搜索互联网上公开内容的工具"
    return_direct: bool = False
    args_schema: Type[BaseModel] = SearchArgs

    def _run(self, query) -> str:
        try:
            print("执行Tavily搜索工具，输入参数为：", query)

            response = tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=3,
                include_answer=True,
                include_raw_content=False
                )
            # 优先返回 Tavily 直接总结的 answer
            if response.get("answer"):
                return response["answer"]

            # 如果没有 answer，就拼接搜索结果内容
            results = response.get("results", [])

            if not results:
                return "没有搜索到任何内容。"

            contents = []
            for item in results:
                title = item.get("title", "")
                url = item.get("url", "")
                content = item.get("content", "")

                contents.append(
                    f"标题：{title}\n链接：{url}\n内容：{content}"
                )

            return "\n\n".join(contents)

        except Exception as e:
            print("搜索失败：", e)
            return "搜索工具调用失败，没有获取到内容"


my_tool = MySearchTool()
print(my_tool.name)
print(my_tool.description)
print(my_tool.args_schema.model_json_schema())
print(my_tool.run("如何使用 langchain"))
