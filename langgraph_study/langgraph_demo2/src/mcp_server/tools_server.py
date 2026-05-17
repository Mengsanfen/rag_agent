from fastmcp import FastMCP
from fastmcp.prompts import Message


server = FastMCP(name='wyx_mcp', instructions='我自己的采用Python实现MCP服务器')


@server.tool(description="打招呼，传入名字，返回字符串信息")
def say_hello(name: str) -> str:
    return f"hello, {name}"

@server.tool(description="搜索，传入查询条件，返回字符串信息")
def my_search(query: str) -> str:
    return f"search result for {query}"

@server.prompt
def ask_about_topic(topic: str) -> str:
    """生成一个关于主题的提问"""
    return f"请你能否解释一下'{topic}',这个概念"

@server.resource("resource://config")
def get_config() -> dict:
    """获取配置信息"""
    return {"name": "wyx_mcp", "age": 18}