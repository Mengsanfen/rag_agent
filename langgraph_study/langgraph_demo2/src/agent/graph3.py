from typing import Any, Dict, List
from langchain_core.messages import ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import json
from langgraph.graph import MessagesState, StateGraph
from agent.my_llm import llm
from langgraph.constants import START, END
from langgraph.prebuilt import ToolNode, tools_condition

chart_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/2e40061f467743/sse",
    "transport": "sse",
}

my12306_mcp_server_config = {
    "url": "https://mcp-.api-inference.modelscope.net/sse",
    "transport": "sse",
}

tavily_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/92b6ce89cfdf42/sse",
    "transport": "sse",
}

mcp_client = MultiServerMCPClient(
    {
        "chart": chart_mcp_server_config,   # type: ignore
        "my12306": my12306_mcp_server_config,
        "tavily": tavily_mcp_server_config,
    }
)


class State(MessagesState):
    """状态类，用于存储输入输出消息
    """
    pass

async def create_graph():
    tools = await mcp_client.get_tools()

    builder = StateGraph(State)

    llm_with_tools = llm.bind_tools(tools)

    async def chatbot(state: State):
        return {"messages": [await llm_with_tools.ainvoke(state["messages"])]}
    
    builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=tools)
    builder.add_node("tools", tool_node)    # type: ignore

    builder.add_conditional_edges(
        "chatbot",
        # route_tools_func,
        tools_condition,
    )

    builder.add_edge("tools", "chatbot")
    builder.add_edge(START, "chatbot")

    graph = builder.compile()
    return graph

agent = asyncio.run(create_graph())