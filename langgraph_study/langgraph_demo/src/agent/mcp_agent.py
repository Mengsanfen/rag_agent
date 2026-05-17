from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
from agent.my_llm import llm
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio 

python_mcp_config = {
    'url': 'http://127.0.0.1:5000/sse',
    'transport': 'sse',
}


mcp_client = MultiServerMCPClient(
    {
        'python_mcp': python_mcp_config,    # type: ignore
    }
)

async def create_agent():
    mcp_tools = await mcp_client.get_tools()

    return create_react_agent(
        llm,
        tools=mcp_tools,
        prompt='你是一个智能助手，尽可能地调用工具回答我的问题。'
    )

agent = asyncio.run(create_agent())