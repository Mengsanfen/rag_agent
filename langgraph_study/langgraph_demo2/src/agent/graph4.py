from typing import Any, Dict, List
from langchain_core.messages import ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import json
from langgraph.graph import MessagesState, StateGraph
from agent.my_llm import llm
from langgraph.constants import START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

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
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer, interrupt_before=['tools'])
    return graph

# agent = asyncio.run(create_graph())

async def run_graph():
    graph = await create_graph()
    # 配置参数
    config = {
        "configurable": {
            # 检查点由session_id访问
            "thread_id": "1234567890"
        }
    }

    def print_message(event, result: str) -> str:
        messages = event.get("messages")
        if messages:
            if isinstance(messages, list):
                message = messages[-1]
            if message.__class__.__name__ == 'AIMessage':
                if message.content:
                    result += message.content
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > 1500:
                msg_repr = msg_repr[:1500] + '...'
            print(msg_repr)
        return result

    def get_answer(tool_message, user_answer):
        """让人工接入，并且给一个问题的答案."""
        tool_name = tool_message.tool_calls[0]["name"]
        answer = (
            f"人工强制终止了工具：{tool_name}的执行, 拒绝的理由是: {user_answer}"
        )
        # 创建一个消息
        new_message = [
            ToolMessage(content=answer, tool_call_id=tool_message.tool_calls[0]["id"]),
            AIMessage(content=answer)
        ]

        # 把人新造的消息，添加到工作流的state中
        graph.update_state(
            config=config,
            values={"messages": new_message},
        )
    

    async def execute_graph(user_input: str) -> str:
        """执行工作流"""
        result = ''
        if user_input.strip().lower() != 'y':   # 正常的用户提问
            current_state = graph.get_state(config)
            if current_state.next:  # 如果有下一步，则当前工作流处于中断状态
                tools_script_message = current_state.values['messages'][-1]
                get_answer(tools_script_message, user_input)
                message = graph.get_state(config).values['messages'][-1]
                result = message.content
                return result
            else :
                async for chunk in graph.astream({'messages': ('user', user_input)}, config, stream_mode='values'):
                    result = print_message(chunk, result)

        else:   # 用户输入是y，则执行工作流
            async for chunk in graph.astream(None, config=config, stream_mode='values'):
                result = print_message(chunk, result)  
        
        current_state = graph.get_state(config)
        if current_state.next:  # 如果有下一步，则当前工作流处于中断状态
            ai_message = current_state.values['messages'][-1]
            tool_name = ai_message.tool_calls[0]["name"]
            result = f"AI助手根据你要求，执行{tool_name}工具。您是否批准执行？(y/n)"
            return result

        return result

    # 执行工作流
    while True:
        user_input = input("用户：")
        res = await execute_graph(user_input)
        print("AI: ", res)


if __name__ == "__main__":
    asyncio.run(run_graph())