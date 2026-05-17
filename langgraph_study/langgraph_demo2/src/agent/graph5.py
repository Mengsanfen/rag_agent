from typing import Any, Dict, List
from langchain_core.messages import ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import json
from langgraph.graph import MessagesState, StateGraph
from agent.my_llm import llm
from langgraph.constants import START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
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

class BasicToolsNode:
    """异步工具节点，用于并发执行AIMessage中请求的工具调用.

    功能：
    1. 接收工具列表并建立名称索引
    2. 并发执行消息中的工具调用请求
    3. 自动处理同步/异步工具适配
    """
    def __init__(self, tools: list):
        """初始化工具节点

        Args:
            tools (list): 工具列表
        """
        self.tools_by_name = {tool.name: tool for tool in tools}
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, List[ToolMessage]]:
        """异步调用工具入口
        
        Args:
            state (Dict[str, Any]): 状态字典（输入字典），需要包含"messages"字段
        
        Returns:
            Dict[str, List[ToolMessage]]: 工具执行结果
        
        Raises:
            ValueError: 输入字典中缺少"messages"字段
        """

        # 1. 输入验证
        if "messages" not in state:
            raise ValueError("输入字典中缺少'messages'字段")
        messages = state["messages"]
        message: AIMessage = messages[-1]

        tool_name = message.tool_calls[0]["name"] if message.tool_calls else None
        if tool_name == 'webSearchStd' or tool_name == 'webSearchSogou':
            response = interrupt(
                f"AI大模型尝试调用工具`{tool_name}`, \n"
                "请审核并选择：批准(y)或直接给我工具执行的答案。"
            )
            # response:由人工输入的，批准（y），工具执行的答案或者拒绝执行工具的理由
            # 根据人工响应类型处理
            if response["answer"] == "y":
                pass
            else:
                return {
                    "messages": [
                        ToolMessage(
                            content=f'人工终止了工具的调用，给出的理由为{response['answer']}',
                            name=tool_name,
                            tool_call_id=message.tool_calls[0]["id"],
                        )
                    ]
                }
        # 2. 并发执行工具调用
        outputs = await self._excute_tool_calls(message.tool_calls)
        return {"messages": outputs}
    
    async def _excute_tool_calls(self, tool_calls: List[Dict]) -> List[ToolMessage]:
        """并发执行工具调用
        
        Args:
            tool_calls (List[Dict]): 工具调用列表
        
        Returns:
            List[ToolMessage]: 工具执行结果
        """

        async def _invoke_tool(tool_call: Dict) -> ToolMessage:
            """执行单个工具调用
            Args:
                tool_call (Dict): 工具调用
            Returns:
                ToolMessage: 工具执行结果
            Raises:
                KeyError: 工具为注册时
                RuntimeError: 工具调用失败
            """

            try:
                # 3. 异步调用工具
                tool = self.tools_by_name.get(tool_call["name"])
                if not tool:
                    raise KeyError(f"工具{tool_call['name']}未注册")
                
                if hasattr(tool, 'ainvoke'):
                    tool_result = await tool.ainvoke(tool_call["args"])
                else:   # 同步工具通过线程池转成异步
                    loop = asyncio.get_running_loop()
                    tool_result = await loop.run_in_executor(None, tool.invoke, tool_call["args"])

                # 4. 构建工具执行结果
                return ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            except Exception as e:
                # 5. 构建工具执行失败结果
                raise RuntimeError(
                    f"工具{tool_call['name']}执行失败: {e}",
                )
            
        try:
            return await asyncio.gather(*[_invoke_tool(tool_call) for tool_call in tool_calls])
        except Exception as e:
            raise RuntimeError(f"工具执行失败: {e}")

class State(MessagesState):
    """状态类，用于存储输入输出消息
    """
    pass

def route_tools_func(state: State) -> str:
    """路由工具调用，若是AIMessage中包含工具调用，则路由到工具节点，否则结束
    
    Args:
        state (State): 状态
    
    Returns:
        str: 路由结果
    """
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

async def create_graph():
    tools = await mcp_client.get_tools()

    builder = StateGraph(State)

    llm_with_tools = llm.bind_tools(tools)

    async def chatbot(state: State):
        return {"messages": [await llm_with_tools.ainvoke(state["messages"])]}
    
    builder.add_node("chatbot", chatbot)

    tool_node = BasicToolsNode(tools)
    builder.add_node("tools", tool_node)    # type: ignore

    builder.add_conditional_edges(
        "chatbot",
        route_tools_func,
        {"tools": "tools", END: END}
    )

    builder.add_edge("tools", "chatbot")
    builder.add_edge(START, "chatbot")

    graph = builder.compile()
    return graph

# agent = asyncio.run(create_graph())

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


