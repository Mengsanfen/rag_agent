from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from typing import Annotated
from agent.my_state import CustomState
from langgraph.prebuilt import InjectedState

@tool
def get_user_name(tool_call_id: Annotated[str, InjectedToolCallId],
                  config: RunnableConfig) -> Command:
    """获取用户的username，以便生成祝福语句."""
    user_name = config.get("configurable", {}).get("user_name", "zhangsan")
    print(f"调用工具，传入的用户名为：{user_name}")
    # 模拟
    return Command(update={
        "username": user_name,   # 更新状态中的用户名
        # 更新一条工具执行后的消息: ToolMessage类型
        "messages": [
            ToolMessage(
                content=f"获取用户{user_name}的所有信息成功",
                tool_call_id=tool_call_id
            )
        ]
    })


@tool
def greet_user(state: Annotated[CustomState, InjectedState]) -> str:
    """在获取用户的username后，生成一个祝福语"""
    username=state['username']  # 获取状态中的用户名

    return f'祝贺你：{username}!'
