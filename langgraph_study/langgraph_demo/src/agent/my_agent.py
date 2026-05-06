from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from agent.tools.tool_demo2 import calculate2
from agent.tools.tool_demo5 import runnable_tool
from agent.tools.tool_demo6 import MySearchTool
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
from agent.tools.tool_demo8 import get_user_name, greet_user
from agent.my_state import CustomState

llm = ChatTongyi(model="qwen3-max") # type: ignore

# 创建一个网络搜索工具
my_search_tool = MySearchTool()

# def get_weather(city: str) ->str:
#     """get weather for a given city."""
#     return f"Tt's always sunny in {city}"  

# 提示词模板的函数：由用户传入内容，组成一个动态提示词
def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    """构造系统提示词."""
    # user_name = config['configurable'].get("user_name", "zhangsan")
    user_name = config.get("configurable", {}).get("user_name", "zhangsan")
    system_message = f'你是一个智能的助手，当前用户的名字是: {user_name}'
    return [SystemMessage(content=system_message)] + state["messages"]  # type:ignore


graph = create_react_agent(
    llm,
    tools=[calculate2, runnable_tool, my_search_tool, get_user_name, greet_user],
    prompt=prompt,  # type:ignore
    state_schema=CustomState,   # 指定自定义状态类
)