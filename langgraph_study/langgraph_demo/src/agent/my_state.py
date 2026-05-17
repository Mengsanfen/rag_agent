from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import MessagesState
# 自己定义的智能体的状态类
class CustomState(AgentState):
    username: str # 用户名


# MessagesState