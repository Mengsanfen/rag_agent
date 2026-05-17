from langgraph.prebuilt import create_react_agent
from agent.my_llm import llm
from langchain_openai import ChatOpenAI
from agent.tools.tool_demo5 import runnable_tool
from langgraph_study.langgraph_demo.src.agent.mcp_agent import my_search_tool
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig 
from langgraph.prebuilt.chat_agent_executor import AgentState
from agent.tools.tool_demo8 import get_user_name, greet_user
from agent.my_state import CustomState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

# checkpointer = InMemorySaver() # 短期记忆，保存到内存中

DB_URL = 'postgresql://postgres:123456@localhost:5432/langgraph_db'

with (
    PostgresSaver.from_conn_string(DB_URL) as checkpointer,
    PostgresStore.from_conn_string(DB_URL) as store,
):
    # checkpointer.setup()    # 第一次使用时使用此行代码，进行数据库初始化，后续使用注释掉
    store.setup()
    agent = create_react_agent(
        llm,
        tools=[runnable_tool, my_search_tool],
        prompt="你是一个智能助手，尽可能地调用工具回答用户的问题",
        checkpointer=checkpointer,
        store=store,
    )

    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    rest = list(agent.get_state(config=config)) #type: ignore
    rest = list(agent.get_state_history(config=config)) #type: ignore
    print(rest)

    resp1 = agent.invoke(
        {"messages": [{"role": "user", "content": "请计算1+1"}]}, 
        config, # type: ignore
    )

    print(resp1['messages'][-1].content)

    resp2 = agent.invoke(
        {"messages": [{"role": "user", "content": "根据上段话，再来个反面的回答"}]},
        config, # type: ignore
    )