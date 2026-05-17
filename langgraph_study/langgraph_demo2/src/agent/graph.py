from typing import TypedDict, Literal
from langgraph.graph import StateGraph
from agent.my_llm import llm
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langgraph.constants import START, END

class State(TypedDict):
    joke: str   # 生成的冷笑话内容
    topic: str  # 用户指定的主题
    feedback: str   # 改进建议
    funny_or_not: str   # 幽默评级

# 结构化输出模型（用于LLM评估反馈）
class FeedBack(BaseModel):
    """使用此工具来结构化响应."""
    grade: Literal["funny", "not funny"] = Field(
        description=" joke 的等级， either 'funny' or 'not funny'.",
        examples=["funny", "not funny"]
    )
    feedback: str = Field(
        description="若是不幽默，提出改进建议",
        examples=["可以加入双语关或者意外结局"]
    )

# 节点函数
def generator_func(state: State):
    """由大模型生成冷笑话的节点."""
    prompt = (
        f"根据反馈改进笑话：{state['feedback']}\n主题：{state['topic']}"
        if state.get("feedback", None) 
        else f"创作一个关于{state['topic']}的笑话"
    )
    # 第一种
    # resp = llm.invoke(prompt)
    # return {"joke": resp.content}
    # 第二种
    chain = llm | StrOutputParser()
    resp = chain.invoke(prompt)
    return {"joke": resp}

# 节点函数
def evaluator_func(state: State):
    """评估状态中的冷笑话."""
    # 第一种
    chain = llm.with_structured_output(FeedBack)
    resp = chain.invoke(f"请对下面笑话进行评价：{state['joke']}")
    return {
        "funny_or_not": resp.grade, # type: ignore
        "feedback": resp.feedback   # type: ignore
    }
    # 第二种
    # chain = llm.bind_tools([FeedBack])
    # evaluation = chain.invoke(f"请对下面笑话进行评价：{state['joke']}")
    # evaluation = evaluation.tool_calls[-1]['args']
    # return {
    #     "funny_or_not": evaluation['grade'],
    #     "feedback": evaluation['feedback'],
    # }


# 条件边的路由函数
def route_func(state: State) -> str:
    """动态路由决策函数."""
    # return END if state.get("funny_or_not", None) == "funny" else "generator"
    # 上行即可，为了方便在调试时，将生成和评估的节点连接起来查看可视化graph，将节点动态映射在后面
    return 'Accepted' if state.get("funny_or_not", None) == "funny" else "Rejected + Feedback"


# 构建一个工作流
builder = StateGraph(State)

builder.add_node('generator', generator_func)
builder.add_node('evaluator', evaluator_func)

builder.add_edge(START, 'generator')
builder.add_edge('generator', 'evaluator')
builder.add_conditional_edges(
    'evaluator',
    route_func,
    {
        "Accepted": END,    # 合格则结束
        "Rejected + Feedback": 'generator'  # 不合格则循环优化
    }
)

graph = builder.compile()