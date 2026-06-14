

# 1. 导入需要的模块
import os 
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi



# 4. 初始化大模型（和LangChain案例一样）
llm = ChatTongyi(
    model="qwen3-max"
    ) # type: ignore


# 5. 定义 State
class WorkflowState(TypedDict, total=False):
    user_role: str  # 存储用户角色
    original_advice: str  # 存储原始学习建议
    simplified_advice: str  # 存储精简后的建议
    translated_advice: str  # 存储翻译后的建议

# 6. 定义节点
def generate_advice(state: WorkflowState):
    prompt = f"给{state['user_role']}写一段50字左右的LangChain学习建议。"
    result = llm.invoke(prompt)
    return {"original_advice": result.content}

def simplify_advice(state: WorkflowState):
    prompt = f"把下面的学习建议精简到30字以内：{state['original_advice']}"
    result = llm.invoke(prompt)
    return {"simplified_advice": result.content}

def translate_advice(state: WorkflowState):
    prompt = f"把下面的学习建议翻译成英文：{state['simplified_advice']}"
    result = llm.invoke(prompt)
    return {"translated_advice": result.content}

# 7. 构建工作流
workflow = StateGraph(WorkflowState)

workflow.add_node("generate", generate_advice)
workflow.add_node("simplify", simplify_advice)
workflow.add_node("translate", translate_advice)

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "simplify")
workflow.add_edge("simplify", "translate")
workflow.add_edge("translate", END)

app = workflow.compile()

# 8. 执行
result = app.invoke({"user_role": "高校学生"})

# 9. 输出
print("原始学习建议：")
print(result["original_advice"])
print("\n精简后学习建议：")
print(result["simplified_advice"])
print("\n翻译后的学习建议：")
print(result["translated_advice"])