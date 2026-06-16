# ================== 导入核心依赖 ==================
import random
import os
import json
import textwrap
from typing import TypedDict, List, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_community.chat_models.tongyi import ChatTongyi
# ================== 初始化大模型 ==================
load_dotenv()

llm = ChatTongyi(
    model="qwen3-max",
    temperature=0.7,
    max_tokens=500,
    ) # type: ignore

parser = StrOutputParser()

# ================== 1. 定义游戏状态 ==================
class GameState(TypedDict):
    """
    游戏状态字典，存储整个游戏的所有关键数据
    TypedDict：提供类型提示，避免键名错误
    """
    civilian_word: str  # 平民词语
    undercover_word: str  # 卧底词语
    role_assignment: dict  # 角色分配：{agent1: ("平民"/"卧底", 词语), ...}
    speeches: dict  # 当前轮发言：{agent1: "发言内容", ...}
    history_speeches: List[Dict[str, str]]  # 历史发言列表：[第1轮发言, 第2轮发言, ...]
    speech_reasoning: dict  # 发言策略理由：{agent1: "理由", ...}
    votes: dict  # 当前轮投票：{agent1: "投给agent2", ...}
    vote_reasoning: dict  # 投票理由：{agent1: "理由", ...}
    game_status: str  # 游戏状态：running（进行中）/end（结束）
    winner: str  # 获胜方：civilian（平民）/undercover（卧底）
    eliminated: List[str]  # 被淘汰的玩家列表
    round: int  # 当前游戏轮次

def init_game_state() -> GameState:
    return {
        "civilian_word": "",
        "undercover_word": "",
        "role_assignment": {},
        "speeches": {},
        "history_speeches": [],
        "speech_reasoning": {},
        "votes": {},
        "vote_reasoning": {},
        "game_status": "running",
        "winner": "",
        "eliminated": [],
        "round": 1
    }

# ================== 2. 节点函数 ==================
def generate_words(state: GameState) -> GameState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是专业的「谁是卧底」游戏出题人，需生成一组高质量的词语对。
核心要求：
1. 词语类型：日常物品/食品/场景（如：奶茶-果汁、牙刷-牙膏），避免生僻词
2. 语义关系：平民词与卧底词高度相似但核心特征不同，有足够博弈空间
3. 难度适配：适合4人游戏，既不轻易暴露也能通过描述区分
4. 输出格式：必须严格按照 JSON 格式输出，示例：{{"civilian": "奶茶", "undercover": "果汁"}}
禁止输出任何额外文字，只返回JSON字符串！"""),
        ("user", "生成一组符合要求的谁是卧底词语对")
    ])
    chain = prompt | llm | parser
    result = chain.invoke({})

    try:
        word_data = json.loads(result.strip())
        civilian_word = word_data["civilian"]
        undercover_word = word_data["undercover"]
    except (json.JSONDecodeError, KeyError):
        fallback_pairs = [
            ("奶茶", "果汁"), ("牙刷", "牙膏"), ("米饭", "面条"),
            ("手机", "平板"), ("篮球", "足球"), ("咖啡", "红茶")
        ]
        civilian_word, undercover_word = random.choice(fallback_pairs)

    state["civilian_word"] = civilian_word
    state["undercover_word"] = undercover_word
    print(f"\n🎯 词语生成完成：平民词={civilian_word} ｜ 卧底词={undercover_word}")
    return state

# ---- 节点2：分配角色 ----
def assign_roles(state: GameState) -> GameState:
    agents = ["agent1", "agent2", "agent3", "agent4"]
    undercover = random.choice(agents)
    for agent in agents:
        if agent == undercover:
            state["role_assignment"][agent] = ("卧底", state["undercover_word"])
        else:
            state["role_assignment"][agent] = ("平民", state["civilian_word"])

    print("\n🎭 角色分配完成：")
    for a, (r, w) in state["role_assignment"].items():
        print(f"  {a}：{r}（词语：{w}）")
    return state

# ---- 节点3：发言----
def generate_speeches(state: GameState) -> GameState:
    """
    节点3：生成智能体发言（发言/策略均不截断，仅Prompt引导10-100字）
    核心逻辑：
    1. 结合历史发言制定本轮发言策略（避免重复/矛盾）
    2. Prompt层面引导发言长度10-100字，不做强制截断
    3. 不同角色（平民/卧底）采用差异化发言策略
    4. 发言和策略理由完全保留原始内容，不做任何截断处理
    """
    speeches = {}
    reasoning = {}
    current_round = state["round"]
    
    # 格式化历史发言（多轮记忆核心：让智能体参考前轮发言）
    history_context = ""
    if state["history_speeches"]:
        history_context = "【历史发言记录】\n"
        for idx, round_speeches in enumerate(state["history_speeches"], 1):
            history_context += f"第{idx}轮发言：\n"
            for agent, speech in round_speeches.items():
                if agent not in state["eliminated"]:
                    history_context += f"- {agent}：{speech}\n"
        history_context += "\n"

    # 强化Prompt字数引导（不做后续截断，全靠LLM遵守）
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""你是「谁是卧底」游戏的资深玩家，当前是第{current_round}轮发言，需结合历史发言制定策略。
【核心规则】
1. 发言要求：
   - 字数：必须严格控制在10-100个汉字（不含标点），无需截断，直接生成符合长度的完整内容
   - 内容：描述词语特征，但绝对不能直接说出词语；结合历史发言调整策略，避免重复自己/他人的描述
   - 风格：自然口语化，句子完整通顺，逻辑清晰
   - 完整性：确保发言是完整的句子，语义完整不截断
2. 角色策略：
   - 平民：描述核心特征，帮助其他平民识别卧底；避免重复前轮发言，找出发言矛盾的玩家
   - 卧底：模仿平民的描述风格，模糊核心差异；避免与前轮自己的发言矛盾，同时不暴露身份
3. 输出格式：必须严格按照JSON格式输出，示例：
   {{{{"speech": "这是一种日常饮用的饮品，有多种口味可选，不同品牌的口感差异不大，平时在家或外出都经常能喝到", "reason": "作为平民，详细描述饮品特征，避免重复前轮发言，帮助其他平民识别卧底"}}}}
禁止输出任何额外文字，只返回JSON字符串！
{history_context}"""),
        ("user", "你的角色是{role}，拿到的词语是{word}")
    ])
    chain = prompt | llm | parser

    print(f"\n🗣 第{current_round}轮发言阶段（建议发言长度：10-100字）：")
    for agent, (role, word) in state["role_assignment"].items():
        if agent in state["eliminated"]:
            continue
        # 调用LLM生成符合角色策略的发言
        output = chain.invoke({"role": role, "word": word})
        
        try:
            # 解析LLM输出的JSON格式数据
            speech_data = json.loads(output.strip())
            raw_speech = speech_data["speech"]
            raw_reason = speech_data["reason"]
            
            # 核心修改1：移除发言截断，仅保留长度提示（不修改内容）
            speech = raw_speech
            # 长度提示（友好提醒，不强制修改）
            if len(speech) > 100:
                print(f"⚠️  {agent}（{role}）发言超过100字（实际{len(speech)}字），内容完整保留")
            elif len(speech) < 10:
                print(f"⚠️  {agent}（{role}）发言不足10字（实际{len(speech)}字），内容完整保留")
                
            # 兜底补充逻辑：仅补充内容，不截断（若仍需补充）
            if len(speech) < 10:
                if role == "平民":
                    speech = f"{speech}，是日常生活中很常见的物品，使用场景非常广泛，几乎每个人都接触过"
                else:
                    speech = f"{speech}，大家在生活中经常能见到或用到，不同场景下的用法基本一致，不容易区分"
                print(f"🔧 {agent}（{role}）发言补充后：{speech}（长度{len(speech)}字）")
                
        except (json.JSONDecodeError, KeyError):
            # LLM输出解析失败时的兜底发言（完整内容，不截断）
            if role == "平民":
                speech = f"第{current_round}轮发言：这是日常能用到的东西，使用频率很高，不同品牌的款式略有差异，但核心功能是一样的，几乎每个家庭都有这类物品，是生活中不可或缺的常用品"
                raw_reason = f"平民兜底发言，第{current_round}轮避免重复前轮，完整描述物品核心特征，不做截断处理"
            else:
                speech = f"第{current_round}轮发言：这是大家都熟悉的物品，平时使用场景很多，外观和功能都比较相似，很难快速区分不同类型，生活中随处可见，几乎每个人都使用过这类物品"
                raw_reason = f"卧底兜底发言，第{current_round}轮伪装平民，完整模糊描述特征避免暴露身份，不截断"
        
        reason = raw_reason

        # 保存当前智能体的发言和策略理由（完整内容）
        speeches[agent] = speech
        reasoning[agent] = reason
        # 打印发言结果（清晰展示角色和完整内容）
        print(f"\n{agent}（{role}）")
        print(f"  发言：{speech}")
        print(f"  策略：{reason}")

    # 将本轮发言存入历史（完整内容，供下一轮参考）
    state["history_speeches"].append(speeches.copy())
    state["speeches"] = speeches
    state["speech_reasoning"] = reasoning
    return state

def vote_undercover(state: GameState) -> GameState:
    votes = {}
    reasons = {}
    current_agents = [a for a in state["role_assignment"] if a not in state["eliminated"]]
    current_round = state["round"]
    
    # 格式化发言上下文
    speech_context = f"【第{current_round}轮发言】\n"
    speech_context += "\n".join([f"{agent}：{speech}" for agent, speech in state["speeches"].items()])
    
    if state["history_speeches"]:
        speech_context += "\n\n【历史发言参考】\n"
        for idx, round_speeches in enumerate(state["history_speeches"][:-1], 1):
            speech_context += f"第{idx}轮：\n"
            for agent, speech in round_speeches.items():
                if agent in current_agents:
                    speech_context += f"- {agent}：{speech}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是「谁是卧底」游戏的理性玩家，需基于当前轮+历史发言分析并投票。
【分析规则】
1. 投票依据：
   - 对比玩家当前轮和历史发言，找出矛盾/异常的描述（卧底常出现前后矛盾）
   - 平民：重点关注发言前后不一致、描述偏离词语特征的玩家
   - 卧底：找出看起来像平民的玩家投票，避免自己被怀疑，保持投票理由连贯
2. 输出格式：必须严格按照JSON格式输出，示例：
   {{{{"vote": "agent2", "reason": "agent2本轮和上轮发言矛盾，描述不符合平民词特征"}}}}
禁止输出任何额外文字，只返回JSON字符串！
{speech_context}"""),
        ("user", """你的角色：{role}
你的词语：{word}
请选择你要投票的玩家并说明理由（理由控制在50字内）""")
    ])
    chain = prompt | llm | parser

    print(f"\n🗳 第{current_round}轮投票阶段：")
    for agent, (role, word) in state["role_assignment"].items():
        if agent in state["eliminated"]:
            continue
        output = chain.invoke({
            "role": role,
            "word": word,
            "speech_context": speech_context
        })
        
        try:
            vote_data = json.loads(output.strip())
            vote = vote_data["vote"].strip()
            raw_reason = vote_data["reason"]
            
            reason = raw_reason
            
        except (json.JSONDecodeError, KeyError):
            vote = random.choice([a for a in current_agents if a != agent])
            reason = textwrap.shorten(
                f"第{current_round}轮无有效分析，基于随机策略投票",
                width=50
            )
        
        # 校验投票有效性
        if vote == agent or vote not in current_agents:
            vote = random.choice([a for a in current_agents if a != agent])
        
        votes[agent] = vote
        reasons[agent] = reason
        print(f"\n{agent}（{role}）")
        print(f"  投票给：{vote}")
        print(f"  理由：{reason}")

    state["votes"] = votes
    state["vote_reasoning"] = reasons
    return state

# ---- 节点5：裁决 ----
def judge_result(state: GameState) -> GameState:
    vote_count = {}
    for v in state["votes"].values():
        vote_count[v] = vote_count.get(v, 0) + 1
    max_vote = max(vote_count.values())
    eliminated = random.choice([a for a, c in vote_count.items() if c == max_vote])
    state["eliminated"].append(eliminated)
    role = state["role_assignment"][eliminated][0]
    current_round = state["round"]
    
    print(f"\n❌ 第{current_round}轮淘汰结果：{eliminated}（{role}）")

    remaining = [a for a in state["role_assignment"] if a not in state["eliminated"]]
    civ = sum(1 for a in remaining if state["role_assignment"][a][0] == "平民")
    uc = sum(1 for a in remaining if state["role_assignment"][a][0] == "卧底")

    if role == "卧底":
        state["game_status"] = "end"
        state["winner"] = "civilian"
        print("🎉 平民胜利！")
    elif civ == 1 and uc == 1:
        state["game_status"] = "end"
        state["winner"] = "undercover"
        print("🎉 卧底胜利！")
    else:
        state["game_status"] = "running"
        state["round"] += 1
        print(f"➡ 游戏继续，进入第{state['round']}轮")
    return state

# ---- 节点6：总结 ----
def show_final_result(state: GameState) -> GameState:
    print("\n" + "="*50)
    print("📜 游戏结束 · 总结")
    print(f"胜利方：{'平民' if state['winner'] == 'civilian' else '卧底'}")
    print(f"平民词：{state['civilian_word']} | 卧底词：{state['undercover_word']}")
    print(f"总轮次：{state['round']}")
    print(f"淘汰顺序：{state['eliminated']}")
    print("="*50)
    return state

# ================== 3. 构建 LangGraph ==================
def build_game_graph():
    graph = StateGraph(GameState)
    graph.add_node("generate_words", generate_words)
    graph.add_node("assign_roles", assign_roles)
    graph.add_node("generate_speeches", generate_speeches)
    graph.add_node("vote_undercover", vote_undercover)
    graph.add_node("judge_result", judge_result)
    graph.add_node("show_final_result", show_final_result)
    graph.set_entry_point("generate_words")

    graph.add_edge("generate_words", "assign_roles")
    graph.add_edge("assign_roles", "generate_speeches")
    graph.add_edge("generate_speeches", "vote_undercover")
    graph.add_edge("vote_undercover", "judge_result")

    def route(state: GameState):
        return "generate_speeches" if state["game_status"] == "running" else "show_final_result"
    graph.add_conditional_edges("judge_result", route)
    graph.add_edge("show_final_result", END)
    return graph

# ================== 4. 入口 ==================
if __name__ == "__main__":
    game_graph = build_game_graph()
    game = game_graph.compile()
    print("="*50)
    print("🎮 谁是卧底 · 多智能体多轮策略版 启动")
    print("="*50)
    game.invoke(init_game_state())