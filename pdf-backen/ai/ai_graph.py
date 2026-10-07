"""
LangGraph StateGraph 构建 + 入口函数 (run_ask / run_load)。

把 ai_agent.py 末尾的 build_graph / run_ask / run_load 抽出, 节点实现仍在 ai_agent.py。
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from .ai_active import layer1_node, split_followup_node
from .ai_agent import (
    PaperAIState,
    compose_messages_node,
    flush_track_node,
    llm_call_node,
    load_agent_memory_node,
    load_paper_history_node,
    save_paper_memory_node,
    update_agent_memory_node,
)
from .ai_emotion import emotion_llm_node
from .ai_models import AiAskReq, AiLoadReq


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def build_graph():
    """
    构建 LangGraph StateGraph:
      load_agent_memory -> load_paper_history -> compose_messages
      -> llm_call -> split_followup -> save_paper_memory
      -> update_agent_memory -> emotion_llm -> active_layer1 -> flush_track

    emotion_llm: Lint LLM 异步评估情绪, 每次 chat ask 都触发,
    非 chat 场景 (论文) 跳过。与主链路完全并行, 不阻塞响应。

    active_layer1: Layer1 短期追问 (ai_active.layer1_node)。
    挂载在 emotion_llm 之后, flush_track 之前。
    仅 chat 场景 (pdf_fp == CHAT_FP) 真正启动后台 pipeline; 论文侧
    layer1_node 内部直接 return {} (无副作用)。
    fire-and-forget: layer1_node 内部用 asyncio.create_task 启动后台
    pipeline, 自己立即 return; 主回复不阻塞。

    split_followup (v2 改造): 同步 lint 拆 MR 末尾隐含追问。
    挂载在 llm_call 之后, save_paper_memory 之前 (保证 main_body 写入磁盘前已改 final_answer)。
    仅 chat 侧真正拆; 论文侧直接 return {split_followup: None} 不调 LLM。
    拆出后 layer1_node 会走 split 优先路径 (跳过 judge + compose)。
    """
    g = StateGraph(PaperAIState)
    g.add_node("load_agent_memory",    load_agent_memory_node)
    g.add_node("load_paper_history",     load_paper_history_node)
    g.add_node("compose_messages",      compose_messages_node)
    g.add_node("llm_call",             llm_call_node)
    g.add_node("split_followup",         split_followup_node)
    g.add_node("save_paper_memory",     save_paper_memory_node)
    g.add_node("update_agent_memory",    update_agent_memory_node)
    g.add_node("emotion_llm",           emotion_llm_node)
    g.add_node("active_layer1",          layer1_node)
    g.add_node("flush_track",           flush_track_node)

    g.set_entry_point("load_agent_memory")
    g.add_edge("load_agent_memory",    "load_paper_history")
    g.add_edge("load_paper_history",   "compose_messages")
    g.add_edge("compose_messages",     "llm_call")
    g.add_edge("llm_call",            "split_followup")
    g.add_edge("split_followup",        "save_paper_memory")
    g.add_edge("save_paper_memory",    "update_agent_memory")
    g.add_edge("update_agent_memory",  "emotion_llm")
    g.add_edge("emotion_llm",          "active_layer1")
    g.add_edge("active_layer1",        "flush_track")
    g.add_edge("flush_track",          END)

    return g.compile()


async def run_ask(req: AiAskReq) -> dict:
    """Run LangGraph for Ask 模式 — 返回 state 字典"""
    initial: PaperAIState = {
        "req": req,
        "messages": [],
        "agent_memory": "",
        "paper_history": [],
        "paper_ask_history": [],
        "paper_load_history": [],
        "final_answer": "",
        "split_followup": None,
        "usage": {},
        "dt": "",
        "req_fp": "",
        "res_fp": "",
        "emotion": {},
    }
    graph = _get_graph()
    result = await graph.ainvoke(initial)
    return result


async def run_load(req: AiLoadReq) -> dict:
    """Run LangGraph for Load 模式"""
    initial: PaperAIState = {
        "req": req,
        "messages": [],
        "agent_memory": "",
        "paper_history": [],
        "paper_ask_history": [],
        "paper_load_history": [],
        "final_answer": "",
        "split_followup": None,
        "usage": {},
        "dt": "",
        "req_fp": "",
        "res_fp": "",
        "emotion": {},
    }
    graph = _get_graph()
    result = await graph.ainvoke(initial)
    return result
