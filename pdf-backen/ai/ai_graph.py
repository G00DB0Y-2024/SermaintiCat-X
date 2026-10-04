"""
LangGraph StateGraph 构建 + 入口函数 (run_ask / run_load)。

把 ai_agent.py 末尾的 build_graph / run_ask / run_load 抽出, 节点实现仍在 ai_agent.py。
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

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
      -> llm_call -> save_paper_memory
      -> update_agent_memory -> emotion_llm -> flush_track

    emotion_llm: Lint LLM 异步评估情绪, 每次 chat ask 都触发,
    非 chat 场景 (论文) 跳过。与主链路完全并行, 不阻塞响应。
    """
    g = StateGraph(PaperAIState)
    g.add_node("load_agent_memory",    load_agent_memory_node)
    g.add_node("load_paper_history",     load_paper_history_node)
    g.add_node("compose_messages",      compose_messages_node)
    g.add_node("llm_call",             llm_call_node)
    g.add_node("save_paper_memory",     save_paper_memory_node)
    g.add_node("update_agent_memory",    update_agent_memory_node)
    g.add_node("emotion_llm",           emotion_llm_node)
    g.add_node("flush_track",           flush_track_node)

    g.set_entry_point("load_agent_memory")
    g.add_edge("load_agent_memory",    "load_paper_history")
    g.add_edge("load_paper_history",   "compose_messages")
    g.add_edge("compose_messages",     "llm_call")
    g.add_edge("llm_call",            "save_paper_memory")
    g.add_edge("save_paper_memory",    "update_agent_memory")
    g.add_edge("update_agent_memory",  "emotion_llm")
    g.add_edge("emotion_llm",          "flush_track")
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
        "usage": {},
        "dt": "",
        "req_fp": "",
        "res_fp": "",
        "emotion": {},
    }
    graph = _get_graph()
    result = await graph.ainvoke(initial)
    return result
