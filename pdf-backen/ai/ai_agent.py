"""
LangGraph StateGraph — Crystal 论文问答核心编排。

架构(对应计划中的 mermaid):
  load_paper_memory -> load_agent_memory -> compose_messages -> llm_call
                     -> save_paper_memory -> update_agent_memory (异步)

关键设计:
- 持久化:
    save/{fp}_ai.json         每篇论文的对话历史 [{role, content, ts}]
    save/Crystal_memory.md    Crystal 对用户的认知沉淀(自由 Markdown, 按 agent 名字隔离)
- memorylist(阅读过的论文片段) 不在这里, 由前端在请求时透传, 见 buildAskMessages / buildLoadMessages
- update_agent_memory 用 asyncio.create_task() 异步触发, 不阻塞响应
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import TypedDict, Optional, Any

import httpx

from .ai_models import AiAskReq, AiLoadReq, AiResp
from .prompts import (
    buildAskMessages, buildLoadMessages,
    MEMORY_UPDATE_SYSTEM, buildMemoryUpdateUserPrompt,
)
from utils.log import debug


# ═══════════════════════════════════════════════════════════════════════
# 路径与常量
# ═══════════════════════════════════════════════════════════════════════

def _get_base_dir() -> str:
    """打包环境兼容: PyInstaller 临时目录 vs 开发目录。

    ai/* 子模块的 BASE_DIR 应当回到 pdf-backen 项目根目录,
    这样 save / static 等目录与 PdfBacken.py 一致。
    """
    if getattr(sys, "frozen", False):
        return sys._MEIPASS  # type: ignore[attr-defined]
    # 本文件位于 pdf-backen/ai/<name>.py, 向上一层就是 pdf-backen 项目根
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BASE_DIR = _get_base_dir()
SAVE_DIR = os.path.join(BASE_DIR, "save")
os.makedirs(SAVE_DIR, exist_ok=True)

# Crystal_mem 文件名 — 计划要求按 agent 名字隔离, 默认 Crystal
CRYSTAL_MEMORY_FILE = os.path.join(SAVE_DIR, "persistence", "Crystal_memory.md")

# Crystal_memory.md 内存缓存。
# 避免每次 /ai/ask 或 /ai/load 都重新打开文件读磁盘。
# 后台任务 _update_crystal_memory_async 写文件成功后, 调用 _invalidate_agent_memory_cache() 同步刷新。
# 这样既快又保持一致性: 如果后台写失败, 下次 _get_agent_memory() 会从磁盘回读(可能拿到上次的内容)。
_agent_memory_cache: str | None = None


def _get_agent_memory() -> str:
    """
    读 agent_memory 缓存。首次调用时同步加载磁盘内容。
    LLM 输出通常只读这份缓存(通过 load_agent_memory_node),
    不需要每次都打开 Crystal_memory.md 文件。
    """
    global _agent_memory_cache
    if _agent_memory_cache is None:
        if os.path.exists(CRYSTAL_MEMORY_FILE):
            try:
                with open(CRYSTAL_MEMORY_FILE, "r", encoding="utf-8") as f:
                    _agent_memory_cache = f.read()
            except OSError:
                _agent_memory_cache = ""
        else:
            _agent_memory_cache = ""
    return _agent_memory_cache


def _set_agent_memory_cache(md: str) -> None:
    """后台任务写完文件后调用, 同步刷新缓存, 避免下个请求读到陈旧数据。"""
    global _agent_memory_cache
    _agent_memory_cache = md

# DeepSeek API 需要在 base url 后拼 /chat/completions
DEEPSEEK_MARKER = "deepseek.com"


def _paper_history_path(pdf_fp: str) -> str:
    return os.path.join(SAVE_DIR, f"{pdf_fp}_ai.json")


# ═══════════════════════════════════════════════════════════════════════
# LangGraph State
# ═══════════════════════════════════════════════════════════════════════

class PaperAIState(TypedDict):
    """
    LangGraph 工作区状态。

    req:        入口请求(AiAskReq 或 AiLoadReq)
    messages:    组装好的 OpenAI 格式 messages, 准备送给 llm_call
    paper_messages: 从 {fp}_ai.json 加载的历史对话
    agent_memory: 从 Crystal_memory.md 加载的 Markdown 全文(注入 system prompt)
    memorylist:  用户阅读过的论文片段(由前端透传)
    final_answer: llm_call 返回的最终 content
    usage:        上游 LLM 的 usage 统计
    """
    req: Any  # AiAskReq | AiLoadReq
    messages: list[dict]
    paper_messages: list[dict]
    agent_memory: str
    memorylist: list[str]
    final_answer: str
    usage: dict


# ═══════════════════════════════════════════════════════════════════════
# AI 配置 — 前端 SET_MODEL 时通过 /ai/config 写入, 后续请求直接读取
# ═══════════════════════════════════════════════════════════════════════

ai_config: dict = {
    "api_key": "",
    "api_url": "",
    "model": "",          # 普通文本模型 (ask / load)
    "vision_model": "",   # 视觉模型 (ask + image_base64)
    "deepseek_thinking": False,
}


def update_ai_config(config: dict) -> None:
    """由 /ai/config 端点调用, 更新模块级 ai_config。"""
    global ai_config
    ai_config.update({
        "api_key": config.get("api_key") or "",
        "api_url": config.get("api_url") or "",
        "model": config.get("model") or "",
        "vision_model": config.get("vision_model") or config.get("model") or "",
        "deepseek_thinking": bool(config.get("deepseek_thinking")),
    })
    debug(f"[ai_agent] ai_config updated: model={ai_config['model']} vision={ai_config['vision_model']}")


def get_current_config() -> dict:
    """返回当前 ai_config 快照 (供 /ai/config GET 使用)。"""
    return dict(ai_config)


# ═══════════════════════════════════════════════════════════════════════
# 工具函数 — URL 解析 / chat 调用
# ═══════════════════════════════════════════════════════════════════════

def _is_deepseek(api_url: str) -> bool:
    return DEEPSEEK_MARKER in api_url


def _resolve_api_url(api_url: str) -> str:
    base = api_url.rstrip("/")
    if _is_deepseek(api_url):
        return f"{base}/chat/completions"
    return base


async def _call_llm(
    messages: list[dict],
    vision_model: bool = False,
) -> tuple[str, dict]:
    """
    直接调上游 LLM, 配置全部从 ai_config 读取。
    vision_model 参数仅用于判断 DeepSeek thinking 开关(仅非视觉模式有效)。
    """
    api_url = ai_config["api_url"]
    api_key = ai_config["api_key"]
    model = ai_config["vision_model"] if vision_model else ai_config["model"]

    if not api_url or not api_key or not model:
        raise ValueError(
            f"ai_config 未完整配置: api_url={bool(api_url)} "
            f"api_key={bool(api_key)} model={bool(model)}"
        )

    request_url = _resolve_api_url(api_url)
    is_ds = _is_deepseek(api_url)

    request_body: dict = {"model": model, "messages": messages}

    # DeepSeek thinking 参数仅兼容非视觉模型, 视觉模式下不传
    if is_ds and not vision_model and ai_config["deepseek_thinking"]:
        request_body["thinking"] = {"type": "enabled"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "APP-Code": "DMQU5622",
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        res = await client.post(request_url, json=request_body, headers=headers)
        res.raise_for_status()
        data = res.json()

    content = (
        data.get("choices", [{}])[0].get("message", {}).get("content")
        or data.get("choices", [{}])[0].get("text")
        or json.dumps(data, ensure_ascii=False)
    )
    usage = data.get("usage") or {}
    return content, usage


# ═══════════════════════════════════════════════════════════════════════
# LangGraph 节点
# ═══════════════════════════════════════════════════════════════════════

async def load_paper_memory_node(state: PaperAIState) -> dict:
    """读 save/{fp}_ai.json, 加载按论文隔离的对话历史(暂不直接喂给 LLM, 仅持久化时回写)"""
    req = state["req"]
    fp = req.pdf_fp
    path = _paper_history_path(fp)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except (json.JSONDecodeError, OSError):
            data = []
    else:
        data = []
    return {"paper_messages": data}


async def load_agent_memory_node(state: PaperAIState) -> dict:
    """读 agent_memory 缓存(首次才打磁盘)。作为 system prompt 注入。

    缓存策略: 模块级 _agent_memory_cache, 首次调用时同步从 Crystal_memory.md 加载,
    之后 _update_crystal_memory_async 写完文件会同步刷新缓存。
    """
    return {"agent_memory": _get_agent_memory()}


async def compose_messages_node(state: PaperAIState) -> dict:
    """
    根据 req 类型 (Ask / Load) 调用 buildAskMessages / buildLoadMessages 组装 messages。
    额外把 Crystal_memory.md 内容追加到 system prompt 末尾(让 Crystal 越来越懂用户)。
    """
    req = state["req"]
    memorylist = state.get("memorylist") or []

    if isinstance(req, AiAskReq):
        messages = buildAskMessages(req, memorylist)
    elif isinstance(req, AiLoadReq):
        messages = buildLoadMessages(req, memorylist)
    else:
        raise ValueError(f"Unknown req type: {type(req)}")

    # 注入 agent memory — 拼到 system 消息末尾
    if state.get("agent_memory"):
        appended = (
            "\n\n【关于这位用户的认知(Crystal 私人笔记, 不要对用户直述)】\n"
            + state["agent_memory"]
        )
        messages[0]["content"] = messages[0]["content"] + appended

    return {"messages": messages}


async def llm_call_node(state: PaperAIState) -> dict:
    """
    调上游 LLM。模型选择由 ai_config 决定:
      - AiAskReq + image_base64 非空 -> vision_model
      - 否则 -> model (普通 ask / load)
    """
    req = state["req"]
    messages = state["messages"]

    is_vision = isinstance(req, AiAskReq) and req.image_base64 is not None

    content, usage = await _call_llm(
        messages=messages,
        vision_model=is_vision,
    )

    return {"final_answer": content, "usage": usage}


async def save_paper_memory_node(state: PaperAIState) -> dict:
    """把本轮 user + assistant 写入 save/{fp}_ai.json(ADD 模式)"""
    req = state["req"]
    fp = req.pdf_fp
    answer = state["final_answer"]

    # Ask 模式有 req.ask; Load 模式用 req.chosen_text 充当 user 文本
    if isinstance(req, AiAskReq):
        user_text = req.ask
    else:
        user_text = req.chosen_text

    path = _paper_history_path(fp)
    history = state.get("paper_messages") or []
    if not isinstance(history, list):
        history = []

    now_ms = int(time.time() * 1000)
    history.append({"role": "user", "content": user_text, "ts": now_ms})
    history.append({"role": "assistant", "content": answer, "ts": now_ms + 1})

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
        debug(f"[ai_agent] save_paper_memory ok: fp={fp} len={len(history)}")
    except OSError as e:
        debug(f"[ai_agent] save_paper_memory FAIL: {e} | fp={fp}")

    return {}


async def update_agent_memory_node(state: PaperAIState) -> dict:
    """
    Ask 模式: 异步触发, 顺便把前端传来的 load_buffer 一并喂给 LLM。
    Load 模式: 不更新 Crystal_memory.md(划词不维护持久化记忆)。
    """
    req = state["req"]
    answer = state["final_answer"]

    if isinstance(req, AiAskReq):
        user_msg = req.ask
        load_buffer = req.load_buffer or []
        # 异步执行, 不 await
        task = asyncio.create_task(
            _update_crystal_memory_async(user_msg, answer, load_buffer)
        )
        debug(
            f"[crystal_memory] scheduled [ask]: "
            f"user_len={len(user_msg)} asst_len={len(answer)} "
            f"load_buffer={len(load_buffer)} task_id={id(task)}"
        )
    else:
        # Load 模式: 不维护 Crystal_memory.md
        debug("[crystal_memory] skipped [load]")

    return {}


async def _update_crystal_memory_async(
    user_msg: str,
    assistant_msg: str,
    load_buffer: list[dict],
) -> None:
    """
    后台任务: 读 Crystal_memory.md, 把 load_buffer + 本轮对话一起喂 LLM, 写回。
    复用 ai_config 中的 api_key / api_url / model。

    异常静默, 不影响用户响应。Crystal_mem.md 是锦上添花, 损坏不应阻塞主链路。
    """
    try:
        if not ai_config["api_key"] or not ai_config["api_url"]:
            debug("[crystal_memory] skip: ai_config 未设置")
            return

        # 读当前 Markdown
        current = ""
        if os.path.exists(CRYSTAL_MEMORY_FILE):
            try:
                with open(CRYSTAL_MEMORY_FILE, "r", encoding="utf-8") as f:
                    current = f.read()
            except OSError:
                current = ""

        new_md = await _call_memory_update_llm(
            current,
            user_msg,
            assistant_msg,
            load_buffer,
        )
        if new_md:
            with open(CRYSTAL_MEMORY_FILE, "w", encoding="utf-8") as f:
                f.write(new_md)
            # 同步刷新内存缓存, 下次 /ai/ask /ai/load 立刻拿到新内容
            _set_agent_memory_cache(new_md)
            prev_len = len(current)
            new_len = len(new_md)
            delta = new_len - prev_len
            debug(
                f"[crystal_memory] WRITE ok: "
                f"prev_len={prev_len} new_len={new_len} delta={delta:+d} "
                f"path={CRYSTAL_MEMORY_FILE}"
            )
    except Exception as e:
        debug(f"[crystal_memory] update FAIL: {type(e).__name__}: {e}")


async def _call_memory_update_llm(
    current_md: str,
    user_msg: str,
    assistant_msg: str,
    load_buffer: list[dict],
) -> str:
    """调 LLM 让它合并更新 Crystal_mem.md, 返回新的 Markdown 文本。配置从 ai_config 读取。"""
    messages = [
        {"role": "system", "content": MEMORY_UPDATE_SYSTEM()},
        {"role": "user", "content": buildMemoryUpdateUserPrompt(
            current_md, user_msg, assistant_msg, load_buffer
        )},
    ]
    content, _ = await _call_llm(messages=messages, vision_model=False)
    return content.strip()


# ═══════════════════════════════════════════════════════════════════════
# Graph 构建
# ═══════════════════════════════════════════════════════════════════════

def build_graph():
    """
    构建 LangGraph StateGraph:
      load_paper_memory -> load_agent_memory -> compose_messages
      -> llm_call -> save_paper_memory -> update_agent_memory
    """
    # LangGraph 在新版是 langgraph.graph.StateGraph
    # 这里延迟 import, 避免冷启动开销
    from langgraph.graph import StateGraph, END  # type: ignore

    g = StateGraph(PaperAIState)
    g.add_node("load_paper_memory", load_paper_memory_node)
    g.add_node("load_agent_memory", load_agent_memory_node)
    g.add_node("compose_messages", compose_messages_node)
    g.add_node("llm_call", llm_call_node)
    g.add_node("save_paper_memory", save_paper_memory_node)
    g.add_node("update_agent_memory", update_agent_memory_node)

    g.set_entry_point("load_paper_memory")
    g.add_edge("load_paper_memory", "load_agent_memory")
    g.add_edge("load_agent_memory", "compose_messages")
    g.add_edge("compose_messages", "llm_call")
    g.add_edge("llm_call", "save_paper_memory")
    g.add_edge("save_paper_memory", "update_agent_memory")
    g.add_edge("update_agent_memory", END)

    return g.compile()


# ═══════════════════════════════════════════════════════════════════════
# 入口 — ai_routes 直接调用 run_ask / run_load
# ═══════════════════════════════════════════════════════════════════════

_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


async def run_ask(req: AiAskReq, memorylist: Optional[list[str]] = None) -> dict:
    """Run LangGraph for Ask 模式 — 返回 state 字典"""
    initial: PaperAIState = {
        "req": req,
        "messages": [],
        "paper_messages": [],
        "agent_memory": "",
        "memorylist": memorylist or [],
        "final_answer": "",
        "usage": {},
    }
    graph = _get_graph()
    result = await graph.ainvoke(initial)
    return result


async def run_load(req: AiLoadReq, memorylist: Optional[list[str]] = None) -> dict:
    """Run LangGraph for Load 模式"""
    initial: PaperAIState = {
        "req": req,
        "messages": [],
        "paper_messages": [],
        "agent_memory": "",
        "memorylist": memorylist or [],
        "final_answer": "",
        "usage": {},
    }
    graph = _get_graph()
    result = await graph.ainvoke(initial)
    return result
