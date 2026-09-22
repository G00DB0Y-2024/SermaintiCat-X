"""
LangGraph StateGraph — Crystal 论文问答核心编排。

架构(对应计划中的 mermaid):
  load_paper_memory -> load_agent_memory -> compose_messages -> llm_call
                     -> save_paper_memory -> update_agent_memory (异步)

关键设计:
- 持久化:
    save/{fp}_ai.json        每篇论文的对话历史 [{role, content, ts}]
    ai/memory/Crystal_memory.md   Crystal 对用户的认知沉淀(自由 Markdown)
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
    get_current_time_context,
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

# Crystal_mem 文件名
CRYSTAL_MEMORY_FILE = os.path.join(BASE_DIR, "ai", "memory", "Crystal_memory.md")

# Crystal_memory.md 内存缓存。
# 避免每次 /ai/ask 或 /ai/load 都重新打开文件读磁盘。
# 后台任务 _update_crystal_memory_async 写文件成功后, 调用 _invalidate_agent_memory_cache() 同步刷新。
# 这样既快又保持一致性: 如果后台写失败, 下次 _get_agent_memory() 会从磁盘回读(可能拿到上次的内容)。
_agent_memory_cache: str | None = None


def _ensure_memory_dir() -> None:
    """冷启动: 确保 ai/memory/ 目录存在, 首次调用时执行一次。"""
    mem_dir = os.path.dirname(CRYSTAL_MEMORY_FILE)
    if not os.path.exists(mem_dir):
        os.makedirs(mem_dir, exist_ok=True)


def _get_agent_memory() -> str:
    """
    读 agent_memory 缓存。首次调用时同步加载磁盘内容, 必要时创建目录和空文件。
    LLM 输出通常只读这份缓存(通过 load_agent_memory_node),
    不需要每次都打开 Crystal_memory.md 文件。
    """
    global _agent_memory_cache
    if _agent_memory_cache is None:
        _ensure_memory_dir()
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


def _load_paper_history(fp: str, limit: int) -> list[dict]:
    """
    读 save/{fp}_ai.json, 返回最近 limit 轮的 messages (OpenAI 格式)。

    - 每轮 = 1 条 user + 1 条 assistant (limit 轮 = limit*2 条 entry)
    - 按 ts 倒序截取后 reverse 回正序 (OpenAI 要求时间正序)
    - limit=0 或文件不存在 / 异常 → 返回 []
    - 不影响主链路, 异常静默
    """
    if limit <= 0:
        return []
    path = _paper_history_path(fp)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        # data: [{role, content, ts}, ...] — 按 ts 已经正序
        # 取最后 limit*2 条 (即最近 limit 轮), 反转为正序
        tail = data[-(limit * 2):]
        # 仅保留 role+content (剥掉 ts, 防止脏数据塞进 messages)
        cleaned = []
        for e in tail:
            if not isinstance(e, dict):
                continue
            role = e.get("role")
            content = e.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                cleaned.append({"role": role, "content": content})
        return cleaned
    except (json.JSONDecodeError, OSError):
        return []


# ═══════════════════════════════════════════════════════════════════════
# LangGraph State
# ═══════════════════════════════════════════════════════════════════════

class PaperAIState(TypedDict):
    """
    LangGraph 工作区状态。

    req:           入口请求(AiAskReq 或 AiLoadReq)
    messages:      组装好的 OpenAI 格式 messages, 准备送给 llm_call
    agent_memory:  从 Crystal_memory.md 加载的 Markdown 全文(注入 system prompt)
    paper_history: 当前 pdf_fp 最近 N 轮 paper_history (Ask 20 / Load 10)
    final_answer:  llm_call 返回的最终 content
    usage:         上游 LLM 的 usage 统计
    """
    req: Any  # AiAskReq | AiLoadReq
    messages: list[dict]
    agent_memory: str
    paper_history: list[dict]
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

async def load_agent_memory_node(state: PaperAIState) -> dict:
    """读 agent_memory 缓存(首次才打磁盘)。作为 system prompt 注入。

    缓存策略: 模块级 _agent_memory_cache, 首次调用时同步从 Crystal_memory.md 加载,
    之后 _update_crystal_memory_async 写完文件会同步刷新缓存。
    """
    return {"agent_memory": _get_agent_memory()}


async def load_paper_history_node(state: PaperAIState) -> dict:
    """
    按 req 类型决定取多少轮 paper history:
      - Ask 模式: 20 轮 (40 条 entry)
      - Load 模式: 10 轮 (20 条 entry)
    不分 mode, Ask/Load 全部加载, 按时间倒序截取后 reverse 回正序。
    """
    req = state["req"]
    limit = 20 if isinstance(req, AiAskReq) else 10
    history = _load_paper_history(req.pdf_fp, limit)
    debug(f"[paper_history] loaded: fp={req.pdf_fp} mode={'ask' if isinstance(req, AiAskReq) else 'load'} rounds={len(history)//2}")
    return {"paper_history": history}


async def compose_messages_node(state: PaperAIState) -> dict:
    """
    根据 req 类型 (Ask / Load) 调用 buildAskMessages / buildLoadMessages 组装 messages。
    Ask 模式额外把 Crystal_memory.md 内容追加到 system prompt 末尾。
    Load 模式不注入 agent_memory。
    """
    req = state["req"]
    history = state.get("paper_history") or []
    agent_mem = state.get("agent_memory") or ""

    # 动态生成时间上下文
    time_context = get_current_time_context()

    if isinstance(req, AiAskReq):
        messages = buildAskMessages(req, history, time_context)
    elif isinstance(req, AiLoadReq):
        messages = buildLoadMessages(req, history, time_context)
    else:
        raise ValueError(f"Unknown req type: {type(req)}")

    # Ask 模式才注入 agent memory — 拼到 system 消息末尾
    if isinstance(req, AiAskReq) and agent_mem:
        messages[0]["content"] = messages[0]["content"] + (
            "\n\n【关于这位用户的认知(Crystal 私人笔记, 不要对用户直述)】\n"
            + agent_mem
        )

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
    """把本轮 user + assistant 追加写入 save/{fp}_ai.json (ADD 模式)"""
    req = state["req"]
    fp = req.pdf_fp
    answer = state["final_answer"]

    # Ask 模式有 req.ask; Load 模式用 req.chosen_text 充当 user 文本
    if isinstance(req, AiAskReq):
        user_text = req.ask
    else:
        user_text = req.chosen_text

    path = _paper_history_path(fp)
    # 节点间不共享 paper_messages,这里直接读磁盘,确保多请求并发也只追加自己的两条
    history: list = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                history = data
        except (json.JSONDecodeError, OSError):
            pass

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
    Ask 模式: 异步触发, 用 state.paper_history 作为更新参考(包含上次 Ask 到本轮的所有轨迹),
              用本轮 req.ask + final_answer 作为核心更新依据。
    Load 模式: 不更新 Crystal_memory.md(划词不维护持久化记忆)。
    """
    req = state["req"]
    answer = state["final_answer"]

    if isinstance(req, AiAskReq):
        user_msg = req.ask
        # paper_history 已经在 load_paper_history_node 加载过 (Ask = 20 轮),
        # 天然覆盖"上次 Ask 到这次 Ask"之间的所有对话轨迹
        intermediate_history = state.get("paper_history") or []
        # 异步执行, 不 await
        task = asyncio.create_task(
            _update_crystal_memory_async(user_msg, answer, intermediate_history)
        )
        debug(
            f"[crystal_memory] scheduled [ask]: "
            f"user_len={len(user_msg)} asst_len={len(answer)} "
            f"intermediate_rounds={len(intermediate_history)//2} task_id={id(task)}"
        )
    else:
        # Load 模式: 不维护 Crystal_memory.md
        debug("[crystal_memory] skipped [load]")

    return {}


async def _update_crystal_memory_async(
    user_msg: str,
    assistant_msg: str,
    intermediate_history: list[dict],
) -> None:
    """
    后台任务: 读 Crystal_memory.md, 把 intermediate_history (上次 Ask 到本轮的完整对话轨迹)
    + 本轮对话一起喂 LLM, 写回。

    复用 ai_config 中的 api_key / api_url / model。

    异常静默, 不影响用户响应。Crystal_mem.md 是锦上添花, 损坏不应阻塞主链路。
    """
    try:
        if not ai_config["api_key"] or not ai_config["api_url"]:
            debug("[crystal_memory] skip: ai_config 未设置")
            return

        # 确保目录存在 (冷启动场景)
        _ensure_memory_dir()

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
            intermediate_history,
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
    intermediate_history: list[dict],
) -> str:
    """调 LLM 让它合并更新 Crystal_mem.md, 返回新的 Markdown 文本。配置从 ai_config 读取。"""
    messages = [
        {"role": "system", "content": MEMORY_UPDATE_SYSTEM()},
        {"role": "user", "content": buildMemoryUpdateUserPrompt(
            current_md, user_msg, assistant_msg, intermediate_history
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
      load_agent_memory -> load_paper_history -> compose_messages
      -> llm_call -> save_paper_memory -> update_agent_memory
    """
    # LangGraph 在新版是 langgraph.graph.StateGraph
    # 这里延迟 import, 避免冷启动开销
    from langgraph.graph import StateGraph, END  # type: ignore

    g = StateGraph(PaperAIState)
    g.add_node("load_agent_memory", load_agent_memory_node)
    g.add_node("load_paper_history", load_paper_history_node)
    g.add_node("compose_messages", compose_messages_node)
    g.add_node("llm_call", llm_call_node)
    g.add_node("save_paper_memory", save_paper_memory_node)
    g.add_node("update_agent_memory", update_agent_memory_node)

    g.set_entry_point("load_agent_memory")
    g.add_edge("load_agent_memory", "load_paper_history")
    g.add_edge("load_paper_history", "compose_messages")
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


async def run_ask(req: AiAskReq) -> dict:
    """Run LangGraph for Ask 模式 — 返回 state 字典"""
    initial: PaperAIState = {
        "req": req,
        "messages": [],
        "agent_memory": "",
        "paper_history": [],
        "final_answer": "",
        "usage": {},
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
        "final_answer": "",
        "usage": {},
    }
    graph = _get_graph()
    result = await graph.ainvoke(initial)
    return result
