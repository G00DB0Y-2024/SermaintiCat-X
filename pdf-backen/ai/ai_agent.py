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
from typing import TypedDict, Optional, Any

import httpx

from .ai_models import AiAskReq, AiLoadReq, AiResp
from .prompts import (
    buildAskMessages, buildLoadMessages, buildGlobalTrackContext,
    MEMORY_UPDATE_SYSTEM, buildMemoryUpdateUserPrompt,
    get_current_time_context, now_ms, format_dt_second, format_dt_minute,
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


# ═══════════════════════════════════════════════════════════════════════
# Crystal_memory.md (全局记忆) — 缓存 + 读写函数
# ═══════════════════════════════════════════════════════════════════════
CRYSTAL_MEMORY_FILE = os.path.join(BASE_DIR, "ai", "memory", "Crystal_memory.md")
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


# ═══════════════════════════════════════════════════════════════════════
# Crystal_track (跨论文 Ask 全局追踪) — 缓存 + 读写函数
# ═══════════════════════════════════════════════════════════════════════
CRYSTAL_TRACK_FILE = os.path.join(BASE_DIR, "ai", "memory", "Crystal_track.json")
MAX_TRACK_SIZE = 20
_agent_track_list: list[dict] | None = None      # None = 未加载
_agent_track_dirty: bool = False                  # True = 有 append 待写盘


def _get_track() -> list[dict]:
    """
    读 agent_track 缓存 (懒加载)。
    首次调用时同步加载磁盘内容, 返回内部 cache 引用 (调用方不应原地修改!)。
    如果要追加请用 _append_track()。
    """
    global _agent_track_list
    if _agent_track_list is None:
        _ensure_memory_dir()
        if os.path.exists(CRYSTAL_TRACK_FILE):
            try:
                with open(CRYSTAL_TRACK_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    _agent_track_list = data[-MAX_TRACK_SIZE:]
                else:
                    _agent_track_list = []
            except (json.JSONDecodeError, OSError):
                _agent_track_list = []
        else:
            _agent_track_list = []
    return _agent_track_list


def _set_track_cache(track: list[dict]) -> None:
    """外部 (如 reload/清空) 同步刷新缓存。"""
    global _agent_track_list, _agent_track_dirty
    _agent_track_list = list(track)
    _agent_track_dirty = True  # 缓存与磁盘不一定一致, 标 dirty 待下次 flush


def _append_track(entry: dict) -> None:
    """
    追加一条 Ask 记录到 cache, FIFO 裁剪到最多 20 条。

    注意: 此函数只改内存 cache + 标记 dirty, 不立即写盘!
    写盘由 flush_track_node 在对话结束后统一执行 (避免高频小 I/O)。

    异常静默: track 是锦上添花, 不影响主链路。
    """
    global _agent_track_dirty
    try:
        track = list(_get_track())  # 复制一份, 避免外部拿到引用后被我们原地改
        track.append(entry)
        # FIFO 裁剪到最近 20 条 (防止 cache 无限增长)
        if len(track) > MAX_TRACK_SIZE:
            track = track[-MAX_TRACK_SIZE:]
        _set_track_cache(track)
        _agent_track_dirty = True
        debug(f"[crystal_track] APPEND cached: fp={entry.get('pdf_fp', '')[:8]} total={len(track)}")
    except Exception as e:  # noqa: BLE001 — track 锦上添花, 不可阻塞主链路
        debug(f"[crystal_track] APPEND FAIL (cache): {e}")


def _flush_track_to_disk() -> None:
    """
    把内存 cache 写回 Crystal_track.json (覆盖式)。
    仅在 dirty=True 时执行 — 减少无谓写盘。
    异常静默: track 是锦上添花, 写盘失败不应影响主链路 (下次冷启动时 cache 会回读旧数据)。
    """
    global _agent_track_dirty
    if not _agent_track_dirty:
        return
    try:
        _ensure_memory_dir()
        track = _get_track()
        # 再裁一次, 防止历史 cache 异常增长
        track_to_write = track[-MAX_TRACK_SIZE:]
        with open(CRYSTAL_TRACK_FILE, "w", encoding="utf-8") as f:
            json.dump(track_to_write, f, ensure_ascii=False)
        _agent_track_dirty = False
        debug(f"[crystal_track] FLUSH ok: total={len(track_to_write)}")
    except OSError as e:
        debug(f"[crystal_track] FLUSH FAIL: {e}")



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
    dt:            服务端时间字符串 (北京时区, 秒级, 来自 now_ms 单源时间),
                   通过 AiResp.dt 透传给前端, 保证前后端时间一致。
    """
    req: Any  # AiAskReq | AiLoadReq
    messages: list[dict]
    agent_memory: str
    paper_history: list[dict]
    final_answer: str
    usage: dict
    dt: str


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
    Ask 模式还会把跨论文全局 track (Crystal_track.json, 最近 20 条)
    作为 assistant 角色消息注入 (在 paper_history 之后、本轮 user 之前),
    让 LLM 在 assistant 位置上感知全局提问脉络。
    Load 模式不注入 agent_memory 也不注入 track。
    """
    req = state["req"]
    history = state.get("paper_history") or []
    agent_mem = state.get("agent_memory") or ""

    # 动态生成时间上下文
    time_context = get_current_time_context()

    # 注入 Ask 模式的 track (作为 assistant 角色的对话历史消息 ——
    # 注入位置放在 paper_history 之后、本轮 user 之前, 保持历史对话连贯性)
    track_block = ""
    if isinstance(req, AiAskReq):
        track = _get_track()
        if track:
            track_block = buildGlobalTrackContext(track)

    if isinstance(req, AiAskReq):
        messages = buildAskMessages(req, history, time_context)
    elif isinstance(req, AiLoadReq):
        messages = buildLoadMessages(req, history, time_context)
    else:
        raise ValueError(f"Unknown req type: {type(req)}")

    # Ask 模式把 track 作为 assistant 消息插入 (在 paper_history 之后、本轮 user 之前)
    if track_block:
        # messages 结构: [system, ...paper_history, user(本轮)]
        # 插入位置: paper_history 末尾之后、本轮 user 之前
        # 即 -2 位置 (因为末尾是本轮 user)
        insert_idx = len(messages) - 1
        messages.insert(insert_idx, {"role": "assistant", "content": track_block})

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

    同时产出服务端时间 (单源 now_ms()) → state.dt, 最终透传给前端。
    """
    req = state["req"]
    messages = state["messages"]

    is_vision = isinstance(req, AiAskReq) and req.image_base64 is not None

    content, usage = await _call_llm(
        messages=messages,
        vision_model=is_vision,
    )

    return {
        "final_answer": content,
        "usage": usage,
        "dt": format_dt_second(now_ms()),
    }


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

    now_ms_paper = now_ms()
    history.append({"role": "user", "content": user_text, "ts": now_ms_paper})
    history.append({"role": "assistant", "content": answer, "ts": now_ms_paper + 1})

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
        debug(f"[ai_agent] save_paper_memory ok: fp={fp} len={len(history)}")
    except OSError as e:
        debug(f"[ai_agent] save_paper_memory FAIL: {e} | fp={fp}")

    return {}


async def update_agent_memory_node(state: PaperAIState) -> dict:
    """
    Ask 模式: 异步触发, 用全局 track (Crystal_track.json) 作为辅助上下文,
              加上本轮 req.ask + final_answer 作为核心更新依据。
    Load 模式: 不更新 Crystal_memory.md(划词不维护持久化记忆)。
    """
    req = state["req"]
    answer = state["final_answer"]

    if isinstance(req, AiAskReq):
        user_msg = req.ask
        # 异步执行, 不 await
        task = asyncio.create_task(
            _update_crystal_memory_async(user_msg, answer)
        )
        debug(
            f"[crystal_memory] scheduled [ask]: "
            f"user_len={len(user_msg)} asst_len={len(answer)} "
            f"task_id={id(task)}"
        )

        # 同步追加到全局 track cache (Ask 完成后, 立即追加一条跨论文记录)
        # 注意: 只改内存 cache + 标 dirty, 不立即写盘!
        # 写盘由 flush_track_node 在对话结束后统一执行 (避免高频小 I/O)。
        try:
            ts = now_ms()
            _append_track({
                "ts": ts,                                                  # 毫秒级 Unix 时间戳 (单源 now_ms())
                "ts_str": format_dt_second(ts),                            # 秒级可读: YYYY-MM-DD HH:MM:SS (北京)
                "pdf_fp": req.pdf_fp,
                "user": user_msg[:200],
                "assistant": answer[:200],
            })
        except Exception as e:
            debug(f"[crystal_track] FAIL (non-fatal): {e}")
    else:
        # Load 模式: 不维护 Crystal_memory.md, 也不追加 track
        debug("[crystal_memory] skipped [load]")

    return {}


async def _update_crystal_memory_async(
    user_msg: str,
    assistant_msg: str,
) -> None:
    """
    后台任务: 读 Crystal_memory.md, 把全局 track (Crystal_track.json, 最近 20 条
    跨论文 Ask 记录) + 本轮对话一起喂 LLM, 写回。

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

        # 取全局 track 作为辅助上下文 (取代原来的 intermediate_history)
        track = _get_track()

        new_md = await _call_memory_update_llm(
            current,
            user_msg,
            assistant_msg,
            track,
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
    track: list[dict] | None = None,
) -> str:
    """
    调 LLM 让它合并更新 Crystal_mem.md, 返回新的 Markdown 文本。配置从 ai_config 读取。

    注入精确到秒的时间戳到 user prompt (buildMemoryUpdateUserPrompt 的 current_timestamp 参数),
    让 LLM 在修改或新增条目末尾追加时间戳。

    track 提供跨论文全局上下文 (替代原来的 intermediate_history)。
    """
    ts = now_ms()
    current_time_str = format_dt_second(ts)

    messages = [
        {"role": "system", "content": MEMORY_UPDATE_SYSTEM()},
        {"role": "user", "content": buildMemoryUpdateUserPrompt(
            current_md, user_msg, assistant_msg, current_time_str, track
        )},
    ]
    content, _ = await _call_llm(messages=messages, vision_model=False)
    return content.strip()


# ═══════════════════════════════════════════════════════════════════════
# Graph 构建
# ═══════════════════════════════════════════════════════════════════════

async def flush_track_node(state: PaperAIState) -> dict:
    """
    对话结束后的最后一个节点: 把 track 缓存一次性写盘 (覆盖式)。

    之前 update_agent_memory_node 用 _append_track 累积到内存 cache +
    标 dirty; 这里统一做一次 flush_track_to_disk(), 避免每次 Ask 都打开
    Crystal_track.json 写盘 (高频小 I/O 浪费)。
    """
    _flush_track_to_disk()
    return {}


def build_graph():
    """
    构建 LangGraph StateGraph:
      load_agent_memory -> load_paper_history -> compose_messages
      -> llm_call -> save_paper_memory -> update_agent_memory -> flush_track
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
    g.add_node("flush_track", flush_track_node)

    g.set_entry_point("load_agent_memory")
    g.add_edge("load_agent_memory", "load_paper_history")
    g.add_edge("load_paper_history", "compose_messages")
    g.add_edge("compose_messages", "llm_call")
    g.add_edge("llm_call", "save_paper_memory")
    g.add_edge("save_paper_memory", "update_agent_memory")
    g.add_edge("update_agent_memory", "flush_track")
    g.add_edge("flush_track", END)

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
        "dt": "",
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
        "dt": "",
    }
    graph = _get_graph()
    result = await graph.ainvoke(initial)
    return result
