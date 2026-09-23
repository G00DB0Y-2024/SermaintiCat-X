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

from .ai_models import AiAskReq, AiLoadReq, AiResp, AiPaperEntry
from .prompts import (
    MEMORY_UPDATE_SYSTEM, buildMemoryUpdateUserPrompt,
    get_current_time_context, now_ms, format_dt_second, format_dt_minute,
    SYSTEM_ASK, SYSTEM_LOAD,
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
# Crystal_track (跨论文 Ask+Load 全局追踪) — 双轨缓存 + 读写函数
# ═══════════════════════════════════════════════════════════════════════
ASK_TRACK_FILE  = os.path.join(BASE_DIR, "ai", "memory", "Crystal_track_ask.json")
LOAD_TRACK_FILE = os.path.join(BASE_DIR, "ai", "memory", "Crystal_track_load.json")
MAX_ASK_TRACK  = 5   # ChatView 注入上限 (track-ask)
MAX_LOAD_TRACK = 3   # ChatView 注入上限 (track-load)

# ── 上下文窗口参数 ──
# PDFAI 论文侧: 仅本论文上下文, 不加载全局 track
ASK_LOCAL_LIMIT  = 10   # 本论文 Ask 历史最多取 10 对 (user+assistant)
LOAD_LOCAL_LIMIT = 5    # 本论文 Load 历史最多取 5 对
# ChatView 侧: Chat 本地窗口
CHAT_LOCAL_LIMIT = 10   # ChatView 本地近 Z=10 对 (user+assistant)
# Memory update 节流: 每 N 次论文 ask 触发一次 memory LLM
# N = ASK_LOCAL_LIMIT // 2 = 5, 即第 5/10/15... 次 ask 触发
MEMORY_UPDATE_EVERY_N = ASK_LOCAL_LIMIT // 2

# Crystal_memory.md 加载上限 (预留, 暂不限制, 由后续方案处理)
# MEMORY_MAX_CHARS = 4000  # 占位, 当前不启用截断

# ChatView 标识 (前端通过 pdf_fp=="crystal_chat" 调用 Chat 上下文)
CHAT_FP = "crystal_chat"

_ask_track_list:  list[dict] | None = None
_load_track_list: list[dict] | None = None
_ask_dirty:  bool = False
_load_dirty: bool = False

# 按 fp 分桶的 ask 计数 (用于 memory update 节流, 跨论文隔离)
_ask_count_by_fp: dict[str, int] = {}

# paper history 读盘缓存: {(path, mtime): data}
_paper_history_cache: dict[tuple[str, float], Any] = {}


def _get_track(mode: str) -> list[dict]:
    """
    读 track 缓存 (懒加载)。
    mode: "ask" | "load"
    返回内部 cache 引用 (调用方不应原地修改!)
    """
    global _ask_track_list, _load_track_list
    _ensure_memory_dir()

    track_file = ASK_TRACK_FILE if mode == "ask" else LOAD_TRACK_FILE
    track_list_ref = (_ask_track_list  if mode == "ask"  else _load_track_list)

    if track_list_ref is None:
        if os.path.exists(track_file):
            try:
                with open(track_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    if mode == "ask":
                        _ask_track_list = data[-MAX_ASK_TRACK:]
                    else:
                        _load_track_list = data[-MAX_LOAD_TRACK:]
                else:
                    if mode == "ask":
                        _ask_track_list = []
                    else:
                        _load_track_list = []
            except (json.JSONDecodeError, OSError):
                if mode == "ask":
                    _ask_track_list = []
                else:
                    _load_track_list = []
        else:
            if mode == "ask":
                _ask_track_list = []
            else:
                _load_track_list = []

    return _ask_track_list if mode == "ask" else _load_track_list


def _append_track(mode: str, entry: dict) -> None:
    """
    追加一条记录到 cache (ask 或 load), FIFO 裁剪。

    注意: 只改内存 cache + 标 dirty, 不立即写盘!
    写盘由 flush_track_node 在对话结束后统一执行。

    过滤规则: Crystal 全局 track 仅追踪论文场景。
    当 entry.pdf_fp == CHAT_FP ("crystal_chat") 时, 表示这是 ChatView 的对话,
    不写入论文 track (避免污染 ChatView 自身的上下文)。
    """
    global _ask_dirty, _load_dirty
    if entry.get("pdf_fp") == CHAT_FP:
        debug(f"[crystal_track] SKIP (chat fp): mode={mode}")
        return
    track = list(_get_track(mode))  # 复制
    max_size = MAX_ASK_TRACK if mode == "ask" else MAX_LOAD_TRACK
    track.append(entry)
    if len(track) > max_size:
        track = track[-max_size:]
    if mode == "ask":
        global _ask_track_list
        _ask_track_list = track
        _ask_dirty = True
    else:
        global _load_track_list
        _load_track_list = track
        _load_dirty = True
    debug(f"[crystal_track] APPEND cached: mode={mode} fp={entry.get('pdf_fp', '')[:8]} total={len(track)}")


def _flush_track_to_disk(mode: str) -> None:
    """
    把内存 cache 写回对应 track 文件 (覆盖式)。
    仅在 dirty=True 时执行。
    """
    if mode == "ask":
        global _ask_dirty
        if not _ask_dirty:
            return
        track = _get_track("ask")
        track_to_write = track[-MAX_ASK_TRACK:]
        track_file = ASK_TRACK_FILE
        _ask_dirty = False
    else:
        global _load_dirty
        if not _load_dirty:
            return
        track = _get_track("load")
        track_to_write = track[-MAX_LOAD_TRACK:]
        track_file = LOAD_TRACK_FILE
        _load_dirty = False

    try:
        _ensure_memory_dir()
        with open(track_file, "w", encoding="utf-8") as f:
            json.dump(track_to_write, f, ensure_ascii=False)
        debug(f"[crystal_track] FLUSH ok: mode={mode} total={len(track_to_write)}")
    except OSError as e:
        debug(f"[crystal_track] FLUSH FAIL: {e}")



# DeepSeek API 需要在 base url 后拼 /chat/completions
DEEPSEEK_MARKER = "deepseek.com"


def _paper_history_path(pdf_fp: str) -> str:
    return os.path.join(SAVE_DIR, f"{pdf_fp}_ai.json")


def _read_json_safe(path: str, default: Any) -> Any:
    """
    读 JSON 文件，异常静默返回 default。
    加 in-memory 缓存: 以 (path, mtime) 为 key, mtime 变了才重读, 避免同一文件
    在单次请求中重复打磁盘 (paper_history_node 与 compose 之间的链路已用 state 传递,
    但 _read_json_safe 仍被多处复用, 缓存兜底)。
    """
    if not os.path.exists(path):
        return default
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return default

    cache_key = (path, mtime)
    cached = _paper_history_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return default

    _paper_history_cache[cache_key] = data
    # 缓存表膨胀保护: 简单随机淘汰 (实际工程中 entry 数受论文 fp 数 + track 上限约束,
    # 单进程内通常不会超过数百条)
    if len(_paper_history_cache) > 64:
        # 移除最旧的一批 entry (按 dict 插入顺序)
        first_key = next(iter(_paper_history_cache))
        if first_key != cache_key:
            _paper_history_cache.pop(first_key, None)
    return data


def _invalidate_history_cache(path: str) -> None:
    """
    写盘后调用, 移除该 path 的所有 mtime 缓存条目。
    save_paper_memory_node 写完盘后调用, 确保下个请求读到新内容。
    """
    keys_to_drop = [k for k in _paper_history_cache if k[0] == path]
    for k in keys_to_drop:
        _paper_history_cache.pop(k, None)


def _load_paper_history(fp: str, mode: str, limit: int) -> list[dict]:
    """
    读 save/{fp}_ai.json, 按 mode 取对应 entry, 返回 OpenAI 格式 messages。

    参数:
      fp:    论文指纹
      mode:  "load" → ReqLoad/ResLoad; "ask" → ReqAsk/ResAsk
      limit: 最大轮次 (每轮 2 条)

    返回: [{role: "user"|"assistant", content: str}, ...] 正序

    过滤规则:
      1. Anno entry: 完全屏蔽, 不参与任何加载
      2. Vision Ask (img 非空): 屏蔽, 不进入上下文也不进入 track
         (Vision Ask 正常写盘，但不参与加载)
    """
    if limit <= 0:
        return []
    path = _paper_history_path(fp)
    data = _read_json_safe(path, [])
    if not isinstance(data, list):
        return []

    type_map = {
        "ReqLoad": "user",
        "ResLoad": "assistant",
        "ReqAsk":  "user",
        "ResAsk":  "assistant",
    }
    if mode == "load":
        targets = {"ReqLoad", "ResLoad"}
    else:
        targets = {"ReqAsk", "ResAsk"}

    filtered = []
    skip_next_res = False  # 标记跳过同 ts 的 ResAsk
    for e in data:
        if not isinstance(e, dict):
            continue
        t = e.get("type", "")

        # 规则 1: Anno 屏蔽
        if t == "Anno":
            continue

        # 规则 2: Vision Ask 屏蔽 (img 非空 = 带图片)
        if t == "ReqAsk" and e.get("img"):
            skip_next_res = True
            continue
        if skip_next_res and t == "ResAsk":
            skip_next_res = False
            continue

        if t not in targets:
            continue
        content = e.get("content")
        if isinstance(content, str):
            filtered.append({
                "role": type_map[t],
                "content": content,
                "ts": e.get("ts"),  # 保留 ts 用于后续去重
            })

    # 取最后 limit*2 条（最近 limit 轮），已正序
    tail = filtered[-(limit * 2):]
    return tail


# ═══════════════════════════════════════════════════════════════════════
# Flattened Context 构建 (多轨合并 + 去重 + 时间序 + 占比截取)
# ═══════════════════════════════════════════════════════════════════════

def _build_flattened_context(
    paper_history: list[dict],
    global_track: list[dict],
    pdf_fp: str,
    local_limit: int,
    global_limit: int,
    is_load: bool = False,
) -> list[dict]:
    """
    本论文 history + 全局 track 合并 -> 去重 -> 时间序展平 -> 占比截取。

    输入:
      paper_history: [{role, content, ts}, ...]   来自 _load_paper_history, 已正序, 保留 ts 字段
      global_track:  [{ts, ts_str, pdf_fp, user, assistant, (chosen_text)?}, ...] 来自 _get_track, 已正序
      pdf_fp:        当前论文 fp, 用于去重 key
      local_limit:   本论文最多取 N 对 (user+assistant)
      global_limit:  全局 track 最多补充 N 对
      is_load:       True=Load 轨 (track 字段名为 chosen_text/assistant),
                     False=Ask 轨 (track 字段名为 user/assistant)

    返回:
      去重 + 时间序铺平的 [{role:user, content}, {role:assistant, content}, ...]
      严格遵循: 本论文先按最近 local_limit 对取用, track 再按最近 global_limit 对补足。

    去重:
      - 本论文 history 和全局 track 都用 (pdf_fp, ts) 作为去重 key
      - 同一轮对话同时存在于 history 和 track 时，只会保留一条
    """
    flat: list[dict] = []
    seen: set[tuple[str, int]] = set()

    # 1) 本论文 history: 取最后 local_limit 对 (即 local_limit*2 条)
    #    paper_history 严格 user/assistant 交替, 直接切片即可, 不需要 pending_user
    local_msgs = paper_history[-(local_limit * 2):]
    for msg in local_msgs:
        role = msg.get("role")
        content = msg.get("content", "")
        flat.append({"role": role, "content": content})
        # 用 ts 作为去重 key，与 track 保持一致
        ts = msg.get("ts")
        if ts is not None:
            seen.add((pdf_fp, ts))

    # 2) 全局 track: 按顺序补足, 去重 key 用 (entry.pdf_fp, entry.ts)
    global_pairs = 0
    for entry in global_track:
        if global_pairs >= global_limit:
            break
        e_fp = entry.get("pdf_fp", "")
        e_ts = entry.get("ts", 0)
        key = (e_fp, e_ts)
        if key in seen:
            continue

        if is_load:
            user_text = entry.get("chosen_text", "")
            asst_text = entry.get("assistant", "")
        else:
            user_text = entry.get("user", "")
            asst_text = entry.get("assistant", "")

        if not user_text or not asst_text:
            continue

        seen.add(key)
        flat.append({"role": "user",      "content": user_text})
        flat.append({"role": "assistant", "content": asst_text})
        global_pairs += 1

    return flat


# ═══════════════════════════════════════════════════════════════════════
# LangGraph State
# ═══════════════════════════════════════════════════════════════════════

class PaperAIState(TypedDict):
    """
    LangGraph 工作区状态。

    req:                 入口请求(AiAskReq 或 AiLoadReq)
    messages:            组装好的 OpenAI 格式 messages, 准备送给 llm_call
    agent_memory:        从 Crystal_memory.md 加载的 Markdown 全文(注入 system prompt)
    paper_history:       兼容字段, 论文侧即为 paper_ask_history
    paper_ask_history:   当前 pdf_fp 最近 ASK_LOCAL_LIMIT 对 ask 历史 (role 交替)
    paper_load_history:  当前 pdf_fp 最近 LOAD_LOCAL_LIMIT 对 load 历史 (role 交替)
    final_answer:        llm_call 返回的最终 content
    usage:               上游 LLM 的 usage 统计
    dt:                  服务端时间字符串 (北京时区, 秒级, 来自 now_ms 单源时间),
                         通过 AiResp.dt 透传给前端, 保证前后端时间一致。
    """
    req: Any  # AiAskReq | AiLoadReq
    messages: list[dict]
    agent_memory: str
    paper_history: list[dict]
    paper_ask_history: list[dict]
    paper_load_history: list[dict]
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

    注: 当前不做长度截断, MEMORY_MAX_CHARS 保留为占位 (后续方案处理)。
    """
    return {"agent_memory": _get_agent_memory()}


async def load_paper_history_node(state: PaperAIState) -> dict:
    """
    一次性加载当前 pdf_fp 的 ask + load 两条历史轨。

    设计: 论文侧上下文按 fp 隔离, 不再读全局 track。
    但 ask 轨需要 load 轨作为辅助概要 (论文中的选段总结能帮 LLM 理解上下文),
    所以两个都加载, 各自按 limit 上限截取, 写入 state 供后续 compose_messages_node 拼装。

    ChatView 场景 (pdf_fp == "crystal_chat"):
      - ask_history 作为 Chat 本地上下文 (CHAT_LOCAL_LIMIT 对)
      - load_history 暂不使用 (Chat 不会 Load)

    Vision Ask 和 Anno 在 _load_paper_history 内部已过滤。
    """
    req = state["req"]
    fp = req.pdf_fp
    is_chat = (fp == CHAT_FP)

    if isinstance(req, AiAskReq):
        # Ask 模式 (论文 + Chat): 加载本 fp ask 历史
        ask_limit = CHAT_LOCAL_LIMIT if is_chat else ASK_LOCAL_LIMIT
        ask_history = _load_paper_history(fp, mode="ask", limit=ask_limit)
        # 同时加载本 fp load 历史 (仅论文侧使用)
        if is_chat:
            load_history: list[dict] = []
        else:
            load_history = _load_paper_history(fp, mode="load", limit=LOAD_LOCAL_LIMIT)
        debug(
            f"[paper_history] ask: fp={fp[:12]} is_chat={is_chat} "
            f"ask_rounds={len(ask_history)//2} load_rounds={len(load_history)//2}"
        )
        return {
            "paper_history": ask_history,        # 兼容
            "paper_ask_history": ask_history,
            "paper_load_history": load_history,
        }
    else:
        # Load 模式: 加载本论文 load 历史
        load_history = _load_paper_history(fp, mode="load", limit=LOAD_LOCAL_LIMIT)
        debug(f"[paper_history] load: fp={fp[:12]} load_rounds={len(load_history)//2}")
        return {
            "paper_history": load_history,
            "paper_ask_history": [],
            "paper_load_history": load_history,
        }


def _build_ask_user_content(req: AiAskReq) -> str | list[dict]:
    """构建 Ask 本轮 user content (纯文本或 vision 多模态)。"""
    if req.image_base64 is not None:
        suffix = (
            "请回答用户的询问：" + req.ask
            if req.ask.strip()
            else "对图片进行解释"
        )
        return [
            {"type": "text", "text": "针对给定图片" + suffix},
            {"type": "image_url", "image_url": {"url": req.image_base64}},
        ]
    else:
        if not req.quotes:
            return req.ask
        quote_lines = "\n".join(
            f"{i + 1}.{q.get('quote_msg', '')}"
            for i, q in enumerate(req.quotes)
        )
        return (
            f"用户引用的内容：\n{quote_lines}\n\n"
            f"用户引用的解释：{req.quote_content}\n\n"
            f"用户的询问【{req.ask}】"
        )


def _build_load_user_content(req: AiLoadReq) -> str:
    """构建 Load 本轮 user content。"""
    instruction = "用中文准确概括" if req.added_prompt == "" else req.added_prompt
    return (
        f"请结合上下文和之前的论文内容，将学术内容【{req.chosen_text}】{instruction}，要求如下：\n"
        "- 概括内容简短、简洁明了，突出重点，合理分段或者分点，无需额外说明，不要输出其它内容\n"
        "- 仅在确有必要时进行分条列点，避免分条过细\n"
        "- 对于重要的专业术语，中文翻译后markdown加粗并附全称，"
        "例如：中文(缩写, 英文全称)，但此后再出现相同术语不再附加全称\n"
        "- 对于公式，请在公式后用markdown引用格式解释公式含义或变量解释，不要在其他地方重复解释\n"
    )


async def compose_messages_node(state: PaperAIState) -> dict:
    """
    按场景组装 messages:

    论文 Ask (pdf_fp != CHAT_FP):
      [system]  CrystalPersona + time + agent_mem + "在论文侧回答"
      [ask_msgs]   本论文 ask (ASK_LOCAL_LIMIT 对)
      [load_msgs]  本论文 load (LOAD_LOCAL_LIMIT 对)
      [user]    本轮提问
      -> 上下文按论文 fp 完全隔离

    Chat Ask (pdf_fp == CHAT_FP):
      [system]  CrystalPersona + time + agent_mem + "全局闲聊" + 论文 track 摘要块
      [chat_msgs]  ChatView 本地近 Z=CHAT_LOCAL_LIMIT 对
      [user]    本轮提问
      -> 论文全局 track 通过 _get_track 注入 (track 仅含论文, 因为 _append_track 已过滤)

    Load 模式: 人设 + 本论文 Load 历史 (LOAD_LOCAL_LIMIT 对)
    """
    req = state["req"]
    agent_mem = state.get("agent_memory") or ""
    time_context = get_current_time_context()
    fp = req.pdf_fp
    is_chat = (fp == CHAT_FP)

    if isinstance(req, AiAskReq):
        ask_history = state.get("paper_ask_history") or []
        load_history = state.get("paper_load_history") or []

        if is_chat:
            messages = _compose_chat_messages(
                req, ask_history, agent_mem, time_context
            )
        else:
            messages = _compose_paper_ask_messages(
                req, ask_history, load_history, agent_mem, time_context
            )
    else:
        # Load: 人设 + 本论文 Load 历史 (LOAD_LOCAL_LIMIT 对)
        messages = [
            {"role": "system", "content": SYSTEM_LOAD(time_context)},
        ]
        load_history = state.get("paper_load_history") or []
        messages.extend(load_history)
        messages.append({"role": "user", "content": _build_load_user_content(req)})

    debug(
        f"[compose_messages] is_chat={is_chat} "
        f"msgs={len(messages)} "
        f"first_role={messages[0]['role'] if messages else '-'}"
    )
    return {"messages": messages}


def _compose_paper_ask_messages(
    req: AiAskReq,
    ask_history: list[dict],
    load_history: list[dict],
    agent_mem: str,
    time_context: str,
) -> list[dict]:
    """
    论文侧 Ask 上下文 (按 fp 隔离, 不读全局 track)。
    ask_history / load_history 已由 load_paper_history_node 装入 state,
    这里不重复读盘, 也不再调 _build_flattened_context 的 global 分支。
    """
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_ASK(time_context, agent_mem)},
    ]
    # ask_history 本身已是 role 交替的标准 messages, 直接 extend 即可
    messages.extend(ask_history)
    # load_history 同理 (role=user=论文片段, role=assistant=概括)
    messages.extend(load_history)
    messages.append({"role": "user", "content": _build_ask_user_content(req)})
    return messages


def _compose_chat_messages(
    req: AiAskReq,
    chat_history: list[dict],
    agent_mem: str,
    time_context: str,
) -> list[dict]:
    """
    ChatView 上下文 (脱离具体论文):
      [system]  CrystalPersona + time + agent_mem + 全局闲聊提示 + 论文 track 摘要
      [user/assistant × CHAT_LOCAL_LIMIT 对]  ChatView 本地历史
      [user]  本轮提问

    论文 track (ask+load) 由 _get_track 读出, 因为 _append_track 已经过滤了 chat fp,
    所以 track 里只含论文场景的记录, 正好对应 ChatView "提示 Crystal 全局而言
    和用户聊过什么" 的诉求。
    """
    # 1) 摘要化论文全局 track (避免破坏 user/assistant 交替, 用文本块)
    track_summary = _build_track_summary_block()

    system_content = SYSTEM_ASK(time_context, agent_mem)
    if track_summary:
        system_content += (
            "\n\n【论文场景全局记忆 (仅供你了解用户近期在论文中的关注点, "
            "不要直接复述, 在闲聊时自然关联即可)】\n"
            + track_summary
        )

    messages: list[dict] = [
        {"role": "system", "content": system_content},
    ]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": _build_ask_user_content(req)})
    return messages


def _build_track_summary_block() -> str:
    """
    把 Crystal_track_ask 和 Crystal_track_load 拼成一段摘要文本, 注入 ChatView system。
    因为 track 已经按 MAX_ASK_TRACK / MAX_LOAD_TRACK 上限截取,
    这里不需要再截断。chosen_text 在 _append_track 里已经被 [:200] 截断。
    """
    ask_track = _get_track("ask") or []
    load_track = _get_track("load") or []

    if not ask_track and not load_track:
        return ""

    lines: list[str] = []

    if ask_track:
        lines.append("【近期论文提问 (近 {} 条)】".format(len(ask_track)))
        for e in ask_track:
            ts = e.get("ts_str", "")
            fp_short = (e.get("pdf_fp", "") or "")[:8]
            user = e.get("user", "")
            asst = e.get("assistant", "")
            lines.append(f"- [{ts}][{fp_short}] 用户: {user} | Crystal: {asst}")

    if load_track:
        lines.append("")
        lines.append("【近期论文选段总结 (近 {} 条)】".format(len(load_track)))
        for e in load_track:
            ts = e.get("ts_str", "")
            fp_short = (e.get("pdf_fp", "") or "")[:8]
            chosen = e.get("chosen_text", "")
            asst = e.get("assistant", "")
            lines.append(f"- [{ts}][{fp_short}] 选段: {chosen} | 总结: {asst}")

    return "\n".join(lines)


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
    """
    把本轮 user + assistant 追加写入 save/{fp}_ai.json (新 schema 格式)。

    写两条 entry: ReqAsk/ReqLoad + ResAsk/ResLoad, 统一用 AiPaperEntry 格式。
    Vision Ask 正常写盘（供前端渲染），但不在上下文加载时被 pickup（由 _load_paper_history 过滤）。
    """
    req = state["req"]
    fp = req.pdf_fp
    answer = state["final_answer"]
    usage = state.get("usage") or {}
    token_count = usage.get("total_tokens")

    ts = now_ms()
    dt_str = format_dt_second(ts)

    # --- Req entry ---
    if isinstance(req, AiAskReq):
        req_entry = {
            "type": "ReqAsk",
            "content": req.ask,
            "ts": ts,
            "dt": dt_str,
            "hl": getattr(req, "hl", None),
            "quote_gids": [
                q.get("quote_gid") for q in (req.quotes or [])
                if isinstance(q, dict) and q.get("quote_gid")
            ],
            "img": getattr(req, "image_filename", "") or "",
        }
    else:
        req_entry = {
            "type": "ReqLoad",
            "content": req.chosen_text,
            "ts": ts,
            "dt": dt_str,
        }

    # --- Res entry ---
    res_type = "ResAsk" if isinstance(req, AiAskReq) else "ResLoad"
    res_entry = {
        "type": res_type,
        "content": answer,
        "ts": ts + 1,
        "dt": dt_str,
        "token_count": token_count,
    }

    # --- 读 + 追加 + 写盘 ---
    path = _paper_history_path(fp)
    history: list = _read_json_safe(path, [])
    history.append(req_entry)
    history.append(res_entry)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
        # 写盘后失效该 path 的 mtime 缓存, 保证下次请求读到的就是新内容
        _invalidate_history_cache(path)
        debug(f"[save_paper_memory] ok: fp={fp} total={len(history)}")
    except OSError as e:
        debug(f"[save_paper_memory] FAIL: {e} | fp={fp}")

    return {}



def _should_update_memory(fp: str, is_chat: bool) -> bool:
    """
    memory update 节流:
    - Chat 场景 (pdf_fp == CHAT_FP) 不触发 memory update (memory 是关于论文阅读的沉淀)
    - 论文 Ask: 每 MEMORY_UPDATE_EVERY_N 次 ask 触发一次 (默认 N=5)
      计数按 fp 分桶, 避免跨论文干扰
    """
    if is_chat:
        return False
    return (_ask_count_by_fp.get(fp, 0) % MEMORY_UPDATE_EVERY_N) == 0


async def update_agent_memory_node(state: PaperAIState) -> dict:
    """
    Ask 模式:
      - 论文 fp + 非 Vision: 追加到 ask_track (经 _append_track 自动过滤 chat fp)
      - Chat fp: 不写 track, 不写 mem (论文 track 保持纯净)
      - 论文 fp: 节流触发 _update_crystal_memory_async (默认每 5 次 ask 一次)
      - Vision Ask (img 非空): 不进 track, 但仍可节流触发 memory update
    Load 模式:
      - 论文 fp: 追加到 load_track
      - Chat fp: 不写 track
    """
    req = state["req"]
    answer = state["final_answer"]
    ts = now_ms()
    dt_str = format_dt_second(ts)
    fp = req.pdf_fp
    is_chat = (fp == CHAT_FP)

    if isinstance(req, AiAskReq):
        is_vision = bool(getattr(req, "image_filename", ""))

        if not is_chat:
            # 论文场景: 写 ask_track (内部已过滤 chat fp, 这里仅作为语义清晰度保留判断)
            if not is_vision:
                _append_track("ask", {
                    "ts": ts,
                    "ts_str": dt_str,
                    "pdf_fp": fp,
                    "user": req.ask[:200],
                    "assistant": answer[:200],
                })

        # 论文 ask 计数 + 节流判定
        if not is_chat and not is_vision:
            _ask_count_by_fp[fp] = _ask_count_by_fp.get(fp, 0) + 1
            trigger_mem = _should_update_memory(fp, is_chat)
        else:
            trigger_mem = False

        if trigger_mem:
            task = asyncio.create_task(
                _update_crystal_memory_async(req.ask, answer, dt_str)
            )
            debug(
                f"[crystal_memory] scheduled [ask throttled]: fp={fp[:12]} "
                f"count={_ask_count_by_fp[fp]} "
                f"user_len={len(req.ask)} asst_len={len(answer)} "
                f"task_id={id(task)}"
            )
        else:
            debug(
                f"[crystal_memory] SKIP: is_chat={is_chat} is_vision={is_vision} "
                f"count={_ask_count_by_fp.get(fp, 0)}/{MEMORY_UPDATE_EVERY_N}"
            )
    else:
        # Load 模式: 论文 fp 写 load_track (chat fp 不写)
        if not is_chat:
            _append_track("load", {
                "ts": ts,
                "ts_str": dt_str,
                "pdf_fp": fp,
                "chosen_text": req.chosen_text[:200],
                "assistant": answer[:200],
            })
            debug("[crystal_memory] load mode: appended to load_track")
        else:
            debug("[crystal_memory] load mode: skipped (chat fp)")

    return {}


async def _update_crystal_memory_async(
    user_msg: str,
    assistant_msg: str,
    current_timestamp: str = "",
) -> None:
    """
    后台任务: 读 Crystal_memory.md, 把双轨 track (Ask + Load) + 本轮对话一起喂 LLM, 写回。

    复用 ai_config 中的 api_key / api_url / model。

    异常静默, 不影响用户响应。Crystal_mem.md 是锦上添花, 损坏不应阻塞主链路。
    """
    try:
        if not ai_config["api_key"] or not ai_config["api_url"]:
            debug("[crystal_memory] skip: ai_config 未设置")
            return

        _ensure_memory_dir()

        # 读当前 Markdown
        current = ""
        if os.path.exists(CRYSTAL_MEMORY_FILE):
            try:
                with open(CRYSTAL_MEMORY_FILE, "r", encoding="utf-8") as f:
                    current = f.read()
            except OSError:
                current = ""

        # 取双轨 track 作为辅助上下文
        ask_track = _get_track("ask")
        load_track = _get_track("load")

        new_md = await _call_memory_update_llm(
            current,
            user_msg,
            assistant_msg,
            current_timestamp,
            ask_track,
            load_track,
        )
        if new_md:
            with open(CRYSTAL_MEMORY_FILE, "w", encoding="utf-8") as f:
                f.write(new_md)
            _set_agent_memory_cache(new_md)
            debug(
                f"[crystal_memory] WRITE ok: "
                f"prev_len={len(current)} new_len={len(new_md)} delta={len(new_md)-len(current):+d}"
            )
    except Exception as e:
        debug(f"[crystal_memory] update FAIL: {type(e).__name__}: {e}")


async def _call_memory_update_llm(
    current_md: str,
    user_msg: str,
    assistant_msg: str,
    current_timestamp: str = "",
    ask_track: list[dict] | None = None,
    load_track: list[dict] | None = None,
) -> str:
    """
    调 LLM 更新 Crystal_mem.md, 返回新的 Markdown 文本。

    参数:
      current_timestamp: 秒级可读时间字符串 (来自 now_ms + format_dt_second)
      ask_track: Ask 跨论文轨迹
      load_track: Load 跨论文轨迹
    """
    messages = [
        {"role": "system", "content": MEMORY_UPDATE_SYSTEM()},
        {"role": "user", "content": buildMemoryUpdateUserPrompt(
            current_md, user_msg, assistant_msg,
            current_timestamp, ask_track, load_track,
        )},
    ]
    content, _ = await _call_llm(messages=messages, vision_model=False)
    return content.strip()


# ═══════════════════════════════════════════════════════════════════════
# Graph 构建
# ═══════════════════════════════════════════════════════════════════════

async def flush_track_node(state: PaperAIState) -> dict:
    """
    对话结束后的最后一个节点: 把 ask + load 两条 track 缓存一次性写盘 (覆盖式)。

    之前 update_agent_memory_node 用 _append_track 累积到内存 cache + 标 dirty;
    这里统一做一次 flush, 避免高频小 I/O。
    """
    _flush_track_to_disk("ask")
    _flush_track_to_disk("load")
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
        "paper_ask_history": [],
        "paper_load_history": [],
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
        "paper_ask_history": [],
        "paper_load_history": [],
        "final_answer": "",
        "usage": {},
        "dt": "",
    }
    graph = _get_graph()
    result = await graph.ainvoke(initial)
    return result
