"""
文件 IO + 内存 cache：track / paper_history / memory 文件。

集中管理:
- Crystal_memory.md (CRYSTAL_MEMORY_FILE) 的进程内缓存
- Crystal_track_ask/load.json 的双轨缓存 + dirty 标记 + 写盘策略
- save/{fp}_ai.json 的读取 + mtime 缓存
- 按 fp 桶化计数的 _ask_count_by_fp (memory update 节流用)

所有 cache 均为模块级单例,跨请求保持,跟原来散落在 ai_agent.py 的行为一致。
"""
from __future__ import annotations

import json
import os
from typing import Any

from .ai_config import (
    ASK_TRACK_FILE,
    CHAT_FP,
    CHAT_MEMORY_LIMIT,
    CRYSTAL_MEMORY_FILE,
    CRYSTAL_SELF_FILE,
    LOAD_TRACK_FILE,
    MAX_ASK_TRACK,
    MAX_LOAD_TRACK,
    MEMORY_DIR,
    SAVE_DIR,
)
from .ai_utils import format_dt_second
from utils.log import debug


# ═══════════════════════════════════════════════════════════════════════
# 模块初始化: 保证磁盘目录存在
# ═══════════════════════════════════════════════════════════════════════
os.makedirs(SAVE_DIR, exist_ok=True)


def _ensure_memory_dir() -> None:
    """冷启动: 确保 ai/memory/ 目录存在, 首次调用时执行一次。"""
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Crystal_memory.md (全局记忆) — 缓存 + 读写
# ═══════════════════════════════════════════════════════════════════════

_agent_memory_cache: str | None = None


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
# Crystal_self.md (Crystal 对自己的认知) — 缓存 + 读写
# ═══════════════════════════════════════════════════════════════════════
#
# 与 Crystal_memory.md 的关系:
#   - memory: 给 LLM 看的「用户认知」, 注入 system prompt 的关于用户笔记块。
#   - self:   Crystal 对自己的认知 (性格、定位、愿望), **不**注入 system prompt
#            (Crystal 默认知道自己是谁), 只在 background self-update 流水线内读写。
# 缓存仍建一份, 保证写入后下一次任何路径读到的都是新内容, 即使将来有别的
# 代码想读 self 也不会读到陈旧数据。

_agent_self_cache: str | None = None


def _get_agent_self() -> str:
    """读 agent_self 缓存(首次才打磁盘)。仅供同步自检 / debug 用, 不进入 ask 链路。"""
    global _agent_self_cache
    if _agent_self_cache is None:
        _ensure_memory_dir()
        if os.path.exists(CRYSTAL_SELF_FILE):
            try:
                with open(CRYSTAL_SELF_FILE, "r", encoding="utf-8") as f:
                    _agent_self_cache = f.read()
            except OSError:
                _agent_self_cache = ""
        else:
            _agent_self_cache = ""
    return _agent_self_cache


def _set_agent_self_cache(md: str) -> None:
    """后台 self-update 写完文件后调用, 同步刷新缓存。"""
    global _agent_self_cache
    _agent_self_cache = md


# ═══════════════════════════════════════════════════════════════════════
# Crystal_track (跨论文 Ask+Load 全局追踪) — 双轨缓存 + 读写
# ═══════════════════════════════════════════════════════════════════════

_ask_track_list:  list[dict] | None = None
_load_track_list: list[dict] | None = None
_ask_dirty:  bool = False
_load_dirty: bool = False

# 按 fp 分桶的 ask 计数 (用于 memory update 节流, 跨论文隔离)
_ask_count_by_fp: dict[str, int] = {}


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


# ═══════════════════════════════════════════════════════════════════════
# Paper history IO — save/{fp}_ai.json 读写 + mtime 缓存
# ═══════════════════════════════════════════════════════════════════════

# paper history 读盘缓存: {(path, mtime): data}
_paper_history_cache: dict[tuple[str, float], Any] = {}


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

    返回: [{role: "user"|"assistant", content: str, ts, msg_fp, quotes}, ...] 正序

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
        ts_val = e.get("ts")
        role = type_map[t]
        content = e.get("content")
        if isinstance(content, str):
            filtered.append({
                "role": role,
                "content": content,            # 原样进 LLM, 不加 [ts label] 前缀
                "ts": ts_val,                   # 保留 ts 用于后续去重
                "msg_fp": e.get("msg_fp", ""),  # 透传消息指纹
                "quotes": e.get("quotes", []),  # 透传引用指纹数组
            })

    # 取最后 limit*2 条（最近 limit 轮），已正序
    tail = filtered[-(limit * 2):]
    return tail


def _load_chat_history_for_memory() -> tuple[list[dict], list[dict]]:
    """
    ChatView 场景下, 为 memory update 提供 ask 上下文。

    直接读 save/crystal_chat_ai.json 的最近 CHAT_MEMORY_LIMIT 对 (ReqAsk + ResAsk),
    转成与论文 track 一致的 {ts, ts_str, user, assistant} 字典, 让 _call_memory_update_llm
    的 prompt 拼装逻辑无需分支处理。

    返回: (ask_track, load_track); chat 没有 load, 第二项永远为 []。

    实现细节:
      - 按 ts 配对 (相邻的 ReqAsk 与 ResAsk 视为一对, ts 连续递增)。
      - 时区/dt 沿用 save_paper_memory_node 写盘时的 dt 字段 (前端也消费同一个字段)。
      - content 截断到 200 字符, 保持与论文 track 同样的尺寸约定。
    """
    path = _paper_history_path(CHAT_FP)
    data = _read_json_safe(path, [])
    if not isinstance(data, list) or not data:
        return [], []

    # 把 entry 流配成 user/assistant 对, 取最后 CHAT_MEMORY_LIMIT 对
    pairs: list[tuple[dict, dict]] = []
    pending_user: dict | None = None
    for e in data:
        if not isinstance(e, dict):
            continue
        t = e.get("type", "")
        # 与论文侧 _load_paper_history 一致: vision Ask (img 非空) 跳过
        if t == "ReqAsk" and e.get("img"):
            pending_user = None  # 作废相邻未匹配的 user
            continue
        if t == "ResAsk" and pending_user is not None:
            pairs.append((pending_user, e))
            pending_user = None
        elif t == "ReqAsk":
            pending_user = e

    # 取最后 N 对, 转 {user, assistant}
    tail = pairs[-CHAT_MEMORY_LIMIT:]
    ask_track: list[dict] = []
    for u, a in tail:
        ask_track.append({
            "ts": a.get("ts", 0),
            "ts_str": a.get("dt", ""),
            "pdf_fp": CHAT_FP,
            "user": (u.get("content") or "")[:200],
            "assistant": (a.get("content") or "")[:200],
        })
    return ask_track, []
