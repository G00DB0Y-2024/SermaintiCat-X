"""
文件 IO + 内存 cache：track / paper_history / memory 文件。

集中管理:
- Crystal_memory.md (CRYSTAL_MEMORY_FILE) 的进程内缓存
- Crystal_track_ask/load.json 的双轨缓存 + dirty 标记 + 写盘策略
- save/{fp}_ai.json 的读取 + mtime 缓存
- ai/memory/params.json 的读写 (存 _ask_count_by_fp — memory update 节流计数,
  持久化以跨进程存活; 所有访问都经 _load_ask_counts / _save_ask_counts /
  _inc_ask_count / _get_ask_count, 不要直接摸 _ask_count_by_fp)

所有 cache 均为模块级单例,跨请求保持,跟原来散落在 ai_agent.py 的行为一致。
"""
from __future__ import annotations

import json
import os
import threading
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
    PARAMS_FILE,
    SAVE_DIR,
)
from .ai_utils import format_dt_second, now_ms
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
#
# 持久化: 这个计数决定"第 N 次 ask 触发一次 memory LLM"。如果只放内存,
# 服务器一重启就归零, 节流节奏被打乱 (每次重启后的前 5 轮会连续触发 5 次
# memory 更新, 白烧 token)。所以落盘到 ai/memory/params.json, 进程冷启动时读回。
_ask_count_by_fp: dict[str, int] = {}
# 懒加载标志: None=还没尝试读过, True=已从磁盘载入
_ask_count_loaded: bool = False


def _load_ask_counts() -> None:
    """
    冷启动时把 _ask_count_by_fp 从 params.json 读回内存。

    读失败一律退化成空 dict (视为"全新开始"), 不抛异常 —— 计数只是节流提示,
    丢了顶多多触发/少触发一次 memory, 不该让整个服务起不来。
    """
    global _ask_count_by_fp, _ask_count_loaded
    if _ask_count_loaded:
        return
    _ensure_memory_dir()
    counts: dict[str, int] = {}
    try:
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw = (data or {}).get("ask_count_by_fp", {})
            if isinstance(raw, dict):
                for k, v in raw.items():
                    # 只接受 int 计数, 防止手改文件写入 "abc" / None 之类
                    if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
                        counts[str(k)] = v
    except (json.JSONDecodeError, OSError, AttributeError, TypeError) as e:
        debug(f"[params] load ask_count_by_fp failed, reset to empty: {e}")
        counts = {}
    _ask_count_by_fp = counts
    _ask_count_loaded = True
    debug(f"[params] loaded ask_count_by_fp: {len(counts)} fp(s) {counts}")


def _save_ask_counts() -> None:
    """
    把 _ask_count_by_fp 写回 params.json (覆盖式)。

    与 track 不同: 计数必须**立即**落盘, 不能等 flush_track_node 统一写 ——
    flush 是每轮对话末尾才执行, 而计数在节点入口就变了, 两者时序不同。

    实现: 委托给 _save_params 做"读-改-写", 避免与情绪向量的写盘相互覆盖。
    """
    _save_params({"ask_count_by_fp": _ask_count_by_fp})


def _inc_ask_count(fp: str) -> int:
    """
    fp 的 ask 计数 +1, 立刻落盘, 返回新值。

    唯一写入口 —— ai_agent.update_agent_memory_node 不要再直接改 _ask_count_by_fp,
    否则会漏掉持久化。
    """
    _load_ask_counts()  # 首次调用时冷启动载入
    _ask_count_by_fp[fp] = _ask_count_by_fp.get(fp, 0) + 1
    _save_ask_counts()
    return _ask_count_by_fp[fp]


def _get_ask_count(fp: str) -> int:
    """读 fp 的 ask 计数 (0 = 该 fp 还没聊过)。"""
    _load_ask_counts()
    return _ask_count_by_fp.get(fp, 0)


# ═══════════════════════════════════════════════════════════════════════
# params.json 统一入口 (_save_params) — 所有写盘都走这里
# ═══════════════════════════════════════════════════════════════════════
#
# 设计: 各字段 (ask_count_by_fp / emotion_vector) 都从 _save_params(updates)
# 走, 避免 "读-改-写" 之间漏掉对方字段。情绪写盘和计数写盘并发时不会出现
# 互相覆盖的 race。
#
# 锁策略: 模块级 asyncio.Lock + 同步 with ——
#   - 同步路径 (emotion decay loop) 走同步 acquire;
#   - 异步路径 (LLM 更新情绪) 走 await lock。
# 锁用于 _save_params 的"读-改-写"段落, 不用于 _load_xxx (加载阶段)。
_params_lock: threading.Lock = threading.Lock()


def _save_params(updates: dict) -> bool:
    """
    原子地 "读-改-写" params.json, 仅把 updates 中的字段写回。

    实现:
      1. 读现有 JSON (若解析失败, 退化为 {})
      2. 与 updates 合并
      3. 写回磁盘
    互斥: 全模块一把锁, 同步/异步路径都走它 (FastAPI 默认同步路由里跑,
          异步路径也跑在同一个 event loop 里, 同步锁不会饿死)。

    Args:
        updates: 要写入的字段 dict, e.g. {"ask_count_by_fp": {...}}
                 或 {"emotion_vector": {...}}。每次调用只写 updates 中给出的字段,
                 其他字段原样保留 (这是与原版"覆盖式写整个 dict"的关键区别)。

    Returns:
        bool — True 写盘成功, False 失败 (异常静默, debug 已记录)。
    """
    _ensure_memory_dir()
    with _params_lock:
        try:
            existing: dict = {}
            if os.path.exists(PARAMS_FILE):
                try:
                    with open(PARAMS_FILE, "r", encoding="utf-8") as f:
                        existing = json.load(f) or {}
                except (json.JSONDecodeError, OSError):
                    existing = {}
            if not isinstance(existing, dict):
                existing = {}
            existing.update(updates)
            with open(PARAMS_FILE, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            return True
        except OSError as e:
            debug(f"[params] save failed: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════
# LLM 配置持久化 (Main + Lint) — 写到 params.json.llm_configs
# ═══════════════════════════════════════════════════════════════════════
#
# 与 ask_count_by_fp / emotion_vector 共用一份 params.json, 走同一把锁
# (_save_params 已自带锁)。设计: 把 main + lint 两份 LLM 配置打包在
# params.json.llm_configs 字段下, 持久化 + 跨进程共享。
#
# 调用入口:
#   · Main LLM 更新走 ai_config.update_ai_config → _save_llm_configs({"main": ...})
#   · Lint LLM 更新走 ai_emotion.update_emotion_llm_config → _save_llm_configs({"lint": ...})
#   · 启动恢复走 ai_config.load_ai_config_from_disk / ai_emotion.load_emotion_llm_config_from_disk
#   · 前端 GET 走 ai_routes 的 /ai/config + /ai/emotion/config (模块 dict → 响应)
#
# 之所以单独抽两个函数而不是直接复用 _save_params:
#   · _save_params 接受任意 dict, 调用方需要自己嵌套 {"llm_configs": {...}} 容易漏掉外壳层
#   · 集中到这里便于以后换存储 (比如挪到独立 YAML) 时只改一个地方


def _load_llm_configs() -> dict:
    """
    从 params.json.llm_configs 读出 Main + Lint 两份 LLM 配置。

    Returns:
        dict — {"main": {...}, "lint": {...}}
               缺字段 → 缺的那个 dict 为空 {}; 文件不存在/解析失败 → 全空 {}
               返回的 dict 是深拷贝, 调用方修改不会影响 cache。
    """
    out: dict = {"main": {}, "lint": {}}
    try:
        if not os.path.exists(PARAMS_FILE):
            return out
        with open(PARAMS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f) or {}
        llm_section = raw.get("llm_configs") or {}
        if not isinstance(llm_section, dict):
            return out
        for slot in ("main", "lint"):
            v = llm_section.get(slot)
            if isinstance(v, dict):
                out[slot] = dict(v)
    except (json.JSONDecodeError, OSError, AttributeError, TypeError) as e:
        debug(f"[llm_configs] load failed, return empty: {e}")
    return out


def _save_llm_configs(updates: dict) -> bool:
    """
    把 Main / Lint 任意一份 LLM 配置增量写回 params.json.llm_configs。
    走 _save_params 的"读-改-写"锁, 不会与 ask_count / emotion_vector 写盘互相覆盖。

    Args:
        updates: 形如 {"main": {...}} 或 {"lint": {...}} 或两者同时给
                 缺 slot 的不动, 已有 slot 的整体覆盖 (不是字段级 merge,
                 因为每次 POST 都是整组配置, 没有"只改一个字段"的场景)。

    Returns:
        bool — True 写盘成功, False 失败 (异常静默, debug 已记录)。
    """
    if not isinstance(updates, dict) or not updates:
        return True   # 空更新: no-op, 视为成功
    valid_slots = {k: v for k, v in updates.items() if k in ("main", "lint") and isinstance(v, dict)}
    if not valid_slots:
        return True

    # 先读出当前 llm_configs 全量, 再 merge, 再走 _save_params 落盘
    current = _load_llm_configs()
    current.update(valid_slots)
    return _save_params({"llm_configs": current})


# ═══════════════════════════════════════════════════════════════════════
# 情绪向量 (emotion_vector) — 缓存 + 读写
# ═══════════════════════════════════════════════════════════════════════
#
# Schema:
#   {
#     "valence":  float,   # 情感正负向: [-1.0, 1.0], 正=积极, 负=消极
#     "arousal":  float,   # 唤醒度:        [-1.0, 1.0], 正=激动, 负=低落
#     "novelty":  float,   # 新奇度:        [-1.0, 1.0], 正=惊讶/好奇, 负=无聊
#     "clarity":  float,   # 清晰度:        [-1.0, 1.0], 正=清醒, 负=困惑
#     "last_update_dt": str,   # "YYYY-MM-DD HH:MM:SS" 北京时间, 衰减计算锚点
#     "version":  int,        # schema 版本, 未来扩展用
#   }
#
# 默认值: 全 0.0 (中性) + 当前 dt, 表示"刚启动, 还没任何情绪信号"。
# 持久化字段: last_update_dt 是显式字符串 (来自 format_dt_second 单源时间),
# 不再是 ts 整数 — 避免时区歧义, 与前端 dt 字段完全同源。

EMOTION_AXES: tuple[str, ...] = ("valence", "arousal", "novelty", "clarity")
EMOTION_DEFAULT: dict = {
    "valence": 0.0,
    "arousal": 0.0,
    "novelty": 0.0,
    "clarity": 0.0,
    "last_update_dt": "",  # 首次加载时由 _load_emotion 填上当前 dt
    "version": 1,
}

_emotion_cache: dict | None = None


def _load_emotion() -> dict:
    """
    加载情绪向量。懒加载 + 缓存; 磁盘缺失则返回默认向量 (全 0)。

    异常静默 (退化策略):
      · 文件不存在 → 默认向量
      · JSON 解析失败 → 默认向量 + debug 记录
      · 缺字段 / 字段类型错 → 修补到合法值 (例如 non-float 退 0.0)

    Returns:
        情绪向量 dict 的**副本** (调用方修改不会影响 cache)。前端 GET /ai/emotion
        拿到副本直接 jsonify 即可, 不需要再 copy。
    """
    global _emotion_cache
    if _emotion_cache is None:
        _ensure_memory_dir()
        loaded: dict = {}
        try:
            if os.path.exists(PARAMS_FILE):
                with open(PARAMS_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                emo_section = (raw or {}).get("emotion_vector", {})
                if isinstance(emo_section, dict):
                    loaded = dict(emo_section)
        except (json.JSONDecodeError, OSError, AttributeError, TypeError) as e:
            debug(f"[emotion] load failed, reset to default: {e}")
            loaded = {}

        # 修补: 每个轴必须是 float, 缺/类型错 → 默认 0.0
        out = dict(EMOTION_DEFAULT)
        for axis in EMOTION_AXES:
            v = loaded.get(axis, 0.0)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                # clamp 到 [-1.0, 1.0], 避免脏数据 (例如 LLM 偶尔输出 1.5)
                out[axis] = max(-1.0, min(1.0, float(v)))
            else:
                out[axis] = 0.0
        # dt 缺则填当前 dt (避免衰减时 dt 为空)
        if not isinstance(loaded.get("last_update_dt"), str) or not loaded["last_update_dt"]:
            out["last_update_dt"] = format_dt_second(now_ms())
        else:
            out["last_update_dt"] = loaded["last_update_dt"]
        out["version"] = 1
        _emotion_cache = out
        debug(f"[emotion] loaded: {out}")
    # 返回副本, 防止外部就地改坏 cache
    return dict(_emotion_cache)


def _save_emotion(emotion: dict) -> None:
    """
    把情绪向量写回 params.json (走 _save_params 的"读-改-写",
    保留 ask_count_by_fp 等其他字段)。

    Args:
        emotion: 完整情绪向量 dict, 应至少含 EMOTION_AXES + last_update_dt + version。
                 不做字段级修补 —— 调用方 (apply_decay / apply_emotion_delta)
                 负责构造合法值。

    Returns:
        bool — _save_params 透传 True/False, 暂时丢弃。
    """
    global _emotion_cache
    _emotion_cache = dict(emotion)
    _save_params({"emotion_vector": _emotion_cache})


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
