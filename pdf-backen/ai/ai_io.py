"""
文件 IO + 内存 cache：paper_history / memory 文件。

集中管理:
- Crystal_memory.md (CRYSTAL_MEMORY_FILE) 的进程内缓存
- Crystal_self.md (CRYSTAL_SELF_FILE) 的进程内缓存
- Crystal_explore.md (EXPLORE_FILE, 主动外呼档案) 的进程内缓存 + 整份写盘
- save/{fp}_ai.json 的读取 + mtime 缓存
- ai/memory/params.json 的读写 (存 _ask_count_by_fp / emotion_vector /
  llm_configs, 持久化以跨进程存活; 所有访问都经
  _load_ask_counts / _save_ask_counts / _inc_ask_count / _get_ask_count /
  _save_params, 不要直接摸 _ask_count_by_fp)

所有 cache 均为模块级单例,跨请求保持,跟原来散落在 ai_agent.py 的行为一致。
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

from .ai_config import (
    CHAT_FP,
    CHAT_MEMORY_LIMIT,
    CRYSTAL_MEMORY_FILE,
    CRYSTAL_SELF_FILE,
    EXPLORE_FILE,
    MEMORY_DIR,
    PARAMS_FILE,
    SAVE_DIR,
    FATIGUE_INCREMENT_MAIN,
    FATIGUE_INCREMENT_PROACTIVE,
    FATIGUE_DECAY_WRITE_THRESHOLD,
)
# 2026-10-10: 移除 FATIGUE_HARD_STOP import — 硬停止整套机制已下线。
from .ai_utils import _write_text_file_atomic, format_dt_second, now_ms
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
# Crystal_explore.md (主动外呼档案) — 缓存 + 读写
# ═══════════════════════════════════════════════════════════════════════
#
# 与 memory / self 的区别:
#   - memory (U): 关于「他」的认知, 注入 chat system。
#   - self   (S): Crystal 的自我认知, 注入 chat system。
#   - explore:   关于「主动开口」的档案, 注入 chat system。
#     LLM 拿到 Crystal_explore.md 全文自由重写, 无段落约束 (无固定章节)。
#     prompt 仅约束"只写时间相关认知" (活跃时段 / 沉默含义 / 回应速度)。
#
# 读: _get_agent_explore()  (全文)
# 写: _write_agent_explore() (整份覆盖, 内部无段级操作)

_explore_cache: str | None = None


def _get_agent_explore() -> str:
    """读 Crystal_explore.md 缓存 (首次才打磁盘)。缺失时返回空串, 由调用方决定是否回退。"""
    global _explore_cache
    if _explore_cache is None:
        _ensure_memory_dir()
        if os.path.exists(EXPLORE_FILE):
            try:
                with open(EXPLORE_FILE, "r", encoding="utf-8") as f:
                    _explore_cache = f.read()
            except OSError:
                _explore_cache = ""
        else:
            _explore_cache = ""
    return _explore_cache


def _set_agent_explore_cache(md: str) -> None:
    """Crystal_explore 写盘后同步刷新缓存, 避免下一个请求读到陈旧数据。"""
    global _explore_cache
    _explore_cache = md


def _write_agent_explore(md: str) -> bool:
    """
    整份覆盖 Crystal_explore.md 并刷新缓存。

    与 memory 的写入一样走 atomic 写 (临时文件 + os.replace), 避免后台
    协程写盘时正好被一次读命中半截内容。

    注意: _write_text_file_atomic 返回的是**字符数** (不是字节数), 因为
    文本模式下 f.tell() 返回的是不透明的 cookie, 不能拿来比长度。
    """

    ok = _write_text_file_atomic(EXPLORE_FILE, md) == len(md)
    if ok:
        _set_agent_explore_cache(md)
    return ok


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

    计数必须**立即**落盘 —— 节点入口就变了, 不能依赖任何延迟统一写。
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


# ═══════════════════════════════════════════════════════════════════════
# 疲劳值 (fatigue_vector) — 缓存 + 读写
# ═══════════════════════════════════════════════════════════════════════
#
# Schema:
#   {
#     "value":          float,    # 疲劳度, [0.0, 1.0], 0=精神, 1=极度疲劳
#     "last_update_dt": str,      # "YYYY-MM-DD HH:MM:SS" 北京时间, 衰减锚点
#     "version":        int,
#   }
#
# 与 emotion_vector 的关键区别:
#   · 一维标量 (不是 4 轴向量)
#   · 衰减双态: 工作中不衰减, 静默超过 FATIGUE_REST_THRESHOLD_MIN 才进入休息态衰减
#   · 累加由调用方显式触发: _inc_fatigue_main / _inc_fatigue_proactive
#   · 与 emotion_llm_node / layer1_node 同样只服务 chat 侧 (非 CHAT_FP → 跳过)
#
FATIGUE_DEFAULT: dict = {
    "value": 0.0,
    "last_update_dt": "",  # 首次加载时由 _load_fatigue 填上当前 dt
    "version": 1,
}

_fatigue_cache: dict | None = None


def _load_fatigue() -> dict:
    """
    加载疲劳值。懒加载 + 缓存; 磁盘缺失则返回默认向量 (0.0)。
    与 _load_emotion 同一套语义:
      · 文件不存在 → 默认向量
      · JSON 解析失败 → 默认向量 + debug
      · 缺字段 / 字段类型错 → 修补到合法值 (clamp [0.0, 1.0])
    Returns:
        疲劳值 dict 的**副本** (调用方修改不会影响 cache)。
    """
    global _fatigue_cache
    if _fatigue_cache is None:
        _ensure_memory_dir()
        loaded: dict = {}
        try:
            if os.path.exists(PARAMS_FILE):
                with open(PARAMS_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                section = (raw or {}).get("fatigue_vector", {})
                if isinstance(section, dict):
                    loaded = dict(section)
        except (json.JSONDecodeError, OSError, AttributeError, TypeError) as e:
            debug(f"[fatigue] load failed, reset to default: {e}")
            loaded = {}

        out = dict(FATIGUE_DEFAULT)
        v = loaded.get("value", 0.0)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            # clamp 到 [0.0, 1.0], 避免脏数据
            out["value"] = max(0.0, min(1.0, float(v)))
        else:
            out["value"] = 0.0
        if not isinstance(loaded.get("last_update_dt"), str) or not loaded["last_update_dt"]:
            out["last_update_dt"] = format_dt_second(now_ms())
        else:
            out["last_update_dt"] = loaded["last_update_dt"]
        out["version"] = 1
        _fatigue_cache = out
        debug(f"[fatigue] loaded: {out}")
    return dict(_fatigue_cache)


def _save_fatigue(fatigue: dict) -> None:
    """
    把疲劳值写回 params.json (走 _save_params 的"读-改-写",
    保留 ask_count_by_fp / emotion_vector 等其他字段)。

    Args:
        fatigue: 完整疲劳值 dict, 应至少含 value + last_update_dt + version。
                 不做字段级修补 —— 调用方负责构造合法值。
    """
    global _fatigue_cache
    _fatigue_cache = dict(fatigue)
    _save_params({"fatigue_vector": _fatigue_cache})


def _get_fatigue_value() -> float:
    """读当前 fatigue.value, 缺省 0.0。供 ai_active / ai_emotion 快速读。"""
    return _load_fatigue().get("value", 0.0)


def _set_fatigue_value(new_value: float) -> float:
    """
    直接设置 fatigue.value (clamp + 落盘 + 刷 last_update_dt)。
    返回新 value。供衰减线程 / 调试用。
    """
    cur = _load_fatigue()
    new_v = max(0.0, min(1.0, float(new_value)))
    cur["value"] = new_v
    cur["last_update_dt"] = format_dt_second(now_ms())
    cur["version"] = 1
    _save_fatigue(cur)
    return new_v


def _inc_fatigue(pdf_fp: str, source: str = "main") -> float:
    """
    chat 侧疲劳值累加, 立刻落盘, 返回新 value。

    Args:
        pdf_fp: 必须 == CHAT_FP, 否则不累加, 立即 return 当前 value (chat-only 约束)
        source: "main" (Crystal 出主答) 或 "proactive" (Crystal 主动追问落盘)
                系数由 ai_config.FATIGUE_INCREMENT_* 决定
    Returns:
        累加后的新 fatigue.value (浮点)。
    """
    if pdf_fp != CHAT_FP:
        # 论文侧不累加 (与 emotion_llm_node / layer1_node 同约束)
        return _get_fatigue_value()
    if source == "proactive":
        delta = FATIGUE_INCREMENT_PROACTIVE
    else:
        delta = FATIGUE_INCREMENT_MAIN
    cur = _load_fatigue()
    new_v = max(0.0, min(1.0, cur.get("value", 0.0) + delta))
    cur["value"] = new_v
    cur["last_update_dt"] = format_dt_second(now_ms())
    cur["version"] = 1
    _save_fatigue(cur)
    debug(f"[fatigue] inc source={source} delta=+{delta:.2f} -> value={new_v:.3f}")
    return new_v


def _apply_fatigue_decay() -> bool:
    """
    单次疲劳衰减判定。同步函数, 由后台 _fatigue_decay_thread 每 N 秒调用一次。
    双态动力学:
      · 距 last_update_dt < FATIGUE_REST_THRESHOLD_MIN  → 工作中, 不衰减
      · 距 last_update_dt ≥ FATIGUE_REST_THRESHOLD_MIN  → 休息态, 走半衰期 FATIGUE_REST_HALF_LIFE_MIN
    Returns:
        bool — True 表示有落盘, False 表示无变化或被跳过。
    """
    from .ai_config import (
        FATIGUE_REST_THRESHOLD_MIN,
        FATIGUE_REST_HALF_LIFE_MIN,
        FATIGUE_DECAY_WRITE_THRESHOLD,
    )
    from datetime import datetime

    cur = _load_fatigue()
    last_dt_str = cur.get("last_update_dt", "")
    if not last_dt_str:
        # 缺 dt → 用 now 重置
        cur["last_update_dt"] = format_dt_second(now_ms())
        _save_fatigue(cur)
        debug("[fatigue_decay] last_update_dt missing, reset to now")
        return True

    try:
        last_dt = datetime.strptime(last_dt_str, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        cur["last_update_dt"] = format_dt_second(now_ms())
        _save_fatigue(cur)
        debug("[fatigue_decay] last_update_dt parse-fail, reset to now")
        return True

    now_dt = datetime.strptime(format_dt_second(now_ms()), "%Y-%m-%d %H:%M:%S")
    elapsed_min = (now_dt - last_dt).total_seconds() / 60.0
    if elapsed_min <= 0:
        return False  # 时钟回拨

    old_v = cur.get("value", 0.0)
    if elapsed_min < FATIGUE_REST_THRESHOLD_MIN:
        # 工作中: 不衰减, 只刷新 dt (避免一直不刷 dt 永远不进入休息态)
        new_v = old_v
    else:
        # 休息态: 半衰期衰减
        excess_min = elapsed_min - FATIGUE_REST_THRESHOLD_MIN
        decay = 0.5 ** (excess_min / FATIGUE_REST_HALF_LIFE_MIN)
        new_v = old_v * decay

    if abs(new_v - old_v) < FATIGUE_DECAY_WRITE_THRESHOLD and new_v > 0:
        # 收敛到阈值以下, 直接置 0 (避免极小残留)
        if old_v > FATIGUE_DECAY_WRITE_THRESHOLD and new_v < FATIGUE_DECAY_WRITE_THRESHOLD:
            cur["value"] = 0.0
            cur["last_update_dt"] = format_dt_second(now_ms())
            cur["version"] = 1
            _save_fatigue(cur)
            debug(f"[fatigue_decay] converged to 0 (was {old_v:.3f})")
            return True
        return False

    cur["value"] = max(0.0, min(1.0, new_v))
    cur["last_update_dt"] = format_dt_second(now_ms())
    cur["version"] = 1
    _save_fatigue(cur)
    debug(
        f"[fatigue_decay] applied: elapsed={elapsed_min:.1f}min "
        f"value=[{old_v:+.3f}->{cur['value']:+.3f}]"
    )
    return True


# ═══════════════════════════════════════════════════════════════════════
# Paper history IO — save/{fp}_ai.json 读写 + mtime 缓存
# ═══════════════════════════════════════════════════════════════════════


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
    # 缓存表膨胀保护: 简单随机淘汰 (实际工程中 entry 数受论文 fp 数约束,
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


# ───────────────────────────────────────────────────────────────────────
# 历史加载共用原语 (Layer1 重构, 单一事实源)
# ───────────────────────────────────────────────────────────────────────

# entry.type → role 映射 (ReqLoad/ResLoad/ReqAsk/ResAsk 4 类)
_TYPE_TO_ROLE: dict[str, str] = {
    "ReqLoad": "user",
    "ResLoad": "assistant",
    "ReqAsk":  "user",
    "ResAsk":  "assistant",
}

# assistant 角色的 label 派生 (按 chat item.active 字段)
_LABEL_FOR_ASSISTANT: dict[bool, str] = {
    True:  "Crystal主动说",     # entry.active=True
    False: "Crystal回复说",     # entry.active=False (默认)
}


def _resolve_label(role: str, entry: dict) -> str:
    """user -> "用户说"; assistant -> 按 entry.active 决定 "Crystal主动说" / "Crystal回复"."""
    if role == "user":
        return "用户说"
    return _LABEL_FOR_ASSISTANT[bool(entry.get("active", False))]


def _filter_entries(data: list[dict], targets: set[str]) -> list[dict]:
    """
    统一过滤规则 (供 _load_paper_history / load_chat_messages 共用):
      1. 非 dict 直接丢
      2. Anno entry: 完全屏蔽
      3. Vision Ask (ReqAsk + img 非空): 屏蔽 (这条 user + 紧随其后的 assistant 都不要)
      4. type 不在 targets: 跳过
    返回过滤后的 entry 列表 (原序, 不倒序).
    """
    out: list[dict] = []
    skip_next_res = False
    for e in data:
        if not isinstance(e, dict):
            continue
        t = e.get("type", "")
        if t == "Anno":
            continue
        if t == "ReqAsk" and e.get("img"):
            skip_next_res = True
            continue
        if skip_next_res and t == "ResAsk":
            skip_next_res = False
            continue
        if t in targets:
            out.append(e)
    return out


def load_chat_messages(limit: int = CHAT_MEMORY_LIMIT) -> list[dict]:
    """
    通用入口. OpenAI chat.completions 严格 messages:
      [{role: "user"|"assistant", content: <正文>}, ...]

    死字段全去: ts / msg_fp / quotes
    活字段: role + content (entry.active 通过 role=assistant 区分, 不再嵌入 content)。

    来源: save/crystal_chat_ai.json (CHAT_FP), 取最近 limit 轮 (limit*2 条 entry).

    【v3 2026-10-10 改造】不再嵌入 "[<ts_str> 用户说/Crystal主动说/Crystal回复]" 前缀。
      旧版这么做的初衷是给 Crystal 感知时间和角色, 但 LLM 实际行为是:
        1) OpenAI messages 协议本身不携带时间结构, 每个 message 的时间在 LLM
           视野里是不可识别的;
        2) LLM 看到这种格式后会"复读" —— 在自己的输出里也加上
           "[2026-10-10 19:18 Crystal回复说] ..." 前缀, 污染落盘的 ResAsk.content。
      时间感知由 system prompt 的 get_current_time_context() 统一提供 (绝对时间),
      角色感知由 OpenAI 协议 role 字段 (user / assistant) 提供, 主动/被动语义
      在 system prompt 的 chat_audit 块里以"已发/待观察"区分。
      综上: per-message "[ts label]" 元信息是冗余 + 污染, 移除。

    异常/边界:
      - 文件不存在 / 解析失败 -> _read_json_safe 返回 default=[], 函数返回 []
      - data 不是 list (异常结构) -> []
      - content 为空字符串 -> 仍发送 (空 content 在 prompt 里就是空, 无副作用)
    """
    data = _read_json_safe(_paper_history_path(CHAT_FP), [])
    filtered = _filter_entries(data, {"ReqAsk", "ResAsk"})
    tail = filtered[-(limit * 2):]

    out: list[dict] = []
    for e in tail:
        role = _TYPE_TO_ROLE[e["type"]]
        body = e.get("content") or ""
        out.append({
            "role": role,
            "content": body,
        })
    return out


def _load_paper_history(fp: str, mode: str, limit: int) -> list[dict]:
    """
    读 save/{fp}_ai.json, 按 mode 取对应 entry, 返回 OpenAI 格式 messages。

    参数:
      fp:    论文指纹
      mode:  "load" → ReqLoad/ResLoad; "ask" → ReqAsk/ResAsk
      limit: 最大轮次 (每轮 2 条)

    返回: [{role, content, ts, msg_fp, quotes}, ...] 正序

    过滤规则 (与 _filter_entries 共用):
      1. Anno entry: 完全屏蔽, 不参与任何加载
      2. Vision Ask (img 非空): 屏蔽, 不进入上下文也不进入 track
         (Vision Ask 正常写盘，但不参与加载)
    """
    if limit <= 0:
        return []
    data = _read_json_safe(_paper_history_path(fp), [])
    if not isinstance(data, list):
        return []

    if mode == "load":
        targets = {"ReqLoad", "ResLoad"}
    else:
        targets = {"ReqAsk", "ResAsk"}

    filtered = _filter_entries(data, targets)
    tail = filtered[-(limit * 2):]

    out: list[dict] = []
    for e in tail:
        role = _TYPE_TO_ROLE[e["type"]]
        content = e.get("content")
        if not isinstance(content, str):
            continue
        out.append({
            "role": role,
            "content": content,            # 原样进 LLM, 不加 [ts label] 前缀
            "ts": e.get("ts"),             # 保留 ts 用于后续去重
            "msg_fp": e.get("msg_fp", ""), # 透传消息指纹
            "quotes": e.get("quotes", []), # 透传引用指纹数组
            # 主动标记 (新语义): entry.active=True ⇒ Crystal主动说
            "active": bool(e.get("active", False)) if e["type"] == "ResAsk" else False,
        })
    # 取最后 limit*2 条（最近 limit 轮），已正序
    return out


# ═══════════════════════════════════════════════════════════════════════
# 主动追问落盘 (ai_active 唯一对外入口)
# ═══════════════════════════════════════════════════════════════════════
#
# Layer1 完成后, ai_active 把生成的追问 ResAsk 经此函数落盘到
# save/crystal_chat_ai.json。关键: 必须带 active=True, 否则下游
# (track 归纳 / prompts_context / ai_emotion) 无法区分"我主动找
# 他"和"他找我我应答"。

def _append_active_resask(
    content: str,
    *,
    msg_fp: str,
    ts_ms: int,
    intent: str = "",
) -> bool:
    """
    追加一条主动追问 ResAsk 到 crystal_chat_ai.json, 标记 active=True。

    Args:
        content:  追问文本 (已通过 lint, 非空)
        msg_fp:   短指纹, 前端 `/msg` 接口用它去重
        ts_ms:    发送时间 (epoch ms, 北京时区)
        intent:   触发意图 (例 "他两天没上线", 写进 entry 用于回溯)

    Returns:
        bool — 落盘是否成功
    """
    entry: dict = {
        "type": "ResAsk",
        "content": content,
        "ts": ts_ms,
        "dt": format_dt_second(ts_ms),
        "msg_fp": msg_fp,
        "active": True,        # 核心字段: 标识这条是 Crystal 主动发起
        "intent": intent,      # 调试/回溯用, 不进 LLM prompt
    }

    path = _paper_history_path(CHAT_FP)
    _ensure_memory_dir()
    # 读现有列表 (绕过 mtime cache, 因为写入要立刻看到自己刚加的)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = []
    except (OSError, json.JSONDecodeError):
        data = []

    data.append(entry)
    try:
        ok = _write_text_file_atomic(path, json.dumps(data, ensure_ascii=False, indent=2)) > 0
    except (OSError, TypeError, ValueError) as e:
        debug(f"[active_resask] write FAIL: {e}")
        return False

    # 清缓存, 让下一次 _load_paper_history / load_chat_messages 读到新条目
    _paper_history_cache.pop((path, os.path.getmtime(path)) if os.path.exists(path) else (path, 0), None)
    debug(f"[active_resask] APPEND ok: fp={msg_fp} intent={intent[:40]}")
    return ok
