"""
Crystal 情绪状态机。

设计:
  · 状态量: 4 维向量 [valence, arousal, novelty, clarity], 每轴 ∈ [-1.0, 1.0]。
  · 情绪更新: 每次 chat ask 后异步调用弱模型 (Lint LLM), 基于
      (C: 对话上下文, S: 自我认知, E: 旧情绪向量) → 输出 delta。
  · 衰减: 后台守护协程每 EMOTION_DECAY_INTERVAL_SEC 秒跑一次, 按半衰期
          EMOTION_DECAY_HALF_LIFE_MIN 向 0 收敛; 收敛到 0.01 阈值以下不再落盘。
  · L1/L2 词袋阶段已删除 — 全部由 Lint LLM 统一判断。

文件职责:
  ai_emotion.py  ← Lint LLM 调用 / 衰减 / 守护协程
  ai_io.py       ← _load_emotion / _save_emotion / cache
  ai_config.py   ← EMOTION_DECAY_* 常量
  ai_agent.py    ← LangGraph 节点 (emotion_llm_node)
  ai_graph.py    ← 把新节点接入 run_ask 流水线
  ai_routes.py   ← GET /ai/emotion, GET/POST /ai/emotion/config, 启动守护协程
"""
from __future__ import annotations

import asyncio
import json as _json
import math
import os
from datetime import datetime
from typing import Optional

from .ai_config import (
    EMOTION_DECAY_HALF_LIFE_MIN,
    EMOTION_DECAY_INTERVAL_SEC,
    EMOTION_DECAY_WRITE_THRESHOLD,
    EMOTION_DECAY_FACTOR,
    BASE_DIR,
    MEMORY_DIR,
)
from .ai_io import (
    EMOTION_AXES,
    EMOTION_DEFAULT,
    _load_emotion,
    _save_emotion,
)
from .ai_llm import _call_llm
from .ai_models import AiAskReq
from .ai_utils import format_dt_second, now_ms
from utils.log import debug


# ═══════════════════════════════════════════════════════════════════════
# Lint LLM 配置 (由 /ai/emotion/config 端点写入)
# ═══════════════════════════════════════════════════════════════════════

EMOTION_LLM_CONFIG: dict = {
    "api_key": "",
    "api_url": "",
    "model": "deepseek-flash",
}


def update_emotion_llm_config(config: dict) -> None:
    """
    由 /ai/emotion/config POST 端点调用, 更新模块级 EMOTION_LLM_CONFIG。

    同步把配置持久化到 ai/memory/params.json.llm_configs.lint (与 Main LLM 同盘),
    这样后端重启不丢、多端任意一端更新另一端下次访问时拉到最新值。
    """
    global EMOTION_LLM_CONFIG
    EMOTION_LLM_CONFIG.update({
        "api_key": config.get("api_key", ""),
        "api_url": config.get("api_url", ""),
        "model":   config.get("model", "deepseek-flash"),
    })
    # 落盘 (静默失败, 内存已更新, 持久化只是 bonus)
    try:
        from .ai_io import _save_llm_configs
        _save_llm_configs({"lint": dict(EMOTION_LLM_CONFIG)})
    except Exception:
        pass


def load_emotion_llm_config_from_disk() -> bool:
    """
    进程启动时从 params.json.llm_configs.lint 把 Lint LLM 配置读回模块变量。
    由 PdfBacken.py 的 lifespan 调用。

    Returns:
        bool — True 表示从磁盘读到非空配置, False 表示磁盘无配置或读取失败
               (False 时保持默认空值, 由前端推送兜底)
    """
    try:
        from .ai_io import _load_llm_configs
        cfg = _load_llm_configs().get("lint") or {}
        if not cfg or not cfg.get("api_key"):
            return False
        global EMOTION_LLM_CONFIG
        EMOTION_LLM_CONFIG.update({
            "api_key": cfg.get("api_key", ""),
            "api_url": cfg.get("api_url", ""),
            "model":   cfg.get("model", "deepseek-flash"),
        })
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════
# 衰减 — 半衰期指数衰减
# ═══════════════════════════════════════════════════════════════════════
#
# 公式: new_v = old_v * 0.5 ^ (elapsed_min / HALF_LIFE_MIN)
#   · base half_life = 4h → 每 4h 各轴衰减到当前值的一半
#   · 各轴实际半衰期 = base × EMOTION_DECAY_FACTOR[axis]
#   · 触发: 后台守护协程每 EMOTION_DECAY_INTERVAL_SEC 跑一次
#   · 落盘: 任一轴变化 > EMOTION_DECAY_WRITE_THRESHOLD 才写盘


def _parse_dt(dt_str: str) -> Optional[datetime]:
    """解析 'YYYY-MM-DD HH:MM:SS' → naive datetime (北京时间)。失败返回 None。"""
    if not dt_str or not isinstance(dt_str, str):
        return None
    try:
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def apply_decay_once() -> bool:
    """
    单次衰减判定。同步函数, 由后台守护协程每 N 秒调用一次。

    Returns:
        bool — True 表示有落盘, False 表示无变化或被跳过。
    """
    emo = _load_emotion()
    last_dt = _parse_dt(emo.get("last_update_dt"))
    if last_dt is None:
        # 缺 dt → 用 now 重置 (兜底)
        emo["last_update_dt"] = format_dt_second(now_ms())
        _save_emotion(emo)
        debug("[emotion_decay] last_update_dt missing/parse-fail, reset to now")
        return True

    # elapsed_min 基于 now_ms 派生 — 与 format_dt_second 完全同源 (北京时区)
    now_dt = datetime.strptime(format_dt_second(now_ms()), "%Y-%m-%d %H:%M:%S")
    elapsed_min = (now_dt - last_dt).total_seconds() / 60.0
    if elapsed_min <= 0:
        # 时钟回拨 / NTP 跳变; 不衰减, 也不写盘
        return False

    # 各轴独立半衰期: axis_half_life = base × EMOTION_DECAY_FACTOR[axis]。
    base_hl = EMOTION_DECAY_HALF_LIFE_MIN
    new_emo = dict(emo)
    changed = False
    for axis in EMOTION_AXES:
        factor = EMOTION_DECAY_FACTOR.get(axis, 1.0)
        if factor <= 0:
            factor = 1.0
        axis_half_life = base_hl * factor
        decay = 0.5 ** (elapsed_min / axis_half_life)
        old_v = emo.get(axis, 0.0)
        new_v = old_v * decay
        if abs(new_v - old_v) > EMOTION_DECAY_WRITE_THRESHOLD:
            new_emo[axis] = new_v
            changed = True
        else:
            # 收敛到阈值以下, 直接置 0 (避免极小残留值持续被算进 |change| 噪声)
            new_emo[axis] = 0.0 if abs(old_v) <= EMOTION_DECAY_WRITE_THRESHOLD else old_v

    if not changed:
        return False

    new_emo["last_update_dt"] = format_dt_second(now_ms())
    new_emo["version"] = 1
    _save_emotion(new_emo)
    debug(
        f"[emotion_decay] applied: elapsed={elapsed_min:.1f}min "
        f"v=[{emo['valence']:+.3f}->{new_emo['valence']:+.3f} hl={base_hl*EMOTION_DECAY_FACTOR.get('valence', 1.0):.0f}m] "
        f"a=[{emo['arousal']:+.3f}->{new_emo['arousal']:+.3f} hl={base_hl*EMOTION_DECAY_FACTOR.get('arousal', 1.0):.0f}m] "
        f"n=[{emo['novelty']:+.3f}->{new_emo['novelty']:+.3f} hl={base_hl*EMOTION_DECAY_FACTOR.get('novelty', 1.0):.0f}m] "
        f"c=[{emo['clarity']:+.3f}->{new_emo['clarity']:+.3f} hl={base_hl*EMOTION_DECAY_FACTOR.get('clarity', 1.0):.0f}m]"
    )
    return True

def _emotion_decay_loop() -> None:
    """
    后台守护协程: 每 EMOTION_DECAY_INTERVAL_SEC 跑一次 apply_decay_once。
    用同步函数 + time.sleep 简单实现, 不需要额外库。

    异常隔离: 每次 apply_decay_once 在 try/except 中调用,
    单次失败不影响后续 tick。
    """
    debug(
        f"[emotion_decay] loop START: interval={EMOTION_DECAY_INTERVAL_SEC}s "
        f"base_half_life={EMOTION_DECAY_HALF_LIFE_MIN}min"
    )
    while True:
        try:
            apply_decay_once()
        except Exception as e:
            debug(f"[emotion_decay] tick FAIL: {type(e).__name__}: {e}")
        try:
            import time as _time
            _time.sleep(EMOTION_DECAY_INTERVAL_SEC)
        except Exception as e:
            debug(f"[emotion_decay] sleep FAIL: {type(e).__name__}: {e}")
            _time.sleep(1)  # 兜底, 不让循环空转


_emotion_decay_thread: Optional[object] = None  # threading.Thread 引用


def start_emotion_decay_thread() -> None:
    """
    启动情绪衰减后台守护线程 (daemon=True, 进程退出自动结束)。

    启动策略:
      · PdfBacken.py 的 lifespan 里调一次。
      · 模块级幂等: 已启动时直接 return, 重复调用安全。

    之所以选线程而不是 asyncio task:
      · apply_decay_once 是纯 CPU+文件 IO, 没有 await; 用线程更简单。
      · 与 FastAPI 异步事件循环解耦, 不会阻塞请求处理。
    """
    global _emotion_decay_thread
    if _emotion_decay_thread is not None and _emotion_decay_thread.is_alive():
        return
    import threading
    t = threading.Thread(
        target=_emotion_decay_loop,
        name="emotion-decay-loop",
        daemon=True,
    )
    _emotion_decay_thread = t
    t.start()
    debug("[emotion_decay] thread started")


# ═══════════════════════════════════════════════════════════════════════
# 应用 delta (同步路径, emotion_llm 共用)
# ═══════════════════════════════════════════════════════════════════════

def apply_emotion_delta(delta: dict) -> dict:
    """
    把 delta 增量叠加到当前情绪向量, 裁到 [-1.0, 1.0], 更新 last_update_dt 并落盘。

    Args:
        delta: 4 轴增量 dict, e.g. {"valence": +0.10, "arousal": +0.05, ...}
               缺轴视为 0.0。

    Returns:
        新的情绪向量 dict (副本)。
    """
    emo = _load_emotion()
    new_emo = dict(emo)
    for axis in EMOTION_AXES:
        d = delta.get(axis, 0.0)
        if not isinstance(d, (int, float)) or isinstance(d, bool):
            d = 0.0
        cur = new_emo.get(axis, 0.0)
        nv = cur + float(d)
        new_emo[axis] = max(-1.0, min(1.0, nv))
    new_emo["last_update_dt"] = format_dt_second(now_ms())
    new_emo["version"] = 1
    _save_emotion(new_emo)
    debug(
        f"[emotion_delta] applied: "
        f"v=[{emo['valence']:+.3f}->{new_emo['valence']:+.3f}] "
        f"a=[{emo['arousal']:+.3f}->{new_emo['arousal']:+.3f}] "
        f"n=[{emo['novelty']:+.3f}->{new_emo['novelty']:+.3f}] "
        f"c=[{emo['clarity']:+.3f}->{new_emo['clarity']:+.3f}] "
        f"delta={delta}"
    )
    return new_emo


# ═══════════════════════════════════════════════════════════════════════
# Lint LLM — 单节点异步情绪判断
# ═══════════════════════════════════════════════════════════════════════
#
# 输入 (C/S/E 三元):
#   C — 对话上下文: 近 N 轮 user↔assistant 对话摘要
#   S — 自我认知: Crystal_memory.md 内容摘要 (仅首 N 字, 防止 prompt 膨胀)
#   E — 旧情绪向量: 当前 4 轴值
#
# 输出:
#   delta dict:  {"valence": ±x, "arousal": ±x, "novelty": ±x, "clarity": ±x}
#   每轴 delta ∈ [-0.30, +0.30]; 禁止单轮大幅跳变。
#
# 设计原则:
#   · 完全异步: 不阻塞主链路响应, 在 save_paper_memory 之后单独节点跑。
#   · 无节流: 每次 chat ask 都调用 (Lint LLM 轻量, 不怕烧 token)。
#   · 无 L1 词袋: 全部委托给 Lint LLM, 保证语义一致性。

_LINT_SYSTEM_PROMPT = """你是 Crystal 的情感分析模块。你的任务是根据以下信息,输出情绪向量变化。

【对话上下文 C】
(近几轮 user↔assistant 对话, 按时间顺序排列)

【Crystal 自我认知 S】
(Crystal 对自身的认知摘要, 由 user prompt 注入。请注意 S 是 Crystal 自己的人设/语气/性格认知,
 不是关于用户的认知 —— 若 user prompt 给的内容明显是用户偏好而非自我人设, 优先把它当参考背景,
 避免把"用户喜欢 X"误读为"自我状态 X"。)

【旧情绪向量 E】
valence: {v}  arousal: {a}  novelty: {n}  clarity: {c}
(各轴 ∈ [-1.0, +1.0], 含义 (axis 各自独立, 不要混淆):
  valence  — 情感正负向: +开心/温暖/满足, -生气/伤心/失望
  arousal  — 唤醒度 (能量级): +激动/兴奋/精神, -低落/平静/疲倦
  novelty  — 新奇度 (信息新鲜感): +好奇/意外/有新鲜感, -平淡/无聊/习以为常
  clarity  — 清晰度 (思路状态): +清醒/坚定/思路清晰, -困惑/迷茫/思路混沌)

【任务】
分析对话上下文和 Crystal 的自我认知, 判断情绪应当如何变化。
输出两种模式之一:
1. 若情绪明显变化:
   {{"valence": +x.x, "arousal": +x.x, "novelty": +x.x, "clarity": +x.x}}
   (4轴delta, 每轴 ∈ [-0.30, +0.30])
2. 若情绪基本不变:
   {{"no_change": true}}

【约束】
· 仔细体会对话的整体情绪色调, 不要被单句带偏
· 主流情绪即可, 不需要过度解读
· delta 每次不超过 ±0.30, 保持平滑过渡
· 4 轴彼此独立: 例如"疲倦"主要影响 arousal(负向), 不要把它同时叠到 valence 上
· 如果对话中性平淡, 返回 {{"no_change": true}}
· 输出必须是严格 JSON, 不要任何额外说明文字
"""


def _load_self_cognition() -> str:
    """
    读取 Crystal 自我认知文件, 返回前 800 字摘要 (防 prompt 膨胀)。

    注意: 路径必须对应 Crystal_self.md (自我认知), 不是 Crystal_memory.md (用户认知)。
    之前版本写错路径 — Lint LLM 实际看到的是用户笔记, 不是 self 笔记, 会导致情绪判断
    把"用户偏好"误读为"自我状态", 引起不必要的情绪扰动。
    """
    try:
        path = os.path.join(MEMORY_DIR, "Crystal_self.md")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                content = f.read()
            return content[:800] + ("…" if len(content) > 800 else "")
    except Exception:
        pass
    return "(无自我认知数据)"


def _build_conversation_context(state: dict) -> str:
    """
    从 state 构建对话上下文摘要字符串。

    取 paper_ask_history / paper_history 最近 5 轮,
    拼成 user ↔ assistant 交替的格式。
    """
    ask_hist: list = state.get("paper_ask_history", [])
    res_hist: list = state.get("paper_history", [])
    lines = []
    # 最近 5 轮, 奇数步取 ask, 偶数步取对应 response
    n = min(5, len(ask_hist))
    for i in range(n):
        ask_entry = ask_hist[-(i + 1)]
        res_entry = res_hist[-(i + 1)] if -(i + 1) >= -len(res_hist) else None
        ask_text = (ask_entry.get("ask") or "")[:300]
        res_text = (res_entry.get("content") or "")[:300] if res_entry else "(无回复)"
        lines.append(f"User: {ask_text}\nCrystal: {res_text}")
    lines.reverse()
    return "\n\n".join(lines) if lines else "(无对话历史)"


async def _call_emotion_llm(
    context: str,
    self_cognition: str,
    emotion: dict,
) -> Optional[dict]:
    """
    调 Lint LLM 评估情绪 delta。

    Args:
        context:      对话上下文字符串 (C)
        self_cognition: 自我认知摘要 (S)
        emotion:      当前情绪向量 dict (E)

    Returns:
        delta dict {"valence": ..., "arousal": ..., "novelty": ..., "clarity": ...}
        若无变化返回 {"no_change": true}
        失败返回 None (不抛异常)。
    """
    v = emotion.get("valence", 0.0)
    a = emotion.get("arousal", 0.0)
    n = emotion.get("novelty", 0.0)
    c = emotion.get("clarity", 0.0)

    user_prompt = (
        f"【对话上下文 C】\n{context}\n\n"
        f"【Crystal 自我认知 S】\n{self_cognition}\n\n"
        f"【旧情绪向量 E】\n"
        f"valence: {v:+.2f}  arousal: {a:+.2f}  novelty: {n:+.2f}  clarity: {c:+.2f}\n\n"
        "请分析本轮对话的情绪变化, 输出严格 JSON。"
    )

    try:
        # 使用 emotion 专用配置 (api_key/api_url/model) 调 Lint LLM
        content, _ = await _call_llm(
            messages=[
                {"role": "system", "content": _LINT_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            vision_model=False,
            disable_thinking=True,
            # --- 以下参数传给 _call_llm 的 ai_config 覆盖 ---
            emotion_config=EMOTION_LLM_CONFIG,
        )
    except Exception as e:
        debug(f"[emotion_llm] LLM call FAIL: {type(e).__name__}: {e}")
        return None

    text = (content or "").strip()
    # 去掉 ```json 包裹
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = _json.loads(text)
    except Exception as e:
        debug(f"[emotion_llm] JSON parse FAIL: {e} | raw={text[:100]}")
        return None

    # no_change 模式
    if isinstance(parsed, dict) and parsed.get("no_change"):
        return {"valence": 0.0, "arousal": 0.0, "novelty": 0.0, "clarity": 0.0}

    if not isinstance(parsed, dict):
        return None

    # 解析 4 轴 delta
    out: dict = {}
    for axis in EMOTION_AXES:
        v = parsed.get(axis, 0.0)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[axis] = max(-0.30, min(0.30, float(v)))
        else:
            out[axis] = 0.0
    return out


async def emotion_llm_node(state: dict) -> dict:
    """
    LangGraph 节点: Lint LLM 异步评估情绪。

    触发: 每次 chat ask (pdf_fp == crystal_chat) 都会调用。
    非 chat 场景 (论文) 跳过 — 论文侧节奏快, 情绪不在这类场景更新。

    不阻塞主链路: 本节点在 save_paper_memory 之后, 与 flush_track 并行。
    """
    req = state.get("req")
    if not isinstance(req, AiAskReq):
        return {}
    if req.pdf_fp != "crystal_chat":
        return {}

    # 检查 Lint LLM 是否已配置
    if not EMOTION_LLM_CONFIG.get("api_key") or not EMOTION_LLM_CONFIG.get("model"):
        debug("[emotion_llm] SKIP — Lint LLM not configured (api_key or model empty)")
        return {}

    context       = _build_conversation_context(state)
    self_cogn     = _load_self_cognition()
    cur_emo       = _load_emotion()

    delta = await _call_emotion_llm(context, self_cogn, cur_emo)
    if delta is None:
        debug("[emotion_llm] no delta (LLM returned None)")
        return {}

    # no_change (全 0) 也走 apply_emotion_delta, 让 last_update_dt 刷新
    new_emo = apply_emotion_delta(delta)
    debug(f"[emotion_llm] applied: delta={delta} new={new_emo}")
    return {"emotion": new_emo}


# ═══════════════════════════════════════════════════════════════════════
# Prompt 注入 — 把情绪向量转成自然语言描述
# ═══════════════════════════════════════════════════════════════════════
#
# 设计:
#   · 决策模型: 与前端 decodeEmotion.js **完全同源** (4 轴加权和选 family, L2 能量算 level),
#     避免前端立绘与后端 LLM 提示词错位 (e.g. 前端给用户看 positive, 后端告诉 LLM "困倦"
#     就会导致 LLM 按"困倦"语气回复, 但立绘是开心脸 — 体验割裂)。
#   · 中性阈值: 4 轴 L2 能量 < 0.15 — 与前端 magnitude < 0.15 完全一致, 中性不注入。
#   · family 平分时按 positive > negative > curious > confuse 顺序选,
#     与前端 Object.entries 声明顺序一致 (Python 3.7+ dict 与 ES2015+ 都保留插入序)。
#   · level 档位: magnitude * 3 四舍五入到 1~3 — 与前端 `Math.round(magnitude * 3)` 同源。
#   · 文本指令是"间接影响" (语气/用词/节奏), 不要让 LLM 直接说自己
#     "开心了 / 伤心了" 等元描述 — 否则会很假。

# family 加权和 (与前端 decodeEmotion.js familyScores 1:1 对齐)
_EMOTION_FAMILY_SCORES: dict[str, dict[str, float]] = {
    "positive": {"valence": 1.0, "arousal": 0.0, "novelty": 0.0, "clarity": 0.5},
    "negative": {"valence": -1.0, "arousal": -0.3, "novelty": 0.0, "clarity": -0.5},
    "curious":  {"valence": 0.0, "arousal": 1.0, "novelty": 1.0, "clarity": 0.0},
    "confuse":  {"valence": 0.0, "arousal": -1.0, "novelty": -1.0, "clarity": -0.5},
}

# family → (中文主词, 中文描述) — 给 LLM 用的拟人描述
_EMOTION_LABEL: dict[str, tuple[str, str]] = {
    "positive": ("温暖", "整体偏正向, 情绪是积极、明亮的"),
    "negative": ("低落", "整体偏负向, 情绪带有灰调或失落感"),
    "curious":  ("好奇", "对当下话题有探索欲, 想多聊几句或换个角度深挖"),
    "confuse":  ("困倦或迷茫", "情绪平静或思路混沌, 节奏会偏慢、句子会偏柔"),
}


def _classify_level_by_magnitude(magnitude: float) -> str:
    """
    强中弱三档: 由 L2 综合能量 magnitude 推导 (与前端 level = round(magnitude*3) 同源)。
    level 1 → "轻度", 2 → "中等", 3 → "强烈"。
    """
    level = max(1, min(3, round(magnitude * 3)))
    return {1: "轻度", 2: "中等", 3: "强烈"}[level]


def build_emotion_context_block(emotion: dict | None) -> str:
    """
    把当前情绪向量转成一段给 LLM 看的自然语言提示词。

    Args:
        emotion: 4 轴 dict, 由 _load_emotion() 返回; None 或非法 dict → 返回 ""。

    Returns:
        多行字符串, 以 `\n\n` 拼接到 system content 末尾;
        中性 (L2 能量 < 0.15) 返回 "" (空字符串, 调用方决定是否拼入)。

    用法 (compose_chat_messages):
        emo = _load_emotion()
        block = build_emotion_context_block(emo)
        if block:
            system_content += "\n\n" + block
    """
    if not emotion or not isinstance(emotion, dict):
        return ""

    # 取 4 轴 float (容错)
    axis_values: dict[str, float] = {}
    for axis in EMOTION_AXES:
        v = emotion.get(axis, 0.0)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        axis_values[axis] = v

    # L2 综合能量 (与前端 decodeEmotion.js magnitude 同源)
    magnitude = math.sqrt(sum(v * v for v in axis_values.values()))
    if magnitude < 0.15:
        return ""  # 中性 — 与前端 crystal_normal.png 对齐

    # family 加权和: 与前端 decodeEmotion.js familyScores 同源。
    # 4 个 family 都按线性组合算得分, 取最高分; 平分时按 positive > negative > curious > confuse
    # 顺序选 (dict 插入序, 与前端 Object.entries 行为一致)。
    best_family = "positive"
    best_score = -float("inf")
    for family, weights in _EMOTION_FAMILY_SCORES.items():
        score = sum(weights.get(axis, 0.0) * axis_values.get(axis, 0.0) for axis in EMOTION_AXES)
        if score > best_score:
            best_score = score
            best_family = family

    label_word, descr = _EMOTION_LABEL[best_family]
    level = _classify_level_by_magnitude(magnitude)

    return (
        "【当前情绪状态 (Crystal 自身情绪, 不要对用户直述, "
        "仅在语气、用词、节奏上自然地体现)】\n"
        f"此刻的主导情绪是「{label_word}」({level}, {descr}), "
        f"综合强度 {magnitude:.2f}。\n"
        "请让回复自然地贴合这个情绪基调 —— "
        "例如节奏快慢、用词温度、对用户的态度等; "
        f"但不要直接说出「我现在{label_word}了」, 也不要破坏信息的准确性。"
    )
