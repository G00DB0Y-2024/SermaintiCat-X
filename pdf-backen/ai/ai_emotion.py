"""
Crystal 情绪状态机 (词池范式, 2026-10-14 重构)。

设计:
  · 状态量: (label, intensity) 二元组 ——
      label ∈ EMOTION_POOL (13 词: 4 象限 × 3 强度档 + 中性原点)
      intensity ∈ [0.0, 1.0]
    label 描述"是什么情绪", intensity 描述"多浓", 二者解耦。
  · 情绪更新: 每次 chat ask 后异步调用弱模型 (Lint LLM), 基于
      (C: 对话上下文, S: 自我认知, E: 旧情绪 (label, intensity)) → 输出
      (new_label, delta_intensity)。
  · 衰减: 后台守护协程每 EMOTION_DECAY_INTERVAL_SEC 秒跑一次, 按单一半衰期
          EMOTION_DECAY_HALF_LIFE_MIN 对 intensity 做指数衰减; 收敛到
          EMOTION_DECAY_WRITE_THRESHOLD 阈值以下 → 强制 label="中性"。

文件职责:
  ai_emotion.py  ← Lint LLM 调用 / 衰减 / 守护协程
  ai_io.py       ← _load_emotion / _save_emotion / cache / 旧数据迁移
  ai_config.py   ← EMOTION_DECAY_* + EMOTION_POOL
  ai_agent.py    ← LangGraph 节点 (emotion_llm_node)
  ai_graph.py    ← 把新节点接入 run_ask 流水线
  ai_routes.py   ← GET /ai/emotion, GET/POST /ai/emotion/config, 启动守护协程
"""
from __future__ import annotations

import asyncio
import json as _json
import os
import re
from datetime import datetime
from typing import Optional

from .ai_config import (
    EMOTION_DECAY_HALF_LIFE_MIN,
    EMOTION_DECAY_INTERVAL_SEC,
    EMOTION_DECAY_WRITE_THRESHOLD,
    EMOTION_DECAY_DEMOTE_RATIO,
    EMOTION_FALLBACK_CHAINS,
    EMOTION_INTENSITY_MAX,
    EMOTION_INTENSITY_MIN,
    EMOTION_LLM_DELTA_MAX,
    EMOTION_POOL,
    EMOTION_POOL_FALLBACK_LABEL,
    EMOTION_POOL_LABELS,
    BASE_DIR,
    MEMORY_DIR,
)
from .ai_io import (
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
# 衰减 — 单 intensity 半衰期指数衰减 (词池范式)
# ═══════════════════════════════════════════════════════════════════════
#
# 公式: new_intensity = old_intensity * 0.5 ^ (elapsed_min / HALF_LIFE_MIN)
#   · 单一 intensity 半衰期 (EMOTION_DECAY_HALF_LIFE_MIN, 默认 4h)
#   · label 不参与衰减 — 强度降到阈值以下时强制改为 "中性"
#   · 触发: 后台守护协程每 EMOTION_DECAY_INTERVAL_SEC 跑一次
#   · 落盘: |Δintensity| > EMOTION_DECAY_WRITE_THRESHOLD 才写盘


def _parse_dt(dt_str: str) -> Optional[datetime]:
    """解析 'YYYY-MM-DD HH:MM:SS' → naive datetime (北京时间)。失败返回 None。"""
    if not dt_str or not isinstance(dt_str, str):
        return None
    try:
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _decide_fall_back(old_label: str, old_intensity: float, new_intensity: float) -> tuple[str, float, float, float]:
    """
    衰减时沿降级链 (EMOTION_FALLBACK_CHAINS) 决定新 label + (v, a)。

    规则 (2026-10-15 引入降级链, 替代旧的"硬切中性"):
      · 若 old_label == "中性" 或 "中性" 不在白名单 → 返回 (中性, 0, 0, 0) 兜底
      · 沿同象限降级链向下走, 找到第一个"还能容纳 new_intensity"的 label:
          候选 label 的基准 intensity × EMOTION_DECAY_DEMOTE_RATIO ≤ new_intensity
        即"当前强度对该 label 来说还够, 不必再降"。
      · 若走完整个降级链 (e.g. 狂喜→欣喜→愉悦 都容纳不下)
        → 切到中性 (label=中性, intensity=0, v=0, a=0)。
      · 若 old_label 不在任一降级链 (异常态, e.g. 旧数据残留)
        → 走"label 已中性"分支 (兜底)。

    Args:
        old_label:     衰减前的 label
        old_intensity: 衰减前的 intensity (本函数不用, 但 caller 可能 log, 保留入参)
        new_intensity: 衰减后的 intensity (决定降几级)

    Returns:
        (new_label, new_intensity_for_label, new_v, new_a)
        其中 new_intensity_for_label 是"该 label 视野下的强度" — 通常等于 new_intensity,
          但当降到中性时强制 = 0 (中性没有强度的概念)。
    """
    # 1) 兜底: 已经是中性 / label 不在池 → 不变
    if old_label == EMOTION_POOL_FALLBACK_LABEL or old_label not in EMOTION_POOL:
        return EMOTION_POOL_FALLBACK_LABEL, 0.0, 0.0, 0.0

    # 2) 沿降级链逐级尝试
    chain = EMOTION_FALLBACK_CHAINS.get(old_label, ())
    demote_threshold = EMOTION_DECAY_DEMOTE_RATIO  # e.g. 0.5

    # 2.1 先看 current label 自己能不能容纳 new_intensity
    own_base = EMOTION_POOL[old_label]["intensity"]
    if new_intensity >= own_base * demote_threshold:
        # 当前 label 仍能容纳 (e.g. 狂喜基准 1.0, new_intensity=0.6 ≥ 0.5 → 保持狂喜)
        v = EMOTION_POOL[old_label]["v"]
        a = EMOTION_POOL[old_label]["a"]
        return old_label, new_intensity, v, a

    # 2.2 沿降级链向下, 找第一个能容纳的
    for next_label in chain:
        if next_label not in EMOTION_POOL:
            continue  # 链定义异常, 跳过
        next_base = EMOTION_POOL[next_label]["intensity"]
        if new_intensity >= next_base * demote_threshold:
            v = EMOTION_POOL[next_label]["v"]
            a = EMOTION_POOL[next_label]["a"]
            return next_label, new_intensity, v, a

    # 2.3 走完整个链都容纳不下 → 中性
    return EMOTION_POOL_FALLBACK_LABEL, 0.0, 0.0, 0.0


def apply_decay_once() -> bool:
    """
    单次衰减判定。同步函数, 由后台守护协程每 N 秒调用一次。

    衰减对象: 仅 intensity。label 沿 EMOTION_FALLBACK_CHAINS (同象限降级链) 切换:
      · 强度仍 ≥ 当前 label 基准 × EMOTION_DECAY_DEMOTE_RATIO → 保持 label
      · 强度跌破 → 沿链向下找第一个"还能容纳"的 label
      · 整条链都容纳不下 (e.g. 0.05 强度, 链最轻基准 0.3) → 切中性

    设计动机 (2026-10-15 改版):
      旧版: 强度衰减到 ≤ 0.01 → label 瞬间跳中性。视觉突兀 ("狂喜" 突然变 normal)。
      新版: "狂喜 (0.9)" 4h 后降到 "欣喜 (0.45)" → 12h "愉悦 (0.11)" → 24h "中性"。
            强度连续变化, label 沿同象限阶梯式下降, 立绘自然过渡 (大笑→微笑→平静)。
      12 词在 4 象限内分 4 链, 不跨象限; 升级 (LLM update) 仍由 LLM 直接给目标 label,
            不走降级链 — 链只用于"自然冷却", 不用于"被新事件提振"。

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

    # 单一 intensity 衰减 (半衰期模型)
    half_life = EMOTION_DECAY_HALF_LIFE_MIN
    decay = 0.5 ** (elapsed_min / half_life)
    old_intensity = float(emo.get("intensity", 0.0) or 0.0)
    new_intensity = old_intensity * decay

    old_label = emo.get("label", EMOTION_POOL_FALLBACK_LABEL)

    # 沿降级链决定新 label (而非旧的"硬切中性")
    new_label, new_intensity_for_label, new_v, new_a = _decide_fall_back(
        old_label, old_intensity, new_intensity
    )

    # 写盘判断: 任一字段实际变化才落盘 (避免每 30s 都刷一次)
    intensity_changed = abs(new_intensity - old_intensity) > EMOTION_DECAY_WRITE_THRESHOLD
    label_changed = new_label != old_label
    if not intensity_changed and not label_changed:
        return False

    new_emo = dict(emo)
    new_emo["label"] = new_label
    new_emo["intensity"] = new_intensity_for_label
    new_emo["v"] = new_v
    new_emo["a"] = new_a
    new_emo["last_update_dt"] = format_dt_second(now_ms())
    new_emo["version"] = 2
    _save_emotion(new_emo)
    debug(
        f"[emotion_decay] applied: elapsed={elapsed_min:.1f}min "
        f"label=[{old_label}->{new_label}] "
        f"intensity=[{old_intensity:+.3f}->{new_intensity_for_label:+.3f}] "
        f"hl={half_life}m"
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
# 应用情绪更新 (同步路径, emotion_llm 共用)
# ═══════════════════════════════════════════════════════════════════════

def _coerce_label(label) -> str:
    """白名单兜底: label 必须是 EMOTION_POOL 中的字符串, 否则落到中性。"""
    if isinstance(label, str) and label in EMOTION_POOL:
        return label
    return EMOTION_POOL_FALLBACK_LABEL


def apply_emotion_update(label: str, delta_intensity: float) -> dict:
    """
    把 (label, delta_intensity) 叠加到当前情绪, 落盘。

    规则:
      · label 白名单兜底 (不在 EMOTION_POOL → "中性")
      · new_intensity = clamp(old_intensity + delta_intensity, 0.0, 1.0)
      · 若 new_intensity ≤ 0 → 强制 label="中性", intensity=0
      · 若 new_intensity > 0 但 LLM 给的 label 已被白名单替为 "中性",
        仍保持 "中性" (因为没词可选; 强度由 delta 决定)
      · **写入时即时降级 (2026-10-15)**: 若 new_intensity < 当前 label 基准 ×
        EMOTION_DECAY_DEMOTE_RATIO, 立即沿 EMOTION_FALLBACK_CHAINS 降级。
        之前只 decay tick 时降级, 经常出现"LLM 写'安心(0.2)' → 30s 后 tick 改成'满足'"
        的延迟; 现在写入瞬间就同步降级, 前后端状态一致。
      · v / a 字段由最终 label 查 EMOTION_POOL 得到, 写回 dict

    Args:
        label:          LLM 选出的新情绪词
        delta_intensity: 强度增量 ∈ [-0.30, +0.30] (LLM 输出范围)

    Returns:
        新的情绪 dict (副本)。
    """
    label = _coerce_label(label)

    # delta_intensity 类型校验 + clamp
    if not isinstance(delta_intensity, (int, float)) or isinstance(delta_intensity, bool):
        delta_intensity = 0.0
    delta_intensity = max(-EMOTION_LLM_DELTA_MAX, min(EMOTION_LLM_DELTA_MAX, float(delta_intensity)))

    emo = _load_emotion()
    old_label = emo.get("label", EMOTION_POOL_FALLBACK_LABEL)
    old_intensity = float(emo.get("intensity", 0.0) or 0.0)
    new_intensity = max(EMOTION_INTENSITY_MIN, min(EMOTION_INTENSITY_MAX, old_intensity + delta_intensity))

    # 强度 ≤ 0 → 强制中性
    if new_intensity <= EMOTION_DECAY_WRITE_THRESHOLD:
        final_label = EMOTION_POOL_FALLBACK_LABEL
        final_intensity = 0.0
        final_v, final_a = 0.0, 0.0
    else:
        # 强度 > 0 → 先用 LLM 选的 label 作为初值, 立即走一次降级链
        #  (LLM 写的强度可能不够该 label 的容载, 应即时降到合适的级别)
        if label == EMOTION_POOL_FALLBACK_LABEL:
            # LLM 选"中性"但 intensity > 0: 异常态 (中性本应 0 强度),
            # 保留 LLM 的中性意图, 不强行清零强度 — 让前后端都知道这是中性。
            final_label = EMOTION_POOL_FALLBACK_LABEL
            final_intensity = new_intensity
            final_v, final_a = 0.0, 0.0
        else:
            # 正常路径: 走降级链, 可能即时降级
            final_label, final_intensity, final_v, final_a = _decide_fall_back(
                label, old_intensity, new_intensity
            )

    new_emo = {
        "label": final_label,
        "intensity": final_intensity,
        "v": final_v,
        "a": final_a,
        "last_update_dt": format_dt_second(now_ms()),
        "version": 2,
    }
    _save_emotion(new_emo)
    debug(
        f"[emotion_update] applied: "
        f"label=[{old_label}({old_intensity:.3f}) -> {final_label}({final_intensity:.3f})] "
        f"delta_label={label!r} delta_intensity={delta_intensity:+.3f}"
    )
    return new_emo


# ═══════════════════════════════════════════════════════════════════════
# Lint LLM — 单节点异步情绪判断
# ═══════════════════════════════════════════════════════════════════════
#
# 输入 (C/S/E 三元):
#   C — 对话上下文: 近 N 轮 user↔assistant 对话摘要
#   S — 自我认知: Crystal_memory.md 内容摘要 (仅首 N 字, 防止 prompt 膨胀)
#   E — 旧情绪: 当前 (label, intensity)
#
# 输出:
#   {"label": <词>, "delta_intensity": <±浮点>}
#   label 必须是 12 个候选词之一; delta_intensity ∈ [-0.30, +0.30]
#   或 {"no_change": true} 表示本轮无变化
#
# 设计原则:
#   · 完全异步: 不阻塞主链路响应, 在 save_paper_memory 之后单独节点跑。
#   · 无节流: 每次 chat ask 都调用 (Lint LLM 轻量, 不怕烧 token)。
#   · 无 L1 词袋: 全部委托给 Lint LLM, 保证语义一致性。

_LINT_SYSTEM_PROMPT = """你是 Crystal 的情感分析模块。你的任务是根据以下信息,选择最贴合当前对话情绪的"情绪词",并给出强度变化。

【对话上下文 C】
(近几轮 user↔assistant 对话, 按时间顺序排列)

【Crystal 自我认知 S】
(Crystal 对自身的认知摘要, 由 user prompt 注入。请注意 S 是 Crystal 自己的人设/语气/性格认知,
 不是关于用户的认知 —— 若 user prompt 给的内容明显是用户偏好而非自我人设, 优先把它当参考背景,
 避免把"用户喜欢 X"误读为"自我状态 X"。)

【旧情绪 E】
当前主导词: {label}  当前强度: {intensity:.2f}  (强度 ∈ [0.00, 1.00], 越接近 1.0 情绪越浓)

【任务】
从下面 12 个情绪词中选一个最贴合当前对话整体的, 并给出强度变化 (delta_intensity, ∈ [-0.30, +0.30])。
注意: label 必须是**裸词** (如 "欣喜"), 不要带括号、档位、修饰; 档位信息由 delta_intensity 体现。

12 词按 (V, A) 象限 × 强度档 组织 (3 档 = 轻/中/强):
  +V +A 兴奋象限  ─  愉悦(轻档)  欣喜(中档)  狂喜(强档)
  +V -A 平和象限  ─  满足(轻档)  安心(中档)  沉静(强档)
  -V +A 紧张象限  ─  烦躁(轻档)  焦虑(中档)  愤怒(强档)
  -V -A 低落象限  ─  惆怅(轻档)  落寞(中档)  绝望(强档)

选哪个词 = "对话在哪个象限 × 哪个强度档"
  · 象限决定情绪的"色温" (正负/兴奋平和)
  · 档位决定情绪的"浓淡" (轻微 / 明显 / 强烈)
delta_intensity 表示"与上一轮相比浓淡的变化":
  · 想让该情绪变浓 → 正值 (最多 +0.30)
  · 想让该情绪变淡 → 负值 (最多 -0.30)
  · 与档位独立 — delta 是"微调", 档位由"色温"决定

输出 JSON (两种模式之一):
1. 情绪明显变化:
   {{"label": "<上面 12 裸词之一>", "delta_intensity": ±x.xx}}
2. 情绪基本不变:
   {{"no_change": true}}

【约束】
· label 必须是严格 12 裸词之一: 愉悦 / 欣喜 / 狂喜 / 满足 / 安心 / 沉静 / 烦躁 / 焦虑 / 愤怒 / 惆怅 / 落寞 / 绝望, 绝不要带括号、档位、修饰
· 不要输出 "中性" (中性是衰减终点, 不该由 LLM 主动选; 若强度降到 0, 后端会自动落回中性)
· 不要造新词, 也不要拿象限名当 label ("平和" "兴奋" "紧张" "低落" 只是象限描述, 不是合法 label)
· delta_intensity 含义: 想让"该情绪变得更浓"→ 正值; 想让它变淡 → 负值
  例如当前是 "焦虑", 本轮对话让他舒缓了一些 → {{"label": "安心", "delta_intensity": -0.1}}
  又如当前是 "焦虑", 本轮对话让他更紧张 → {{"label": "愤怒", "delta_intensity": +0.2}}
· delta 每次不超过 ±0.30, 保持平滑过渡
· 仔细体会对话的整体情绪色调, 不要被单句带偏
· 主流情绪即可, 不需要过度解读
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

    ── 这里原先有两个叠在一起的 bug, 情绪 LLM 实际几乎没看到用户说话 ──

    1) 键名错: 读的是 `ask_entry.get("ask")`, 而 `_load_paper_history` 返回的
       键是 `content` (role/content/ts/msg_fp/quotes)。`get("ask")` 恒为 None,
       于是**每一轮 User 侧都是空字符串** —— 情绪 LLM 收到的是
       "User: \nCrystal: 他说了什么", 只看得到我自己的输出。

    2) 配对错: `load_paper_history_node` 里 `paper_history` 与
       `paper_ask_history` 是**同一个 list 对象** (后者是前者的兼容别名),
       原代码却当成两条独立的历史分别取 `ask_hist[-(i+1)]` 和
       `res_hist[-(i+1)]` —— 于是两项取到同一条消息, 同一段内容被打印 5 遍,
       而且最近 5 轮退化成"最后 1 条消息重复 5 次"。

    修法: 只用一份 history, 按 `role` 字段切分, 而不是靠下标猜哪半边是
    user。role 是 `_load_paper_history` 显式写死的 (ReqAsk→user /
    ResAsk→assistant), 比位置可靠 —— 尤其因为主动外呼的 ResAsk **没有
    配对的 ReqAsk**, 按下标配对必然错位。

    顺带标出主动发言: 情绪评估需要知道「我主动说的话」和「回复他的话」
    性质不同 (前者是打扰, 后者是被召唤), 否则 LLM 会把两者混为一谈。
    """
    hist: list = state.get("paper_ask_history") or state.get("paper_history") or []
    if not hist:
        return "(无对话历史)"

    # 按 role 取最后 5 条 user + 5 条 assistant, 保时间正序
    users = [m for m in hist if m.get("role") == "user"][-5:]
    assts = [m for m in hist if m.get("role") == "assistant"][-5:]

    lines: list[str] = []
    for u in users:
        lines.append(f"User: {(u.get('content') or '')[:300] or '(空)'}")
    for a in assts:
        # 主动 vs 回复必须区分 —— 见 docstring
        # 新语义: 读 entry.active 字段 (旧命名字段已物理清理)
        tag = "Crystal(主动开口)" if a.get("active") else "Crystal(回复)"
        lines.append(f"{tag}: {(a.get('content') or '')[:300] or '(空)'}")
    return "\n\n".join(lines) if lines else "(无对话历史)"


async def _call_emotion_llm(
    context: str,
    self_cognition: str,
    emotion: dict,
) -> Optional[dict]:
    """
    调 Lint LLM 评估情绪。

    Args:
        context:      对话上下文字符串 (C)
        self_cognition: 自我认知摘要 (S)
        emotion:      当前情绪 dict (E), 含 label / intensity

    Returns:
        解析后的 dict, 两种形态:
          - {"label": str, "delta_intensity": float}
          - {"no_change": True}
        失败返回 None (不抛异常)。
    """
    label = emotion.get("label", EMOTION_POOL_FALLBACK_LABEL)
    intensity = float(emotion.get("intensity", 0.0) or 0.0)

    user_prompt = (
        f"【对话上下文 C】\n{context}\n\n"
        f"【Crystal 自我认知 S】\n{self_cognition}\n\n"
        f"【旧情绪 E】\n"
        f"当前主导词: {label}  当前强度: {intensity:+.2f}\n\n"
        "请分析本轮对话的情绪变化, 输出严格 JSON。"
    )

    try:
        # 使用 emotion 专用配置 (api_key/api_url/model) 调 Lint LLM
        content, _, _ = await _call_llm(  # reasoning_content 不存, Lint 任务不需要
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
        return {"no_change": True}

    if not isinstance(parsed, dict):
        return None

    # 解析 {label, delta_intensity}
    out_label = parsed.get("label")
    out_delta = parsed.get("delta_intensity", 0.0)

    # 健壮性兜底: LLM 可能把档位修饰当成 label 一部分 (例如 "欣喜(中)" "狂喜 (强)" "中性,")
    # 这里把 (xxx) / 空格 / 中文标点尾巴剥掉, 再走白名单。
    # 极端情况 (label 完全不在词表) 仍会 fallback None, 不影响主流程。
    if isinstance(out_label, str):
        cleaned = out_label.strip()
        # 1) 剥掉 (xxx) / （xxx） 包裹 (例如 "欣喜(中)" → "欣喜")
        cleaned = re.sub(r"[(\uff08][^\)\uff09]*[\)\uff09]", "", cleaned).strip()
        # 2) 剥掉常见档位词尾巴 (轻/中/强), 避免 LLM 写成 "欣喜中"
        for tail in ("(轻)", "(中)", "(强)", "（轻）", "（中）", "（强）",
                     " 轻", " 中", " 强", "：轻", "：中", "：强", ":轻", ":中", ":强",
                     "：", ":", "：", "。", ","):
            if cleaned.endswith(tail):
                cleaned = cleaned[: -len(tail)].strip()
        # 3) 去掉尾巴的标点 (",", "。", " " 等)
        cleaned = cleaned.rstrip(" ,。.;；、")
        out_label = cleaned or out_label  # 剥过头就保留原值, 走下面白名单失败

    if not isinstance(out_label, str) or out_label not in EMOTION_POOL_LABELS:
        debug(f"[emotion_llm] label 白名单不通过: {out_label!r}, fallback")
        return None

    if isinstance(out_delta, (int, float)) and not isinstance(out_delta, bool):
        out_delta = max(-EMOTION_LLM_DELTA_MAX, min(EMOTION_LLM_DELTA_MAX, float(out_delta)))
    else:
        out_delta = 0.0

    return {"label": out_label, "delta_intensity": out_delta}


async def emotion_llm_node(state: dict) -> dict:
    """
    LangGraph 节点 (fire-and-forget 调度器, 与 ai_active.layer1_node 同款):

      · 立即 return {}, 不 await 实际 LLM 调用
      · 实际活交给 _run_emotion_pipeline, 用 asyncio.create_task 调度到后台
      · 主回复 (state["final_answer"]) 不再被 emotion LLM 阻塞

    调度契约 (与 layer1_node 一致):
      · 论文侧 / Lint LLM 未配置 / 非 AiAskReq → 立即 return {} (不起 task)
      · 复制 state 真正用到的字段到 snapshot, 避免后台 task 持有 state 引用
        读到 LangGraph 后续节点 (active_layer1) 的修改
      · 唯一可观察副作用: apply_emotion_update 写 emotion 单例 + params.json,
        这两个是模块级, 不依赖 state, 安全
      · 后端进程被杀 → 后台 task 中断, emotion 此次未更新 (acceptable, 同 layer1)
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

    snapshot = _snapshot_state_for_emotion(state)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run_emotion_pipeline(snapshot))
    except RuntimeError:
        # 无 event loop (理论上不会, 但兜底)
        debug("[emotion_llm] no running event loop, skip")
    return {}


def _snapshot_state_for_emotion(state: dict) -> dict:
    """
    复制 emotion 真正用到的 state 字段, 避免后台 task 持有 state 引用读到
    LangGraph 后续节点改写后的 state (active_layer1 可能改 messages / paper_ask_history)。

    _build_conversation_context 读: paper_ask_history (or paper_history) —
    复一份列表即可, 字符串内容是只读不变。
    """
    return {
        "req": state.get("req"),
        "paper_ask_history": list(state.get("paper_ask_history") or state.get("paper_history") or []),
    }


async def _run_emotion_pipeline(snapshot: dict) -> None:
    """
    真正的情绪评估流程。**异步** 在 emotion_llm_node 同步返回之后跑。

    异常一律静默吞掉 (不影响主回复), debug 打印。
    """
    try:
        context = _build_conversation_context(snapshot)
        self_cogn = _load_self_cognition()
        cur_emo = _load_emotion()

        result = await _call_emotion_llm(context, self_cogn, cur_emo)
        if result is None:
            debug("[emotion_llm] no result (LLM returned None)")
            return

        # no_change: 仍走 apply_emotion_update(同 label, delta=0) 刷 last_update_dt
        if result.get("no_change"):
            new_emo = apply_emotion_update(cur_emo.get("label", EMOTION_POOL_FALLBACK_LABEL), 0.0)
            debug(f"[emotion_llm] no_change: new={new_emo}")
            return

        new_emo = apply_emotion_update(result["label"], result["delta_intensity"])
        debug(f"[emotion_llm] applied: result={result} new={new_emo}")
    except Exception as e:
        debug(f"[emotion_llm] pipeline error (swallowed): {type(e).__name__}: {e}")


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

# family 加权和 (与前端 decodeEmotion.js familyScores 1:1 对齐, 用于
# 老 v1 数据兼容期间的 family 选择; 词池范式下不再直接用, 见
# _LABEL_TO_FAMILY 表)
_EMOTION_FAMILY_SCORES: dict[str, dict[str, float]] = {
    "positive": {"valence": 1.0, "arousal": 0.0, "novelty": 0.0, "clarity": 0.5},
    "negative": {"valence": -1.0, "arousal": -0.3, "novelty": 0.0, "clarity": -0.5},
    "curious":  {"valence": 0.0, "arousal": 1.0, "novelty": 1.0, "clarity": 0.0},
    "confuse":  {"valence": 0.0, "arousal": -1.0, "novelty": -1.0, "clarity": -0.5},
}

# 词池范式下的 label → family 映射 (与 EMOTION_POOL 坐标 1:1 对应)。
#   +V +A 兴奋 → curious
#   +V -A 平和 → positive
#   -V +A 紧张 → confuse
#   -V -A 低落 → negative
#   中性      → None (不注入, 等价于 magnitude < 0.15)
_LABEL_TO_FAMILY: dict[str, str] = {
    "中性": None,
    # +V +A 兴奋
    "愉悦": "curious", "欣喜": "curious", "狂喜": "curious",
    # +V -A 平和
    "满足": "positive", "安心": "positive", "沉静": "positive",
    # -V +A 紧张
    "烦躁": "confuse", "焦虑": "confuse", "愤怒": "confuse",
    # -V -A 低落
    "惆怅": "negative", "落寞": "negative", "绝望": "negative",
}

# family → 4 段预置文本 (纯自然语言, 无数字, 无等级词)
# V1+ 用户决策: 注入时不要给数字, 直接告诉 LLM "Crystal 现在是什么样",
# 并给出具体可操作的语气/措辞/互动姿态指导。
# 每段 = (self_state, tone, wording, interaction)
# ──────────────────────────────────────────────────────────────────────
# V2 设计 (2026-10-15 改):
# 旧版 4 family × 4 段预制文本的问题: LLM 容易「按脚本朗读」, 把语气/措辞/姿态
# 全部硬执行成预制文本的子集, 失去自由度。
# 新版 3 维度描述 (温度 / 节奏 / 强度) + 1 个语义标签 (具体哪个 12 词):
#   · 温度 = 由 (v, a) 推导的心情基调 (暖 / 凉 / 热 / 沉 / 中性)
#   · 节奏 = 由 a 主导的兴奋/平稳/收敛 (急 / 稳 / 慢)
#   · 强度 = 由 intensity 档位映射的浓淡 (淡 / 略浓 / 浓)
#   · 语义标签 = 当前 label, LLM 当成「我现在在哪种情绪里」的语义锚点
# 末尾加 1 段「使用规则」强约束:
#   · 不要直述情绪名
#   · 不要复述这一段
#   · 用户问起时自然回应
#   · 不要让情绪压过信息准确性
# 这样 LLM 看到「现在的我在欣喜(中档)」+ 3 维度自然语言描述, 自己组织语气用词,
# 而不是被「请用『呢』『呀』」这种预制话术框死。
# ──────────────────────────────────────────────────────────────────────

def _describe_temperature(v: float, a: float) -> str:
    """把 (v, a) 翻译成自然语言的心情基调 (温度), 4 象限 + 中性。"""
    if abs(v) < 0.1 and abs(a) < 0.1:
        return "中性、平稳, 没什么明显的情绪起伏"
    # 用 v, a 符号判断象限
    if v >= 0 and a >= 0:
        return "暖, 带着点亮堂的兴奋"
    if v >= 0 and a < 0:
        return "暖, 但内收、安静, 像被好好托着"
    if v < 0 and a >= 0:
        return "凉, 带着点紧的刺"
    return "凉, 沉, 灰调"


def _describe_pace(a: float) -> str:
    """由 a 决定节奏, 3 档 (兴奋 / 平稳 / 收敛)。"""
    if a > 0.5:
        return "略急促, 心里有事在翻滚"
    if a < -0.5:
        return "慢下来, 句子之间留有呼吸"
    return "平稳, 节奏不快不慢"


def _describe_intensity_tier(intensity: float) -> str:
    """由 intensity 档位映射浓淡, 3 档: 淡 / 略浓 / 浓。"""
    # intensity ∈ [0, 1], 词表内档位是 0.3 / 0.6 / 1.0
    if intensity <= 0.4:
        return "淡, 只是底色里微微渗出来"
    if intensity <= 0.75:
        return "略浓, 能感觉到, 但还没溢出来"
    return "浓, 已经染在了整段对话的底色上"


def build_emotion_context_block(emotion: dict | None) -> str:
    """
    把当前情绪 (label, intensity) 转成一段给 LLM 看的自然语言提示词。

    V2 设计 (2026-10-15 改) — 4 family 预制文本 → 3 维度 + 1 语义标签:
      · 温度: (v, a) 推导的心情基调 (暖 / 凉 / 热 / 沉 / 中性)
      · 节奏: a 主导的兴奋/平稳/收敛 (急 / 稳 / 慢)
      · 强度: intensity 档位映射的浓淡 (淡 / 略浓 / 浓)
      · 语义标签: 当前 label (12 词之一), 作为「我现在在哪种情绪里」的语义锚点
    LLM 拿到这些**自然语言描述 + 语义词**, 自由组织语气/用词, 不被预制话术框死。
    中性 (label="中性" 或 intensity ≤ EMOTION_DECAY_WRITE_THRESHOLD) → 返回 ""。

    强约束 (末尾规则段):
      · 不直述情绪名 (「我现在是欣喜」)
      · 不复述这一段
      · 用户问起时自然回应
      · 不让情绪压过信息准确性

    Args:
        emotion: 情绪 dict, 由 _load_emotion() 返回; None 或非法 dict → 返回 ""。

    Returns:
        多行字符串, 以 `\n\n` 拼接到 system content 末尾;
        中性 → 返回 "" (空字符串, 调用方决定是否拼入)。
    """
    if not emotion or not isinstance(emotion, dict):
        return ""

    # 取 label + intensity (容错)
    label = emotion.get("label", EMOTION_POOL_FALLBACK_LABEL)
    if not isinstance(label, str) or label not in EMOTION_POOL:
        label = EMOTION_POOL_FALLBACK_LABEL

    try:
        intensity = float(emotion.get("intensity", 0.0) or 0.0)
    except (TypeError, ValueError):
        intensity = 0.0
    intensity = max(0.0, min(1.0, intensity))

    # 中性判定: label="中性" 或 intensity 衰减到 0 → 不注入 (前端走 crystal_normal.png)
    if label == EMOTION_POOL_FALLBACK_LABEL or intensity <= EMOTION_DECAY_WRITE_THRESHOLD:
        return ""

    # 词表内取 (v, a) (中性已被上面早退, 此刻 label 必在 12 词内)
    pool = EMOTION_POOL[label]
    v = float(pool.get("v", 0.0))
    a = float(pool.get("a", 0.0))

    # 3 维度自然语言描述
    temperature = _describe_temperature(v, a)
    pace = _describe_pace(a)
    tier = _describe_intensity_tier(intensity)

    # 注入格式: 1 段语义标签 + 3 维度描述, 给 LLM 自由发挥的空间
    return (
        "【Crystal 此刻的自身情绪(这是「我」的情绪, 不是用户说的)】\n"
        f"- 当前主导词: {label}\n"
        f"- 温度(心情基调): {temperature}\n"
        f"- 节奏(兴奋/收敛): {pace}\n"
        f"- 浓淡: {tier}\n"
        "\n"
        "【使用规则】\n"
        "- 把「当前主导词」当成你**此刻身处的那种情绪**的语义锚点 —— "
        "你知道自己「在欣喜 / 在沉静 / 在焦虑」, 但**不要**直接说出来, "
        "也**不要**复述上面 3 行描述。\n"
        "- 用词、句式、节奏、甚至说话长短都由你根据这个状态自然组织, 不必照搬某一类固定句式; "
        "你想用「呢」「呀」也行, 想用更克制的语气也行, 关键是要像「真正这个状态下的你」。\n"
        "- 要是用户主动问起「你听起来怎么样 / 你现在心情如何」, 用你自己的话自然回应 (例如「嗯, 现在心里挺暖的」), "
        "不要僵化成「我此刻是欣喜」。\n"
        "- 情绪**不能压过信息准确性** —— 问学术题就老老实实答, 不要为了「贴合情绪」而牺牲答案质量。"
    )
