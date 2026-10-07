"""
AI 模块所有可调参数 + LLM 配置。

分为三块:
- 路径:  BASE_DIR / SAVE_DIR / MEMORY_DIR / CRYSTAL_MEMORY_FILE / EXPLORE_FILE /
- 窗口:  ASK_LOCAL_LIMIT / LOAD_LOCAL_LIMIT / CHAT_LOCAL_LIMIT / MAX_ASK_TRACK / ...
- LLM:   DEEPSEEK_MARKER + ai_config 字典 (由 /ai/config 端点维护)

ai_config 字典 (api_key / api_url / model / vision_model / deepseek_thinking)
由 ai_routes 写入, _call_llm 读取 — 单一可变状态只有这一份。

ACUS 四元素的物理落点 (隔离, 互不串档):
    U = Crystal_memory.md   LLM 读, 关于「他」的认知
    S = Crystal_self.md     LLM 读, Crystal 对自己的认知
    A = explore.md          LLM 读, 主动开口的自然语言判断 (软)
"""
from __future__ import annotations

import os
import sys
from typing import Final


# ═══════════════════════════════════════════════════════════════════════
# 路径
# ═══════════════════════════════════════════════════════════════════════

def _get_base_dir() -> str:
    """
    打包环境兼容: PyInstaller 临时目录 vs 开发目录。

    ai/* 子模块的 BASE_DIR 应当回到 pdf-backen 项目根目录,
    这样 save / static 等目录与 PdfBacken.py 一致。
    """
    if getattr(sys, "frozen", False):
        return sys._MEIPASS  # type: ignore[attr-defined]
    # 本文件位于 pdf-backen/ai/<name>.py, 向上一层就是 pdf-backen 项目根
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BASE_DIR: Final[str] = _get_base_dir()
SAVE_DIR: Final[str] = os.path.join(BASE_DIR, "save")
MEMORY_DIR: Final[str] = os.path.join(BASE_DIR, "ai", "memory")
CRYSTAL_MEMORY_FILE: Final[str] = os.path.join(MEMORY_DIR, "Crystal_memory.md")
CRYSTAL_SELF_FILE:   Final[str] = os.path.join(MEMORY_DIR, "Crystal_self.md")
# appoint.md → explore.md (ACUS 的 A 元素) — 主动外呼档案。
#
# 与 memory / self 的分工 (三者物理隔离, 互不串档):
#   - Crystal_memory.md (U): 关于「他」的一切认知 —— 身份、偏好、忌讳、项目、环境。
#   - Crystal_self.md   (S): Crystal 对自己的认知 —— 性格、语气、愿望。
#   - explore.md        (A): 关于「主动开口」的档案 —— 三段:
#       ## 主动判断 该在什么状态下开口/追问 (intake 追加 + 学习回路归纳)
#       ## 节奏规律 LLM 从历史样本归纳的推断 (软, 可错可推翻)
#       ## 外呼记录 每条外呼/沉默的归档 (自然语言, 不是 JSON)
#
# 为什么叫 explore 而不是 appoint:
#   旧版有一段「## 约定」承载硬承诺(每天早上问好), 靠正则猜时间窗做硬门控。
#   那套已退役。explore.md 只留「自然语言判断」, 归 LLM 读。
#   判据是**谁读数据**: LLM 读 md, 代码读 json。
EXPLORE_FILE:       Final[str] = os.path.join(MEMORY_DIR, "explore.md")
ASK_TRACK_FILE:  Final[str] = os.path.join(MEMORY_DIR, "Crystal_track_ask.json")
LOAD_TRACK_FILE: Final[str] = os.path.join(MEMORY_DIR, "Crystal_track_load.json")
# 运行期可变参数的持久化文件 (与上面的 track 放在一起, 都是"跨进程要留住"的状态)。
# 目前存 _ask_count_by_fp —— memory update 的节流计数。
# 放 ai/memory/ 而不是 save/: 它跟 memory 生命周期待在一起, 便于打包时一起收集。
PARAMS_FILE: Final[str] = os.path.join(MEMORY_DIR, "params.json")


# ═══════════════════════════════════════════════════════════════════════
# 上下文窗口参数
# ═══════════════════════════════════════════════════════════════════════
#
# 论文侧 (按 fp 隔离):
#   ASK_LOCAL_LIMIT  : 本论文 ask 最近 10 对 (user+assistant)
#   LOAD_LOCAL_LIMIT : 本论文 load 最近 5 对
#
# Chat 侧:
#   CHAT_LOCAL_LIMIT : ChatView 近 10 对
#   CHAT_MEMORY_LIMIT: 喂给 memory update 的 Chat 历史窗口 (5 对)
#
# Track (全局跨论文):
#   MAX_ASK_TRACK / MAX_LOAD_TRACK: track 文件累计上限, FIFO 裁剪
#
# 节流:
#   MEMORY_UPDATE_EVERY_N: 每 N 次论文 ask 触发一次 memory LLM

ASK_LOCAL_LIMIT:   Final[int] = 10
LOAD_LOCAL_LIMIT:  Final[int] = 5
CHAT_LOCAL_LIMIT:  Final[int] = 20
CHAT_MEMORY_LIMIT: Final[int] = 10
MAX_ASK_TRACK:     Final[int] = 5
MAX_LOAD_TRACK:    Final[int] = 3
MEMORY_UPDATE_EVERY_N: Final[int] = 3  # = ASK_LOCAL_LIMIT // 2


# ═══════════════════════════════════════════════════════════════════════
# Chat 标识 (前端通过 pdf_fp=="crystal_chat" 调用 Chat 上下文)
# ═══════════════════════════════════════════════════════════════════════

CHAT_FP: Final[str] = "crystal_chat"


# ═══════════════════════════════════════════════════════════════════════
# 情绪状态机参数
# ═══════════════════════════════════════════════════════════════════════
#
# 情绪衰减:
#   EMOTION_DECAY_HALF_LIFE_MIN : 基础半衰期 (分钟)。
#       每轴的实际半衰期 = HALF_LIFE_MIN × EMOTION_DECAY_FACTOR[axis]
#       衰减公式: new_v = old_v * 0.5 ^ (elapsed_min / axis_half_life)
#   EMOTION_DECAY_INTERVAL_SEC : 后台守护协程每 N 秒跑一次衰减判定。
#       跑的时候读 last_update_dt, 算 elapsed_min, 算衰减后的向量;
#       若任一轴 |new - old| > EMOTION_DECAY_WRITE_THRESHOLD 才落盘,
#       避免每 30s 都刷一次 git 噪音。
#   EMOTION_DECAY_FACTOR      : 各轴半衰期倍率, 体现"积累越快的轴衰减也越快"。
#       设计思路:
#         · valence 最易积累 (L1 关键词命中最多), 衰减也最快 (factor < 1)
#         · arousal 是瞬时唤醒度, 中等节奏 (factor ≈ 1)
#         · novelty 是"新印象", 比情绪持久得多 (factor > 1)
#         · clarity 是认知/偏好, 衰减最慢 (factor 最大)
#       改 factor 即可调粒度, 不需要动公式。
EMOTION_DECAY_HALF_LIFE_MIN:  Final[int]   = 240   # 4 小时基础半衰
EMOTION_DECAY_INTERVAL_SEC:   Final[int]   = 30    # 后台守护协程周期
EMOTION_DECAY_WRITE_THRESHOLD: Final[float] = 0.01  # 任一轴变化超过此值才写盘
EMOTION_DECAY_FACTOR: Final[dict[str, float]] = {
    "valence": 0.5,   # 等效半衰 2h:  情绪"转得快"
    "arousal": 1.0,   # 等效半衰 4h:  唤醒度基准节奏
    "novelty": 2.0,   # 等效半衰 8h:  "新印象"持续较久
    "clarity": 3.0,   # 等效半衰 12h: 认知/偏好最持久
}


# ═══════════════════════════════════════════════════════════════════════

# LLM 路由
# ═══════════════════════════════════════════════════════════════════════

# DeepSeek API 需要在 base url 后拼 /chat/completions
DEEPSEEK_MARKER: Final[str] = "deepseek.com"


# ═══════════════════════════════════════════════════════════════════════
# ai_config 模块状态 (由 /ai/config 端点写入)
# ═══════════════════════════════════════════════════════════════════════

_ai_config: dict = {
    "api_key": "",
    "api_url": "",
    "model": "",          # 普通文本模型 (ask / load)
    "vision_model": "",   # 视觉模型 (ask + image_base64)
    "deepseek_thinking": False,
}


def update_ai_config(config: dict) -> None:
    """
    由 /ai/config 端点调用, 更新模块级 _ai_config。
    视觉模型缺省回落到普通 model。

    同步把配置持久化到 ai/memory/params.json.llm_configs.main, 这样:
      · 后端重启后配置不丢
      · 多端 (电脑/手机) 任何一端更新了配置, 都能从磁盘读到最新值
    """
    global _ai_config
    _ai_config.update({
        "api_key": config.get("api_key") or "",
        "api_url": config.get("api_url") or "",
        "model": config.get("model") or "",
        "vision_model": config.get("vision_model") or config.get("model") or "",
        "deepseek_thinking": bool(config.get("deepseek_thinking")),
    })
    # 落盘 (静默失败, 内存已更新, 持久化只是 bonus)
    try:
        from .ai_io import _save_llm_configs
        _save_llm_configs({"main": dict(_ai_config)})
    except Exception:
        # 落盘失败不影响本次请求的处理 (模块级状态已经是最新的)
        pass


def get_current_config() -> dict:
    """返回当前 _ai_config 快照 (供 /ai/config GET 使用)。"""
    return dict(_ai_config)


def load_ai_config_from_disk() -> bool:
    """
    进程启动时从 params.json.llm_configs.main 把配置读回到 _ai_config 模块变量。
    由 PdfBacken.py 的 lifespan 调用。

    这样做的好处:
      · 后端重启 → 配置不丢 (原先只在模块 dict, 重启即丢)
      · 多端 (电脑 + 手机) 任何一端更新了 → 另一端下次访问后端时,
        看到的就是最新的 (前端 GET /ai/config 拉一次即可同步)

    Returns:
        bool — True 表示从磁盘读到非空配置, False 表示磁盘无配置或读取失败
               (False 时 _ai_config 保持默认空值, 由前端推送兜底)
    """
    try:
        from .ai_io import _load_llm_configs
        cfg = _load_llm_configs().get("main") or {}
        if not cfg or not cfg.get("api_key"):
            return False
        global _ai_config
        _ai_config.update({
            "api_key": cfg.get("api_key") or "",
            "api_url": cfg.get("api_url") or "",
            "model": cfg.get("model") or "",
            "vision_model": cfg.get("vision_model") or cfg.get("model") or "",
            "deepseek_thinking": bool(cfg.get("deepseek_thinking")),
        })
        return True
    except Exception:
        return False
