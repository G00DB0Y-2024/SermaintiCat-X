"""
AI 模块所有可调参数 + LLM 配置。

分为三块:
- 路径:  BASE_DIR / SAVE_DIR / MEMORY_DIR / CRYSTAL_MEMORY_FILE / *_TRACK_FILE
- 窗口:  ASK_LOCAL_LIMIT / LOAD_LOCAL_LIMIT / CHAT_LOCAL_LIMIT / MAX_ASK_TRACK / ...
- LLM:   DEEPSEEK_MARKER + ai_config 字典 (由 /ai/config 端点维护)

ai_config 字典 (api_key / api_url / model / vision_model / deepseek_thinking)
由 ai_routes 写入, _call_llm 读取 — 单一可变状态只有这一份。
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
ASK_TRACK_FILE:  Final[str] = os.path.join(MEMORY_DIR, "Crystal_track_ask.json")
LOAD_TRACK_FILE: Final[str] = os.path.join(MEMORY_DIR, "Crystal_track_load.json")


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
CHAT_LOCAL_LIMIT:  Final[int] = 10
CHAT_MEMORY_LIMIT: Final[int] = 5
MAX_ASK_TRACK:     Final[int] = 5
MAX_LOAD_TRACK:    Final[int] = 3
MEMORY_UPDATE_EVERY_N: Final[int] = 1  # = ASK_LOCAL_LIMIT // 2


# ═══════════════════════════════════════════════════════════════════════
# Chat 标识 (前端通过 pdf_fp=="crystal_chat" 调用 Chat 上下文)
# ═══════════════════════════════════════════════════════════════════════

CHAT_FP: Final[str] = "crystal_chat"


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
    """
    global _ai_config
    _ai_config.update({
        "api_key": config.get("api_key") or "",
        "api_url": config.get("api_url") or "",
        "model": config.get("model") or "",
        "vision_model": config.get("vision_model") or config.get("model") or "",
        "deepseek_thinking": bool(config.get("deepseek_thinking")),
    })


def get_current_config() -> dict:
    """返回当前 _ai_config 快照 (供 /ai/config GET 使用)。"""
    return dict(_ai_config)
