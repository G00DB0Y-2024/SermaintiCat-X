"""
ai 子包私有工具层。

当前包含:
- 时间系统: now_ms / get_current_time_context / format_dt_minute / format_dt_second
- 设备感知: get_device_context
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo


# ═══════════════════════════════════════════════════════════════════════
# 原子文本写 (atomic text write) — 跨平台通用
# ═══════════════════════════════════════════════════════════════════════
# 写文本文件, 返回写入字节数。出错抛 OSError。
#
# 注: ai_io 内部 EXPLORE_FILE 等场景要求"不依赖 ai_agent 的 _write_text_file_atomic"
#     (否则循环 import), 因此这里放一份独立副本。语义一致: f.tell() 在 text
#        mode 下返回的是透明 cookie, 文本长度一致时可用来早期失败检测。
def _write_text_file_atomic(path: str, content: str) -> int:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
        return f.tell()


# ═══════════════════════════════════════════════════════════════════════
# 单源时间系统 (Single-source time)
# ═══════════════════════════════════════════════════════════════════════
# 设计: 整个进程共用一个时间源 `now_ms()` —— 返回 UTC Unix 毫秒时间戳。
#
# 为什么不用 `datetime.now().timestamp() * 1000` ?
#   - `datetime.now()` 默认是系统本地时区 (naive datetime)
#   - naive datetime 调 `.timestamp()` 时 Python 会假设它是系统本地时区再转 UTC
#   - 结果依赖系统时区, 在非 UTC 服务器上会与 `time.time()` 不一致
#
# 为什么不用 `time.time() * 1000` ?
#   - 表达力差, 不暴露"毫秒"语义
#   - 调用点分散容易写错 (有人会写 `int(time.time())` 丢精度)
#
# 唯一源: `now_ms()` 返回 UTC epoch ms, 任何时区/格式/星期换算都基于它派生。

def now_ms() -> int:
    """返回当前 UTC Unix 时间戳 (毫秒)。进程内唯一时间源。"""
    return int(time.time() * 1000)


# 北京时间固定时区常量 — 整个进程只用一个 tz, 避免散落 ZoneInfo("Asia/Shanghai")
_BEIJING_TZ = ZoneInfo("Asia/Shanghai")
_BEIJING_WEEKDAY_NAMES = ["一", "二", "三", "四", "五", "六", "日"]


def _to_beijing_dt(ts_ms: int) -> datetime:
    """把 epoch ms 转换为北京本地时间的 datetime 对象 (内部辅助)。"""
    return datetime.fromtimestamp(ts_ms / 1000, tz=_BEIJING_TZ)


def get_current_time_context() -> str:
    """
    返回供 LLM 使用的"当前时间感知"字符串 (北京时区)。
    调用 now_ms() 派生 — 与系统时区无关。
    """
    dt = _to_beijing_dt(now_ms())
    weekday = _BEIJING_WEEKDAY_NAMES[dt.weekday()]
    return (
        f"【当前时间感知】{dt.strftime('%Y年%m月%d日 %H:%M:%S')} 星期{weekday}（北京时间）\n"
    )


def format_dt_minute(ts_ms: int) -> str:
    """
    毫秒时间戳 → 可读字符串 "YYYY-MM-DD HH:MM" (北京时间, 分钟精度)。
    用于 Crystal_memory.md 时间戳 (LLM 写到笔记里)。
    """
    return _to_beijing_dt(ts_ms).strftime("%Y-%m-%d %H:%M")


def format_weekday(ts_ms: int) -> str:
    """
    毫秒时间戳 → 星期几, 返回 "周一" ... "周日" (北京时间)。

    供外呼结算记录使用 —— 让学习回路能归纳「他周末才回」这类规律。
    必须用真实 weekday, 不能截日期字符串的字符 (那是"日", 不是星期)。
    ts_ms <= 0 返回空串 (未发送 / 未知)。
    """
    if not ts_ms or ts_ms <= 0:
        return ""
    return f"周{_BEIJING_WEEKDAY_NAMES[_to_beijing_dt(ts_ms).weekday()]}"


def format_dt_second(ts_ms: int) -> str:
    """
    毫秒时间戳 → 可读字符串 "YYYY-MM-DD HH:MM:SS" (北京时间, 秒级精度)。
    用于 ai 消息的 dt 字段 (前端 / 内部时间戳统一来源) 和 update memory
    的 current_timestamp。
    """
    return _to_beijing_dt(ts_ms).strftime("%Y-%m-%d %H:%M:%S")


# ═══════════════════════════════════════════════════════════════════════
# 设备感知
# ═══════════════════════════════════════════════════════════════════════

_DEVICE_LABELS: dict[str, str] = {
    "desktop": "电脑（用户坐在电脑荧幕前和你聊天）",
    "mobile":  "手机（用户可能在外出或临时使用手机联络）",
    "tablet":  "平板（用户可能在外出或临时使用平板联络）",
}


def get_device_context(device: str | None) -> str:
    """
    返回供 LLM 使用的"设备感知"字符串。
    device: "desktop" | "mobile" | "tablet" 或 None（未知）。
    与 get_current_time_context 并列注入 system prompt。
    """
    if not device:
        return ""
    desc = _DEVICE_LABELS.get(device, device)
    return f"【当前设备感知】用户正在使用{desc}。\n"
