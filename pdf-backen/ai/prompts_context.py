"""
下层：基于历史/track/quote 的最终 prompt 拼装。

包含:
- Memory 更新 user prompt: MEMORY_UPDATE_USER_HEADER / buildMemoryUpdateUserPrompt
- Ask/Load 的 user content 构建: build_ask_user_content / build_load_user_content
- 上下文组装: compose_paper_ask_messages / compose_chat_messages / build_track_summary_block

buildAskMessages / buildLoadMessages 在原 prompts.py 中的版本已废弃,
由本文件中的 compose_*_messages 接管, 不再保留。
"""
from __future__ import annotations

from typing import Optional

from .ai_models import AiAskReq, AiLoadReq
from .ai_io import _get_track
from .ai_utils import format_dt_second, get_device_context
from .prompts_system import SYSTEM_ASK, SYSTEM_LOAD
from utils.log import debug


# ═══════════════════════════════════════════════════════════════════════
# Memory 更新 user prompt
# ═══════════════════════════════════════════════════════════════════════

def MEMORY_UPDATE_USER_HEADER() -> str:
    """
    用于 update_crystal_memory 的 user prompt 静态头部:
    一次性说清楚时间戳规范 + 输出指令 (避免与 system 重复)。
    时间戳值 / 当前笔记内容 / 本轮对话 / track 上下文等动态内容
    由 buildMemoryUpdateUserPrompt 在尾部拼接。
    """
    return (
        "请基于提供的「当前 Crystal_memory.md」「本次用户对话」以及作为补充的「当前论文窗口内的轨迹」,\n"
        "输出压缩并更新后的完整 Crystal_memory.md 文本。\n"
    )


def MEMORY_COMPRESS_USER_HEADER() -> str:
    """
    compress phase 1 的 user prompt 静态头部:
    一次性说清楚: 这是压缩任务, 输出压缩后的 Markdown, 不引入任何新对话内容。
    """
    return (
        "下面是旧的 Crystal_memory.md 全文, 以及当前时间。\n"
        "请按 system 中的「时间分层」规则对其压缩, 输出压缩后的完整 Markdown。\n"
        "不要引入任何新对话、新事件或新的更新时间戳 (那属于后续 update 阶段的工作)。\n"
    )


def buildMemoryCompressUserPrompt(
    current_memory_md: str,
    current_timestamp: str = "",
) -> str:
    """
    构造 compress phase 1 的 user prompt:
      - 静态头部: MEMORY_COMPRESS_USER_HEADER
      - 当前时间
      - 旧 md 全文
    """
    timestamp_section = ""
    if current_timestamp:
        timestamp_section = (
            f"\n【当前时间】{current_timestamp}（北京时间）。\n"
            "请基于此时间计算每条记忆的「距今天数」, 按时间分层压缩。\n"
        )

    memory_block = (
        "【当前 Crystal_memory.md 内容】\n"
        + (current_memory_md if current_memory_md else "(空)\n")
    )

    return (
        MEMORY_COMPRESS_USER_HEADER()
        + timestamp_section
        + "\n"
        + memory_block
    )


def buildMemoryUpdateUserPrompt(
    current_memory_md: str,
    user_msg: str,
    assistant_msg: str,
    current_timestamp: str = "",
    ask_track: list[dict] | None = None,
    load_track: list[dict] | None = None,
) -> str:
    """
    构造 update_crystal_memory 的 user prompt:
      - 静态头部: MEMORY_UPDATE_USER_HEADER (含时间戳规则 + 输出指令)
      - 动态块: 当前笔记 + 双轨 track (ask + load) + 本轮对话 + 更新时间戳

    ask_track / load_track: 当前触发 fp 窗口内的轨迹 (已在调用方按 pdf_fp 过滤 + 截尾)。
        论文场景: 只取当前论文最近 N 条 ask + M 条 load (跨论文 track 已被滤掉)。
        Chat 场景: 调用方不传, 这里也走空块 (chat 不进 track, memory 不更新)。
    """

    timestamp_section = ""
    if current_timestamp:
        timestamp_section = (
            f"\n【本次更新时刻】{current_timestamp}（北京时间）。\n"
            "请将本次更新时刻按上方时间戳规范追加到被修改或新增的条目末尾。\n"
        )

    memory_block = (
        "【当前 Crystal_memory.md 内容】\n"
        "(如果是空字符串, 表示这是首次记录)\n"
        + (current_memory_md if current_memory_md else "(空)\n")
    )

    # ── 当前窗口内的 ask 轨迹 ──
    ask_section = ""
    if ask_track:
        lines = ["【当前论文最近 Ask 轨迹 (按时间倒序)】"]
        for entry in ask_track:
            ts_str = entry.get("ts_str", "")
            user = entry.get("user", "")
            asst = entry.get("assistant", "")
            lines.append(f"- [{ts_str}] 用户: {user}")
            lines.append(f"  Crystal: {asst}")
        ask_section = "\n" + "\n".join(lines) + "\n"

    # ── 当前窗口内的 load 轨迹 ──
    load_section = ""
    if load_track:
        lines = ["【当前论文最近 Load 轨迹 (按时间倒序)】"]
        for entry in load_track:
            ts_str = entry.get("ts_str", "")
            chosen = entry.get("chosen_text", "")
            asst = entry.get("assistant", "")
            lines.append(f"- [{ts_str}] 选区: {chosen}")
            lines.append(f"  Crystal: {asst}")
        load_section = "\n" + "\n".join(lines) + "\n"

    this_turn_block = (
        "【本次用户和你核心的对话内容】\n"
        f"用户: {user_msg}\n"
        f"Crystal: {assistant_msg}\n"
    )

    return (
        MEMORY_UPDATE_USER_HEADER()
        + "\n\n"
        + memory_block
        + ask_section
        + load_section
        + this_turn_block
        + timestamp_section
    )


# ═══════════════════════════════════════════════════════════════════════
# Ask/Load 的 user content 构建
# ═══════════════════════════════════════════════════════════════════════

def build_ask_user_content(
    req: AiAskReq,
    ask_history: Optional[list[dict]] = None,
) -> str | list[dict]:
    """
    构建 Ask 本轮 user content。

    引用处理 (统一通过 fp 反查, 不再依赖前端传纯文本):
      - quotes 非空 -> 在 ask_history 中按 msg_fp 反查每条原文, 用 <quoted_message>
        块逐条包起来 (支持论文场景引用多条论文段落)
      - quotes 为空 -> 无引用, 直接返回 ask

    ask_history: 调用方传入 (load_paper_history_node 已装入 state), 避免此处重复读盘。
    """
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

    # 通过 msg_fp 在 ask_history 中反查引用原文
    quote_msgs: list[str] = []
    if req.quotes and ask_history is not None:
        fp_map: dict[str, str] = {
            m.get("msg_fp", ""): m.get("content", "")
            for m in ask_history
            if m.get("msg_fp")
        }
        # 防御: 前端如果误传非字符串元素(对象残留 / null), 仅 fp-shaped 字符串才参与反查。
        # 这样 quote.gid/quote.quote_msg 之类的旧字段不会污染 fp_map 查询。
        for fp in req.quotes:
            if not isinstance(fp, str) or not fp:
                debug(f"[build_ask_user_content] skip non-fp quote: {fp!r}")
                continue
            content = fp_map.get(fp, "")
            if content:
                quote_msgs.append(content)

    if quote_msgs:
        # 论文场景可能引用多条: 逐条用 <quoted_message>...</quoted_message> 包起来
        # 让 LLM 知道每一条都是引用上下文 (而非用户问题的一部分)。
        # Chat 场景 quotes 最多一条, 行为兼容。
        quoted_blocks = "\n\n".join(
            f"<quoted_message>\n{text}\n</quoted_message>"
            for text in quote_msgs
        )
        return (
            f"{quoted_blocks}\n\n"
            f"用户的询问【{req.ask}】"
        )

    # 无引用: 直接发 ask
    return req.ask


def build_load_user_content(req: AiLoadReq) -> str:
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


# ═══════════════════════════════════════════════════════════════════════
# 上下文组装
# ═══════════════════════════════════════════════════════════════════════

def compose_paper_ask_messages(
    req: AiAskReq,
    ask_history: list[dict],
    load_history: list[dict],
    agent_mem: str,
    time_context: str,
) -> list[dict]:
    """
    论文侧 Ask 上下文 (按 fp 隔离, 不读全局 track)。
    ask_history / load_history 已由 load_paper_history_node 装入 state,
    这里不重复读盘。论文 fp 上下文不再注入全局 track (与 chat 路径分离,
    详见 compose_chat_messages)。
    """
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_ASK(time_context, agent_mem)},
    ]
    # ask_history 本身已是 role 交替的标准 messages, 直接 extend 即可
    messages.extend(ask_history)
    # load_history 同理 (role=user=论文片段, role=assistant=概括)
    messages.extend(load_history)
    messages.append({"role": "user", "content": build_ask_user_content(req, ask_history)})
    return messages


def compose_chat_messages(
    req: AiAskReq,
    chat_history: list[dict],
    agent_mem: str,
    time_context: str,
) -> list[dict]:
    """
    ChatView 上下文 (脱离具体论文):
      [system]  CrystalPersona + time + device + agent_mem + 全局闲聊提示 + 论文 track 摘要
      [user/assistant × CHAT_LOCAL_LIMIT 对]  ChatView 本地历史
      [user]  本轮提问

    论文 track (ask+load) 由 _get_track 读出, 因为 _append_track 已经过滤了 chat fp,
    所以 track 里只含论文场景的记录, 正好对应 ChatView "提示 Crystal 全局而言
    和用户聊过什么" 的诉求。
    """
    # 1) 摘要化论文全局 track (避免破坏 user/assistant 交替, 用文本块)
    track_summary = build_track_summary_block()

    # 2) 设备感知上下文 (前端传入 device 字段)
    device = getattr(req, "device", None) or None
    device_context = get_device_context(device) if device else ""

    system_content = SYSTEM_ASK(time_context, agent_mem)
    if device_context:
        system_content += "\n\n" + device_context
    if track_summary:
        system_content += (
            "\n\n【全局上下文感知】\n"
            + track_summary
        )

    messages: list[dict] = [
        {"role": "system", "content": system_content},
    ]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": build_ask_user_content(req, chat_history)})
    return messages


def build_track_summary_block() -> str:
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
        lines.append(f"【近期对话感知】")
        for e in ask_track:
            ts = e.get("ts_str", "")
            user = e.get("user", "")
            asst = e.get("assistant", "")
            lines.append(f"- [{ts} 用户说]\n  {user}")
            if asst:
                lines.append(f"  [Crystal说]\n  {asst}")

    if load_track:
        lines.append("")
        lines.append(f"【近期用户论文阅读感知】")
        for e in load_track:
            ts = e.get("ts_str", "")
            chosen = e.get("chosen_text", "")
            asst = e.get("assistant", "")
            lines.append(f"- [{ts} 选段]\n  {chosen}")
            if asst:
                lines.append(f"  [总结]\n  {asst}")

    return "\n".join(lines)
