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
from .ai_config import CHAT_FP
from .ai_emotion import build_emotion_context_block
from .ai_io import _get_track, _load_emotion
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
        "【保真提醒】本次更新与压缩均按「保真优先」原则: "
        "合并/淘汰时保留用户在研究方向、工具/方法、表达习惯、忌讳点上的具体事实与示例, "
        "避免把笔记抹平为「用户喜欢 X 方向」这种抽象描述。\n"
    )


def MEMORY_COMPRESS_USER_HEADER() -> str:
    """
    compress phase 1 的 user prompt 静态头部:
    一次性说清楚: 这是压缩任务, 输出压缩后的 Markdown, 不引入任何新对话内容。
    """
    return (
        "下面是旧的 Crystal_memory.md 全文, 以及当前时间。\n"
        "请按 system 中「价值分层 + 保真优先」的原则对其压缩, 输出压缩后的完整 Markdown。\n"
        "默认目标是剔除冗余与过期 (压缩到 80-95% 区间), 保留可回忆的具体细节 "
        "(研究方向子领域、工具/方法的具体名称、具体偏好例子、明确忌讳), "
        "而不是把笔记抹平为抽象描述。\n"
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


# ═══════════════════════════════════════════════════════════════════════
# Crystal_self (Self-Cognition) 更新 user prompt — 与 memory 平行
# ═══════════════════════════════════════════════════════════════════════

def SELF_COMPRESS_USER_HEADER() -> str:
    """
    Crystal_self compress phase 1 的 user prompt 静态头部:
    一次性说清楚: 这是 Crystal 自我笔记的压缩任务, 输出压缩后的 Markdown, 不引入新对话内容。
    """
    return (
        "下面是旧的 Crystal_self.md 全文, 以及当前时间。\n"
        "请按 system 中「价值分层 + 保真优先」的原则对其压缩, 输出压缩后的完整 Markdown。\n"
        "默认目标是剔除冗余与过期 (压缩到 80-95% 区间), 保留可回忆的具体细节 "
        "(具体表达偏好、具体想学/想尝试的事、具体场景示例), "
        "而不是把自我认知抹平为抽象描述。\n"
        "不要引入任何新对话、新事件或新的更新时间戳 (那属于后续 update 阶段的工作)。\n"
    )


def SELF_UPDATE_USER_HEADER() -> str:
    """
    Crystal_self update phase 的 user prompt 静态头部:
    一次性说清楚: 这是在压缩后的旧 self 之上, 融合「刚更新好的 Crystal_memory.md」
    以及「本轮对话 + 双轨 track」, 输出新的 Crystal_self.md。

    与 memory update 的关键差异: self 多喂一个「更新好的 memory」块 —
    让 LLM 知道 "我对用户已经形成了哪些认知", 这会反向影响 Crystal 的自我定位。
    """
    return (
        "请基于提供的「当前 Crystal_self.md (压缩后)」「刚更新好的 Crystal_memory.md」"
        "「本次用户对话」以及作为补充的「当前论文窗口内的轨迹」,\n"
        "输出压缩并更新后的完整 Crystal_self.md 文本。\n"
        "【保真提醒】本次更新与压缩均按「保真优先」原则: "
        "合并/淘汰时保留 Crystal 的具体表达偏好、具体想学/想尝试的事、具体场景示例, "
        "避免把自我认知抹平为「性格开朗/喜欢聊天」这种抽象描述。\n"
    )


def buildSelfCompressUserPrompt(
    current_self_md: str,
    current_timestamp: str = "",
) -> str:
    """
    构造 self compress phase 1 的 user prompt (与 memory compress 完全对称):
      - 静态头部: SELF_COMPRESS_USER_HEADER
      - 当前时间
      - 旧 self md 全文
    """
    timestamp_section = ""
    if current_timestamp:
        timestamp_section = (
            f"\n【当前时间】{current_timestamp}（北京时间）。\n"
            "请基于此时间计算每条自我笔记的「距今天数」, 按时间分层压缩。\n"
        )

    self_block = (
        "【当前 Crystal_self.md 内容】\n"
        + (current_self_md if current_self_md else "(空)\n")
    )

    return (
        SELF_COMPRESS_USER_HEADER()
        + timestamp_section
        + "\n"
        + self_block
    )


def buildSelfUpdateUserPrompt(
    current_self_md: str,
    updated_memory_md: str,
    user_msg: str,
    assistant_msg: str,
    current_timestamp: str = "",
    ask_track: list[dict] | None = None,
    load_track: list[dict] | None = None,
) -> str:
    """
    构造 self update phase 的 user prompt:
      - 静态头部: SELF_UPDATE_USER_HEADER
      - 动态块: 压缩后的旧 self + 刚更新好的 memory + 双轨 track + 本轮对话 + 时间戳

    与 memory update 的差异 (按你定的规则):
      ① **加载旧的 Crystal_self.md** (compressed_self) 作为基础
      ② **额外加载更新过的 Crystal_memory.md** 作为外部参照
      ③ 不向 update memory 的 prompt 中加载 self (那边始终不传 self)

    ask_track / load_track: 当前触发 fp 窗口内的轨迹, 与 memory update 同源同源过
    """
    timestamp_section = ""
    if current_timestamp:
        timestamp_section = (
            f"\n【本次更新时刻】{current_timestamp}（北京时间）。\n"
            "请将本次更新时刻按 [更新时间: YYYY-MM-DD HH:MM] 规范追加到被修改或新增的条目末尾。\n"
        )

    self_block = (
        "【当前 Crystal_self.md 内容 (压缩后)】\n"
        "(如果是空字符串, 表示这是首次记录)\n"
        + (current_self_md if current_self_md else "(空)\n")
    )

    memory_block = (
        "【刚更新好的 Crystal_memory.md (作为外部参照, 帮助 Crystal 校准自我定位)】\n"
        "(这一段不是让 self 变成 memory 的内容, 而是供 LLM 看到「我对用户已经形成了哪些认知」"
        "再决定 Crystal 自己的状态如何适配)\n"
        + (updated_memory_md if updated_memory_md else "(空)\n")
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
        SELF_UPDATE_USER_HEADER()
        + "\n\n"
        + self_block
        + "\n\n"
        + memory_block
        + ask_section
        + load_section
        + this_turn_block
        + timestamp_section
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
    agent_self: str = "",
    agent_explore: str = "",
) -> list[dict]:
    """
    ChatView 上下文 (脱离具体论文), 全量注入三份档案 + emotion:
      [system]  CrystalPersona + time + device + agent_mem(U) + agent_self(S)
                + agent_explore(A) + 全局闲聊提示 + 论文 track 摘要 + emotion(C)
      [user/assistant × CHAT_LOCAL_LIMIT 对]  ChatView 本地历史
      [user]  本轮提问

    ECUS 四元素在本函数里的落点:
      E = agent_explore — 「约定 / 节奏规律 / 外呼记录」三段
      C = time_context + device_context + emotion_context
      U = agent_mem    — 关于他的认知 (不含约定, 已迁至 A)
      S = agent_self   — Crystal 对自己的认知

    论文 track (ask+load) 由 _get_track 读出, 因为 _append_track 已经过滤了 chat fp,
    所以 track 里只含论文场景的记录, 正好对应 ChatView "提示 Crystal 全局而言
    和用户聊过什么" 的诉求。

    agent_self (Crystal_self.md 全文): chat 侧**全量注入**。
    chat 是人格主场 —— 闲聊本来就该有连续性, 让 Crystal 记得"我是谁、我怎么说话、
    我在乎什么" 才能维持persona 一致。与论文侧相反: SYSTEM_ASK 论文分支 /
    SYSTEM_LOAD 都不带 self, 因为客观学术问答不需要 (也不该有) 哲学化自我叙述。
    """
    # 1) 摘要化论文全局 track (避免破坏 user/assistant 交替, 用文本块)
    track_summary = build_track_summary_block()

    # 2) 设备感知上下文 (前端传入 device 字段)
    device = getattr(req, "device", None) or None
    device_context = get_device_context(device) if device else ""

    # 3) 当前情绪上下文 (ChatView 专属 — 论文侧完全跳过, 不影响论文客观问答)
    #    读取时机是 compose_messages_node (LLM 调用前), 因此拿到的是"上一轮
    #    对话沉淀下的情绪快照"; 本轮 emotion_llm_node 在 save_paper_memory_node 之后才
    #    异步更新, 那个新值留给下一轮用。这个回路天然闭环:
    #      上轮 emotion → 影响本轮 LLM 语气 → 本轮 Lint LLM 评估 → 下轮 emotion 基线
    emotion_context = build_emotion_context_block(_load_emotion())

    system_content = SYSTEM_ASK(time_context, agent_mem)
    if device_context:
        system_content += "\n\n" + device_context
    if agent_self:
        system_content += (
            "\n\n【Crystal 对自己的认知(Crystal 私人笔记, 不要对用户直述)】\n"
            + agent_self
            + "\n(以上是你对自己的认知, 用来约束你的语气、态度与话题取舍; "
              "它不是你要讲出来的话题, 绝不要提及'我的笔记里写着...'之类元叙述。"
              "它同样不代表你认同其中每一条 —— 若某条与当下的真实感受冲突, "
              "以当下的感受为准。)"
        )
    # agent_explore 当前未消费 —— explore.md 注入逻辑将在
    # 后续 ai_active 重写 explore 处理时一并重建, 这里保留入参占位。
    if track_summary:
        system_content += (
            "\n\n【全局上下文感知】\n"
            + track_summary
        )
    if emotion_context:
        system_content += "\n\n" + emotion_context

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
            # 【v3 全局化】chat 也进 track 后, 需要区分语境: 论文对话 vs 闲聊对话,
            # 让 LLM 知道"这段记忆发生在什么场景下"。论文 fp 在 _append_track 时
            # 写入, chat fp 同理, 缺省 fallback "chat"。
            pdf_fp = e.get("pdf_fp", "")
            if pdf_fp == CHAT_FP:
                scene = "闲聊"
            elif pdf_fp:
                scene = f"论文:{pdf_fp[:8]}"
            else:
                scene = "未知场景"
            lines.append(f"- [{ts} {scene} 用户说]\n  {user}")
            if asst:
                # 关键区分: 主动开口 vs 回复。新字段是 entry.active
                # (旧命名字段已物理清理, 只有 ai_active
                # 落盘的追问 ResAsk 会写 active=True)。保留这个标签让
                # LLM 能区分"我主动找他"和"他找我我应答" —— 而这两者
                # 对他的打扰程度天差地别, 是节奏规律/分寸归纳的核心依据。
                # 【v4 时间戳一致】label 前也带 [ts], 与上面"用户说"对齐,
                # 让 LLM 一眼看到"这是哪个时点我说的", 方便它读时间序列。
                label = "Crystal主动说" if e.get("active") else "Crystal回复说"
                lines.append(f"  [{ts} {label}]\n  {asst}")

    if load_track:
        lines.append("")
        lines.append(f"【近期用户论文阅读感知】")
        for e in load_track:
            ts = e.get("ts_str", "")
            chosen = e.get("chosen_text", "")
            asst = e.get("assistant", "")
            lines.append(f"- [{ts} 选段]\n  {chosen}")
            if asst:
                # 同样的"label 带 ts"统一, [总结] 之前也带上 ts, 风格与 ask track 一致。
                lines.append(f"  [{ts} 总结]\n  {asst}")

    return "\n".join(lines)

