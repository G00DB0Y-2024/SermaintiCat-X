"""
下层: 基于历史 / quote 的最终 prompt 拼装。

包含:
- Ask/Load 的 user content 构建: build_ask_user_content / build_load_user_content
- 上下文组装: compose_paper_ask_messages / compose_chat_messages / build_chat_audit_block
- 合并 memory + self update: buildMemoryAndSelfUpdateUserPrompt

buildAskMessages / buildLoadMessages 在原 prompts.py 中的版本已废弃,
由本文件中的 compose_*_messages 接管, 不再保留。

【v8 删除】memory / self 的 compress 阶段 (Phase 1) 已移除:
  - 删除: MEMORY_COMPRESS_USER_HEADER / buildMemoryCompressUserPrompt
  - 删除: SELF_COMPRESS_USER_HEADER / buildSelfCompressUserPrompt
  update prompt 现在直接喂当前 md (不做预压缩)。

【2026-10-11 删除】旧版独立两段 update 的 user prompt 拼装:
  - 删除: MEMORY_UPDATE_USER_HEADER / buildMemoryUpdateUserPrompt
  - 删除: SELF_UPDATE_USER_HEADER / buildSelfUpdateUserPrompt
  update 路径只走 buildMemoryAndSelfUpdateUserPrompt, 失败时不 fallback。
"""
from __future__ import annotations

from typing import Optional

from .ai_models import AiAskReq, AiLoadReq
from .ai_config import CHAT_FP, CHAT_AUDIT_LIMIT
from .ai_emotion import build_emotion_context_block
from .ai_io import _load_emotion, _paper_history_path, _read_json_safe, _filter_entries
from .ai_utils import format_dt_second, get_device_context
from .prompts_system import SYSTEM_ASK, SYSTEM_LOAD
from utils.log import debug


# ═══════════════════════════════════════════════════════════════════════
# Crystal_memory + Crystal_self 一次性合并更新 (2026-10-09)
# ═══════════════════════════════════════════════════════════════════════
# 两份 md 全文同时注入, 强制 LLM 看到双方内容,
# 在生成 {updated_memory, updated_self} 时主动避免重复。

def buildMemoryAndSelfUpdateUserPrompt(
    current_memory_md: str,
    current_self_md: str,
    user_msg: str,
    assistant_msg: str,
    current_timestamp: str = "",
    chat_audit: str = "",
) -> str:
    """
    构造「合并 update」user prompt — 一次注入两份 md 全文,
    让 LLM 在 JSON 输出里同时维护两边。

    注入顺序 (按 system prompt 的 Step 1→5 工作流):
      1) 时间戳 (本轮时刻)
      2) 近期对话感知 (chat_audit) — 让 LLM 看到最近 N 条互动的全貌
      3) 本轮对话 (用户 + Crystal)
      4) 当前 Crystal_memory.md 全文
      5) 当前 Crystal_self.md 全文
      6) 任务指令: 输出 JSON

    字段值里的 JSON 字符串禁止再含 ``` 围栏, 已在 system prompt 强约束。
    """
    timestamp_section = ""
    if current_timestamp:
        timestamp_section = (
            f"\n【本次更新时刻】{current_timestamp}（北京时间）。\n"
            "请将本次更新时刻按 [更新时间: YYYY-MM-DD HH:MM] 规范追加到被修改或新增的条目末尾。\n"
        )

    chat_audit_section = ""
    if chat_audit:
        chat_audit_section = (
            "\n【近期对话感知 (来自 crystal_chat_ai.json 近 "
            f"{CHAT_AUDIT_LIMIT} 条)】\n"
            "(用以感知用户在近期对话里的称呼、互动仪式、潜在新偏好/新习惯)\n"
            f"{chat_audit}\n"
        )

    this_turn_block = (
        "【本次用户和你核心的对话内容】\n"
        f"用户: {user_msg}\n"
        f"Crystal: {assistant_msg}\n"
    )

    memory_block = (
        "【当前 Crystal_memory.md 全文 (memory — 关于「他」的认知)】\n"
        "(如果是空字符串, 表示这是首次记录)\n"
        + (current_memory_md if current_memory_md else "(空)\n")
    )

    self_block = (
        "【当前 Crystal_self.md 全文 (self — 关于「我」的认知)】\n"
        "(如果是空字符串, 表示这是首次记录)\n"
        + (current_self_md if current_self_md else "(空)\n")
    )

    task_block = (
        "【任务指令】\n"
        "请严格按 system prompt 定义的「主语判定规则 + 交叉去重工作流」,"
        "在一次响应里同时输出 updated_memory 和 updated_self。\n"
        "同一事实**严禁**在两份笔记里各写一份。\n"
        "输出格式必须是合法 JSON 对象, 仅含两个字段:\n"
        '  {"updated_memory": "<完整 Markdown 全文>", '
        '"updated_self": "<完整 Markdown 全文>"}\n'
    )

    return (
        timestamp_section
        + chat_audit_section
        + this_turn_block
        + memory_block
        + self_block
        + task_block
    )


# ═══════════════════════════════════════════════════════════════════════
# Crystal_memory + Crystal_self 一次性合并 COMPRESS (2026-10-11 引入)
# ═══════════════════════════════════════════════════════════════════════
# 与 buildMemoryAndSelfUpdateUserPrompt 对称, 但任务目标不同:
#   update: "追加本轮新事实"
#   compress: "按时间距离重排整份 md, 旧内容进 Hippocampus / 简化措辞"
# 关键差异:
#   · 不注入本轮对话 (压缩是离线任务, 与本轮对话无关)
#   · 不注入 chat_audit (同理, 压缩是历史重排, 不需要近期 audit)
#   · 显式告诉 LLM 当前 L1/L2 阈值 (24h / 72h), 方便它分档
#   · 输出 JSON 字段名: compressed_memory / compressed_self
#                    (与 update 的 updated_memory / updated_self 区分,
#                     _safe_parse_update_json 通用解析两个 key 都能取)

def buildMemoryAndSelfCompressUserPrompt(
    current_memory_md: str,
    current_self_md: str,
    current_timestamp: str = "",
    l1_hours: int = 24,
    l2_hours: int = 72,
) -> str:
    """
    构造「合并 compress」user prompt。

    注入顺序 (按 system prompt 的工作流):
      1) 时间戳 (本轮时刻) — LLM 用来算"距今多久"
      2) 档位阈值 (L1 / L2 小时数) — LLM 用来分档
      3) 当前 Crystal_memory.md 全文
      4) 当前 Crystal_self.md 全文
      5) 任务指令: 输出 JSON
    """
    timestamp_section = ""
    if current_timestamp:
        timestamp_section = (
            f"\n【本次压缩时刻】{current_timestamp}（北京时间）。\n"
            "请以此为基准, 扫描下方两份 md 中每条 Perception 条目文本前的\n"
            "「YYYY年MM月DD日 HH:MM」时间, 按距此时刻的时长决定档位。\n"
        )

    threshold_section = (
        f"\n【分档阈值 (来自 ai_config.COMPRESS_L*_HOURS)】\n"
        f"  - 第一档 (< {l1_hours}h): 日记体, 不动\n"
        f"  - 第二档 ({l1_hours}~{l2_hours}h): 简化措辞 + 合并相似条目, 仍按时间\n"
        f"  - 第三档 (> {l2_hours}h): 迁入 ## Hippocampus, 按主题分类, 无序列表\n"
    )

    memory_block = (
        "【当前 Crystal_memory.md 全文 (memory — 关于「他」的认知)】\n"
        "(如果是空字符串, 表示这是首次记录, 无需压缩, 直接复制空字符串或写占位说明)\n"
        + (current_memory_md if current_memory_md else "(空)\n")
    )

    self_block = (
        "【当前 Crystal_self.md 全文 (self — 关于「我」的认知)】\n"
        "(如果是空字符串, 表示这是首次记录, 无需压缩, 直接复制空字符串或写占位说明)\n"
        + (current_self_md if current_self_md else "(空)\n")
    )

    task_block = (
        "【任务指令】\n"
        "请严格按 system prompt 定义的「时间分档 + Hippocampus 规范」,\n"
        "在一次响应里同时输出 compressed_memory 和 compressed_self。\n"
        "已迁入 Hippocampus 的旧条目必须从日记区**删除** (避免重复)。\n"
        "输出格式必须是合法 JSON 对象, 仅含两个字段:\n"
        '  {"compressed_memory": "<完整 Markdown 全文>", '
        '"compressed_self": "<完整 Markdown 全文>"}\n'
    )

    return (
        timestamp_section
        + threshold_section
        + memory_block
        + self_block
        + task_block
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
    论文侧 Ask 上下文 (按 fp 隔离)。
    ask_history / load_history 已由 load_paper_history_node 装入 state,
    这里不重复读盘。论文 fp 上下文与 chat 路径分离 (详见 compose_chat_messages)。
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
                + agent_explore(A) + 全局闲聊提示 + [近期对话感知](chat audit) + emotion(C)
      [user/assistant × CHAT_LOCAL_LIMIT 对]  ChatView 本地历史
      [user]  本轮提问

    ECUS 四元素在本函数里的落点:
      E = agent_explore — 「约定 / 节奏规律 / 外呼记录」三段
      C = time_context + device_context + emotion_context
      U = agent_mem    — 关于他的认知 (不含约定, 已迁至 A)
      S = agent_self   — Crystal 对自己的认知

    近期对话感知: build_chat_audit_block 从 save/crystal_chat_ai.json 取
    最近 CHAT_AUDIT_LIMIT 条, 直接呈现"用户/Crystal主动/Crystal回复"
    三类 label 的对话流, 让 Crystal 看清"我与他的近期接触是怎样的"。

    agent_self (Crystal_self.md 全文): chat 侧**全量注入**。
    chat 是人格主场 —— 闲聊本来就该有连续性, 让 Crystal 记得"我是谁、我怎么说话、
    我在乎什么" 才能维持persona 一致。与论文侧相反: SYSTEM_ASK 论文分支 /
    SYSTEM_LOAD 都不带 self, 因为客观学术问答不需要 (也不该有) 哲学化自我叙述。
    """
    # 1) chat 路径的"近期对话感知"直接读 save/crystal_chat_ai.json
    #    (取最近 CHAT_AUDIT_LIMIT 条), 数据不依赖任何 track 子系统。
    chat_audit = build_chat_audit_block()

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
    # emotion 注入位置说明 (2026-10-15 微调):
    #   · 之前拼在 system 末尾 — LLM 对长 system 末尾注意力衰减, 容易把情绪
    #     当成「本轮附带信息」淡化掉。
    #   · 改为紧跟 agent_self 之后 — agent_self 是「Crystal 对自己」, emotion
    #     同样是「Crystal 自己的状态」, 同源同质, 放一起语义连贯; 又不抢 chat_audit
    #     这种「近期对话」段的位置 (chat_audit 涉及具体事件, 离 user query 越近越好)。
    #   · 与 ai_active._build_split_system_prompt 的 emotion 位置保持一致 (也在 self 之后)。
    if emotion_context:
        system_content += "\n\n" + emotion_context
    # agent_explore 当前未消费 —— Crystal_explore.md 注入逻辑将在
    # 后续 ai_active 重写 explore 处理时一并重建, 这里保留入参占位。

    # 【v6 重构】chat 侧注入 [近期对话感知], 直接读 save/crystal_chat_ai.json
    # (CHAT_AUDIT_LIMIT 条), 不依赖任何 track 子系统。
    if chat_audit:
        system_content += (
            "\n\n【近期对话内容感知, 用于参考、摘录、查询、区分角色、计算时间等】\n"
            + chat_audit
        )

    messages: list[dict] = [
        {"role": "system", "content": system_content},
    ]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": build_ask_user_content(req, chat_history)})
    return messages


# ═══════════════════════════════════════════════════════════════════════
# Chat 路径专属: 近期对话感知 (直接读 save/crystal_chat_ai.json)
# ═══════════════════════════════════════════════════════════════════════

def build_chat_audit_block(max_entries: int = CHAT_AUDIT_LIMIT) -> str:
    """
    ChatView 专属的 "近期对话感知" 块, 数据源: save/crystal_chat_ai.json
    (chat 本地缓存, 与 track 子系统解耦)。

    字段语义:
      · ReqAsk       →  "[<ts_str> 用户说]" + 用户内容
      · ResAsk active=False (主答)  →  "[<ts_str> Crystal回复说]" + 内容
      · ResAsk active=True  (主动追问) → "[<ts_str> Crystal主动说]" + 内容
                          (无配对 user, 主动开口本身就是完整事件)

    "谁/是否主动说" 三维标注:
      · 谁:  user / Crystal  (两条独立 label)
      · 是否主动: 仅对 Crystal 区分 (Crystal回复说 / Crystal主动说)
      · 时间:  ts_str 直接放 label 内。

    Args:
        max_entries: 最多取多少条 entry (ReqAsk+ResAsk 合并计数);
                     默认 CHAT_AUDIT_LIMIT (=20), 改一处全表生效。

    Returns:
        形如:
        - [<ts> 用户说]
          <user>
          [<ts> Crystal回复说]
          <asst>
          [<ts> Crystal主动说]
          <asst>
        ...
        或空串 (文件不存在 / 解析失败 / 无 entries)。
    """
    # 1) 读 CHAT_FP 的本地缓存
    path = _paper_history_path(CHAT_FP)
    data = _read_json_safe(path, [])
    if not isinstance(data, list) or not data:
        return ""

    # 2) 复用统一过滤 (屏蔽 Anno / Vision Ask), 保留 ReqAsk + ResAsk
    filtered = _filter_entries(data, {"ReqAsk", "ResAsk"})
    if not filtered:
        return ""

    # 3) 取最近 max_entries 条 (时间正序 → 最新一条在末尾)
    tail = filtered[-max_entries:]

    # 4) 按 entry 类型渲染, label 带时间戳 + 谁/是否主动
    lines: list[str] = []
    for e in tail:
        et = e.get("type", "")
        ts = e.get("dt", "") or ""
        content = (e.get("content") or "")[:200]

        if et == "ReqAsk":
            lines.append(f"- [{ts} 用户说]\n  {content}")
        elif et == "ResAsk":
            # 关键区分: active=True 是 Crystal 主动追问 (layer1 触发),
            # active=False/缺失是 Crystal 回复用户。
            label = "Crystal主动说" if e.get("active") else "Crystal回复说"
            lines.append(f"  [{ts} {label}]\n  {content}")
        # 其他 type 在 _filter_entries 后已不会再出现

    return "\n".join(lines)

