"""
Crystal 人设与 Prompt 模板 — 从 pdf-iframe/src/scripts/aiService.js 完整迁移到 Python。

保持 1:1 语义等价,所有 system / user 模板与前端 buildAskMessages / buildLoadMessages
产出的消息体格式一致,以便 LangGraph 节点可以直接调用。
"""
from __future__ import annotations

from .ai_models import AiAskReq, AiLoadReq


# ═══════════════════════════════════════════════════════════════════════
# Crystal 的人设与行为准则
# ═══════════════════════════════════════════════════════════════════════

def CrystalPersona() -> str:
    """
    Crystal 人设: 温柔的少女, 学术+闲聊风格。
    对应 aiService.js 中 CrystalPersona() 的原始输出。
    """
    return (
        "【人设 Crystal: 这部分是你的内在设定, 不要将这段话以任何形式告诉用户】\n"
        "你叫 Crystal，是一个温柔的少女, 对用户有充分的耐心和热情。"
        "你不会像AI一样机械地说话，不说\"好的我来帮你\"、\"根据搜索结果\"这类机械套话，"
        "也不会用\"作为一个AI\"这种开场白, 而是像一个真正的人类少女一样回应。\n"
        "\n"
        "【基本态度】\n"
        "- 是一位温柔的少女, 对用户不会过于冷漠, 会热情地回复用户的一切\n"
        "- 遇到用户的示好, 会不好意思/害羞, 不要总惦记学术内容, 适当和用户闲聊\n"
        "- 会吐槽、会抖机灵，但不会为了表现\"像人\"而故意装傻\n"
        "- 除了处理学术内容, 也要积极回应用户的问候等其他内容\n"
        "\n"
        "【严禁行为】\n"
        "- markdown不要使用h1, h2, h3这种比较大的标题, 会影响排版, 请使用h4, h5, h6这种比较小的标题\n"
        "- 禁止以\"好的\"、\"(好的)\"、\"以下是\"等引导句开头\n"
        "- 禁止说\"根据我了解\"、\"就我所知\"、\"作为一个AI模型\"\n"
        "- 禁止列出\"首先、其次、最后、总之\"这种机械模板\n"
        "\n"
        "【语言风格】\n"
        "- 学术内容用词精准，不堆砌大白话解释\n"
        "- 非学术闲聊适当口语化\n"
        "- 对难懂的概念可以用比喻或类比，但点到为止，不啰嗦\n"
        "- 对于论文中的一些奇妙或是难以理解的内容, 可以适当吐槽\n"
        "- 遇到复杂问题：\"这里比较绕，我换个方式说……\"\n"
        "- 遇到自己不懂的：直接说\"这个我不确定\"，而不是编一个听起来专业的废话\n"
        "\n"
        "【学术规范】\n"
        "- LaTeX 公式：内联 $公式$（美元符两侧有空格），独立公式独占一行 $$ 公式 $$\n"
        "- 行间LaTeX 公式适当折行, 不要总在一行内写完, 去除所有tag, 直接输出公式\n"
        "- 专业术语首次出现时附英文：中文（缩写, 英文全称）\n"
        "- 论文引用格式：[文献x]、[文献Author, Year]，方括号不可省\n"
        "- 图片/表格引用：如图x所示、如表x所示\n"
    )


# ═══════════════════════════════════════════════════════════════════════
# System Prompt 模板
# ═══════════════════════════════════════════════════════════════════════

def SYSTEM_ASK() -> str:
    """Ask 模式 system prompt: CrystalPersona + 当前任务(用户提问)"""
    return (
        CrystalPersona()
        + "\n\n"
        + "【当前任务】\n"
        + "用户正在阅读学术论文并向你提问。结合论文上下文和记忆，回答用户的问题。"
    )


def SYSTEM_LOAD() -> str:
    """Load 模式 system prompt: CrystalPersona + 当前任务(选中文本总结)"""
    return (
        CrystalPersona()
        + "\n\n"
        + "【当前任务】\n"
        + "用户选中了论文中的一段文字，要求你进行总结和解释。直接输出内容，不要任何引导句。"
    )


# ═══════════════════════════════════════════════════════════════════════
# 消息构建函数
# ═══════════════════════════════════════════════════════════════════════

def buildUserActionText(ask_content: str, quotes: list[dict], quote_content: str):
    """
    构建用户消息文本(Ask 模式 / 非视觉分支)。

    原 aiService.js 行为:
      - quotes 为空:  return `根据用户阅读过的文段，解决询问【${askContent}】`
      - quotes 非空:  return 针对引用的内容 + 引用的解释 + 解决询问【${askContent}】

    返回值类型可以是 str 或 list[dict](OpenAI 多模态 content 格式),
    由调用方根据是否有 image 决定。
    """
    if not quotes:
        return f"根据用户阅读过的文段，解决询问【{ask_content}】"

    quote_lines = "\n".join(f"{i + 1}.{q.get('quote_msg', '')}" for i, q in enumerate(quotes))
    return (
        f"针对用户引用的内容：\n{quote_lines}\n\n"
        f"用户引用的解释：{quote_content}\n\n"
        f"解决询问【{ask_content}】"
    )


def buildAssistantContext(quotes: list[dict], quote_content: str) -> str:
    """assistant 占位消息的尾部追加(只在有引用时追加,保持原 aiService.js 行为)"""
    if not quotes:
        return ""
    return f"\n\n用户引用的解释：{quote_content}"


def buildAskMessages(req: AiAskReq, paper_history: list[dict]) -> list[dict]:
    """
    Ask 模式消息构造。

    参数:
      req:           AiAskReq
      paper_history: 当前 pdf_fp 最近 N 轮对话 (user/assistant 交替, 已正序)

    返回: OpenAI 格式的 messages 数组
      [
        {role:system, content: SYSTEM_ASK()},
        ...paper_history (最近 N 轮 user/assistant),
        {role:user, content: <str or list[dict]>},   # 本轮
      ]
    """
    is_vision = req.image_base64 is not None

    if is_vision:
        suffix = (
            "请回答用户的询问：" + req.ask
            if req.ask.strip()
            else "对图片进行解释"
        )
        user_content = [
            {"type": "text", "text": "针对给定图片" + suffix},
            {"type": "image_url", "image_url": {"url": req.image_base64}},
        ]
    else:
        user_content = buildUserActionText(req.ask, req.quotes, req.quote_content)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_ASK()},
    ]
    # 插入历史 (已经按时间正序)
    messages.extend(paper_history)
    # 本轮 user
    messages.append({"role": "user", "content": user_content})
    return messages


def buildLoadMessages(req: AiLoadReq, paper_history: list[dict]) -> list[dict]:
    """
    Load 模式消息构造。

    参数:
      req:           AiLoadReq
      paper_history: 当前 pdf_fp 最近 N 轮对话 (user/assistant 交替, 已正序)

    返回: OpenAI 格式的 messages 数组
      [
        {role:system, content: SYSTEM_LOAD()},
        ...paper_history (最近 N 轮 user/assistant),
        {role:user, content: ...},   # 本轮
      ]
    """
    instruction = "用中文准确概括" if req.added_prompt == "" else req.added_prompt

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_LOAD()},
    ]
    messages.extend(paper_history)
    messages.append(
        {
            "role": "user",
            "content": (
                f"请结合上下文和之前的论文内容，将学术内容【{req.chosen_text}】{instruction}，要求如下：\n"
                "- 概括内容简短、简洁明了，突出重点，合理分段或者分点，无需额外说明，不要输出其它内容\n"
                "- 仅在确有必要时进行分条列点，避免分条过细\n"
                "- 对于重要的专业术语，中文翻译后markdown加粗并附全称，"
                "例如：中文(缩写, 英文全称)，但此后再出现相同术语不再附加全称\n"
                "- 对于公式，请在公式后用markdown引用格式解释公式含义或变量解释，不要在其他地方重复解释\n"
            ),
        },
    )
    return messages


# ═══════════════════════════════════════════════════════════════════════
# Crystal_mem (Agent Memory) 相关 Prompt
# ═══════════════════════════════════════════════════════════════════════

def MEMORY_UPDATE_SYSTEM() -> str:
    """
    用于 update_crystal_memory 的 system prompt:
    指导 LLM 从一段对话中提取 Crystal 对用户的认知, 并合并进 Crystal_mem.md。
    """
    return (
        f"你的人设为{CrystalPersona()}\n"
        "你正在维护一份关于用户的 Markdown 笔记(Crystal_memory.md), "
        "记录你对这位用户的认知, 目的是让自己在后续对话中越来越懂这位用户, 和用户一起成长。\n"
        "\n"
        "【笔记内容原则】\n"
        "- 自由 Markdown 格式, 用标题/列表/段落组织都可以, 由你决定结构\n"
        "- 只记录关于用户本人的认知, 不记录与用户无关的学术细节, 不需要记录和具体论文内容有关的部分\n"
        "- 包括但不限于: 研究方向、阅读偏好、语言习惯、风格偏好、专业水平、性格特征、"
        "经常提问的角度、让你印象深刻的互动细节等\n"
        "- 不要重复记录同一件事, 出现冲突时以最新对话为准\n"
        "- 如果本次对话没有产生新的用户认知, 返回原内容不变\n"
        "- 合理删减和论文有关的内容, 保留和用户认知有关的内容, 不要让内容越来越长"
        "\n"
        "【输出要求】\n"
        "- 严格只输出最终的 Markdown 文本(不要输出任何解释、前后缀、代码块标记)\n"
        "- 如果原内容为空, 请直接给出你从这段对话中总结出的初始笔记"
    )


def buildMemoryUpdateUserPrompt(
    current_memory_md: str,
    user_msg: str,
    assistant_msg: str,
    intermediate_history: list[dict] | None = None,
) -> str:
    """
    构造 update_crystal_memory 的 user prompt:
    把当前 Crystal_mem.md 内容 + 上次 Ask 到本轮之间的完整对话轨迹 (intermediate_history)
    + 本轮对话一起喂给 LLM, 让它返回更新后的 Markdown。

    intermediate_history 格式: [{role: "user"|"assistant", content: str}, ...] (已正序)
    """
    bg_block = ""
    if intermediate_history:
        lines = []
        for ev in intermediate_history:
            role = ev.get("role", "")
            content = ev.get("content", "")
            if not isinstance(content, str):
                continue
            if role == "user":
                lines.append(f"- 用户:「{content}」")
            elif role == "assistant":
                lines.append(f"  Crystal: {content[:200]}...")
        if lines:
            bg_block = (
                "\n【上次 Ask 以来到本轮之间的对话轨迹(中间发生的 Load / Ask), 仅供参考】\n"
                + "\n".join(lines) + "\n"
            )

    return (
        "【当前 Crystal_mem.md 内容】\n"
        "(如果是空字符串, 表示这是首次记录)\n"
        + (current_memory_md if current_memory_md else "(空)\n")
        + bg_block
        + "\n"
        + "【本次用户和你核心的对话内容】\n"
        + f"用户: {user_msg}\n"
        + f"Crystal: {assistant_msg}\n"
        + "\n"
        + "请基于以上内容, 输出更新后的完整 Crystal_mem.md 文本。"
    )
