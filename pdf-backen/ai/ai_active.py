"""ai_active.py — Crystal 主动发言机制 (本期只实现 Layer1 短期追问)。

挂载位置 (LangGraph):  emotion_llm -> active_layer1 -> END

【重要约束】所有主动功能 (Layer1 / 未来的 Layer2 / Layer3) **只服务于 chat**
(pdf_fp == CHAT_FP = "crystal_chat"),论文侧不触发任何主动逻辑:

  · 论文节奏快、学术问答无 "主动关心" 语义。
  · chat 节奏慢 (数小时级),主动追问才有 "陪伴感" 价值。
  · 这是产品决策,不是技术限制 — 未来若做论文侧主动需另开层。

实现约束:

  · layer1_node / _run_layer1_pipeline 入口都检 pdf_fp != CHAT_FP → return。
  · on_user_msg (WS 上行) 入口同样 guard。
  · 未来加 layer2 / layer3 节点 → 必须先调 _assert_chat_fp(pdf_fp) 显式拒论文。
  · 主动行为产生的一切落盘 (active_resask / plans.json / Crystal_explore.md
    update) 都通过 _assert_chat_fp 二次校验,防漏 guard。

设计核心 (来自 plan §0):

  · E = Crystal_explore.md 全文 (LLM 自由管理,无段落约束)。
  · C = chat_local_limit 内近期对话窗口
        (load_chat_messages 返回 OpenAI messages,content 头部嵌
        [<ts> 用户说/Crystal主动说/Crystal回复] label)。
  · U = Crystal_memory.md 头部,S = Crystal_self.md 头部,
    + 重点强调本轮 MR 对话内容。

不实现:

  · 后台守护 tick 线程 (Layer1 是事件驱动,不需要 tick 轮询)。
  · 已读回执 (Layer1 不需要已读,deadline 是软标记)。
  · plans.json 读写 (无 Layer3 计划)。
  · λ 泊松采样 (无 Layer2 速率)。
  · settle / 重决策 (Layer1 无 "挂起到时间点再发")。

执行语义 (plan §10 fire-and-forget):

  · layer1_node(state) 是 LangGraph 节点签名,但内部用 asyncio.create_task
    启动 _run_layer1_pipeline,自己立即 return {}。
  · 主回复 (state["final_answer"]) 在 emotion_llm 之后立即就绪;
    layer1 后台跑,不阻塞主回复返回前端。
  · 追问生成完后,_broadcast_resactive 单条 WS push 到聊天窗口。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import threading
from dataclasses import dataclass
from typing import Any

from fastapi import WebSocket

from utils.log import debug

from .ai_config import CHAT_FP, CHAT_MEMORY_LIMIT
from .ai_io import (
    _append_active_resask,
    _get_agent_explore,
    _get_agent_self,
    _paper_history_path,
    _read_json_safe,
    _write_text_file_atomic,
    format_dt_second,
    load_chat_messages,
    now_ms,
)
from .ai_llm import _call_llm
from .ai_models import AiAskReq


# === 数据结构 ============================================================

@dataclass
class PendingEntry:
    """Layer1 主动追问 pending 栈元素。"""

    pdf_fp: str
    # 字段名沿用 plan §2.8 的命名 (历史字段名,与落盘 entry.msg_fp 同源)
    msg_fp: str               # 与落盘 entry.msg_fp 同源 (用此字段名,不沿用旧名)
    followup_content: str     # 我刚发出去的追问正文
    predict_window_sec: int   # 软标记,不阻塞
    sent_at_ms: int
    predict_reply_text: str


@dataclass
class LearnSample:
    """explore 反馈回路样本。"""

    pdf_fp: str
    sent_at_ms: int
    followup_content: str
    predict_reply_text: str
    user_replied_at_ms: int | None  # None = 用户沉默
    actual_reply_text: str | None
    hit: bool | None                 # None = 沉默 (无论 late / deadline 内)
    late: bool = False               # 用户答复超出 predict_window_sec 仍属 hit


# 内存 pending 栈 (不落盘,重启即丢)
_pending: list[PendingEntry] = []
_PENDING_LOCK = threading.Lock()

# 学习 batch (供 explore update 读)
_learn_batch: list[LearnSample] = []
_LEARN_LOCK = threading.Lock()

# WebSocket 客户端池
_ws_clients: list[WebSocket] = []
_WS_LOCK = threading.Lock()


# === 工具: Lint LLM 调用包装 =============================================

async def _call_lint_llm(prompt: str, system: str | None = None) -> str | None:
    """调 Lint LLM 做轻量判定 (judge / 落盘是否需要)。

    Args:
        prompt: user role 内容 (规则/判例/MR/JSON schema)。
        system: 可选 system role 内容 (例如 Crystal 人设档案,防止偏移)。
                传 None 或空串 → 退化为单 user message (向后兼容 v7)。

    Lint LLM 的配置在 params.json.llm_configs.lint;缺省时跳过 (return None)。
    """
    from .ai_config import get_current_config
    from .ai_emotion import EMOTION_LLM_CONFIG

    cfg = get_current_config()

    # 优先用 Lint 配置 (emotion module 用的也是它)
    api_key = EMOTION_LLM_CONFIG.get("api_key") or cfg.get("api_key", "")
    api_url = EMOTION_LLM_CONFIG.get("api_url") or cfg.get("api_url", "")
    model = EMOTION_LLM_CONFIG.get("model") or cfg.get("model", "")

    if not (api_key and api_url and model):
        return None

    # 构造 messages: system 可选,放人设/原则;user 放业务内容
    messages: list[dict] = []
    if system and system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        out = await _call_llm(
            messages=messages,
            disable_thinking=True,
            emotion_config={
                "api_key": api_key,
                "api_url": api_url,
                "model": model,
            } if (api_key and api_url and model) else None,
        )
        # _call_llm 返回 (content, usage, reasoning_content);
        # 兼容旧 2 元素元组 (含单元测试 / mock)
        if isinstance(out, tuple):
            content_str = out[0] if len(out) >= 1 else None
        else:
            content_str = out
        return content_str if isinstance(content_str, str) else None
    except Exception as e:
        debug(f"[ai_active] lint LLM error: {type(e).__name__}: {e}")
        return None


async def _parse_json_lenient(raw: str) -> dict | None:
    """容错 JSON 解析 (LLM 输出常带尾巴或包裹)。

    LLM 真实输出常见模式:
      1. ```json ... ``` markdown fence → 剥掉
      2. JSON 之后追加解释/换行/注释 → "Extra data" json.JSONDecodeError
      3. 多个 JSON 拼接 (LLM 重复生成) → 取第一个完整对象
      4. JSON 之前有 prose (LLM 先说一句话再写 JSON) → 取第一个 '{' 起
      5. JSON 之前有 markdown fence ``` 但 raw 不以 ``` 起 → rfind '{' 起
      6. JSON 内有非标字符 (LLM 偶尔写错) → 跳过非标位置

    策略 (按成功率从高到低,4 轮尝试):
      A. 剥 fence + s[0]=='{' 时,json.loads 一次 — 成功则返 (80% case)
      B. 失败 → 配对 '}' 截取第一个完整 JSON object,再 json.loads
      C. 仍失败 → 跳到下一个 '{' 用 raw_decode 一次性截
      D. 仍未果 → 正则粗找 '{"...":...}' 样式,二次尝试

    注: LLM 真在 string 内写未转义 '"' 是无法挽回的 (任何 parser 都解不开),
    这种情况直接 return None 才是诚实。
    """
    if not raw:
        return None

    s = raw.strip()

    # 1. 剥 markdown fence
    if s.startswith("```"):
        parts = s.split("```", 2)
        if len(parts) >= 2:
            s = parts[1]
            if s.startswith("json"):
                s = s[4:]
            s = s.strip().rstrip("`").strip()

    if not s:
        return None

    # 2. 策略 A: 直接 json.loads (80% 的"纯 JSON + 尾部空白/换行" case 直接过)
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. 策略 B: 配对 '}' 截取第一个完整 JSON object
    #    LLM 输出形如 "{...}{...}" 也能取第一个;形如 "prose\n{...}\ntail" 也能取
    first_brace = s.find("{")
    if first_brace < 0:
        return None
    body = s[first_brace:]

    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(body):
        if escape:
            escape = False
            continue
        if in_str:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = body[: i + 1]
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    # 配对截取后还解析失败 → 字符串内可能有非标 '}' 干扰,继续尝试
                    continue

    # 4. 策略 C: 整段都配对不上 → 用 raw_decode 一次性截
    try:
        obj, _ = json.JSONDecoder().raw_decode(s)
        return obj
    except (json.JSONDecodeError, ValueError):
        return None


# === Split 节点 — MR 末尾隐含追问拆解 (v2 改造) ==========================

def _build_split_prompt(mr_text: str, context_block: dict | None = None) -> tuple[str, str]:
    if context_block is None:
        # 向后兼容: 旧 caller (不传 req) 时降级到 V9 注入 (只注 self)
        context_block = {}
        try:
            context_block["self_head"] = _get_agent_self() or ""
        except Exception:
            context_block["self_head"] = ""

    # system role: 完整 chat context 注入 (V9.5+)
    system_prompt = _build_split_system_prompt(context_block)
    # user role: 任务 + 规则 + 判例 + 本轮 MR + JSON schema (V9.5+ 仍走 V9 的 schema 设计)
    user_prompt = _build_split_user_template(mr_text)
    return (system_prompt, user_prompt)


def _build_split_system_prompt(context_block: dict) -> str:
    """Split prompt 的 system role 部分: 注入精简 context (V1 用户决策 + emotion)。

    注入顺序:
      1. CrystalPersona()              (人设,防措辞漂移)
      2. agent_self (Crystal_self.md)  (Crystal 对自己的认知)
      3. chat_audit (max_entries=6)    (近期对话感知)
      4. emotion hint (V1 新增)         (Crystal 自身情绪 → 段数偏好 + 段间风格)

    不注入 (V9.5++ 决策):
      - time_context (split 不需要时段判断,早安/晚安由后续 chat 路径处理)
      - agent_mem    (memory 含大量 "已知话题",反而诱导 LLM 误切;
                      不利于按自然断点判断)

    任何一块为空 → 该块跳过,不注入空标题 (避免污染 LLM 注意力)。
    """
    sections: list[str] = []

    persona = context_block.get("persona") or ""
    if persona:
        sections.append(persona)

    self_head = context_block.get("self_head") or ""
    if self_head:
        sections.append(
            "【Crystal 对自己的认知(Crystal 私人笔记,不要对用户直述)】\n"
            + self_head
            + "\n(以上是你对自己的认知,用来约束你本次拆分时的语气、措辞和话题取舍;"
              "它不是你要讲出来的话题,绝不要提及 '我的笔记里写着...' 之类元叙述。"
              "它同样不代表你认同其中每一条 —— 若某条与当下的真实感受冲突,"
              "以当下的感受为准。)"
        )

    chat_audit = context_block.get("chat_audit") or ""
    if chat_audit:
        sections.append(
            "【近期对话内容感知(用于判断当前 MR 是不是在回应之前的话题,"
            "避免把已经在聊的同一段对话误当成独立气泡)】\n"
            + chat_audit
        )

    # V1+ 注入: Crystal 自身情绪 → 段数偏好 + 段间风格
    # 与 chat 用 emotion 调**生成内容**不同;split 用 emotion 调**物理切割**。
    # 中性 (< 0.15 magnitude) → _build_split_emotion_hint 返空,不注入。
    emotion_hint = _build_split_emotion_hint()
    if emotion_hint:
        sections.append(emotion_hint)

    if not sections:
        return ""  # 所有注入项均为空 → 返空串,caller 走单 user 路径,优雅降级

    return "\n\n".join(sections)


def _build_split_emotion_hint() -> str:
    """V1+ 新增: 把 Crystal 此刻的 emotion 翻译成 split 专用的 "段数偏好" 提示。

    与 chat 注入的差异:
      · chat 用 emotion 调**生成内容**的语气/措辞/姿态 (4 段结构)。
      · split 用 emotion 调**物理切割**的段数偏好和段间风格
        (e.g. positive → 倾向 2-3 段每段软软的;negative → 倾向 1-2 段简短;
         curious → 倾向 2-3 段每段是新角度;confuse → 倾向 1-2 段不要硬拆)。

    实现 (词池范式, 2026-10-14):
      · 从 emotion.label 查 _LABEL_TO_FAMILY 选 family (与 chat 注入同源)
      · **不复用** chat 4 段文本 — 因为 chat 段是给 "内容生成" 看的,
        split 段需要给 "切割决策" 看,语境完全不同。
      · 不输出 magnitude / level 数字 (V1+ 用户决策: 不注入数字)。
      · 中性 (label="中性" 或 intensity ≤ EMOTION_DECAY_WRITE_THRESHOLD) → 返空串,
        不污染 LLM 注意力。

    Returns:
        str — 形如 "【本轮拆分的情绪基调】...",或 "" (中性/失败)。
    """
    try:
        from .ai_emotion import (
            _LABEL_TO_FAMILY,
            EMOTION_POOL,
        )
        from .ai_config import EMOTION_DECAY_WRITE_THRESHOLD, EMOTION_POOL_FALLBACK_LABEL
        from .ai_io import _load_emotion
    except Exception as e:
        debug(f"[split] import emotion fail: {type(e).__name__}: {e}")
        return ""

    # 1) 取 emotion
    try:
        emo = _load_emotion()
    except Exception as e:
        debug(f"[split] _load_emotion fail: {type(e).__name__}: {e}")
        return ""

    if not isinstance(emo, dict):
        return ""

    # 2) 词池范式: label + intensity
    label = emo.get("label", EMOTION_POOL_FALLBACK_LABEL)
    if not isinstance(label, str) or label not in EMOTION_POOL:
        label = EMOTION_POOL_FALLBACK_LABEL
    try:
        intensity = float(emo.get("intensity", 0.0) or 0.0)
    except (TypeError, ValueError):
        intensity = 0.0
    intensity = max(0.0, min(1.0, intensity))

    # 3) 中性 → 不注入
    if label == EMOTION_POOL_FALLBACK_LABEL or intensity <= EMOTION_DECAY_WRITE_THRESHOLD:
        return ""

    # 4) label → family 查表
    family = _LABEL_TO_FAMILY.get(label)
    if family is None:
        return ""

    # 5) family → split 专用的段间风格指引 (V1 精简)
    # 每个 family 只给一段:
    #   split_style: 本轮拆出来的气泡,节奏/温度/段间风格该是什么样;
    #                附带 "不要给拆分多少段做硬约定" 的强提示,
    #                情绪只决定 "怎么切",不决定 "切几段",由自然断点主导。
    # 中性以上,文本里**不出现**数字 magnitude/level — 只用自然语言量词。
    split_hints: dict[str, dict[str, str]] = {
        "positive": {
            "split_style": "本轮 MR 拆出来应当**舒展、温暖**,每段都不长,像在小声絮叨;"
                           "段与段之间留一点小雀跃或软软的呼吸感,允许撒娇式收尾。"
                           "情绪上**不要给拆分多少段做硬约定** — 由自然断点决定,"
                           "情绪只决定「怎么切」,不决定「切几段」。",
        },
        "negative": {
            "split_style": "本轮 MR 拆出来应当**克制、收敛**,每段都短而轻,不要让任何一段显得密集;"
                           "段与段之间不要硬接感叹号或热情词,留一点安静的停顿。"
                           "情绪上**不要给拆分多少段做硬约定** — 由自然断点决定,"
                           "如果只有 1 个连贯话题就保持 1 段,不要为了「凑多段」把陈述改成问号。",
        },
        "curious": {
            "split_style": "本轮 MR 拆出来应当**轻快、有探索感**,每段都是一个新角度或新追问;"
                           "段与段之间允许有反问、联想或小俏皮的转场,但不要变成连珠炮。"
                           "情绪上**不要给拆分多少段做硬约定** — 由自然断点决定,"
                           "如果 MR 里有多个独立角度/追问/小联想,可松散拆几段。",
        },
        "confuse": {
            "split_style": "本轮 MR 拆出来应当**柔软、略带犹豫**,每段都留点缓冲呼吸;"
                           "段与段之间可以用「嗯...」「是这样吗...」这种带缓冲的口吻过渡,"
                           "但不要让犹豫感扩散到每段都像在自言自语。"
                           "情绪上**不要给拆分多少段做硬约定** — 由自然断点决定,"
                           "如果话题本就在绕一个事,保持 1 段即可,硬拆只会让「转不动」的感觉更碎。",
        },
    }

    if family not in split_hints:
        return ""

    hint = split_hints[family]

    # 6) 拼接 — 明确说明是 Crystal 自己的情绪,不要对用户直述
    #    2026-10-15 微调: 加 "系统注入 · 非用户输入" 前缀, 与 chat 路径的 emotion
    #    块保持一致, 防少数 LLM 把 system 末尾的情绪提示误读为"用户输入"。
    return (
        "【Crystal 此刻的自身情绪"
        "(这是「我」的情绪,不是用户的,不直接对用户说,也不要复述这一段)】\n"
        f"- 本轮的段间风格: {hint['split_style']}\n"
        "\n"
        "【使用规则】\n"
        "- 上面一行是**对你(拆分者)的指引**,不是给用户的内容,也不要让 parts 里的任何一段复述它。\n"
        "- 情绪**只调段间风格,不调段数**;段数严格由「自然断点」决定,不要因为「想凑多段」"
        "而把陈述改成问号,也不要因为「情绪低落」就硬压成 1 段。"
    )

def _build_split_user_template(mr_text: str) -> str:
    return f"""# 任务

你是 Crystal 的对话节奏拆分专家 + 元信息清洗员。真实人类在长篇回复时,为了避免对方看着累,会把一段长话"切"成连续发送的多条聊天气泡(第一条发完,稍停顿,接着发第二条;如果有三个自然话题,还可能再追加第三条)。

请分析以下 Crystal 准备发送的一段长消息,判断是否需要进行拆分。
如果需要,请对其进行**气泡级拟人化切割**: 第一条作为主答立刻发,后续的气泡作为追加内容陆续发送。
如果不需要拆分,**也必须**输出清洗过的单段(把 `[<时间> <角色>说]` 之类的元数据前缀剥掉,但保留正文内容),放在 `parts` 数组中(单元素)。

# 🚫【触发门槛:什么时候绝对不要拆 (should_split=false)?】

1. **浑然一体的单话题**:如果整段话都在聊同一件事,且篇幅不长,拆开会显得说话大喘气。
2. **全句即问题 / 纯陈述**:没有明显的话题分界线,或者只是简单的早晚安。
3. **找不到自然断点**:原消息虽然长,但话题连贯/没换行/没明显转折,强拆会割裂语义。

# 🧹【元信息清洗(无论拆不拆都要做,作用在每条 part 上)】

- 无论是 `should_split=true` 的多段,还是 `should_split=false` 的单段,**每条 part 必须是纯口语内容**。
- **必须剥掉**所有的元数据前缀 / 时间戳 / 方括号标签 / "角色名: 说了" 之类的署名。
  - 你在 system 上下文里看到的 `[2026-10-11 21:11 Crystal回复说]` 是系统给你读的对话格式, 用来理解时间先后, **不是要你复刻的输出格式**。
  - **直接给"她说了什么"本身即可**, 就像真人在聊天软件里发出去的消息一样干净。
- 如果原消息本身就是纯口语没有元数据, parts 直接照搬正文内容, 不要画蛇添足。
- **绝对不要**在 part 里复刻 `[时间 角色说] ...` 格式。
- **绝对不要**因为"原文太干净不需要清洗"就直接 `parts = [原文]` 跳过输出 —— 哪怕你最终输出的内容跟原文一字不差, 也必须把正文文字重新放进 parts 数组里作为你"主动清洗后的输出",因为 lint 端要拿到这个数组才能完成清洗回路。

# ✂️【气泡级切割与拟人化重组原则(仅在 should_split=true 时严格遵守)】

1. **【信息对等与拟人化润色】(最核心!)**:
   - 拆分不应是生硬的一刀切。为了让连续发送的气泡更像真人聊天,你可以**在自然断点处适度加入口语衔接词**(但**不要每段都加**,更不要所有段都用同一个词)。
  - 衔接词库(每次从里面选**不同的**词,不要重复):
    - 话题延伸: "而且" "另外" "还有" "顺便说一下"
    - 语气软化: "不过" "其实" "老实说" "说真的"
    - 自然转场: "话说" "哎对" "对了" "想起来" "对了想起"
    - 尽量减少"你", 如果有和用户约定好的称呼, 就多进行称呼
  - **绝对禁止**: 连续两段用同一个衔接词; 连续两段都以称呼开头; 在每条 part 前面都强行插一个衔接词。
  - 衔接词是**调味剂**,不是主菜。原文逻辑清晰时,不加任何衔接词直接分段反而更自然。
   - **人设一致性底线**: 润色时必须严格保持原消息的称呼习惯与情感基调。如果原句很亲昵,拆分出的句子也要充满温度,**绝不能突兀地变成冷冰冰的客服反问(如干瘪的"你")**。
   - `parts` 列表里所有气泡拼起来,**表达的核心话题与总信息量必须约等于原消息**。
   - **绝对禁止**无中生有地捏造原文根本没有的新事实或新问题。
   - **【元数据隔离】`parts` 里每条气泡都必须是纯口语内容, 严禁出现 `[<时间> <角色>说]` 这种元数据前缀 / 时间戳 / 方括号标签 / "角色名: 说了" 之类的署名。** 你在 system 上下文里看到的 `[2026-10-11 21:11 Crystal回复说]` 是系统给你读的对话格式, 用来理解时间先后, 不是要你复刻的输出格式。直接给"她说了什么"本身即可, 就像真人在聊天软件里发出去的消息一样干净。

2. **【寻找自然断点(多段切割)】**:
   - 观察原消息是否包含**多个不同的话题/意图**(例如:问候 + 早餐 + 行李整理 + 小尾巴)。
   - 如果有明确的换行符(`\\n\\n`)或话题转换,这就是最完美的切割点!按话题顺序一段一段切。
   - 每一段的结尾**完全可以是问号或句号**,不要受限于字数。

3. **【陈述与互动揉捏时的剥离】**:
   - 如果原文没有明显的分段,而是"一段长陈述 + 结尾附带了一个小提问",则把陈述给第一段,把提问给第二段。

4. **【parts 段数自然判断(无硬上限,但请慎重)】**:
   - 段数由你**根据自然断点自由决定**,没有硬性上限。
   - **⚠️ 软警告**:人类一次性发 4+ 条气泡会显得在"刷屏",绝大多数情况下 1-3 段最自然。
   - 如果只有 1 个自然断点 → 2 段;如果有 2 个自然断点 → 3 段;**如果找不到自然断点 → 报 should_split=false + parts=单元素数组**。

# 本轮待拆分的长消息全文

{mr_text}

# 严格 JSON 输出 (无 markdown fence):

{{
  "should_split": true|false,
  "parts": ["第一条气泡(主答,立刻发)", ...后续消息(可增加口语化衔接词(衔接词要多样化)/称呼)],
  "predict_reply": "预测用户看完最后一条气泡后会怎么回 (≤30字,若无拆分则空)",
  "predict_window_sec": int,  // 软预测: 用户大概多久会回 (秒), 30~600
  "reason": "≤40字,说明是在哪里找到的话题断点(按顺序列出),或不拆的理由"
}}
"""


async def _split_followup(mr_text: str, req: object | None = None) -> dict | None:
    """调 lint LLM 把 MR 拆成 N 段 (V9.5 任意段数版)。失败 → return None (degrade)。

    Returns:
        None = lint 失败/解析失败 → caller 走 fallback。
        {"should_split": False, "parts": [mr_text], "reason": "..."}
              = lint OK 但 MR 无需拆 (1 段也按不拆处理) → caller 走 fallback。
        {"should_split": True, "parts": [str, str, ...],
         "predict_reply": "...", "predict_window_sec": 60, "reason": "..."}
              = 拆成功 → caller 改 final_answer=parts[0],后续 parts 作追问。

    V9.5 关键变化:
    - LLM 输出 schema 从 {main_body, content} 改为 {parts: [str, ...]}。
    - parts 长度由 LLM 按自然断点自由决定 (无硬上限)。
    - 兜底: LLM 若仍按旧 schema 输出 (main_body + content) → 自动合并成 parts。
    - 末尾标点修复: 对每一段都做 (不再只修 main_body)。
    - predict_reply / predict_window_sec 字段**保留** (用户决策: 先不删,暂不动下游)。
    - V9.5+ 注入升级: 和 chat 同标准 — 注入 CrystalPersona + self + memory +
      time + 部分 chat_audit (max_entries=6),防止拆分时措辞漂移 / 误切上下文。
    """
    if not mr_text:
        return None

    # V9.5+ 注入完整 chat context (graceful degrade) — V1 增加 emotion 注入
    # · 与 chat compose_messages_node 注入对齐 (但更精简: 不注 device/memory/time)
    # · V1: emotion 通过 _build_split_emotion_hint 在 _build_split_system_prompt 内部注入
    #   (调 split 的段数偏好 + 段间风格,与 chat 调生成内容是两条独立链路)
    # · context_block 仍只传 3 个字段 (persona / self / chat_audit),
    #   emotion 在 system_prompt 拼装时单独注入 — 避免与 chat 注入格式耦合
    context_block = _build_split_context_block(req)
    system_prompt, user_prompt = _build_split_prompt(mr_text, context_block=context_block)

    raw = await _call_lint_llm(user_prompt, system=system_prompt)
    if not raw:
        return None

    out = await _parse_json_lenient(raw)
    if out is None:
        debug(f"[split] parse fail: {raw[:80]!r}")
        return None

    should = bool(out.get("should_split"))

    # ── 解析 parts 列表(无论 should_split=true/false 都要解析,
    #    因为 false 时也要用 lint 清洗后的 parts[0] 覆盖 final_answer) ──
    raw_parts = out.get("parts")
    parsed_parts: list[str] = []
    if isinstance(raw_parts, list) and raw_parts:
        parsed_parts = [str(p).strip() for p in raw_parts if str(p).strip()]
    else:
        # 兜底: LLM 仍按旧 schema 输出 main_body + content
        # 合并成 parts: [main_body, content] (顺序与旧版一致)
        legacy_body = str(out.get("main_body", "")).strip()
        legacy_content = str(out.get("content", "")).strip()
        if legacy_body:
            parsed_parts.append(legacy_body)
        if legacy_content:
            parsed_parts.append(legacy_content)

    if not should:
        # 找不到自然断点不拆:但仍用 lint 清洗后的 parts[0] 覆盖 final_answer,
        # 避免 main LLM 残留的元数据前缀 ([<时间> <角色>说] 之类) 直接落到用户眼前。
        # 若 lint 没输出 parts (极少见, 通常 prompt 强制要求输出),
        # 兜底用 mr_text,保留原行为。
        reason = str(out.get("reason", ""))[:80]
        if parsed_parts:
            cleaned = parsed_parts[0]
            debug(
                f"[split] no-split but cleaned: in={mr_text[:30]!r} "
                f"out={cleaned[:30]!r} reason={reason!r}"
            )
            return {
                "should_split": False,
                "parts": [cleaned],   # 用清洗后的内容, 不是 mr_text
                "predict_reply": "",
                "predict_window_sec": 60,
                "reason": reason,
            }
        debug(f"[split] skip: should_split=false + no parts, keep original: reason={reason!r}")
        return {
            "should_split": False,
            "parts": [mr_text],  # lint 没输出 parts 时的最终兜底
            "predict_reply": "",
            "predict_window_sec": 60,
            "reason": reason,
        }

    parts = parsed_parts
    if not parts:
        # 拆出来但 parts 全空 → 视为失败,不改 final_answer
        debug("[split] invalid: should=True but parts all empty")
        return None

    if len(parts) == 1:
        # 约定: parts.length==1 等同于不拆,caller 看到 should=True+1 段会走 fallback
        # 但仍设 should=True,让 caller 记一行 debug log
        debug(f"[split] parts.length==1, treat as no-split: {parts[0][:30]!r}")
        return {
            "should_split": True,  # 仍记 True,便于统计 "LLM 判拆但只拆 1 段" 频率
            "parts": parts,        # 单元素列表
            "predict_reply": "",
            "predict_window_sec": 60,
            "reason": "llm-only-1-part",
        }

    if len(parts) < 1:
        return None

    # ── 末尾标点修复 (对每一段都做) ──
    # 语义分割后,每段都是独立语义单元 — 不做字符串去重。
    # 仅修一个事实问题: LLM 切分时偶会把段尾切断在 ", " 或 "。" 之后,
    # 留一个孤立标点 (例如 "我心里很暖, "),视觉突兀 → 自动 rstrip + 加句号。
    fixed_parts: list[str] = []
    for i, p in enumerate(parts):
        if p and p[-1] in "，。、,.;；:： ":
            new_p = p.rstrip("，。、,.;；:： ").rstrip()
            if new_p and new_p[-1] not in "。！!?？…":
                new_p += "。"
            if new_p != p:
                debug(f"[split] fix-tail-punct part[{i}]: {p!r} -> {new_p!r}")
            fixed_parts.append(new_p)
        else:
            fixed_parts.append(p)

    parts = fixed_parts

    debug(f"[split] ok parts_n={len(parts)} parts0={parts[0][:30]!r}")
    return {
        "should_split": True,
        "parts": parts,                            # V9.5 主输出
        "predict_reply": str(out.get("predict_reply", ""))[:80],   # 暂保留,待后续决策
        "predict_window_sec": int(out.get("predict_window_sec", 60)),  # 暂保留
        "reason": str(out.get("reason", ""))[:80],
    }


def _build_split_context_block(req: object | None) -> dict:
    """V9.5+ 注入 (精简版): 组装 split 需要的 context block。

    注入清单 (V9.5++ 用户决策: 不注入 memory/time,避免干扰拆分判断):
      - persona:    str  (CrystalPersona() 全文,防措辞漂移)
      - self_head:  str  (Crystal_self.md 全文,语气/价值观约束)
      - chat_audit: str  (build_chat_audit_block(6),防 "误把完整对话当独立气泡")

    不注入:
      - memory  (用户决策: memory 含大量 "已知话题",反而会诱导 LLM 误切,
                 不利于按自然断点判断)
      - time    (用户决策: split 不需要时段判断,早安/晚安由后续 chat 路径处理)

    任何注入项失败 → 对应字段为空串,不影响整体调用 (graceful degrade)。
    所有 IO 都有进程内缓存,二次调用零开销。
    """
    ctx = {
        "persona": "",     # 必填,但下面 try 兜底
        "self_head": "",
        "chat_audit": "",
    }

    # 1) CrystalPersona (静态字符串,不会失败)
    try:
        from .prompts_system import CrystalPersona
        ctx["persona"] = CrystalPersona()
    except Exception as e:
        debug(f"[split] read CrystalPersona fail: {type(e).__name__}: {e}")

    # 2) Crystal_self.md (caller 已有 _get_agent_self,这里再读一次,进程内缓存)
    try:
        ctx["self_head"] = _get_agent_self() or ""
    except Exception as e:
        debug(f"[split] read Crystal_self fail: {type(e).__name__}: {e}")

    # 3) chat_audit 部分 (最近 6 条,不依赖 track 子系统)
    #    论文侧 / chat 之外调用 → req=None → build_chat_audit_block 仍可独立调用
    #    (它直接读 save/crystal_chat_ai.json,与 pdf_fp 无关)
    try:
        from .prompts_context import build_chat_audit_block
        ctx["chat_audit"] = build_chat_audit_block(max_entries=6)
    except Exception as e:
        debug(f"[split] read chat_audit fail: {type(e).__name__}: {e}")

    debug(
        f"[split] context_block sizes: "
        f"persona={len(ctx['persona'])} self={len(ctx['self_head'])} "
        f"audit={len(ctx['chat_audit'])}"
    )
    return ctx


async def split_followup_node(state: dict) -> dict:
    """LangGraph 节点: MR 末尾隐含追问拆解。

    挂载位置: llm_call → split_followup → save_paper_memory。
    【chat-only 约束】论文侧直接返回 {split_followup: None},不调 LLM。
    论文侧无追问语义 (层 1 不服务论文),同步 split 同样不服务论文。
    """
    req = state.get("req")
    if not isinstance(req, AiAskReq):
        return {"split_followup": None}

    if not _assert_chat_fp(req.pdf_fp, "[split_node]"):
        return {"split_followup": None}

    mr_text = state.get("final_answer", "") or ""
    if not mr_text:
        return {"split_followup": None}

    try:
        result = await _split_followup(mr_text, req=req)  # V9.5+: 传 req 用于组装 context_block
    except Exception as e:
        debug(f"[split] node error (swallowed): {type(e).__name__}: {e}")
        result = None

    if result is None:
        # lint 失败 → 保持 split_followup=None,不改 final_answer
        return {"split_followup": None}

    parts: list[str] = result.get("parts") or []

    # 清洗 (parts.length == 1) vs 拆分 (parts.length >= 2):
    #   1) parts == []  → 异常,不做任何改动
    #   2) parts == [1] → 不拆, 但**用清洗后的 parts[0] 覆盖 final_answer** (去掉元数据前缀)
    #   3) parts >= [2] → 拆, parts[0] 改 final_answer, 后续作追问陆续发
    if len(parts) < 1:
        debug("[split] skip: parts empty")
        return {"split_followup": None}

    if len(parts) == 1:
        # 清洗路径: 用 lint 清洗后的单段覆盖 final_answer
        cleaned = parts[0]
        if cleaned != mr_text:
            debug(
                f"[split] cleaned-only: in={mr_text[:30]!r} -> out={cleaned[:30]!r}"
            )
        else:
            debug(f"[split] cleaned-only (no change): {cleaned[:30]!r}")
        return {
            "final_answer": cleaned,
            "split_followup": None,
        }

    # 拆成功 → 改 final_answer = parts[0],后续 parts 作追问陆续发
    # 【v3 split-origin】拆分成功时,把原始 mr_text 存进 split_followup.origin,
    # 供 save_paper_memory_node 在主答 ResAsk entry 上写 "origin" 字段。
    # 目的是保留 "拆分前" 的原始消息,方便后续 LLM 复习 / 前端调试 / 论文分析。
    # V9.5 改造: contents 改为 list,每段独立 ResAsk + WS push。
    main_body = parts[0]
    followups = parts[1:]

    debug(
        f"[split] ok fp={req.pdf_fp[:12]} parts_n={len(parts)} "
        f"main={main_body[:30]!r} followups_n={len(followups)}"
    )
    return {
        "final_answer": main_body,
        "split_followup": {
            "contents": [c[:200] for c in followups],   # V9.5: 列表,每段独立落盘+push
            "predict_reply": result["predict_reply"],   # 暂保留 (用户决策: 不删,待后续)
            "predict_window_sec": result["predict_window_sec"],
            "intent": result["reason"],
            "from_split": True,
            "origin": None,  # 2026-10-16 改造: origin 不再存原始 MR, 改由 save_paper_memory_node
                              # 按"完整最终输出"(main_body + 全部 followups)写入
        },
        # 【计数语义 2026-10-14】不再注入 is_user_initiated=False。一次用户主动 ask,
        # 即便 split 拆成多段追问, "主答到达"这件事本身仍算一次成功的用户主动对话
        # → update_agent_memory_node 会按 state 默认值 True +1。后续追问在
        # _emit_followup 里也不会再 +1, 严格保证"一次用户主动对话 = count+1"。
    }


# === Fire-and-forget Pipeline (§10) ======================================

def _assert_chat_fp(pdf_fp: str, caller: str) -> bool:
    """主动功能 chat-only guard。

    所有主动机制 (Layer1 / 未来的 Layer2 / Layer3) 入口必先调此函数,
    论文侧返回 False,调用方必须立即放弃。

    Args:
        pdf_fp: state["req"].pdf_fp 或 on_user_msg 传入的 fp。
        caller: 调试用 caller 名 (例 "[layer1]"),出错时写 debug log。

    Returns:
        True  = chat 侧,可继续。
        False = 论文侧,调用方必须 return。
    """
    if pdf_fp != CHAT_FP:
        # 用 debug 不是 error — 论文侧调过来是合规行为 (节点拓扑保证),
        # 不该每次都打 ERROR 噪音。仅排查时 verbose 用。
        return False
    return True


def _snapshot_state_for_layer1(state: dict) -> dict:
    """复制 layer1 真正用到的字段,避免后台 task 持有 state 引用读到
    LangGraph 后续节点改写后的 state (save_paper_memory / emotion_llm
    都可能改 messages / final_answer)。

    【新增】包含 split_followup 字段,若 split_followup_node 已拆出,
    layer1 走 split 路径直接落盘追问。
    """
    req = state.get("req")
    return {
        "req": req,
        "fp": req.pdf_fp if req else "",
        "final_answer": state.get("final_answer", ""),
        "messages": list(state.get("messages", [])),
        "req_fp": state.get("req_fp", ""),
        "split_followup": state.get("split_followup"),
    }


async def _run_layer1_pipeline(snapshot: dict) -> None:
    """真正的 layer1 流程。**异步** 在 sync state 调用返回之后跑。

    【chat-only 约束】论文侧 (_assert_chat_fp=False) 直接 return,不调 LLM。
    【v2 split-only】仅依赖 split_followup_node 的拆解结果;无兜底分支,
    若 split 未拆出,layer1 整体空跑。
    异常一律静默吞掉 (不影响主回复),debug 打印。
    """
    req = snapshot.get("req")
    fp = snapshot.get("fp", "")
    if not isinstance(req, AiAskReq):
        return
    if not _assert_chat_fp(fp, "[layer1_pipeline]"):
        return  # 论文侧 — layer1 不服务

    try:
        # ── split 唯一路径 (V9.5 多分改造) ──
        # split_followup_node 在 llm_call 后同步拆出,若拆出,直接落盘跳过
        # V9.5: contents 是 list,每段独立落盘+WS push,part_index 控制延迟分配
        split_fu = snapshot.get("split_followup")
        if split_fu and isinstance(split_fu, dict):
            contents = split_fu.get("contents") or []
            if contents:
                total = len(contents)
                for idx, content in enumerate(contents):
                    if not content or not content.strip():
                        continue
                    await _emit_followup(
                        fp=fp,
                        content=content,
                        predict_reply=split_fu.get("predict_reply", ""),
                        predict_window_sec=split_fu.get("predict_window_sec", 60),
                        intent=split_fu.get("intent", "[split]"),
                        source="split",
                        part_index=idx,         # V9.5: 0-based 段号
                        total_parts=total,      # V9.5: 总段数 (供延迟分配)
                    )
        return  # 拆出 / 未拆出: 兜底已删除,层主不做空问
    except Exception as e:
        debug(f"[layer1] pipeline error (swallowed): fp={fp[:12]} {type(e).__name__}: {e}")


def _compute_split_delay_ms(content: str, part_index: int, total_parts: int) -> int:
    """
    V1+ 升级: 段间 delay 由"段位固定区间"改为"段位 base + 字数因子 + 随机 jitter"。

    设计目标:
      · 1 字短段 → ~3000ms (单字也该有"主答刚落屏"的小停顿)
      · 100 字长段 → ~8000ms (Crystal 在打长字,用户感知是"在写", 不觉得卡)
      · 段位节奏: 0(主答 3000) → 1(追问 2000, 略快) → 2(放缓 3000) → 3+(再缓 3500)
                  段间差 1s+, 营造"对话感"
      · jitter ±500ms, 让多次拆出来不至于"机械感"重复同一 delay

    公式:
      delay_ms = base(part_index) + char_factor(part_index) × len(content)
                 + uniform(-jitter, +jitter)
      clamp 到 [1500, 10000] 区间, 保护边界

    Args:
        content: 本段气泡文本 (V1+ 新增字段, 按 len(content) 算 char_factor)
        part_index: 0-based 段号 (0=主答之后第一段追问)
        total_parts: 总段数 (保留参数, 暂不直接用, 留给未来扩展)
                     — V1 决策: 节奏主要由 part_index 决定, total_parts 用作日志
    Returns:
        delay_ms ∈ [1500, 10000]
    """
    # 段位策略表 (V1+ 新增)
    #   base:     段位基础延迟 (ms)
    #   char_ms:  每字增量 (ms/字), 字数因子
    #   jitter:   随机抖动半径 (ms), 中心对称 uniform
    PART_STRATEGY = (
        # (base_ms, char_ms_per_char, jitter_ms)
        (3000, 50,  500),  # part=0: 主答后, 仪式感最强 (V1+ 调高, 撑起开场)
        (2000, 45,  500),  # part=1: 追问稍快, 维持推进 (V1+ 调低, 与 part=0 拉开 1s 差)
        (3000, 40,  500),  # part=2: 放缓, 给用户消化 (V1+ 调高, 与 part=1 拉开 1s 差)
        (3500, 35,  500),  # part=3+: 再缓, 避免刷屏 (V1+ 调高, 段间差更大)
    )
    # 段位越界保护 (e.g. part_index >= 4 用最后一行)
    if part_index < 0:
        part_index = 0
    if part_index >= len(PART_STRATEGY):
        part_index = len(PART_STRATEGY) - 1
    base_ms, char_ms, jitter_ms = PART_STRATEGY[part_index]

    # 字数因子: 按 len(content) 算
    n_chars = len(content) if isinstance(content, str) else 0

    # jitter: 中心对称 uniform
    jitter = random.randint(-jitter_ms, jitter_ms)

    delay_ms = base_ms + char_ms * n_chars + jitter

    # 边界保护: clamp 到 [1500, 10000] 区间, 防止极端值
    # (200 字以上会被截到 10000, 但实际 Crystal 追问极少超过 100 字)
    delay_ms = max(1500, min(10000, delay_ms))

    return delay_ms


async def _emit_followup(
    fp: str,
    content: str,
    predict_reply: str,
    predict_window_sec: int,
    intent: str,
    source: str,  # "split" (历史保留字段,唯一取值)
    part_index: int = 0,    # V9.5: 0-based 段号 (0=主答之后第一段追问)
    total_parts: int = 1,   # V9.5: 总段数 (供延迟分配)
) -> None:
    """追问落盘 + _pending 栈 + WS push 的统一出口。

    【split 唯一路径】v2 改造后 compose 兜底已删除,此函数仅供 split 路径调用。
    异常由 caller 静默吞 (这里是 inner,不该 raise)。

    【V1+ 段数递增 + 字数成正比延迟策略】(V9.5 升级)
    主回复通过 SSE 几乎实时到前端。split 路径下多段追问按
    (段位 base + 字数因子 + 随机 jitter) 综合算 delay,让用户感知
    "主回复 → 短停顿 → 追问1 → 中停顿 → 追问2 → 长停顿 → 追问3..."
    的自然节奏,避免刷屏感:

      · part_index=0:    base 3000ms + 50ms/字 + ±500ms jitter
                         (主答刚落屏, 撑起 "主动" 仪式感, 长段更慢)
      · part_index=1:    base 2000ms + 45ms/字 + ±500ms jitter
                         (追问1 节奏稍快, 维持对话推进感, 与 part=0 拉开 1s 差)
      · part_index=2:    base 3000ms + 40ms/字 + ±500ms jitter
                         (追问2 略放缓, 给用户消化, 与 part=1 拉开 1s 差)
      · part_index>=3:   base 3500ms + 35ms/字 + ±500ms jitter
                         (后续段继续拉长, 避免 "刷屏" 感, 段间差更大)
      · 目标区间: 1 字短段 → ~3000ms, 100 字长段 → ~8000ms
      · 边界保护: clamp 到 [1500, 10000] ms (1 字 base 2500 + jitter +500 = 3000;
                  200 字 base 2500 + 50*200=12500 → 截到 10000, 避免极端长文卡死)
      · 计算函数: _compute_split_delay_ms(content, part_index, total_parts)

    注意: 落盘和入 _pending 保持即时,不受延迟影响 → on_user_msg 的
    hit/miss 判定仍按真实时间轴,不会因为 WS 延迟误判 "沉默超时"。
    """
    # 1. 落盘 ResAsk (active=True)
    # 【计数语义 2026-10-14】不调 _inc_ask_count —— 本函数是 layer1 主动 push 追问,
    # 不是用户发起对话。ask_count 只在 update_agent_memory_node (graph 主链) 里
    # 通过 is_user_initiated 判定后递增,严格表示"用户一次成功主动对话"。
    ts_ms = now_ms()
    msg_fp = hashlib.md5(f"{ts_ms}-{fp}-{content}".encode()).hexdigest()[:12]
    ok = _append_active_resask(
        content,
        msg_fp=msg_fp,
        ts_ms=ts_ms,
        intent=intent,
    )
    if not ok:
        debug(f"[layer1] WARN: _append_active_resask fail fp={fp[:12]} source={source}")
        return

    # 构造同步 entry (用于 _pending + WS push;与磁盘一致)
    entry = {
        "type": "ResAsk",
        "content": content,
        "ts": ts_ms,
        "dt": format_dt_second(ts_ms),
        "msg_fp": msg_fp,
        "active": True,
        "intent": intent,
    }

    # 2. 入 _pending 栈
    with _PENDING_LOCK:
        _pending.append(PendingEntry(
            pdf_fp=fp,
            msg_fp=msg_fp,
            followup_content=content,
            predict_window_sec=predict_window_sec,
            sent_at_ms=ts_ms,
            predict_reply_text=predict_reply,
        ))

    # 3. WS push (段数递增 + 字数成正比 + 随机抖动延迟)
    # V1+ 升级: 旧版 (V9.5) 4 段位固定区间, 与字数无关; 新版按"段位 base + 字数因子 + jitter"算
    # · 目标: 1 字 → ~2500ms (单字短段), 100 字 → ~7500ms (长段 Crystal 思考中)
    # · 段位节奏保留: 0 段 (主答后仪式感) → 1 段 (追问稍快) → 2 段 (放缓)
    #   → 3+ 段 (再缓, 避免"刷屏")
    delay_ms = _compute_split_delay_ms(content, part_index, total_parts)

    debug(
        f"[layer1] delay {delay_ms}ms before push "
        f"(source={source} part={part_index}/{total_parts} chars={len(content)})"
    )
    await asyncio.sleep(delay_ms / 1000.0)
    _broadcast_resactive(fp, entry)

    # 4. explore update 也 fire-and-forget
    asyncio.create_task(_schedule_explore_update_async())
    debug(
        f"[layer1] done: fp={fp[:12]} outbound={msg_fp[:8]} "
        f"source={source} part={part_index}/{total_parts} reason={intent!r}"
    )


async def layer1_node(state: dict) -> dict:
    """LangGraph 节点入口 (fire-and-forget 调度器)。

    不 await 后台 pipeline,立即 return {}。
    【chat-only 约束】论文侧立即跳过,不起后台 task。
    """
    req = state.get("req")
    if not isinstance(req, AiAskReq):
        return {}
    if not _assert_chat_fp(req.pdf_fp, "[layer1_node]"):
        return {}

    snapshot = _snapshot_state_for_layer1(state)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run_layer1_pipeline(snapshot))
    except RuntimeError:
        # 无 event loop (理论上不会,但兜底)
        debug("[layer1] no running event loop, skip")
    return {}


# === 用户回复处理 (on_user_msg) ==========================================

def on_user_msg(pdf_fp: str, content: str, ts_ms: float) -> None:
    """WS 上行入口: 用户发新消息。

    行为:
      1. 查 _pending 栈: 若存在尚未 settle 的追问 (按 fp)。
      2. 调 Lint LLM 判定 hit/miss + 是否 late。
      3. 入 _learn_batch。
      4. 从 _pending 抹去该 entry。

    【chat-only 约束】论文侧 WS 上行 (如果未来出现) 直接 return。
    """
    if not _assert_chat_fp(pdf_fp, "[on_user_msg]"):
        return

    with _PENDING_LOCK:
        # 找到最新的 pending (可能同时有多个,取最近)
        pending = None
        for p in reversed(_pending):
            if p.pdf_fp == pdf_fp:
                pending = p
                break

    if pending is None:
        return  # 无 pending,不记样本

    now_ms_i = int(ts_ms) if ts_ms > 0 else now_ms()
    late = (now_ms_i - pending.sent_at_ms) > (pending.predict_window_sec * 1000)

    # 调 Lint LLM 判定 hit/miss (异步,不阻塞 WS)
    asyncio.create_task(
        _settle_pending_async(pending, content, now_ms_i, late)
    )


async def _settle_pending_async(pending: PendingEntry, content: str, replied_at_ms: int, late: bool) -> None:
    """异步判定 hit/miss,入 _learn_batch,从 _pending 抹去。"""
    try:
        hit = await _judge_hit(pending, content)
        sample = LearnSample(
            pdf_fp=pending.pdf_fp,
            sent_at_ms=pending.sent_at_ms,
            followup_content=pending.followup_content,
            predict_reply_text=pending.predict_reply_text,
            user_replied_at_ms=replied_at_ms,
            actual_reply_text=content[:200],
            hit=hit,
            late=late,
        )
        with _LEARN_LOCK:
            _learn_batch.append(sample)

        # 从 _pending 抹去
        with _PENDING_LOCK:
            try:
                _pending.remove(pending)
            except ValueError:
                pass

        debug(f"[layer1] settle: fp={pending.pdf_fp[:12]} hit={hit} late={late}")
    except Exception as e:
        debug(f"[layer1] settle error (swallowed): {type(e).__name__}: {e}")


async def _judge_hit(pending: PendingEntry, actual: str) -> bool | None:
    """Lint LLM 判定实际回复 vs 追问意图。

    返回 True (命中) / False (未命中) / None (判定失败)。
    """
    if not actual.strip():
        return False  # 空内容视为未命中

    prompt = f"""# 任务: 判定 hit/miss

Crystal 主动追问用户: {pending.followup_content!r}
Crystal 预测用户会回: {pending.predict_reply_text!r}
用户实际回复: {actual!r}

判定: 用户实际回复是否在语义上回应了 Crystal 的追问?
返回严格 JSON: {{"hit": true|false}}

"""
    raw = await _call_lint_llm(prompt)
    if not raw:
        return None

    try:
        out = await _parse_json_lenient(raw)
        if out is None:
            return None
        return bool(out.get("hit", False))
    except (ValueError, TypeError):
        return None


# === explore 反馈回路 (§2.6) ==============================================

async def _schedule_explore_update_async() -> None:
    """触发 Crystal_explore.md 重写。

    共享 _memory_updating 互斥锁 + MEMORY_UPDATE_EVERY_N 计数 (此处简化:
    不去争用 memory 互斥, 单独跑 — explore 重写是独立契约)。

    输入: Crystal_explore.md 全文 + _learn_batch 内的 hit/miss/late 样本。
    输出: LLM 重写整份 Crystal_explore.md → _write_agent_explore。
    """
    with _LEARN_LOCK:
        if not _learn_batch:
            return
        # 拷贝后清空 (避免 race)
        batch = list(_learn_batch)
        _learn_batch.clear()

    explore = _get_agent_explore() or ""

    # 构造 prompt
    samples_text = "\n".join(
        f"- 追问时间: {format_dt_second(s.sent_at_ms)}, 实际回复时间: "
        f"{format_dt_second(s.user_replied_at_ms) if s.user_replied_at_ms else '沉默'}, "
        f"hit={s.hit}, late={s.late}, "
        f"预测用户回: {s.predict_reply_text!r}, 实际: {s.actual_reply_text!r}"
        for s in batch[-10:]  # 最近 10 条
    )

    prompt = f"""# 任务: 重写 Crystal_explore.md

Crystal_explore.md 是关于**用户时间相关认知**的档案,只写:

  - 活跃时段 (他通常什么时候活跃)
  - 沉默含义 (他不回 = 在忙 / 不想聊 / 没看到?)
  - 回应速度习惯 (他通常多久会回,早回/晚回有没有规律)
  - 追问长度偏好 (他喜欢短答/中答/长答)
  - 节奏规律 (追问后多久他会回,哪些时间点他不回)

**不写**:
  - 单一问答过程 / 具体消息内容 / 用户偏好细节 (那些归 Crystal_memory)
  - 不要保留任何旧章节约束 (固定的三段标题已经废止)

# 当前 Crystal_explore.md

{explore if explore else "(空 — 这是首次写入)"}

# 新增样本 (本批)

{samples_text}

# 输出

重写**整份** Crystal_explore.md,Markdown 格式。返回**严格 JSON**:

{{"content": "重写后的 Crystal_explore.md 全文"}}

"""

    raw = await _call_lint_llm(prompt)
    if not raw:
        debug("[explore] LLM no response, skip write")
        return

    try:
        out = await _parse_json_lenient(raw)
        if out is None:
            debug(f"[explore] parse fail: not valid JSON  raw={raw[:80]!r}")
            return

        new_md = str(out.get("content", "")).strip()
        if not new_md:
            return

        # 落盘
        from .ai_config import EXPLORE_FILE
        _ensure_dir(EXPLORE_FILE)
        if _write_text_file_atomic(EXPLORE_FILE, new_md) == len(new_md.encode("utf-8")):
            from .ai_io import _set_agent_explore_cache
            _set_agent_explore_cache(new_md)
            debug(f"[explore] rewritten: {len(new_md)} chars")
    except (ValueError, TypeError) as e:
        debug(f"[explore] parse fail: {e}  raw={raw[:80]!r}")


def _ensure_dir(path: str) -> None:
    """确保父目录存在。"""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


# === WebSocket ===========================================================

def register_ws_client(ws: WebSocket) -> None:
    with _WS_LOCK:
        if ws not in _ws_clients:
            _ws_clients.append(ws)
    debug(f"[ws] client registered (total={len(_ws_clients)})")


def unregister_ws_client(ws: WebSocket) -> None:
    with _WS_LOCK:
        try:
            _ws_clients.remove(ws)
        except ValueError:
            pass
    debug(f"[ws] client unregistered (total={len(_ws_clients)})")


def _broadcast_resactive(fp: str, entry: dict) -> None:
    """WS push 主动追问 entry 到所有客户端。

    payload: {"type": "ResActive", "pdf_fp": fp, "entry": entry}
    """
    payload = json.dumps({
        "type": "ResActive",
        "pdf_fp": fp,
        "entry": entry,
    }, ensure_ascii=False)

    with _WS_LOCK:
        clients = list(_ws_clients)

    if not clients:
        # 客户端没连 WS → push 落空,落盘的 entry 只能等下次刷新通过 history 拉到。
        # 这是排查 WS 链路的核心信号 (Vite 没配 /ws proxy / 前端没 connect 都会出现)。
        debug("[ws] broadcast SKIP: no clients connected (entry 落盘,仅刷新可见)")
        return

    async def _push_all():
        for ws in clients:
            try:
                await ws.send_text(payload)
            except Exception as e:
                debug(f"[ws] push fail: {type(e).__name__}: {e}")

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_push_all())
    except RuntimeError:
        # 无 event loop,同步丢弃 (acceptable — 主流程不依赖此 push)
        pass


# === 对外只读接口 (供 frontend / 调试) ===================================

def get_pending_count() -> int:
    with _PENDING_LOCK:
        return len(_pending)


def get_learn_batch_count() -> int:
    with _LEARN_LOCK:
        return len(_learn_batch)