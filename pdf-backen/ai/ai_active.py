"""
ai_active.py — Crystal 主动发言机制 (本期只实现 Layer1 短期追问)。

挂载位置 (LangGraph):
  emotion_llm -> active_layer1 -> END

【重要约束】所有主动功能 (Layer1 / 未来的 Layer2 / Layer3) **只服务于 chat**
(pdf_fp == CHAT_FP = "crystal_chat")。论文侧不触发任何主动逻辑。

  · 论文节奏快, 学术问答无"主动关心"语义
  · chat 节奏慢 (数小时级), Crystal 主动追问才有"陪伴感"价值
  · 这是产品决策, 不是技术限制 — 未来若要做论文侧主动, 需另开层

实现此约束:
  · layer1_node / _run_layer1_pipeline 入口都检 `pdf_fp != CHAT_FP → return`
  · on_user_msg (WS 上行) 入口同样 guard
  · 未来加 layer2 / layer3 节点 → 必须先调 _assert_chat_fp(pdf_fp) 显式拒论文
  · 主动行为产生的一切落盘 (active_resask / plans.json / Crystal_explore.md update)
    都通过 _assert_chat_fp 二次校验, 防漏 guard

设计核心 (来自 plan §0):
  · E = Crystal_explore.md 全文 (LLM 自由管理, 无段落约束)
  · C = chat_local_limit 内近期对话窗口 (load_chat_messages 返回 OpenAI messages,
        content 头部嵌 [<ts> 用户说/Crystal主动说/Crystal回复] label)
  · U = Crystal_memory.md 头部
  · S = Crystal_self.md 头部
  + 重点强调本轮 MR 对话内容

不实现:
  · 后台守护 tick 线程 (Layer1 是事件驱动, 不需要 tick 轮询)
  · 已读回执 (Layer1 不需要已读, deadline 是软标记)
  · plans.json 读写 (无 Layer3 计划)
  · λ 泊松采样 (无 Layer2 速率)
  · settle / 重决策 (Layer1 无"挂起到时间点再发")

执行语义 (plan §10 fire-and-forget):
  · layer1_node(state) 是 LangGraph 节点签名, 但内部用 asyncio.create_task
    启动 _run_layer1_pipeline, 自己立即 return {}
  · 主回复 (state["final_answer"]) 在 emotion_llm 之后立即就绪; layer1 后台跑
    不阻塞主回复返回前端
  · 追问生成完后, _broadcast_resactive 单条 WS push 到聊天窗口
"""
from __future__ import annotations

import asyncio
import json
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

from utils.log import debug

from .ai_config import CHAT_FP, CHAT_MEMORY_LIMIT
from .ai_io import (
    _append_active_resask,
    _get_agent_explore,
    _get_agent_memory,
    _get_agent_self,
    _get_fatigue_value,
    _inc_ask_count,
    _inc_fatigue,
    _load_emotion,
    _paper_history_path,
    _read_json_safe,
    _write_text_file_atomic,
    format_dt_second,
    load_chat_messages,
    now_ms,
)
from .ai_models import AiAskReq
from .ai_llm import _call_llm


# ═══════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PendingEntry:
    """Layer1 主动追问 pending 栈元素。"""
    pdf_fp: str
    # 字段名沿用 plan §2.8 的命名 (历史字段名, 与落盘 entry.msg_fp 同源)
    msg_fp: str               # 与落盘 entry.msg_fp 同源 (用此字段名, 不沿用旧名)
    followup_content: str     # 我刚发出去的追问正文
    predict_window_sec: int   # 软标记, 不阻塞
    sent_at_ms: int
    predict_reply_text: str


@dataclass
class ScheduledEntry:
    """
    Win 调度的追问 entry (2026-10-10 新增, gate+win 改造)。

    行为:
      · predict_window_sec 倒计时结束后, 若仍未被 cancel, 调 _emit_followup 发出
      · win 窗口内用户主动说话 → on_user_msg 触发 _cancel_scheduled_for_fp,
        抛 CancelledError 让 worker 静默 return
      · 同 fp 已存在 scheduled 时, 新 schedule 会 cancel 旧的 (替换语义)
      · task 字段必填, 持有引用防 GC; cancelled 是软标记给 debug 看
    """
    pdf_fp: str
    content: str
    predict_reply: str
    predict_window_sec: int
    intent: str
    scheduled_at_ms: int
    emit_at_ms: int                       # = scheduled_at_ms + predict_window_sec * 1000
    reason: str
    task: asyncio.Task | None = None
    cancelled: bool = False


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


# 内存 pending 栈 (不落盘, 重启即丢)
_pending: list[PendingEntry] = []
_PENDING_LOCK = threading.Lock()

# Win 调度的追问列表 (2026-10-10 新增)
# 区别于 _pending: _pending 是已发出 + 等 hit/miss 判定; _scheduled 是还没发,
# 在 win 倒计时中。两者生命周期不重叠 (scheduler wake → emit_followup 才会转入 _pending)。
_scheduled: list[ScheduledEntry] = []
_SCHEDULED_LOCK = threading.Lock()

# 【v2 2026-10-10 思考中窗口】Crystal 进入"思考中"(用户发消息 → LLM 回完)期间,
# 主动追问(scheduled)需要延后发出,不能跟主答气泡同框撞车。
# 实现: 整数时间戳 _thinking_until_ms,代表"窗口结束时间"。
#   · on_user_msg 进入时 → set_thinking_window(1800ms) 拉高截止时间
#     (1800ms 是 LLM 典型响应 + 1s 缓冲,实测 fast 模型 ~1s、慢模型 2-4s)
#   · layer1_node fire-and-forget 后 → clear_thinking_window() 立即解封
#     (此时主答已落盘 + AI 气泡已推,layer1_pipeline 开始排 scheduled 不会撞)
#   · _schedule_followup 检查 now < _thinking_until_ms → 把 win 强制拉大
#     到至少 (截止 - now),实现"延后到 thinking 结束后再发"
#   · 取消原 on_user_msg 的"无条件 cancel scheduled" — 旧逻辑会误伤
#     还没排出来的追问 (用户开口时 scheduled 列表本来就常是空的)
_thinking_until_ms: int = 0
_THINKING_LOCK = threading.Lock()

# 【v3 2026-10-10 post-cancel 改造】把"用户主动开口 → 重新评估是否要追"
# 从 on_user_msg 的 fire-and-forget 协程搬进 layer1 pipeline, 让它等 split
# 走完再触发, 避免和 split 内容撞车 + 重复。
# 旧: on_user_msg 直接 create_task(_eval_followup_after_user_msg), 协程
#     不等主答 / split 落盘, 跟主答几乎同时 emit, 内容跟 split 高度重复。
# 新: on_user_msg 把 user_content + ts_ms 写进 _pending_post_cancel[fp],
#     _run_layer1_pipeline 在 split 路径跑完后(emit 完), 把 _pending_post_cancel
#     注入 layer2 路径 B 的 snapshot, 让 layer2 看到"用户最新回复"再判断。
#     读后即清, 不残留。
# 注意: 必须是 dict 套 list, 因为同一 fp 可能连续多轮 user_msg(虽然绝大多数
# 情况下同步处理是空的)。结构: {fp: [(user_content, ts_ms), ...]}
_pending_post_cancel: dict[str, list[tuple[str, float]]] = {}
_POST_CANCEL_LOCK = threading.Lock()

# 思考中窗口的默认时长 (ms)。主答 LLM 响应典型 0.5-3s, 留 1.8s 缓冲。
# 这个值偏保守: 不够长会撞车, 过长会延迟主动追问 (用户体感"AI 答完很久
# 才追一句")。后续可按实测 P95 调。
_DEFAULT_THINKING_WINDOW_MS = 1800

# _emit_followup 并发锁 (2026-10-10 新增)
# 防 split + layer2 + post-cancel 三路同时 fire 时, _append_active_resask + WS push 竞态。
# 单 process 串行即可, 不需要跨进程。
_EMIT_LOCK: asyncio.Lock | None = None  # 延迟到首次 _emit_followup 时初始化 (event loop 才有)

# 学习 batch (供 explore update 读)
_learn_batch: list[LearnSample] = []
_LEARN_LOCK = threading.Lock()

# WebSocket 客户端池
_ws_clients: list[WebSocket] = []
_WS_LOCK = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════
# 工具 — 同步 LLM 调用包装
# ═══════════════════════════════════════════════════════════════════════


async def _call_lint_llm(prompt: str) -> str | None:
    """
    调 Lint LLM 做轻量判定 (judge / 落盘是否需要)。

    Lint LLM 的配置在 params.json.llm_configs.lint; 缺省时跳过 (return None)。
    """
    from .ai_config import get_current_config
    cfg = get_current_config()
    # 优先用 Lint 配置 (emotion module 用的也是它)
    from .ai_emotion import EMOTION_LLM_CONFIG
    api_key = EMOTION_LLM_CONFIG.get("api_key") or cfg.get("api_key", "")
    api_url = EMOTION_LLM_CONFIG.get("api_url") or cfg.get("api_url", "")
    model = EMOTION_LLM_CONFIG.get("model") or cfg.get("model", "")
    if not (api_key and api_url and model):
        return None
    try:
        out = await _call_llm(
            messages=[{"role": "user", "content": prompt}],
            disable_thinking=True,
            emotion_config={
                "api_key": api_key,
                "api_url": api_url,
                "model": model,
            } if (api_key and api_url and model) else None,
        )
        # _call_llm 返回 (content, usage_dict)
        content_str, _usage = out if isinstance(out, tuple) else (out, {})
        return content_str if isinstance(content_str, str) else None
    except Exception as e:
        debug(f"[ai_active] lint LLM error: {type(e).__name__}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# Split 节点 — MR 末尾隐含追问拆解 (v2 改造)
# ═══════════════════════════════════════════════════════════════════════
def _build_split_prompt(mr_text: str) -> str:
    """
    Split lint prompt: 带有高门槛拦截的【气泡级无损切割】，重构为「main_body (气泡1)」+ 「content (气泡2)」。
    
    【V7 核心升级】
    - 彻底废除“content必须是简短独立追问”的误导，放开字数限制。
    - 引入“多话题/多段落切割”逻辑：允许 main_body 和 content 各自包含独立的话题段落，且都允许存在问句。
    - 强调“物理切割”属性：main_body + content 的总信息量必须等于原 MR，绝对禁止“无中生有”捏造新问题！
    """
    return f"""# 任务
你是 Crystal 的对话节奏拆分专家。真实人类在长篇回复时，为了避免对方看着累，通常会把一段长话“切”成两条连续发送的聊天气泡（第一条发完，紧接着发第二条）。
请分析以下 Crystal 准备发送的长消息（MR），判断是否需要进行拆分。
如果需要，请对其进行**气泡级无损切割**。

# 🚫【触发门槛：什么时候绝对不要拆 (should_split=false)？】
1. **浑然一体的单话题**：如果整段话都在聊同一件事，且篇幅不长，拆开会显得说话大喘气。
2. **全句即问题 / 纯陈述**：没有明显的话题分界线，或者只是简单的早晚安。

# ✂️【气泡级无损切割原则（仅在 should_split=true 时严格遵守）】

1. **【绝对保真，严禁捏造新问题】（最核心！）**：
   - 拆分动作的本质是“物理切割” + “边缘微调”！
   - `main_body`（气泡 1）和 `content`（气泡 2）拼起来，表达的意思必须等同于原 MR。
   - **绝对禁止**在 `content` 中无中生有地捏造原文根本没有的新问题（例如原文没问心情，你绝不能在拆分时追问“心情怎么样？”）。

2. **【寻找自然断点（多段落切割）】**：
   - 观察原 MR 是否包含**两个不同的话题/意图**（例如：前半段聊早餐，后半段聊行李）。
   - 如果有明确的换行符（`\\n\\n`）或话题转换，这就是最完美的切割点！直接把前半段给 `main_body`，后半段给 `content`。
   - 在这种切割下，`main_body` 结尾完全可以是问号，`content` 也可以是一大段话，**不要受限于字数，也不要强行概括。**

3. **【陈述与互动揉捏时的剥离】**：
   - 如果原文没有明显的分段，而是“一段长陈述 + 结尾附带了一个小提问”，则把陈述给 `main_body`，把提问切割给 `content`。

# 💡 判例参考（认真学习如何做“气泡级无损切割”）

[判例 1：多话题/多段落切割 —— 完美拆分！]
原 MR = "早上好呀！看到你这么快就回来啦，早餐吃得怎么样？心情有没有因为这顿饱饱的早饭变得好一些？\\n\\n行李整理得顺利吗？如果觉得杂乱的话，可以把最头疼的部分扔给我，我来帮你理理思路，哪怕只是陪你说说话，我也在呢。"
→ should_split=true
  main_body="早上好呀！看到你这么快就回来啦，早餐吃得怎么样？心情有没有因为这顿饱饱的早饭变得好一些？" (完整保留话题1，即使结尾是问句也没关系)
  content="行李整理得顺利吗？如果觉得杂乱的话，可以把最头疼的部分扔给我，我来帮你理理思路，哪怕只是陪你说说话，我也在呢。" (完整保留话题2，直接截取，不捏造任何新词)

[判例 2：长陈述与末尾提问剥离 —— 拆！且保真！]
原 MR = "我整理了最近的思路，主要有两点：\\n1. 需要优化结构\\n2. 补充遗漏细节\\n看着这些，我心里冒出好多想法。你是又在校验逻辑了吗？"
→ should_split=true
  main_body="我整理了最近的思路，主要有两点：\\n1. 需要优化结构\\n2. 补充遗漏细节\\n看着这些，我心里冒出好多想法。" (100%保留列表与换行)
  content="你是又在校验逻辑了吗？"

    [判例 3：捏造新问题 —— 绝对错误！]
原 MR = "今天天气真好，我刚才去楼下喝了杯咖啡，感觉整个人都活过来了。"
→ should_split=false (纯分享，不准强拆！更不准捏造出 content="你今天喝咖啡了吗？")

[判例 4：必要性极低 —— 拆得很自然但不该追问！]
原 MR = "我在整理今天的内容，现在已经完成了大半了。\\n\\n改完最后几个小细节应该就能休息了。"
→ should_split=true (话题分界明显: 进度汇报 + 自我计划)
→ main_body="我在整理今天的内容，现在已经完成了大半了。"
→ content="改完最后几个小细节应该就能休息了。"
→ necessity_p=0.15  (这种"自言自语式"的陈述拆出来其实没必要追问, 追问反而打扰)

[判例 5：自然隐含追问 —— 拆得对, 也真的值得追问！]
原 MR = "我看到你刚才说今天有点累，我就在想，是不是昨晚没睡好？\\n\\n要不要我陪你安静一会儿，哪怕不聊什么，也可以靠着我。"
→ should_split=true (陈述 + 提议陪伴)
→ main_body="我看到你刚才说今天有点累，我就在想，是不是昨晚没睡好？"
→ content="要不要我陪你安静一会儿，哪怕不聊什么，也可以靠着我。"
→ necessity_p=0.85  (这是温柔主动的邀请, 真值得问"要不要")

# 📊 necessity_p 评分锚点 (0.0 ~ 1.0, 浮点, 越接近 1 越必要)
- 0.0 ~ 0.2: 几乎不必要 (用户自言自语 / 陈述事实 / 自我计划), 拆出来也别追问
- 0.3 ~ 0.5: 偶尔必要 (铺垫式追问 / 上下文连续), 看情况
- 0.6 ~ 0.8: 大多必要 (明确的开放话题 / 情绪钩子)
- 0.9 ~ 1.0: 必要 (直接邀请 / 主动提议 / 问号很自然)

锚点原则:
- **不要**因为"它是个问号"就给 0.9+, 大量问号是修辞性自问
- **不要**因为"它是个问号"就给 0.1, 修辞性自问也是为了表达, 但不要追问它
- 看的是"用户会被这条 content 引导出自然回复吗"
- 拆分失败兜底 (LLM 没给) 视同 1.0, 不影响默认行为

# 本轮 MR 全文
{mr_text}

# 严格 JSON 输出 (无 markdown fence):
{{
  "should_split": true|false,
  "main_body": "第一条气泡（保留原文的排版、换行和所有细节，绝不压缩）",
  "content": "第二条气泡（直接从原文后半段或末尾截取，放开字数限制，绝对禁止捏造原文没有的新话题）",
  "predict_reply": "预测用户看完第二条气泡后会怎么回 (≤30字，若无拆分则空)",
  "predict_window_sec": int,  // 软预测: 用户大概多久会回 (秒)。
      // 2026-10-10: 调整为短期对话锚点 5~120, 极少到 600。
      // 这是 split 隐含追问的 Win 维度, 含义是"等多久再发这条追问":
      //   - 5~30s:  "用户应该马上能回" (默认, 多数情况)
      //   - 30~120s: "用户在做一件短时间会完成的事" (深呼吸/喝口水/看一眼手机)
      //   - 120~600s: 仅限用户明确在做的事 (泡茶/取快递), 需 reason 解释
      //   - >600s: 不要。短期对话里长时间后再问 = 多半已经换话题, 体感出戏
      // 30 分钟 (1800) 是硬上限, 用了会被代码层截。
  "reason": "≤40字，说明是在哪里找到的话题断点，或不拆的理由",
  "necessity_p": float  // 0.0~1.0, 你自评这条追问的必要性 (按上方锚点)
}}
"""


async def _split_followup(mr_text: str) -> dict | None:
    """
    调 lint LLM 拆 MR 末尾隐含追问。失败 → return None (degrade to no split)。

    Returns:
        None = lint 失败/解析失败 → caller 走 fallback
        {"should_split": False} = lint OK 但 MR 无追问 → caller 走 fallback
        {"should_split": True, "main_body": "...", "content": "...",
         "predict_reply": "...", "predict_window_sec": 60, "reason": "..."}
                = 拆成功 → caller 改 final_answer + 落盘
    """
    if not mr_text:
        return None
    raw = await _call_lint_llm(_build_split_prompt(mr_text))
    if not raw:
        return None
    out = await _parse_json_lenient(raw)
    if out is None:
        # 改进可观测性: 之前只打前 80 字符, 看不到尾巴无法诊断。
        # 现在打全文 (LLM 输出通常 < 2KB, 不爆) + 错误定位。
        debug(
            f"[split] parse fail (len={len(raw)}): "
            f"HEAD={raw[:200]!r} TAIL={raw[-200:]!r}"
        )
        # 兜底 2: 既然 LLM 给我们的是合法意图 (should_split=true) 但 JSON 烂了,
        # 尝试用正则塌缩抽出核心字段。LLM 真实坏数据模式几乎只有一种:
        #   · string 内有未转义的 " 或 \n 让 json.loads 整体失败
        # 此时用宽松正则 + 字段裁切抢救。抢救失败才彻底放弃。
        salvaged = _salvage_split_payload(raw)
        if salvaged is not None:
            debug(f"[split] salvaged via regex: keys={list(salvaged.keys())}")
            out = salvaged
        else:
            return None
    should = bool(out.get("should_split"))
    if not should:
        debug(f"[split] skip: should_split=false reason={out.get('reason','')!r}")
        return {
            "should_split": False,
            "main_body": mr_text,
            "content": "",
            "predict_reply": "",
            "predict_window_sec": 60,
            "reason": str(out.get("reason", ""))[:80],
            "necessity_p": 0.0,  # 不拆时, 必要性 0
        }
    content = str(out.get("content", "")).strip()
    main_body = str(out.get("main_body", "")).strip()
    if not content:
        # 拆出来但 content 空 → 视为失败, 不改 final_answer
        debug(f"[split] invalid: should=True but empty content")
        return None
    if not main_body:
        # main_body 空 → 拆的太激进, 整段都是追问?
        main_body = mr_text  # fallback 用原 MR

    # 唯一后处理: 末尾标点修复。
    # 语义分割后, main_body 与 content 是独立语义单元 — 不做字符串去重。
    # 仅修一个事实问题: LLM 重写时偶会把 main_body 切断在 ", " 或 "。" 之后,
    # 留一个孤立标点 (例如 "我心里很暖, "), 视觉突兀 → 自动 rstrip + 加句号。
    if main_body and main_body[-1] in "，。、,.;；:： ":
        main_body = main_body.rstrip("，。、,.;；:： ").rstrip()
        if main_body and not main_body[-1] in "。！!?？…":
            main_body += "。"
        debug(f"[split] fix-tail-punct: ended with broken punctuation")

    # 2026-10-10: 读 LLM 自评的 necessity_p (浮点 0.0~1.0, 越高越必要)
    # 缺省 1.0 (LLM 没给 / 解析失败 → 视同"必要", 保持旧行为, 概率门控的"baseline")
    try:
        necessity_p = float(out.get("necessity_p", 1.0))
    except (TypeError, ValueError):
        necessity_p = 1.0
    necessity_p = max(0.0, min(1.0, necessity_p))
    if necessity_p != 1.0:
        debug(f"[split] LLM necessity_p={necessity_p:.2f}")

    return {
        "should_split": True,
        "main_body": main_body,
        "content": content,
        "predict_reply": str(out.get("predict_reply", ""))[:80],
        "predict_window_sec": int(out.get("predict_window_sec", 60)),
        "reason": str(out.get("reason", ""))[:80],
        "necessity_p": necessity_p,
    }


async def split_followup_node(state: dict) -> dict:
    """
    LangGraph 节点: MR 末尾隐含追问拆解。

    挂载位置: llm_call → split_followup → save_paper_memory

    【chat-only 约束】论文侧直接返回 {split_followup: None}, 不调 LLM。
    论文侧无追问语义 (层 1 不服务论文), 同步 split 同样不服务论文。
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
        result = await _split_followup(mr_text)
    except Exception as e:
        debug(f"[split] node error (swallowed): {type(e).__name__}: {e}")
        result = None

    if result is None:
        # lint 失败 → 保持 split_followup=None, 不改 final_answer
        return {"split_followup": None}

    if not result.get("should_split"):
        # 无追问 → split_followup=None, final_answer 不动
        return {"split_followup": None}

    # 拆成功 → 改 final_answer = main_body, 写 split_followup 给 layer1
    # 【v3 split-origin】拆分成功时, 把原始 mr_text (含 main_body + content) 存进
    # split_followup.origin, 供 save_paper_memory_node 在主答 ResAsk entry 上
    # 写 "origin" 字段。目的是保留"拆分前"的原始消息, 方便后续:
    #   - LLM 复习: 被拆分的主答回灌 prompt 时, 若需要还原原始语义链, 可对照
    #   - 前端调试: 开发者工具直接看到 main_body 与原 MR 的 diff
    #   - 论文分析: 统计 split 拆分稳定性 / main_body 重写质量
    # 字段命名沿用 save_paper_memory_node 的 entry 字段风格 (扁平字符串)。
    debug(
        f"[split] ok fp={req.pdf_fp[:12]} "
        f"main_body={result['main_body'][:30]!r} content={result['content'][:30]!r}"
    )
    return {
        "final_answer": result["main_body"],
        "split_followup": {
            "content": result["content"][:200],
            "predict_reply": result["predict_reply"],
            "predict_window_sec": result["predict_window_sec"],
            "intent": result["reason"],
            "from_split": True,
            "origin": mr_text,   # v3: 原始 MR 全文 (含 main_body + content)
            "necessity_p": result["necessity_p"],  # v3 10-10: LLM 自评必要性
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Fire-and-forget Pipeline (§10)
# ═══════════════════════════════════════════════════════════════════════


def _assert_chat_fp(pdf_fp: str, caller: str) -> bool:
    """
    主动功能 chat-only guard。

    所有主动机制 (Layer1 / 未来的 Layer2 / Layer3) 入口必先调此函数,
    论文侧返回 False, 调用方必须立即放弃。

    Args:
        pdf_fp: state["req"].pdf_fp 或 on_user_msg 传入的 fp
        caller: 调试用 caller 名 (例 "[layer1]"), 出错时写 debug log

    Returns:
        True = chat 侧, 可继续
        False = 论文侧, 调用方必须 return
    """
    if pdf_fp != CHAT_FP:
        # 用 debug 不是 error — 论文侧调过来是合规行为 (节点拓扑保证),
        # 不该每次都打 ERROR 噪音。仅排查时 verbose 用。
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════
# 思考中窗口 (thinking window) — 主答 LLM 跑完前, 主动追问延后发出
# ═══════════════════════════════════════════════════════════════════════


def set_thinking_window(duration_ms: int = _DEFAULT_THINKING_WINDOW_MS) -> int:
    """
    设置"思考中"窗口截止时间 (now + duration_ms)。

    调用方: on_user_msg (用户发消息 → LLM 即将跑 → 拉高窗口)。
    重复调用取较大值, 避免被短窗口覆盖 (用户连续发消息时只取最长的那一次)。
    """
    new_until = now_ms() + int(duration_ms)
    with _THINKING_LOCK:
        global _thinking_until_ms
        if new_until > _thinking_until_ms:
            _thinking_until_ms = new_until
            debug(f"[thinking] window SET until={format_dt_second(new_until)} (+{duration_ms}ms)")
    return _thinking_until_ms


def clear_thinking_window() -> None:
    """
    立即清空"思考中"窗口 (主答已落盘 + AI 气泡已推时调)。

    调用方: layer1_node (LangGraph 节点返回 {} 之前), 此时 _run_layer1_pipeline
    即将开始排 scheduled, 不应该被 thinking 窗口继续打压。
    """
    global _thinking_until_ms
    with _THINKING_LOCK:
        if _thinking_until_ms > 0:
            debug(f"[thinking] window CLEAR (was until={format_dt_second(_thinking_until_ms)})")
        _thinking_until_ms = 0


def is_in_thinking_window() -> tuple[bool, int]:
    """
    查询当前是否在思考中窗口内。

    Returns:
        (in_window, until_ms):
          · in_window: True = 还在窗口内 (now < until), 主动追问应延后
          · until_ms:  当前截止时间戳 (供 _schedule_followup 算"延后多久")
    """
    now = now_ms()
    with _THINKING_LOCK:
        until = _thinking_until_ms
    return (now < until, until)


def thinking_delay_ms() -> int:
    """
    返回当前还需要延后多少 ms (max(0, until - now))。
    _schedule_followup 把这个值叠加到 win 上, 实现"延后到 thinking 结束再发"。
    """
    _, until = is_in_thinking_window()
    if until <= 0:
        return 0
    return max(0, until - now_ms())


# ═══════════════════════════════════════════════════════════════════════
# post-cancel 队列 — on_user_msg 入队, _run_layer1_pipeline path B 出队
# ═══════════════════════════════════════════════════════════════════════


def push_post_cancel(fp: str, user_content: str, ts_ms: float) -> None:
    """
    on_user_msg 调用: 把"用户最新回复"入队, 供 layer2 路径 B 消费。

    设计:
      · 同一 fp 短时间内多次 push: 累加 list, 让 layer2 看到全部 (理论上
        不会发生, 因为 on_user_msg 是同步处理, 但保留 append 语义)
      · ts_ms ≤ 0: 视为 now_ms() 兜底
    """
    ts = ts_ms if ts_ms > 0 else float(now_ms())
    with _POST_CANCEL_LOCK:
        bucket = _pending_post_cancel.setdefault(fp, [])
        bucket.append((user_content, ts))
    debug(
        f"[post-cancel] push fp={fp[:12]} content={user_content[:30]!r} "
        f"ts={format_dt_second(int(ts))} queue_len={len(_pending_post_cancel.get(fp, []))}"
    )


def pop_post_cancel(fp: str) -> list[tuple[str, float]]:
    """
    _run_layer1_pipeline 路径 B 调用: 拉出并清空该 fp 的所有 post-cancel 记录。

    返回: 该 fp 累积的 (user_content, ts_ms) 列表 (按入队顺序)。
    找不到: 返回空列表。
    """
    with _POST_CANCEL_LOCK:
        bucket = _pending_post_cancel.pop(fp, [])
    if bucket:
        debug(f"[post-cancel] pop fp={fp[:12]} got {len(bucket)} entries")
    return bucket


def _snapshot_state_for_layer1(state: dict) -> dict:
    """
    复制 layer1 真正用到的字段, 避免后台 task 持有 state 引用读到
    LangGraph 后续节点改写后的 state (save_paper_memory / emotion_llm
    都可能改 messages / final_answer)。

    【新增】包含 split_followup 字段, 若 split_followup_node 已拆出, layer1 走
    split 优先路径, 跳过 judge + compose。
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


async def _parse_json_lenient(raw: str) -> dict | None:
    """
    容错 JSON 解析 (LLM 输出常带尾巴或包裹)。

    LLM 真实输出常见模式:
      1. ```json ... ``` markdown fence → 剥掉
      2. JSON 之后追加解释/换行/注释 → "Extra data" json.JSONDecodeError
      3. 多个 JSON 拼接 (LLM 重复生成) → 取第一个完整对象
      4. JSON 之前有 prose (LLM 先说一句话再写 JSON) → 取第一个 '{' 起
      5. JSON 之前有 markdown fence ``` 但 raw 不以 ``` 起 → rfind '{' 起
      6. JSON 内有非标字符 (LLM 偶尔写错) → 跳过非标位置

    策略 (按成功率从高到低, 4 轮尝试):
      A. 剥 fence + s[0]==='{' 时, json.loads 一次 — 成功则返 (80% case)
      B. 失败 → 配对 '}' 截取第一个完整 JSON object, 再 json.loads
      C. 仍失败 → 跳到下一个 '{' 用 raw_decode 一次性截
      D. 仍未果 → 正则粗找 '{"...":...}' 样式, 二次尝试

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
    #    LLM 输出形如 "{...}{...}" 也能取第一个; 形如 "prose\n{...}\ntail" 也能取
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
                    # 配对截取后还解析失败 → 字符串内可能有非标 '}' 干扰
                    # 跳到下一个 '}' 位置继续尝试
                    continue
    # 4. 策略 C: 整段都配对不上 → 用 raw_decode 一次性截
    try:
        obj, _ = json.JSONDecoder().raw_decode(s)
        return obj
    except (json.JSONDecodeError, ValueError):
        return None


def _salvage_split_payload(raw: str) -> dict | None:
    """
    LLM 输出 JSON 烂掉时的塌缩兜底: 用宽松正则抽出 4 个核心字段。

    只在 [split] 路径使用, 通用 _parse_json_lenient 不调用。

    设计取舍:
      · 不尝试重建完整 JSON, 只挖 (should_split, main_body, content, predict_reply)
      · string 内的未转义 " 是塌缩唯一能解的问题 (因为我们用非贪婪匹配 + 启发式边界)
      · 必须 4 字段全部命中才算成功, 缺一返 None
      · 启发式边界:
          - should_split: 取 "should_split" 后的第一个 true/false
          - main_body/content/predict_reply: 取对应 key 后的第一个 JSON-style string
            启发式: 从第一个非空字符起, 跳到下一个 " (含反斜杠转义计数), 或到行尾 / 下一个 key

    成功率估计: 在 80% 的 "string 内未转义引号" case 上能救回 (因为多数情况
    string 边界正好是换行或下一个 key)。
    """
    import re

    if not raw:
        return None

    # 1. should_split: 简单, 必能命中
    m_should = re.search(r'"should_split"\s*:\s*(true|false)', raw, re.IGNORECASE)
    if not m_should:
        return None
    should_split = m_should.group(1).lower() == "true"

    # 2. main_body / content / predict_reply: 用启发式 string 抽取
    def _extract_string_field(key: str) -> str | None:
        # 找 "key": " 起始位置 (允许任意空白)
        m = re.search(rf'"{key}"\s*:\s*"', raw)
        if not m:
            return None
        i = m.end()  # 已到 string 内容起点 (跳过 ")
        out_chars: list[str] = []
        while i < len(raw):
            ch = raw[i]
            if ch == "\\":
                # 处理转义
                if i + 1 < len(raw):
                    nxt = raw[i + 1]
                    if nxt == "n":
                        out_chars.append("\n")
                    elif nxt == "t":
                        out_chars.append("\t")
                    elif nxt == "r":
                        out_chars.append("\r")
                    elif nxt == '"':
                        out_chars.append('"')
                    elif nxt == "\\":
                        out_chars.append("\\")
                    elif nxt == "/":
                        out_chars.append("/")
                    else:
                        out_chars.append(nxt)
                    i += 2
                    continue
            if ch == '"':
                # string 结束。但坏 case 下可能是未转义的误闭合,
                # 此时后面不是 ',', '}', '\n' 或 EOF, 说明是坏 string, 继续吃。
                if i + 1 < len(raw) and raw[i + 1] in ",\n\r \t}":
                    return "".join(out_chars)
                # 坏 case: 误闭合, 把这个 " 当成 string 内容继续
                out_chars.append(ch)
                i += 1
                continue
            out_chars.append(ch)
            i += 1
        return "".join(out_chars) if out_chars else None

    main_body = _extract_string_field("main_body")
    content = _extract_string_field("content")
    predict_reply = _extract_string_field("predict_reply") or ""

    # 必须 main_body 和 content 都拿到
    if main_body is None or content is None:
        return None

    return {
        "should_split": should_split,
        "main_body": main_body.strip(),
        "content": content.strip(),
        "predict_reply": predict_reply.strip()[:80],
        # 2026-10-10: 塌缩时也抽 necessity_p (浮点比 string 好抽, 正则简单)
        # 缺省 1.0, 与 _split_followup 缺省语义一致
        "necessity_p": _extract_float_field(raw, "necessity_p") or 1.0,
    }


def _extract_float_field(raw: str, key: str) -> float | None:
    """
    2026-10-10 新增: 从脏 JSON 里抽浮点字段, 用于 _salvage_split_payload。
    只用于塌缩路径, 通用 _parse_json_lenient 不调用。
    """
    import re
    m = re.search(rf'"{key}"\s*:\s*(-?\d+(?:\.\d+)?)', raw)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


async def _run_layer1_pipeline(snapshot: dict) -> None:
    """
    真正的 layer1 流程。**异步** 在 sync state 调用返回之后跑。

    【chat-only 约束】论文侧 (_assert_chat_fp=False) 直接 return, 不调 LLM。

    【v3 改造】两条独立路径, 互不影响, 都跑都生效:
      · 路径 A: split 路径 (v2 既有) — split_followup_node 拆出即落盘
      · 路径 B: layer2 概率追问 (本期新增) — _run_followup_pipeline 迭代乘积
    即使 split 已经发了一条, layer2 仍会跑, 由 LLM 的概率衰减决定要不要继续追。
    两条路径的落盘顺序由 _emit_followup 内部 500-1800ms 随机延迟撑开。

    异常一律静默吞掉 (不影响主回复), debug 打印。
    """
    req = snapshot.get("req")
    fp = snapshot.get("fp", "")
    if not isinstance(req, AiAskReq):
        return
    if not _assert_chat_fp(fp, "[layer1_pipeline]"):
        return  # 论文侧 — layer1 不服务

    try:
# ═══ 路径 A: split 唯一路径 (v2 既有) ═══
        # split_followup_node 在 llm_call 后同步拆出, 若拆出, 进入概率门控
        # 疲劳硬停止: 已禁用 (2026-10-10)
        #   旧: value ≥ FATIGUE_HARD_STOP (0.80) 时 split 路径也直接跳过
        #   新: 移除硬停, 改为 _should_emit_split() 概率门控:
        #       P_split = _fatigue_discount_factor(); P ≥ 0.3 必发, < 0.3 摇骰。
        #   理由: 硬停是抢答, 用户体感是"突然不问了", 与疲劳语气的渐进调整割裂;
        #         概率门控 + 语气档位 + LLM 自判三层渐进衰减, 比 hard stop 自然;
        #         split 与 layer2 现在行为对称: 都受 _fatigue_discount_factor 调制。
        split_fu = snapshot.get("split_followup")
        split_emitted = False  # 标记是否实际发, 透传给 layer2 决定是否 seed history
        # 从 split_fu 里读 necessity_p, 缺省 1.0 (历史数据 / 缺字段都视同"必要")
        try:
            necessity_p = float(split_fu.get("necessity_p", 1.0)) if isinstance(split_fu, dict) else 1.0
        except (TypeError, ValueError):
            necessity_p = 1.0
        necessity_p = max(0.0, min(1.0, necessity_p))  # clamp
        if (
            split_fu
            and isinstance(split_fu, dict)
            and split_fu.get("content")
            and _should_emit_split(necessity_p)
        ):
            # 2026-10-10: 改 emit 为 schedule (gate+win 改造)
            # split 路径 LLM 已经在 _build_split_prompt 里输出 predict_window_sec,
            # 直接用, 不再强制"立即发"。win ≤ 5s 时 _schedule_followup 内部会走
            # 立即 emit, 行为与改造前一致。
            await _schedule_followup(
                fp=fp,
                content=split_fu["content"],
                predict_reply=split_fu.get("predict_reply", ""),
                predict_window_sec=split_fu.get("predict_window_sec", 60),
                intent=split_fu.get("intent", "[split]"),
            )
            split_emitted = True
        elif split_fu and isinstance(split_fu, dict) and split_fu.get("content"):
            # 拆出来了但被概率门控吞掉: silent, 不发, 不加疲劳
            d = _fatigue_discount_factor()
            p = necessity_p * d
            debug(
                f"[split] SKIP: probability gate "
                f"(necessity_p={necessity_p:.2f} x discount={d:.2f} = {p:.3f}, "
                f"value={_get_fatigue_value():.2f})"
            )

# ═══ 路径 B: layer2 概率追问 (本期新增, 互不影响) ═══
        # 不论 split 是否拆出, 都独立跑概率追问; LLM 看到 history 自行决定 P_continue。
        # 2026-10-10: 把 split_emitted 透传, 让 layer2 知道 split 是否真的发出过,
        #   没发出时不 seed history (避免 LLM 以为自己已经追过, 跳过同一话题)。

        # 【v3 2026-10-10 post-cancel 串行化】拉出 on_user_msg 入队的 post-cancel,
        # 等 split 路径的 emit 延迟跑完, 再启动 layer2。
        # 物理上保证"split 先发 → layer2 后发", 内容上 LLM 也能看到 split 已发
        # 的追问和用户最新回复, 不会和 split 重复。
        post_cancel = pop_post_cancel(fp)
        if post_cancel:
            # 1. 等 split 的 emit 延迟(1500-3000ms)+ WS push 完成, 再启动 layer2。
            #    最坏情况 split emit 没卡顿(立即), 这里 sleep 仍能保证 split 在
            #    用户视觉上出现后再跑 layer2 的 LLM(本身 1-2s)→ 实际追问答在
            #    主答后 3-5s 出现, 节奏自然。
            if split_emitted:
                await asyncio.sleep(_SPLIT_EMIT_MIN_MS / 1000.0)
            # 2. 把 post-cancel 列表注入 snapshot, layer2 据此重排 history
            snapshot = {
                **snapshot,
                "post_cancel": post_cancel,
            }

        await _run_followup_pipeline({**snapshot, "split_emitted": split_emitted})
    except Exception as e:
        debug(f"[layer1] pipeline error (swallowed): fp={fp[:12]} {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Layer2 概率追问 — 迭代乘积版 (v1)
# ═══════════════════════════════════════════════════════════════════════
#
# 设计核心:
#   · 每次迭代调一次 LLM, 同调输出 (should_continue, content, P_continue, reason)
#   · 累计通过率 R = product(P_continue[1..n]), 初始 1.0
#   · 终止条件 (任一):
#       (1) LLM 输出 should_continue=false 或 content=SKIP
#       (2) random() >= R (累计概率被拒绝)
#       (3) n 达到 5 (防死循环兜底, 不暴露为产品参数)
#   · 不设 P 参考区间, 不写时段约束, 不写固定间隔
#   · 终止的克制性完全靠 LLM 看到 history 后自我判断
#   · memory / self / explore 全文注入, 不截断 (因为不是按时间顺序写的)
#
# 复用既有资产 (不重写):
#   · _emit_followup: source="followup" 自动走 compose 延迟 (500-1500ms)
#   · _pending / on_user_msg / _judge_hit / _learn_batch: 追问落盘后自然进入反馈回路
#   · _schedule_explore_update_async: 累计 10 个 LearnSample 触发 explore 重写
#   · _call_lint_llm: 同调 LLM 调用 (复用 emotion LLM 配置)
#   · _parse_json_lenient: 容错 JSON 解析
#   · _assert_chat_fp: 论文侧拦截
# ───────────────────────────────────────────────────────────────────────

# 极保守兜底: 防 LLM 死给 1.0 时跑飞。无产品含义, 仅保护进程。
_FOLLOWUP_SOFT_MAX_ITER = 5


# 2026-10-10: Win 边界常量 (gate+win 改造)
#   · WIN_HARD_MIN=5: ≤5s 视为"立即发", 不进 scheduler (避免 sleep 比 emit 延迟还短)
#   · WIN_HARD_MAX=1800: 30min 上限, 再长 LLM 就是在凑数, 用户早就忘了
#   · 软锚 5~60s 是 prompt 给 LLM 的典型值, 硬限是防 LLM 飞
WIN_HARD_MIN = 5
WIN_HARD_MAX = 1800

# 【v3 2026-10-10】_run_layer1_pipeline 在 split 路径跑完后, 等这么久再启动
# layer2。值取 _emit_followup 内部 split 路径的"最小延迟" (1500ms),
# 给 split 的 emit/WS push 留出落地时间。物理上保证 layer2 的 LLM 不会在
# split 气泡出现前就跑完, 也就不会撞车。
_SPLIT_EMIT_MIN_MS = 1500


def _clamp_win(llm_value: int) -> int:
    """
    把 LLM 给的 predict_window_sec clamp 到 [WIN_HARD_MIN, WIN_HARD_MAX]。
    解析失败/类型错/缺省 → 走 60s 默认 (与原 split_fu.predict_window_sec 缺省一致)。
    """
    try:
        v = int(llm_value)
    except (TypeError, ValueError):
        return 60
    return max(WIN_HARD_MIN, min(WIN_HARD_MAX, v))


def _cancel_scheduled_for_fp(pdf_fp: str, reason: str) -> int:
    """
    Cancel 同 fp 所有 scheduled。返回取消个数, 供 debug 观测。

    实现: 给每个未完成的 task 发 cancel 信号, worker 会在下个 sleep tick 抛
    CancelledError 静默 return。cancelled 软标记置 True 防重入。
    """
    cancelled = 0
    with _SCHEDULED_LOCK:
        targets = [s for s in _scheduled if s.pdf_fp == pdf_fp and not s.task.done()]
        for s in targets:
            s.cancelled = True
            if s.task is not None:
                s.task.cancel()
            cancelled += 1
    if cancelled:
        debug(
            f"[scheduled] CANCEL x{cancelled} fp={pdf_fp[:12]} reason={reason!r}"
        )
    return cancelled


# 2026-10-10: 移除 _fatigue_should_hard_stop 与 FATIGUE_HARD_STOP 整套硬停机制。
#   旧: fatigue ≥ 0.80 时 split + layer2 followup 路径直接跳过
#   新: 疲劳只通过 _fatigue_discount_factor (R *= P × discount) 自然衰减,
#       再叠加 build_fatigue_context_block 的语气档位注入, 不再硬停。
#   理由: 硬停是"突然不问", 折扣是"渐进变少", 后者对用户更自然;
#         LLM 自身在疲劳时也会判 P_continue 低, 多层兜底会互相打架。


# 2026-10-10: split 路径加概率调制 (与 layer2 路径行为对称)
#   · 旧 (v1): P_split 由 _fatigue_discount_factor 派生, 纯公式
#   · 旧 (v2): 硬停 (已删)
#   · 新 (v3): P_split = necessity_p × fatigue_discount_factor
#       · necessity_p: split LLM 在拆分时同时输出的"必要性"判断 (浮点 0.0-1.0)。
#         来源: _build_split_prompt 加 prompt 段要求 LLM 自评"这条追问是否真的有必要",
#         LLM 觉得追问很生硬 / 强拆 / 用户其实不需要时给低分, 自然追问给高分。
#         缺省 1.0 (LLM 没给, 视为必要)。
#       · fatigue_discount_factor: 疲劳折扣 (0.4~1.0)
#       · 二者乘积即 P_split, random < P_split 才发
#   · 理由: 必要性是 LLM 自己的判断, 比公式派生的"看起来该问"更准; 疲劳只做调制器。
#     之前"v=0.5 必发"等公式阈值, 本质是把"必要性"硬编码成"疲劳档位", 拆出就不拆了;
#     改 v3 后, LLM 可以主动表态"这个追问不太自然", 即使疲劳低也跳过, 反之亦然。
#   · 互补: split 偶尔跳过时, layer2 仍会独立跑; 反过来 layer2 跳过时, split 也仍可发。
def _should_emit_split(necessity_p: float) -> bool:
    """
    2026-10-10 v3: split 路径的概率门控 (LLM 自评 × 疲劳折扣)。

    Args:
        necessity_p: split LLM 输出的必要性 (0.0~1.0), 缺省 1.0。

    Returns:
        True = 发第二条气泡, False = silent (final_answer 已只剩 main_body)。
    """
    import random as _random
    p = float(necessity_p) * _fatigue_discount_factor()
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return _random.random() < p


def _fatigue_discount_factor() -> float:
    """
    2026-10-10 v3: 疲劳乘性折扣系数, 全 v 平滑, 移除 0.30 突变点。
    2026-10-10 v4: 同步 SLOPE 调整说明 (常量保留, 值改了)。

    旧 (v2, 硬阈值):
        v <= 0.30 → 1.0  (硬阈值, 突变)
        v >  0.30 → 1 - 0.60 * (v - 0.30)

    新 (v3, 全 v 平滑):
        discount = 1 - FATIGUE_DISCOUNT_SLOPE * v   # SLOPE = 0.42
        v=0 → 1.0, v=0.30 → 0.874, v=1.0 → 0.58, 全范围连续, 无突变。
    SLOPE 调 0.42 (不是 0.60) 是为了保持 v=0.30 时的折扣与 v2 完全一致 (0.88)。

    返回: [0.0, 1.0]
    """
    from .ai_config import FATIGUE_DISCOUNT_SLOPE
    v = _get_fatigue_value()
    factor = 1.0 - FATIGUE_DISCOUNT_SLOPE * v
    return max(0.0, min(1.0, factor))


# ── 疲劳语气档位表 (2026-10-10 新) ──
# 把 fatigue 0~1 切成 4 档, 注入到 layer2 决策 prompt 顶部,
# 让 LLM 知道"这是个系统级资源, 不只是情绪四元组里的一个数字"。
# 档位是语气/字数/可问性的综合指导, 不是硬阈值 (硬阈值由 _fatigue_discount_factor 负责)。
_FATIGUE_TIER_PROMPT = [
    # (min, max, label, instruction)
    (0.00, 0.30, "轻松", "对话刚刚开始, Crystal 精神充沛, 自由发挥, 正常语气正常长度, 完全可以追问。"),
    (0.30, 0.55, "略累", "Crystal 已经聊了一阵子, 语气开始收一点, 追问可以短一些 (≤40字), 主动性降一档, 不是每条都问。"),
    (0.55, 0.75, "明显累", "Crystal 已经累了, 语气安静、停顿感, 主动追问要克制: 除非有真新角度, 否则给 P_continue ≤ 0.20 甚至直接 SKIP。"),
    (0.75, 1.01, "疲惫", "Crystal 已经很疲惫, 几乎不主动开口, 语气词少、句短、最多一句轻声关心 (≤20字), 默认 SKIP, P_continue ≤ 0.08。"),
]


def _build_fatigue_block(fatigue_value: float) -> str:
    """
    生成"对话疲劳度"专用块, 注入 layer2 决策 prompt, 让 LLM 在判定 should_continue
    和写 content 时, 主动把"crystal 现在累不累"考虑进去。

    关键: 这块必须**先于** emotion 四元组, 单独成段, 否则 LLM 会把它当成
    "情绪向量的一个分量"读, 意识不到这是"系统资源/对话时长"的信号。

    v2 2026-10-10 升级: 数字 + 档位文字 + 行为指引三件套, 抛弃"只丢一个浮点数"
    的旧风格。LLM 看到 "疲劳 0.42 / 略累档" 比看到 "fatigue=0.42" 决策精度高一个量级。
    """
    v = max(0.0, min(1.0, float(fatigue_value)))
    # 找到当前档位
    tier_label = ""
    tier_instr = ""
    for lo, hi, lbl, inst in _FATIGUE_TIER_PROMPT:
        if lo <= v < hi:
            tier_label = lbl
            tier_instr = inst
            break
    else:
        # 兜底: v >= 1.0
        tier_label = "疲惫"
        tier_instr = _FATIGUE_TIER_PROMPT[-1][3]

    return (
        f"【对话疲劳度: {v:.2f} / 1.00 — 档位: {tier_label}】\n"
        "  · 含义: Crystal 与用户累计对话的『系统疲劳值』, 0=精力充沛, 1=完全累垮。\n"
        "  · 跟 emotion 情绪向量无关 — 这不是『心情』, 这是『还能聊多久』的资源条。\n"
        f"  · 当前档位行为指引: {tier_instr}\n"
        "  · 硬性提示: 当前档位越高, should_continue 默认越倾向 false, P_continue 默认越低;\n"
        "    反过来, 档位低时 LLM 不要因为『crystal 累了』的理由随便 SKIP。\n"
    )


# ── 情绪 4 维 → 文字档位映射 (2026-10-10 新, 配套疲劳档位, 一起做"数字 → 文字"升级) ──
# 之前直接喂 "valence=0.20 arousal=0.04 novelty=-0.07 clarity=0.95",
# LLM 把它当噪声跳过的概率很高, 而且也判断不出 "0.20 该对应啥语气"。
# 现在切成文字档位 + 数字参考, LLM 可以直接用档位做决策。

def _emo_tier(v: float) -> str:
    """把 [-1, 1] 浮点切成 5 档文字标签"""
    if v <= -0.5:
        return "明显负面"
    if v <= -0.1:
        return "偏负面"
    if v < 0.1:
        return "中性"
    if v < 0.5:
        return "偏正面"
    return "明显正面"


def _build_emotion_block(valence: float, arousal: float, novelty: float, clarity: float) -> str:
    """
    把 4 维情绪向量转成"档位文字 + 数字 + 行为含义"块。
    注入到 layer2 决策 prompt, 让 LLM 真正能"读懂"情绪。
    """
    v_lab = _emo_tier(valence)
    a_lab = _emo_tier(arousal)
    n_lab = _emo_tier(novelty)
    c_lab = _emo_tier(clarity)

    # 简单行为映射: 告诉 LLM 看到这些档位, 追问语气该往哪个方向走
    v_instr = {
        "明显负面": "用户处于低谷, 追问要极度克制或直接 SKIP, 真要问只能给最轻的一句『我在』。",
        "偏负面":   "用户有点低落, 追问要稳, 不要带笑, 不要信息追问, 多给陪伴。",
        "中性":     "用户情绪平稳, 正常节奏, 按本轮氛围 + 疲劳判断即可。",
        "偏正面":   "用户心情不错, 语气可以柔和带温度, 追问可以稍微主动。",
        "明显正面": "用户很开心, 语气可以明亮一点, 但不要喧宾夺主, 让用户继续发光。",
    }[v_lab]
    a_instr = {
        "明显负面": "用户极压抑, 追问要慢、轻、留白。",
        "偏负面":   "用户能量低, 追问节奏慢。",
        "中性":     "用户节奏平稳。",
        "偏正面":   "用户有点兴奋, 跟着用户的节奏, 但不要比用户更激动。",
        "明显正面": "用户高能量, 别泼冷水也别过度附和, 同频即可。",
    }[a_lab]
    n_instr = {
        "明显负面": "用户觉得无聊/重复, 别再绕同样话题, 切新角度或 SKIP。",
        "偏负面":   "内容有点重复, 给新信息或新角度。",
        "中性":     "内容新鲜度一般, 按正常流程。",
        "偏正面":   "有新鲜感, 可以顺着深一点。",
        "明显正面": "用户对当前话题很兴奋, 可以再推一层。",
    }[n_lab]
    c_instr = {
        "明显负面": "用户头脑很糊, 别问复杂问题, 一句轻声最合适。",
        "偏负面":   "用户思路有点乱, 追问要简短明确。",
        "中性":     "用户思路正常, 按本轮氛围。",
        "偏正面":   "用户思路清晰, 可以聊得有结构。",
        "明显正面": "用户思路非常清晰, 可以一起分析问题。",
    }[c_lab]

    return (
        "【Crystal 当前情绪 — 4 维向量 (档位文字 + 数字参考)】\n"
        f"  · valence (心境正负): {valence:+.2f} → {v_lab}\n"
        f"      含义/影响: {v_instr}\n"
        f"  · arousal (能量高低): {arousal:+.2f} → {a_lab}\n"
        f"      含义/影响: {a_instr}\n"
        f"  · novelty (新鲜度):   {novelty:+.2f} → {n_lab}\n"
        f"      含义/影响: {n_instr}\n"
        f"  · clarity (思路清晰): {clarity:+.2f} → {c_lab}\n"
        f"      含义/影响: {c_instr}\n"
    )


_FOLLOWUP_ITERATION_PROMPT = """你正在决定 Crystal 是否要在本轮对话末尾, 主动向用户追问。

【当前是第 {n} 次决策】

【本轮已发出的追问】(第 1 次时为空)
{history_block}

{user_recent_block}
{forbidden_block}
【Follow 的本质 — 重新定义 (2026-10-10 v2)】
Follow 不是"接住用户的话往下问", 也不是"延伸当前对话" ——
Follow 是【以本轮主答 (final_answer) 或 split 追问 为锚, 替用户多想一步,
抛出一个**用户大概率会感兴趣、但自己还没说出来**的新话题】。

核心思路 — "读者视角 + 关联联想":
  1. 读完主答/split, 你 (LLM) 作为一个【很了解这个用户的亲密伙伴】, 会想到什么?
  2. 那个"想到的东西"如果抛出来, 用户会不会觉得"诶我也想聊这个 / 哦对, 我都没想到"?
  3. 那就是 follow 应该写的内容。

  · 好的 follow 例子 (从主答/split 自然长出来):
    - 主答"今天能撑过去, 你自己先有的那股劲儿" → follow "那股劲儿, 你自己怎么发现的?"
      (从"自己先有"这个洞察延伸)
    - split "下次也可以先把那股劲儿写下来" → follow "写下来的时候, 你通常会先写哪一句?"
      (从"写下来"这个动作展开)
    - 主答"算法练习 + 汇报进度 是最舒服的时候" → follow "你练习算法的时候, 哪类题会特别有劲?"
      (从"练习算法"这个细节切出)
  · 不好的 follow 例子 (跟主答/split 没关系 / 套话):
    - "今天感觉怎么样?"  → ❌ 客服式, 跟主答没关联
    - "你有没有想说的?"   → ❌ 空, 没从主答里抽
    - 复述主答里的话再换词问一遍 → ❌ 重复

  · 怎么"长出"新话题 (3 条思维路径, 至少满足一条):
    (A) 【细节放大】主答/split 提了一个具体物/动作/场景, 把它放大成小话题
        (例: "那股劲儿" → "那股劲儿具体长什么样?")
    (B) 【后续推演】主答/split 描述了一个状态/事件, 自然的后续会是什么?
        (例: "撑过去了" → "撑过去之后你通常会做什么?")
    (C) 【关联联想】主答/split 的话题, 在 memory/self 里有没有相关但**还没聊过**的细节?
        (例: 主答聊"练算法", memory 里有"她之前提过喜欢边听白噪音边写" →
         follow "你写算法的时候, 一般会听什么?")

  · 严禁的写法:
    - 跟主答/split **完全没有可追溯的关联** (用户会觉得"你从哪冒出来的")
    - 把主答/split 复述一遍再换个问号 (重复)
    - 套话开场白 (今天感觉怎么样 / 有没有不舒服 / 在吗 / 吃了吗 / 睡了吗)
    - 问诊式 (这个跟你身体有关吗 / 你是不是因为 X 才 Y)

  · 找不到"能长出来的新话题" → 默认 SKIP 或给 P_continue ≤ 0.08
    强行编一个 = 假聪明, 用户一下就能看出来

【情绪轴一致性约束 — 写 content 之前必读】
你的追问必须落在【与本轮主答/用户最新 post-cancel 同一情绪轴上】, 严禁"突然换轴":
  · 本轮氛围 = 陪伴/感激/亲密/肯定  → 追问也必须是陪伴/感激/亲密/肯定轴
    例: post-cancel 是"谢谢你陪着我", 你又写"今天情绪还好吗" → ❌ 从陪伴轴突然跳到问诊轴
  · 本轮氛围 = 信息追问/讨论/解释     → 追问可以是同话题追问或新角度
  · 本轮氛围 = 用户刚道别/收束        → 默认 SKIP, 不要硬追
  · 严禁以下"客服模板开场白":
    - "今天/此刻 感觉怎么样"
    - "有没有哪里不舒服"
    - "在吗" / "吃了吗" / "睡了吗"
    这类"关心式问候"在没有新角度时 = 套话, 体感很差
  · 真正的新角度 = 同轴上的切角, 例如:
    - 陪伴轴上: "下次你也可以先把那股劲儿写下来, 我会认真看"
    - 肯定轴上: "那股撑过去的劲儿, 是你自己先有的, 我只是在这儿"
    才是"同轴新方向", 才给 P_continue 0.3-0.6
  · 没有新角度 → 直接 SKIP 或给 P_continue ≤ 0.08

【本次对话的"时间定位"】(用户上次说话距今多久)
{time_gap_str}

【近期对话上下文】(最近 5 轮, 已按时间正序排列)
{context}

{fatigue_block}
{emotion_block}
【Crystal 对用户的记忆】(全文, 未截断)
{memory_full}

【Crystal 对自己的认知】(全文, 未截断)
{self_full}

【Crystal 主动外呼档案】(全文, 未截断)
{explore_full}

【你的任务】
综合以上所有信息, 自行判断:
1. 要不要继续追问? (should_continue: bool)
2. 如果要追问, 内容是什么?
   【重读顶部"Follow 的本质"节, 严格按那 3 条思维路径 (细节放大/后续推演/关联联想) 写】
   长度 ≤80字, 像微信里真人会自然说出的话, 不要"在吗"这种空开场。
   写之前**先在心里过一遍**: "这个话题从主答或 split 的哪句话/哪个细节长出来的?"
   答得出来 → 写; 答不出来 → SKIP (不要硬编)。
3. 给出 P_continue: 0.0-1.0, 表达"这条追问之后还值不值得再追问一次"的置信度。
   - 话题已被充分覆盖 → 给低 (例如 0.05-0.20)
   - 仍有未完结的活头 → 适当给高 (例如 0.30-0.60)
   - 你看到"已发出追问"里已经有内容时, 自然会知道再说会显得重复/烦人 → 自然降
   - 不写参考区间, 不套公式, 完全根据当前上下文判断
4. 给出 predict_window_sec: 5~1800 之间的整数, 表示"你想等多
   久再发出这条追问"。**这是 2026-10-10 新增的 Win 维度**:

   【核心约束 — 这是短期对话,不是长期陪伴】
   ┌──────────────────────────────────────────┐
   │ win 主要服务于"现在 → 几分钟内"的节奏,    │
   │ 不要想着"几小时后再问"或"明天提醒一下"。   │
   │ 短期对话场景下,长 win 几乎总是错的选择:     │
   │   · 用户多半已经换话题/去做别的事了         │
   │   · win 越长, hit 率越低 (被 cancel 概率越大)│
   │   · 用户体感"过了很久突然问一句"很出戏       │
   └──────────────────────────────────────────┘
   所以默认请控制在 5~120 秒之间 (秒级~两分钟级),
   极少数情况下 (用户明确提到"我去做个事,等会儿") 才考虑 120~600 秒,
   600 秒以上 (10 分钟+) 基本不要用, 30 分钟 (1800) 是绝对硬上限, 用了就被截。

   - 何时给小 win (5~30s): "用户应该马上能回" (对方刚说完一句没说完的话;
     用户刚做了个动作/表情, 立刻接一句最自然)
   - 何时给中 win (30~120s): "用户在做某件短时间内会完成的事" (深呼吸、
     喝口水、看一眼手机) — 这是最常见的"延迟高价值"场景
   - 何时给大 win (120~600s): 仅限用户**明确**在做一件耗时几分钟的事
     (泡茶、出门取快递), 还要附 reason 说明
   - 何时给超大 win (>600s): 不要。除非你能给出非常强的人身关联理由,
     且会通过反馈回路被 Crystal_explore 记住 — 否则 LLM 在偷懒凑数

【P_continue × predict_window_sec 联合语义 — 务必协调这两个值】
- (P 高 [0.5~0.8], win 小 [5~30s])   → "立即深追": 现在问, 后面还想问
- (P 中 [0.3~0.6], win 中 [30~120s]) → "延迟高价值": 等会儿问, 这条值得
- (P 低 [0.05~0.2], win 中~大)        → "随口提一句": 等会儿问, 大概率不再追
- (P 高, win 大)                     → 矛盾, 请重新判断:
     真要等很久就降 P (因为久后再追多半也断了);
     真值得继续就降 win (短期对话就别拖)
- (P 低, win 极小 [<5s])              → 矛盾, 请重新判断:
     真立刻问就升 P (既然值得问就别太敷衍);
     真不重要就升 win (不重要就别立刻问)
   锚点: 这两个字段是**同一个判断的两个面**, 不要各答各的。

【判断优先级 — 务必遵守】
   A. 【最近一条 ReqAsk 的实际内容】 > 【memory 里的历史相似情境】
      - memory 里的"用户睡过觉/惊恐发作/需要休息"是历史快照, 不是当前事实
      - 必须先读"近期对话上下文"里【用户最新说的那句】, 那是 ground truth
      - 如果 memory 写"用户惊恐发作后睡了"但 4 小时后用户清醒写题回来, 必须以清醒写题为准
   B. 【本次对话的"时间定位"】里的"X 小时前"是决定性事实
      - 跨小时级 gap 时, 默认假设"上次的约定/状态已过期", 除非用户在新一轮里重新提起
      - 不要沿用上次的"睡吧/醒了跟我说一声"等模板, 用户已经回来了, 不是在睡
   C. 【本轮已覆盖主题块】(顶部 ⚠️ 块) > 【近期对话上下文里的旧轮】
      - 主答 (final_answer) 和本轮 split 追问是"本轮刚刚已经说出口的内容",
        主题上和它们重复或近义 = 用户看到第二条同主题追问会觉得"你刚才不是说过了吗",
        体感非常差, 应直接 SKIP 或给 P_continue ≤ 0.10
      - 例: 主答说"知道原因就不慌了, 你挺棒的" → 你又写"知道自己焦虑的原因了吧,
        你已经很棒了" → 主题/句式双重复, 应 SKIP
      - 例: split 问"你感觉好点了吗" → 你又问"你现在感觉怎么样" → 同义改写, 应 SKIP
      - 真正新方向: 见顶部【Follow 的本质】节, 至少满足"细节放大/后续推演/关联联想"中一条,
        且与本轮主答或 split 在话题/情绪轴上**可追溯地关联** → 允许, 给 P_continue=0.3-0.6
        写不出这种关联的, 才是"同话题/同情绪延伸的重复", 应 SKIP
   D. 【用户最新 post-cancel(顶部"用户最新说的话"块) > 一切】
      - 用户在主答发出后主动说话 = 这是当前 ground truth, 优先级高于 memory / history / 情绪
      - post-cancel 含"感谢/确认/道别/再分享" → 强烈建议 SKIP 或 P_continue ≤ 0.08:
        · "谢谢你""谢谢你的陪伴""我挺过来了""晚安""我去睡了""我先走了"
        · "感觉好多了""我知道了""明白了""嗯""好的""对"
      - post-cancel 引入新信息(具体事件/具体情绪/具体想法) → 可以问"跟进"角度,
        但仍需符合"情绪轴一致性约束"和 forbidden_block
      - 严禁: 用户刚说完"谢谢你陪着我" → 追问"今天感觉怎么样" = 把亲密陪伴
        拉回问诊开场, 情绪轴断层, 用户体感非常差

【输出严格 JSON】
{{"should_continue": bool, "content": "追问正文, 或 SKIP", "P_continue": float, "predict_window_sec": int, "reason": "≤30字, 说明判断依据"}}
"""


def _short_log(s: str, n: int) -> str:
    """
    给 debug 日志用的"短但带省略号 + 原长"截断器。

    行为:
      - 长度 <= n → 返回 repr(s) (带引号, 完整显示)
      - 长度 >  n → 返回 repr(s[:n] + '…(共 K 字)') (让读日志的人一眼看出被截了)

    用例:
      _short_log("用户刚说完...", 30)
      → 长度 8 → "'用户刚说完...'"
      → 长度 80 → "'用户刚刚明确表示要吃汉堡放松,并设定了工作...(共 80 字)'"

    为什么不用 s[:n] + '...':
      因为后面接 !r / 直接拼字符串都可能再嵌引号, 一个会让人误以为原文以 '...' 结尾。
      '…(共 K 字)' 是中文省略号 + 总长, 不会被误读为内容本身。
    """
    s = str(s)
    if len(s) <= n:
        return repr(s)
    return repr(s[:n] + f"…(共 {len(s)} 字)")


def _compute_time_gap_str() -> str:
    """
    算"用户上次说话 → 现在"的时间差, 写成中文描述注入 layer2 prompt。

    目的: 让 LLM 知道 memory 里的"用户惊恐发作/午睡"是 4 小时前还是 4 天前,
    避免被过期 memory 锚定, 答非所问。

    规则:
      · < 10 分钟: "刚说完" (无 gap, 不提)
      · 10 分钟 ~ 4 小时: "X 小时前说完的" (半天内)
      · 4 ~ 24 小时: "X 小时前说完, 隔了一(几)夜"
      · 1 ~ 7 天: "X 天前"
      · > 7 天: "X 天前 (超过一周)"
      · 文件空/无 ReqAsk: "(无历史)"

    返回: 单行中文, 末尾带换行, 可直接拼进 prompt。
    """
    try:
        path = _paper_history_path(CHAT_FP)
        data = _read_json_safe(path, [])
        if not isinstance(data, list) or not data:
            return "(无历史对话)\n"

        # 从尾部倒着找最近一条 ReqAsk
        last_req_ts: int | None = None
        for entry in reversed(data):
            if isinstance(entry, dict) and entry.get("type") == "ReqAsk":
                ts = entry.get("ts")
                if isinstance(ts, (int, float)) and ts > 0:
                    last_req_ts = int(ts)
                break

        if last_req_ts is None:
            return "(无历史 ReqAsk)\n"

        delta_ms = now_ms() - last_req_ts
        if delta_ms < 0:
            delta_ms = 0
        delta_min = delta_ms / 60_000

        if delta_min < 10:
            return "用户刚说完, 本次是连续对话。\n"
        if delta_min < 240:  # < 4 小时
            hours = delta_min / 60
            return f"用户上次说话在 {hours:.1f} 小时前, 本次是回来后的第一句。\n"
        if delta_min < 1440:  # < 24 小时
            hours = delta_min / 60
            return f"用户上次说话在 {hours:.0f} 小时前, 隔了大半天/一夜。\n"
        if delta_min < 10080:  # < 7 天
            days = delta_min / 1440
            return f"用户上次说话在 {days:.1f} 天前。\n"
        days = delta_min / 1440
        return f"用户上次说话在 {days:.0f} 天前 (超过一周)。\n"
    except Exception as e:
        debug(f"[followup] time_gap 计算失败: {type(e).__name__}: {e}")
        return "(时间差计算失败, 请根据 context 里的 [ts] 自判)\n"


def _build_followup_context(snapshot: dict) -> dict:
    """
    拼追问 prompt 的 EMSC 注入段。

    【不截断】 memory / self / explore 全部用 _get_agent_*() 全文。
    因为这三份文件不是按时间顺序写的 (LLM 自由增删), 截断头部会丢最新认知,
    截断尾部会丢早期锚点, 都会失真。

    【窗口限制】 对话上下文用最近 5 轮 (这是窗口截断, 仅控制 prompt 长度,
    与 memory 全文注入是两码事)。

    【time gap】 单独算"用户上次说话距今"时长, 让 LLM 知道 memory 里的
    "惊恐发作/午睡"是几小时前的过期历史, 不被锚定 (2026-10-10 v1: 加)。

    【v3 2026-10-10 禁重复主题块】从 snapshot 读 main_answer + split_followup
    (split 可能没拆出来 / 拆了但被概率门控吞了), 显式作为
    "禁止重复追问的主题范围" 注入 prompt 顶部 (在 [近期对话上下文] 之前,
    视觉上紧贴 history_block)。LLM 必须确认新追问的"主题关键词"既不在
    main_answer 也不在 split 追问里, 才算合规。
    """
    history = snapshot.get("followup_history", []) or []
    history_block = "\n".join(f"- {h}" for h in history) or "(无)"

    # ── 时间差 (用户上次发言 → now) ──
    time_gap_str = _compute_time_gap_str()

    # ── 5 轮对话 (load_chat_messages 返回 OpenAI messages, content 头部含 [ts] label) ──
    msgs = load_chat_messages(limit=5) or []
    context_lines: list[str] = []
    for m in msgs:
        body = (m.get("content") or "").strip()
        if not body:
            continue
        role_tag = "用户" if m.get("role") == "user" else "我"
        context_lines.append(f"[{role_tag}] {body}")
    context = "\n".join(context_lines) or "(无对话历史)"

    # ── v3 禁重复主题块: 主答 + split 追问作为"已被本轮覆盖/提出的主题"清单 ──
    main_answer = (snapshot.get("final_answer") or "").strip()
    split_fu_raw = snapshot.get("split_followup") or {}
    split_fu_content = (
        str(split_fu_raw.get("content", "")).strip()
        if isinstance(split_fu_raw, dict) else ""
    )

    # ── v3 用户最新 post-cancel 块: 用户在主答后主动说的话, 优先级最高 ──
    #     不只塞 history_block 里被淹没, 这里单独抽出来顶到顶部, 强制 LLM 先看
    post_cancel = snapshot.get("post_cancel") or []
    user_recent_block = ""
    if isinstance(post_cancel, list) and post_cancel:
        lines = []
        for uc, uts in post_cancel:
            body = str(uc or "").strip()
            if not body:
                continue
            ts_str = format_dt_second(int(uts)) if uts else "?"
            lines.append(f"[{ts_str}] {body[:200]}")
        if lines:
            user_recent_block = (
                "【用户最新说的话(本轮主答发出后才说的, 时间上最晚, 权重最高)】\n"
                + "\n".join(lines)
                + "\n"
            )

    if main_answer and split_fu_content:
        forbidden_block = (
            f"""【⚠️ 本轮已覆盖主题 — 你的追问主题不得与以下任何一条重复/高度相似】
1. 本轮主答(主答正文): {main_answer[:300]}
2. 本轮 split 追问(已发出, 用户可能正在看): {split_fu_content[:200]}

判别规则:
  · 同话题/同情绪延伸 (例如主答"安慰肯定", 你又"安慰肯定") → 重复, 应 SKIP
  · 同信息追问 (例如主答说"原因", 你又问"具体什么身体原因") → 重复, 应 SKIP
  · 同时间框架内的重复 (例如 split 已经问"你感觉好点了吗", 你又问类似) → 重复, 应 SKIP
  · 真正的新方向 (新话题 / 切角 / 主动观察) → 允许, 给 P_continue=0.3-0.6
  · 没有新方向 → 强烈建议 SKIP 或给 P_continue ≤ 0.10 (自然衰减)
"""
        )
    elif main_answer:
        forbidden_block = (
            f"""【⚠️ 本轮已覆盖主题 — 你的追问主题不得与以下主答重复/高度相似】
1. 本轮主答(主答正文): {main_answer[:300]}

判别规则:
  · 同话题/同情绪延伸 → 重复, 应 SKIP
  · 同信息追问 → 重复, 应 SKIP
  · 真正的新方向 (新话题 / 切角) → 允许, 给 P_continue=0.3-0.6
  · 没有新方向 → 强烈建议 SKIP 或给 P_continue ≤ 0.10
"""
        )
    else:
        forbidden_block = ""

    # ── 情绪 (最新 4 维, _load_emotion 返回副本, 直接读) ──
    emo = _load_emotion() or {}
    valence = float(emo.get("valence", 0.0) or 0.0)
    arousal = float(emo.get("arousal", 0.0) or 0.0)
    novelty = float(emo.get("novelty", 0.0) or 0.0)
    clarity = float(emo.get("clarity", 0.0) or 0.0)

    # ── 疲劳块 (v3 2026-10-10 新) ──
    # 单独抽成独立块, 跟 emotion 区分开, 避免 LLM 把"对话疲劳"误读为"心情向量的一维"。
    fatigue_value = _get_fatigue_value()
    fatigue_block = _build_fatigue_block(fatigue_value)

    # ── 情绪块 (v2 2026-10-10 升级) ──
    # 把 4 维向量转成"档位文字 + 数字 + 行为含义"块, 替换旧的 "valence=0.20 arousal=0.04"
    # 纯数字显示, LLM 之前的判读精度很差。
    emotion_block = _build_emotion_block(valence, arousal, novelty, clarity)

    # ── memory / self / explore 全文注入 ──
    memory_full = _get_agent_memory() or "(空)"
    self_full = _get_agent_self() or "(空)"
    explore_full = _get_agent_explore() or "(空)"

    return {
        "history_block": history_block,
        "user_recent_block": user_recent_block,
        "time_gap_str": time_gap_str,
        "forbidden_block": forbidden_block,
        "fatigue_block": fatigue_block,
        "fatigue_value": fatigue_value,
        "emotion_block": emotion_block,
        "context": context,
        "v": valence,
        "a": arousal,
        "n_v": novelty,
        "c": clarity,
        "memory_full": memory_full,
        "self_full": self_full,
        "explore_full": explore_full,
    }


def _format_followup_prompt(n: int, ctx: dict) -> str:
    """把 ctx 喂进 prompt 模板, 返回最终 LLM 输入。"""
    try:
        return _FOLLOWUP_ITERATION_PROMPT.format(n=n, **ctx)
    except (KeyError, IndexError) as e:
        debug(f"[followup] prompt format error: {type(e).__name__}: {e}")
        # 兜底: 用简化版 (只保证核心字段, 不让 prompt 构造失败中断整个 pipeline)
        return (
            f"你正在决定 Crystal 是否要追问 (第 {n} 次)。\n"
            f"时间定位: {ctx.get('time_gap_str', '(无)').strip()}\n"
            f"已发出追问:\n{ctx.get('history_block', '(无)')}\n"
            f"近期对话:\n{ctx.get('context', '(无)')}\n"
            f"情绪: v={ctx.get('v', 0):.2f} a={ctx.get('a', 0):.2f} "
            f"n={ctx.get('n_v', 0):.2f} c={ctx.get('c', 0):.2f}\n"
            f"判断优先级: 读最新 ReqAsk 的实际内容, memory 是历史不是当前。\n"
            f"输出严格 JSON: {{\"should_continue\": bool, \"content\": \"...\", "
            f"\"P_continue\": float, \"reason\": \"...\"}}"
        )


async def _call_followup_iteration(n: int, ctx: dict) -> dict | None:
    """
    调一次 LLM, 同调拿到 (should_continue, content, P_continue, predict_window_sec, reason)。

    复用 _call_lint_llm (走 EMOTION_LLM_CONFIG, 缺省时返 None 跳过)。
    解析失败 (LLM 输出非 JSON) → return None, 上层 break。

    2026-10-10: 增 predict_window_sec 字段 (gate+win 改造)。
      · 缺省 → 60 (与 split_fu.predict_window_sec 缺省一致)
      · 解析/类型错 → 60
      · 上层 _run_followup_pipeline 调 _schedule_followup 时再 _clamp_win
        到 [5, 1800], 这里不做硬限 (LLM 给超界的会走 clamp 兜底)。
    """
    prompt = _format_followup_prompt(n, ctx)
    raw = await _call_lint_llm(prompt)
    if not raw:
        debug(f"[followup] iter {n}: LLM returned empty, STOP")
        return None

    parsed = await _parse_json_lenient(raw)
    if not isinstance(parsed, dict):
        debug(f"[followup] iter {n}: parse fail, raw={raw[:80]!r}, STOP")
        return None

    # 字段容忍: P_continue 缺省视为 0 (直接 break)
    if "P_continue" not in parsed:
        parsed["P_continue"] = 0.0
    # 字段容忍: predict_window_sec 缺省 → 60 (与 _schedule_followup 默认一致)
    if "predict_window_sec" not in parsed:
        parsed["predict_window_sec"] = 60
    return parsed


async def _run_followup_pipeline(snapshot: dict) -> None:
    """
    Layer2 概率追问 — 迭代乘积版。

    每轮: 一次 LLM 调用同时输出 content + P_continue。
    累计通过率 R = product(P_continue), 衰减到 ~0 自然停。
    没有 MAX_ITERATIONS 硬上限 (除 _FOLLOWUP_SOFT_MAX_ITER 防死循环兜底),
    没有固定间隔, 没有 P 参考区间, 没有时段约束。

    终止的三种情况:
      1) LLM 自报 should_continue=false 或 content=SKIP → break
      2) random() >= R (累计通过率衰减到被拒绝) → break
      3) 迭代 n 达到 _FOLLOWUP_SOFT_MAX_ITER → break (纯技术兜底)

    注意: 路径 A (split) 跟本路径互不影响, 都跑都生效。本函数在 layer1
    pipeline 末尾与 split 路径并列触发, 由 _emit_followup / _schedule_followup
    内部延迟 + win 调度撑开时序。

    2026-10-10: emit 改 schedule (gate+win 改造)。LLM 在每次迭代里输出
    predict_window_sec, 决定这条追问等多久再发出。win=0/小值走"立即发"分支,
    行为与改造前一致; win>5s 走 _scheduled_emit_worker 倒计时, 用户在 win
    窗口内主动说话 → on_user_msg 自动 cancel。
    """
    fp = snapshot.get("fp", "")
    if not _assert_chat_fp(fp, "[followup_pipeline]"):
        return  # 论文侧 — layer2 不服务

    # 疲劳硬停止: 已禁用 (2026-10-10)
    #   旧: value ≥ FATIGUE_HARD_STOP 时 layer2 完全跳过
    #   新: 移除硬停, 让疲劳只通过 _fatigue_discount_factor (R *= P * discount) 自然衰减主动概率。
    #   理由: 硬停 + 折扣 是双层兜底, 抢 LLM 自己的 should_continue 决策;
    #         疲劳时 LLM 也会倾向 P_continue 低, 不需要 hard stop 替它做主;
    #         (2026-10-10 二期) split 路径也已同步移除 hard stop, 全栈统一走"折扣+语气"两层。
    # if _fatigue_should_hard_stop():
    #     debug(f"[followup] SKIP: fatigue hard stop (value={_get_fatigue_value():.2f})")
    #     return

    cumulative_R = 1.0
    history: list[str] = []
    emitted_count = 0

    # 【v3 2026-10-10 post-cancel 注入】从 snapshot 读出 _run_layer1_pipeline
    # 路径 A 末尾 push 进来的 post-cancel 列表(用户在主答后主动说了什么),
    # seed 到 history 头部。让 layer2 第 1 次 LLM 决策时就知道"用户最新回复",
    # 自然在 prompt 里把"用户说了 X" 摆在最前面, LLM 倾向于 should_continue=false
    # (除非话题真的还有活头), 大幅降低和 split 重复的概率。
    post_cancel = snapshot.get("post_cancel") or []
    if isinstance(post_cancel, list):
        for uc, uts in post_cancel:
            if not uc:
                continue
            seed = f"[post-cancel @ {format_dt_second(int(uts))}] 用户回复: {uc[:120]!r}"
            history.append(seed)
        if post_cancel:
            debug(
                f"[followup] seeded history with {len(post_cancel)} post-cancel entries"
            )

    # P0 修复: 把 split 路径已经发出的追问 seed 进 history, 避免 layer2 重复追问同一话题。
    # 时序: split_followup_node 在 llm_call 之后同步执行, 把 {content, predict_reply, ...}
    # 写到 state["split_followup"]。layer1_node 把 state snapshot 整体传给 _run_layer1_pipeline,
    # snapshot 里就带着 split_followup。我们在这里把 content 注入到 history 头部,
    # 让 layer2 第 1 次 LLM 决策时就能看到 "已经发了这条", 直接 should_continue=false 或
    # 换个新话题。
    #   · 关键: 不走 _emit_followup, 只在 LLM 视野里出现。实际落盘在路径 A 已完成, 这里
    #     只负责"告知 LLM 已有此条", 避免双发。
    #   · 标记: 用 "[split] " 前缀, 与本轮 followup 路径自己 emit 的(无前缀)区分, 让 LLM
    #     一眼看出"这是路径 A 已发的, 不是 layer2 自己发的"。
    # 2026-10-10 二期: split 路径加了概率门控, 拆出来不一定发。这里只 seed "实际发过"的,
    #   看 snapshot.get("split_emitted") 而不是 split_followup 的存在与否。
    split_emitted = bool(snapshot.get("split_emitted"))
    if split_emitted:
        split_fu = snapshot.get("split_followup") or {}
        if isinstance(split_fu, dict) and split_fu.get("content"):
            split_content = str(split_fu["content"]).strip()
            if split_content:
                history.append(f"[split] {split_content}")
                debug(
                    f"[followup] seeded history with split content: "
                    f"{_short_log(split_content, 40)}"
                )
    else:
        # split 拆出来了但被概率门控吞掉, layer2 不知道有这件事, 可能重复追问。
        # 注入一个 "[split-suppressed]" 标记让 LLM 知道"用户已经看到一条隐含问句了, 别再追"。
        split_fu = snapshot.get("split_followup") or {}
        if isinstance(split_fu, dict) and split_fu.get("content"):
            split_content = str(split_fu["content"]).strip()
            if split_content:
                history.append(f"[split-suppressed] {split_content}")
                debug(
                    f"[followup] seeded history with SUPPRESSED split: "
                    f"{_short_log(split_content, 40)}"
                )

    fatigue_factor = _fatigue_discount_factor()  # 整轮共享一个折扣系数

    for n in range(1, _FOLLOWUP_SOFT_MAX_ITER + 1):
        # 把 history 注入 snapshot, 喂给 _build_followup_context
        ctx_snapshot = {**snapshot, "followup_history": history}
        ctx = _build_followup_context(ctx_snapshot)

        result = await _call_followup_iteration(n=n, ctx=ctx)
        if result is None:
            # LLM 失败 / JSON 解析失败 / P_continue 缺省 → 自然停
            break

        should = bool(result.get("should_continue"))
        content = str(result.get("content") or "").strip()
        try:
            P = float(result.get("P_continue", 0.0))
        except (TypeError, ValueError):
            P = 0.0
        # clamp P 到 [0, 1] 防止 LLM 偶尔输出 1.5/-0.2
        P = max(0.0, min(1.0, P))
        reason = str(result.get("reason") or "")[:30]

        if (not should) or content == "SKIP" or not content:
            debug(f"[followup] iter {n}: SKIP, reason={_short_log(reason, 30)}")
            break

        # ── 累计概率衰减 ──
        # 累乘 P 之外, 再乘 fatigue_factor (疲劳折扣):
        #   · value ≤ 0.30 → factor=1.0 (无影响)
        #   · value=1.00 → factor=0.4 (R 直接打 4 折)
        cumulative_R *= P * fatigue_factor
        debug(
            f"[followup] iter {n}: P={P:.2f} fatigue={fatigue_factor:.2f} "
            f"R={cumulative_R:.3f} content={content[:40]!r}"
        )

        # ── roll: 累计通过率决定本轮是否真的发出去 ──
        if random.random() >= cumulative_R:
            debug(
                f"[followup] iter {n}: roll FAIL (R={cumulative_R:.3f}), STOP"
            )
            break

        # ── 抽中: schedule (2026-10-10: emit → schedule, gate+win 改造) ──
        # LLM 在迭代里也输出 predict_window_sec, 透传给 _schedule_followup。
        # win=0/小值会被 _clamp_win 兜底, ≤5s 走"立即 emit"分支, 与改造前同语义。
        # source 不再传 (改走 source="scheduled" 由 _schedule_followup 内部定),
        # _emit_followup 的延迟档会变成 300-800ms (scheduled 档)。
        llm_win = result.get("predict_window_sec", 60)
        await _schedule_followup(
            fp=fp,
            content=content,
            predict_reply="",          # 概率追问不预测用户回复
            predict_window_sec=llm_win,
            intent=f"[iter={n} P={P:.2f} R={cumulative_R:.3f} win={llm_win}s] {reason}",
        )
        emitted_count += 1
        history.append(content)

        # 迭代间不设固定延迟, 立即进下一轮 (累计概率本身决定节奏)

    debug(f"[followup] DONE: emitted={emitted_count} messages")


# ═══════════════════════════════════════════════════════════════════════
# Win 调度器 (2026-10-10 新增, gate+win 改造)
# ═══════════════════════════════════════════════════════════════════════
async def _scheduled_emit_worker(entry: ScheduledEntry) -> None:
    """
    Win 倒计时 worker。

    行为:
      · 1s 粒度循环 sleep (防 cancel 信号延迟), 每 tick 检查 entry.cancelled
      · win 秒到 + 未被 cancel → 调 _emit_followup (走 _EMIT_LOCK 串行)
      · 任何阶段被 cancel → 静默 return, 不发
      · 异常一律吞 (fire-and-forget 性质, 不该冒到 caller)

    为什么 1s 粒度而不是单次 long sleep:
      asyncio.sleep 是阻塞的, 期间收到 task.cancel() 信号后, 下一次
      await 边界才会响应。单次 sleep(300) 意味着最坏要等 300s 才感知 cancel。
      拆成 1s 一循环, cancel 响应延迟 ≤ 1s, 用户体感"立即闭嘴"。
    """
    try:
        for _ in range(entry.predict_window_sec):
            await asyncio.sleep(1)
            if entry.cancelled:
                return
    except asyncio.CancelledError:
        return
    # 倒计时结束 + 未被 cancel → emit
    try:
        await _emit_followup(
            fp=entry.pdf_fp,
            content=entry.content,
            predict_reply=entry.predict_reply,
            predict_window_sec=entry.predict_window_sec,
            intent=f"[win +{entry.predict_window_sec}s] {entry.intent}",
            source="scheduled",
        )
    except Exception as e:
        debug(f"[scheduled] worker emit error: {type(e).__name__}: {e}")
    finally:
        with _SCHEDULED_LOCK:
            try:
                _scheduled.remove(entry)
            except ValueError:
                pass


async def _schedule_followup(
    fp: str,
    content: str,
    predict_reply: str,
    predict_window_sec: int,
    intent: str,
) -> None:
    """
    统一 schedule 入口 (split 路径 + layer2 路径共用)。

    行为:
      1. clamp win 到 [5, 1800], 失败/缺省 → 60
      2. cancel 同 fp 已存在的 scheduled (替换语义)
      3. win ≤ WIN_HARD_MIN 走"立即 emit", 不进 scheduler (避免 sleep 比 emit 延迟还短)
      4. 否则: 启动 _scheduled_emit_worker, 入 _scheduled
      5. event loop 不可用 → fallback 立即 emit
    """
    win = _clamp_win(predict_window_sec)
    # 【v2 2026-10-10 思考中窗口】若仍在 thinking 窗口内, 强制把 win 拉大
    # 到至少"窗口剩余时长"。这样 scheduled 不会在主答气泡还没出完前撞车。
    # 例: LLM 正在跑 (剩余 1.2s), split 想立即发追问 (win=5s) → 实际 win
    # 被拉到 1.2s, 追问等到主答落盘后 ~1s 才发出, 不跟主答气泡同框。
    delay = thinking_delay_ms()
    if delay > 0:
        win = max(win, delay)
        debug(
            f"[scheduled] thinking-window: bumped win from "
            f"{_clamp_win(predict_window_sec)}s to {win}s (delay={delay}ms)"
        )
    # 替换: 取消同 fp 已有的 scheduled
    _cancel_scheduled_for_fp(fp, reason="superseded")
    if win <= WIN_HARD_MIN:
        # 5s 内视为"立即发", 走 _emit_followup 自身的延迟档
        await _emit_followup(
            fp=fp,
            content=content,
            predict_reply=predict_reply,
            predict_window_sec=win,
            intent=intent,
            source="scheduled",
        )
        return
    # 真正的 schedule
    now = now_ms()
    entry = ScheduledEntry(
        pdf_fp=fp,
        content=content,
        predict_reply=predict_reply,
        predict_window_sec=win,
        intent=intent,
        scheduled_at_ms=now,
        emit_at_ms=now + win * 1000,
        reason=intent,
    )
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(_scheduled_emit_worker(entry))
        entry.task = task
        with _SCHEDULED_LOCK:
            _scheduled.append(entry)
        debug(
            f"[scheduled] SCHEDULED fp={fp[:12]} win={win}s "
            f"content={content[:40]!r}"
        )
    except RuntimeError:
        debug("[scheduled] no event loop, fallback to immediate emit")
        await _emit_followup(
            fp=fp,
            content=content,
            predict_reply=predict_reply,
            predict_window_sec=win,
            intent=intent,
            source="scheduled",
        )


async def _eval_followup_after_user_msg(
    pdf_fp: str, user_content: str, ts_ms: float,
) -> None:
    """
    【v3 2026-10-10 弃用】win 取消后的补问评估。

    历史:
      v1: on_user_msg 直接 fire-and-forget 调本函数 → 跟主答同框 + 重复 split
      v2: 加 await thinking_delay_ms() 仍不解决, 因为 layer1_node 早 clear window
      v3 (现在): 弃用。改用 on_user_msg → push_post_cancel 入队 →
                _run_layer1_pipeline 路径 A 跑完 split → 路径 B 读出 →
                注入 layer2 history 头部 → LLM 自然判断要不要追。
                物理时序: split emit (1.5-3s) → layer2 sleep 1.5s →
                layer2 LLM (1-2s) → 实际 layer2 emit 在主答后 3-5s,
                节奏自然, 不会撞车。

    本函数保留为 no-op 兜底, 万一旧代码路径调到不会报错。
    """
    debug(
        f"[post-cancel] _eval_followup_after_user_msg is DEPRECATED (v3); "
        f"fp={pdf_fp[:12]} ignored. Use push_post_cancel from on_user_msg instead."
    )
    return


async def _emit_followup(
    fp: str,
    content: str,
    predict_reply: str,
    predict_window_sec: int,
    intent: str,
    source: str,  # "split" / "compose" / "scheduled"
) -> None:
    """
    追问落盘 + _pending 栈 + WS push 的统一出口。

    给 layer1 异步 pipeline 复用 (split 路径 + compose 兜底路径 + scheduled 路径)。
    异常由 caller 静默吞 (这里是 inner, 不该 raise)。

    【随机延迟策略】主回复通过 SSE 几乎实时到前端, split 路径下追问
    也是落盘即 push → 两条瞬间同现, 视觉突兀。对 WS push 追加随机延迟,
    让用户感知到"主回复 → 短暂停顿 → 追问"的节奏:
      · split 路径: 主回复刚落屏, 追问需要更明显停顿才能撑起"主动"仪式感
        → 1500-3000ms 区间
      · compose 路径: judge+compose 本身已花 1-3s, 与主回复已错开, 只需
        轻微微调避免极端挨近 → 500-1500ms 区间
      · scheduled 路径 (2026-10-10 新增): win 已被推迟过, 用户预期"晚点来",
        只需轻微微调避免极端挨近 → 300-800ms 区间
    注意: 落盘和入 _pending 保持即时, 不受延迟影响 → on_user_msg 的
    hit/miss 判定仍按真实时间轴, 不会因为 WS 延迟误判"沉默超时"。

    【并发锁】 2026-10-10 新增: 多 scheduled 同时 wake + split 路径同时 fire
    时, 用 _EMIT_LOCK 串行化落盘 + WS push。_EMIT_LOCK 延迟初始化
    (event loop 才有 asyncio.Lock), 这里用 try/except + 锁占位 fallback。
    """
    global _EMIT_LOCK
    # 懒初始化 asyncio.Lock (必须在 event loop 里)
    if _EMIT_LOCK is None:
        try:
            _EMIT_LOCK = asyncio.Lock()
        except RuntimeError:
            _EMIT_LOCK = None  # 无 loop, 走无锁分支 (acceptable, 进程级单 emitter 顺序自然串行)
    # 1. 递增 ask_count (与 user ask 同桶, 在 WS push 前)
    _inc_ask_count(fp)

    if _EMIT_LOCK is not None:
        async with _EMIT_LOCK:
            await _emit_followup_locked(
                fp=fp, content=content, predict_reply=predict_reply,
                predict_window_sec=predict_window_sec, intent=intent, source=source,
            )
    else:
        await _emit_followup_locked(
            fp=fp, content=content, predict_reply=predict_reply,
            predict_window_sec=predict_window_sec, intent=intent, source=source,
        )


async def _emit_followup_locked(
    fp: str,
    content: str,
    predict_reply: str,
    predict_window_sec: int,
    intent: str,
    source: str,
) -> None:
    """
    _emit_followup 的实际工作函数, 假定 caller 已持锁 (或无锁 fallback)。
    拆出来是为了让锁的边界清晰: 落盘 → 入 _pending → sleep → WS push 整段原子。
    """
    # 2. 落盘 ResAsk (active=True)
    ts_ms = now_ms()
    import hashlib
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

    # 疲劳累加: 主动追问发出, 仅 chat 侧 (_inc_fatigue 内部 gate)
    _inc_fatigue(fp, source="proactive")

    # 构造同步 entry (用于 _pending + WS push; 与磁盘一致)
    entry = {
        "type": "ResAsk",
        "content": content,
        "ts": ts_ms,
        "dt": format_dt_second(ts_ms),
        "msg_fp": msg_fp,
        "active": True,
        "intent": intent,
    }

    # 3. 入 _pending 栈
    with _PENDING_LOCK:
        _pending.append(PendingEntry(
            pdf_fp=fp,
            msg_fp=msg_fp,
            followup_content=content,
            predict_window_sec=predict_window_sec,
            sent_at_ms=ts_ms,
            predict_reply_text=predict_reply,
        ))

    # 4. WS push (随机延迟)
    # - split 路径: 主回复刚落屏, 给 1.5-3.0s 缓冲撑起"主动"仪式感
    # - compose 路径: judge+compose 已耗 1-3s, 只需 0.5-1.5s 避免极端挨近
    # - scheduled 路径: win 已被推迟过, 用户预期"晚点来", 0.3-0.8s 微调
    if source == "split":
        delay_ms = random.randint(1500, 3000)
    elif source == "scheduled":
        delay_ms = random.randint(300, 800)
    else:  # "compose"
        delay_ms = random.randint(500, 1500)
    debug(f"[layer1] delay {delay_ms}ms before push (source={source})")
    await asyncio.sleep(delay_ms / 1000.0)
    _broadcast_resactive(fp, entry)

    # 5. explore update 也 fire-and-forget
    asyncio.create_task(_schedule_explore_update_async())

    debug(
        f"[layer1] done: fp={fp[:12]} outbound={msg_fp[:8]} "
        f"source={source} reason={intent!r}"
    )


async def layer1_node(state: dict) -> dict:
    """
    LangGraph 节点入口 (fire-and-forget 调度器)。
    不 await 后台 pipeline, 立即 return {}.

    【chat-only 约束】论文侧立即跳过, 不起后台 task。
    """
    req = state.get("req")
    if not isinstance(req, AiAskReq):
        return {}
    if not _assert_chat_fp(req.pdf_fp, "[layer1_node]"):
        return {}

    snapshot = _snapshot_state_for_layer1(state)
    # 【v2 2026-10-10 思考中窗口】主答已落盘, AI 气泡即将/已经 push。
    # 此时清空 thinking window, 让 _run_layer1_pipeline 后续排的
    # scheduled 不被延后(它们是"主答已出 + 再过 win 秒" 的自然节奏)。
    # 注意: clear 在 create_task 之前同步执行, _run_layer1_pipeline 是
    # 异步 fire-and-forget, 跑起来时 _thinking_until_ms 已经是 0。
    clear_thinking_window()
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run_layer1_pipeline(snapshot))
    except RuntimeError:
        # 无 event loop (理论上不会, 但兜底)
        debug("[layer1] no running event loop, skip")
    return {}


# ═══════════════════════════════════════════════════════════════════════
# 用户回复处理 (on_user_msg)
# ═══════════════════════════════════════════════════════════════════════


def on_user_msg(pdf_fp: str, content: str, ts_ms: float) -> None:
    """
    WS 上行入口: 用户发新消息。

    行为:
      1. 【v2 2026-10-10 改】拉高"思考中"窗口 (主答 LLM 跑完前, 主动追问延后)
         - 旧版在这里 cancel 同 fp 所有 scheduled, 但用户开口时 scheduled
           列表本来就常是空的 (上一轮 layer1 早已 emit/timeout) → 大量空 cancel
         - 新版改用 thinking window 机制: 窗口期内的 scheduled 自动延后,
           直到主答落盘 (layer1_node clear) 才解封
      2. 【v3 2026-10-10 改】post-cancel 评估从 fire-and-forget 改为入队
         - 旧: 直接 create_task(_eval_followup_after_user_msg), 协程不等
           主答/split 跑完就 emit → 跟主答/同 split 内容撞车 + 重复
         - 新: push_post_cancel 入队, 由 _run_layer1_pipeline 路径 B 在
           split 路径跑完后消费, 串行化"split → layer2" 顺序, 物理上
           保证 post-cancel 不会撞主答, 内容上保证 LLM 看到 split 已发
           不会重复
      3. 查 _pending 栈: 若存在尚未 settle 的追问 (按 fp)
      4. 调 Lint LLM 判定 hit/miss + 是否 late
      5. 入 _learn_batch
      6. 从 _pending 抹去该 entry

    【chat-only 约束】论文侧 WS 上行 (如果未来出现) 直接 return。
    """
    if not _assert_chat_fp(pdf_fp, "[on_user_msg]"):
        return

    # ── 1. 拉高"思考中"窗口 (1800ms, 主答 LLM 典型响应 + 1s 缓冲) ──
    #   在 thinking 窗口期间, _schedule_followup 会被自动延后 (见
    #   thinking_delay_ms), 所以不需要再 cancel scheduled — scheduled
    #   列表此时本来就常空 (上一轮 layer1_pipeline 早 emit/timeout 完毕)。
    set_thinking_window()

    # ── 2. post-cancel 入队 (供 layer1 路径 B 消费, 不再 fire-and-forget) ──
    push_post_cancel(pdf_fp, content, ts_ms)

    # ── 3-6. 原 _pending 结算逻辑 (保持原顺序) ──
    with _PENDING_LOCK:
        # 找到最新的 pending (可能同时有多个, 取最近)
        pending = None
        for p in reversed(_pending):
            if p.pdf_fp == pdf_fp:
                pending = p
                break

    if pending is None:
        return  # 无 pending, 不记样本

    now_ms_i = int(ts_ms) if ts_ms > 0 else now_ms()
    late = (now_ms_i - pending.sent_at_ms) > (pending.predict_window_sec * 1000)

    # 调 Lint LLM 判定 hit/miss (异步, 不阻塞 WS)
    asyncio.create_task(
        _settle_pending_async(pending, content, now_ms_i, late)
    )


async def _settle_pending_async(pending: PendingEntry, content: str, replied_at_ms: int, late: bool) -> None:
    """异步判定 hit/miss, 入 _learn_batch, 从 _pending 抹去。"""
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
    """
    Lint LLM 判定实际回复 vs 追问意图。
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


# ═══════════════════════════════════════════════════════════════════════
# explore 反馈回路 (§2.6)
# ═══════════════════════════════════════════════════════════════════════


async def _schedule_explore_update_async() -> None:
    """
    触发 Crystal_explore.md 重写。

    与 _call_memory_update_llm 共享 _memory_updating 互斥锁 + MEMORY_UPDATE_EVERY_N
    计数 (此处简化: 不去争用 memory 互斥, 单独跑 — explore 重写是独立契约)。

    输入: Crystal_explore.md 全文 + _learn_batch 内的 hit/miss/late 样本
    输出: LLM 重写整份 Crystal_explore.md → _write_agent_explore
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
Crystal_explore.md 是关于**用户时间相关认知**的档案, 只写:
  - 活跃时段 (他通常什么时候活跃)
  - 沉默含义 (他不回 = 在忙 / 不想聊 / 没看到?)
  - 回应速度习惯 (他通常多久会回, 早回/晚回有没有规律)
  - 追问长度偏好 (他喜欢短答/中答/长答)
  - 节奏规律 (追问后多久他会回, 哪些时间点他不回)

**不写**:
  - 单一问答过程 / 具体消息内容 / 用户偏好细节 (那些归 Crystal_memory)
  - 不要保留任何旧章节约束 (固定的三段标题已经废止)

# 当前 Crystal_explore.md
{explore if explore else "(空 — 这是首次写入)"}

# 新增样本 (本批)
{samples_text}

# 输出
重写**整份** Crystal_explore.md, Markdown 格式。返回**严格 JSON**:
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
    import os
    p = path
    parent = os.path.dirname(p)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# WebSocket
# ═══════════════════════════════════════════════════════════════════════


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
    """
    WS push 主动追问 entry 到所有客户端。
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
        # 客户端没连 WS → push 落空, 落盘的 entry 只能等下次刷新通过 history 拉到。
        # 这是排查 WS 链路的核心信号 (Vite 没配 /ws proxy / 前端没 connect 都会出现)。
        debug(f"[ws] broadcast SKIP: no clients connected (entry落盘, 仅刷新可见)")
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
        # 无 event loop, 同步丢弃 (acceptable — 主流程不依赖此 push)
        pass


# ═══════════════════════════════════════════════════════════════════════
# 对外只读接口 (供 frontend / 调试)
# ═══════════════════════════════════════════════════════════════════════


def get_pending_count() -> int:
    with _PENDING_LOCK:
        return len(_pending)


def get_scheduled_count() -> int:
    """
    2026-10-10 新增: 实时查看 scheduled 数量 (win 倒计时中但还没发)。

    用途: 调试 / 前端观测 / 排查"为什么追问没发"。
    """
    with _SCHEDULED_LOCK:
        return len(_scheduled)


def get_learn_batch_count() -> int:
    with _LEARN_LOCK:
        return len(_learn_batch)
