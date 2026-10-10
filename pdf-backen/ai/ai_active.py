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
    _inc_ask_count,
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

# 本轮 MR 全文
{mr_text}

# 严格 JSON 输出 (无 markdown fence):
{{
  "should_split": true|false,
  "main_body": "第一条气泡（保留原文的排版、换行和所有细节，绝不压缩）",
  "content": "第二条气泡（直接从原文后半段或末尾截取，放开字数限制，绝对禁止捏造原文没有的新话题）",
  "predict_reply": "预测用户看完第二条气泡后会怎么回 (≤30字，若无拆分则空)",
  "predict_window_sec": int,  // 软预测: 用户大概多久会回 (秒), 30~600
  "reason": "≤40字，说明是在哪里找到的话题断点，或不拆的理由"
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
        debug(f"[split] parse fail: {raw[:80]!r}")
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

    return {
        "should_split": True,
        "main_body": main_body,
        "content": content,
        "predict_reply": str(out.get("predict_reply", ""))[:80],
        "predict_window_sec": int(out.get("predict_window_sec", 60)),
        "reason": str(out.get("reason", ""))[:80],
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


async def _run_layer1_pipeline(snapshot: dict) -> None:
    """
    真正的 layer1 流程。**异步** 在 sync state 调用返回之后跑。

    【chat-only 约束】论文侧 (_assert_chat_fp=False) 直接 return, 不调 LLM。

    【v2 split-only】仅依赖 split_followup_node 的拆解结果; 无兜底分支,
    若 split 未拆出, layer1 整体空跑。

    异常一律静默吞掉 (不影响主回复), debug 打印。
    """
    req = snapshot.get("req")
    fp = snapshot.get("fp", "")
    if not isinstance(req, AiAskReq):
        return
    if not _assert_chat_fp(fp, "[layer1_pipeline]"):
        return  # 论文侧 — layer1 不服务

    try:
# ═══ split 唯一路径 (v2 改造) ═══
        # split_followup_node 在 llm_call 后同步拆出, 若拆出, 直接落盘跳过
        split_fu = snapshot.get("split_followup")
        if split_fu and isinstance(split_fu, dict) and split_fu.get("content"):
            await _emit_followup(
                fp=fp,
                content=split_fu["content"],
                predict_reply=split_fu.get("predict_reply", ""),
                predict_window_sec=split_fu.get("predict_window_sec", 60),
                intent=split_fu.get("intent", "[split]"),
                source="split",
            )
        return  # 拆出 / 未拆出: 兜底已删除, 层主不做空问
    except Exception as e:
        debug(f"[layer1] pipeline error (swallowed): fp={fp[:12]} {type(e).__name__}: {e}")


async def _emit_followup(
    fp: str,
    content: str,
    predict_reply: str,
    predict_window_sec: int,
    intent: str,
    source: str,  # "split" or "compose"
) -> None:
    """
    追问落盘 + _pending 栈 + WS push 的统一出口。

    给 layer1 异步 pipeline 复用 (split 路径 + compose 兜底路径)。
    异常由 caller 静默吞 (这里是 inner, 不该 raise)。

    【随机延迟策略】主回复通过 SSE 几乎实时到前端, split 路径下追问
    也是落盘即 push → 两条瞬间同现, 视觉突兀。对 WS push 追加随机延迟,
    让用户感知到"主回复 → 短暂停顿 → 追问"的节奏:
      · split 路径: 主回复刚落屏, 追问需要更明显停顿才能撑起"主动"仪式感
        → 800-1800ms 区间
      · compose 路径: judge+compose 本身已花 1-3s, 与主回复已错开, 只需
        轻微微调避免极端挨近 → 300-900ms 区间
    注意: 落盘和入 _pending 保持即时, 不受延迟影响 → on_user_msg 的
    hit/miss 判定仍按真实时间轴, 不会因为 WS 延迟误判"沉默超时"。
    """
    # 1. 递增 ask_count (与 user ask 同桶, 在 WS push 前)
    _inc_ask_count(fp)

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
    # - split 路径: 主回复刚落屏, 给 0.8-1.8s 缓冲撑起"主动"仪式感
    # - compose 路径: judge+compose 已耗 1-3s, 只需 0.3-0.9s 避免极端挨近
    if source == "split":
        delay_ms = random.randint(1500, 3000)
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
      1. 查 _pending 栈: 若存在尚未 settle 的追问 (按 fp)
      2. 调 Lint LLM 判定 hit/miss + 是否 late
      3. 入 _learn_batch
      4. 从 _pending 抹去该 entry

    【chat-only 约束】论文侧 WS 上行 (如果未来出现) 直接 return。
    """
    if not _assert_chat_fp(pdf_fp, "[on_user_msg]"):
        return

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


def get_learn_batch_count() -> int:
    with _LEARN_LOCK:
        return len(_learn_batch)
