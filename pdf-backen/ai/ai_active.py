"""
ai_active.py — Crystal 主动发言机制 (本期只实现 Layer1 短期追问)。

挂载位置 (LangGraph):
  emotion_llm -> active_layer1 -> flush_track

设计核心 (来自 plan §0):
  · E = explore.md 全文 (LLM 自由管理, 无段落约束)
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
# ECUS 构造
# ═══════════════════════════════════════════════════════════════════════


def _build_layer1_system(req: AiAskReq, mr_text: str) -> str:
    """
    Layer1 ECUS 提示词 (system 段)。

    拼接 E + C (运行时再 extend) + U + S + 上下文强调 (本轮 MR)。
    """
    explore = _get_agent_explore()
    memory = _get_agent_memory()[:1500]   # memory 截断
    self_md = _get_agent_self()[:500]     # self 截断

    return f"""# 角色
你是 Crystal 的人格 (S 段)。你在评估「本轮 main reply (MR) 之后, 是否要主动追问用户」。

# S — Crystal 自我认知
{self_md}

# U — 关于用户的认知
{memory}

# E — 主动外呼档案 (explore.md)
以下是关于用户**时间相关认知**的档案 (活跃时段 / 沉默含义 / 回应速度习惯等)。
这是你判断"是否要追问"以及"追多紧"的关键依据。
{explore if explore else "(空 — 暂无档案)"}

# 本轮 MR 上下文
{mr_text[:1500]}

# 任务: 评估是否要主动追问
如果用户在 MR 之后沉默, 你是否会想再问一句?
返回 **严格 JSON** (无 markdown fence):
{{
  "should_followup": true|false,
  "reason": "简述, ≤30 字",
  "predict_window_sec": 60,      // 软预测: 用户大概多久会回 (秒), 30~600
  "predict_reply_length": "短"|"中"|"长"
}}
"""


def _build_layer1_compose_user(judge: dict) -> str:
    """追问构造 user 段: 给 main LLM 的方向提示。"""
    return f"""# 任务: 写一句主动追问
基于上面 S/U/E 档案 + 本轮 MR 上下文, 写一句**口语化、简短**的追问。
要求:
  - 长度 ≤ 50 字
  - 延续 MR 的语气, 不要重新开话题
  - 不要复述 MR 已经说过的内容
  - 留出用户回应的空间 (开放/收口皆可)

Lint 裁判理由: {judge.get("reason", "")}
预测回复长度: {judge.get("predict_reply_length", "短")}

返回 **严格 JSON** (无 markdown fence):
{{
  "content": "追问正文, ≤50 字",
  "predict_reply": "预测用户会怎么回 (≤30 字)",
  "predict_window_sec": {judge.get("predict_window_sec", 60)}
}}
"""


# ═══════════════════════════════════════════════════════════════════════
# Fire-and-forget Pipeline (§10)
# ═══════════════════════════════════════════════════════════════════════


def _snapshot_state_for_layer1(state: dict) -> dict:
    """
    复制 layer1 真正用到的字段, 避免后台 task 持有 state 引用读到
    LangGraph 后续节点改写后的 state (save_paper_memory / emotion_llm / flush_track
    都可能改 messages / final_answer)。
    """
    req = state.get("req")
    return {
        "req": req,
        "fp": req.pdf_fp if req else "",
        "final_answer": state.get("final_answer", ""),
        "messages": list(state.get("messages", [])),
        "req_fp": state.get("req_fp", ""),
    }


async def _parse_json_lenient(raw: str) -> dict | None:
    """
    容错 JSON 解析 (LLM 输出常带尾巴)。

    LLM 真实输出常见 3 种"额外数据":
      1. ```json ... ``` markdown fence → 剥掉
      2. JSON 之后追加解释/换行/注释 → "Extra data" json.JSONDecodeError
      3. 多个 JSON 拼接 (LLM 重复生成) → 取第一个完整对象

    策略:
      1. 剥 ``` fence
      2. 找到第一个 '{' 起, 配对 '}' 截取第一个完整 JSON object
      3. 截取后仍解析失败 → return None
    """
    if not raw:
        return None
    s = raw.strip()
    # 1. 剥 markdown fence
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip().rstrip("`").strip()
    if not s or s[0] != "{":
        return None
    # 2. 配对 '}' 截取第一个完整 JSON object
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                s = s[: i + 1]
                break
    else:
        # 整个字符串 depth 都没归零 → 没有完整对象
        return None
    # 3. 解析截取后的纯 JSON
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


async def _judge_followup(ecus_system: str) -> dict | None:
    """二元裁判: 是否追问。失败/缺配置 → 跳过 (return None)。"""
    raw = await _call_lint_llm(ecus_system)
    if not raw:
        return None
    try:
        judge = await _parse_json_lenient(raw)
        if judge is None or "should_followup" not in judge:
            debug(f"[ai_active] judge parse fail: missing should_followup  raw={raw[:80]!r}")
            return None
        return {
            "should_followup": bool(judge["should_followup"]),
            "reason": str(judge.get("reason", ""))[:80],
            "predict_window_sec": int(judge.get("predict_window_sec", 60)),
            "predict_reply_length": str(judge.get("predict_reply_length", "短")),
        }
    except (ValueError, TypeError) as e:
        debug(f"[ai_active] judge parse fail: {e}  raw={raw[:80]!r}")
        return None


async def _compose_followup(system: str, user: str) -> dict | None:
    """Main LLM 构造追问正文。失败/缺配置 → 跳过。"""
    raw = await _call_lint_llm(system + "\n\n" + user)
    if not raw:
        return None
    try:
        out = await _parse_json_lenient(raw)
        if out is None:
            debug(f"[ai_active] compose parse fail: not valid JSON  raw={raw[:80]!r}")
            return None
        content = str(out.get("content", "")).strip()
        if not content:
            return None
        return {
            "content": content[:200],
            "predict_reply": str(out.get("predict_reply", ""))[:80],
            "predict_window_sec": int(out.get("predict_window_sec", 60)),
        }
    except (ValueError, TypeError) as e:
        debug(f"[ai_active] compose parse fail: {e}  raw={raw[:80]!r}")
        return None


async def _run_layer1_pipeline(snapshot: dict) -> None:
    """
    真正的 layer1 流程。**异步** 在 sync state 调用返回之后跑。

    异常一律静默吞掉 (不影响主回复), debug 打印。
    """
    req = snapshot.get("req")
    fp = snapshot.get("fp", "")
    if not isinstance(req, AiAskReq):
        return
    if fp != CHAT_FP:
        return  # Layer1 只在 chat 场景

    try:
        mr_text = snapshot.get("final_answer", "") or ""
        system = _build_layer1_system(req, mr_text)

        # 1. 二元裁判
        judge = await _judge_followup(system)
        if judge is None or not judge["should_followup"]:
            debug(f"[layer1] skip: judge=no fp={fp[:12]}")
            return

        # 2. 构造追问
        compose_user = _build_layer1_compose_user(judge)
        followup = await _compose_followup(system, compose_user)
        if followup is None:
            debug(f"[layer1] skip: compose fail fp={fp[:12]}")
            return

        # 3. 递增 ask_count (与 user ask 同桶, 在 WS push 前)
        _inc_ask_count(fp)

        # 4. 落盘 ResAsk (active=True)
        ts_ms = now_ms()
        # 生成 short msg_fp (12 char)
        import hashlib
        msg_fp = hashlib.md5(f"{ts_ms}-{fp}-{followup['content']}".encode()).hexdigest()[:12]
        ok = _append_active_resask(
            followup["content"],
            msg_fp=msg_fp,
            ts_ms=ts_ms,
            intent=judge.get("reason", ""),
        )
        if not ok:
            debug(f"[layer1] WARN: _append_active_resask fail fp={fp[:12]}")
            return
        # 构造同步 entry (用于 _pending + WS push; 与磁盘一致)
        entry = {
            "type": "ResAsk",
            "content": followup["content"],
            "ts": ts_ms,
            "dt": format_dt_second(ts_ms),
            "msg_fp": msg_fp,
            "active": True,
            "intent": judge.get("reason", ""),
        }

        # 5. 入 _pending 栈
        with _PENDING_LOCK:
            _pending.append(PendingEntry(
                pdf_fp=fp,
                msg_fp=msg_fp,
                followup_content=followup["content"],
                predict_window_sec=followup["predict_window_sec"],
                sent_at_ms=ts_ms,
                predict_reply_text=followup["predict_reply"],
            ))

        # 6. WS push
        _broadcast_resactive(fp, entry)

        # 7. explore update 也 fire-and-forget
        asyncio.create_task(_schedule_explore_update_async())

        debug(f"[layer1] done: fp={fp[:12]} outbound={msg_fp[:8]} reason={judge.get('reason', '')!r}")
    except Exception as e:
        debug(f"[layer1] pipeline error (swallowed): fp={fp[:12]} {type(e).__name__}: {e}")


async def layer1_node(state: dict) -> dict:
    """
    LangGraph 节点入口 (fire-and-forget 调度器)。
    不 await 后台 pipeline, 立即 return {}.

    唯一前置过滤: 仅 chat 场景 (pdf_fp == CHAT_FP) 启动; 论文侧跳过。
    """
    req = state.get("req")
    if not isinstance(req, AiAskReq):
        return {}
    if req.pdf_fp != CHAT_FP:
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
    """
    if pdf_fp != CHAT_FP:
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
    触发 explore.md 重写。

    与 _call_memory_update_llm 共享 _memory_updating 互斥锁 + MEMORY_UPDATE_EVERY_N
    计数 (此处简化: 不去争用 memory 互斥, 单独跑 — explore 重写是独立契约)。

    输入: explore.md 全文 + _learn_batch 内的 hit/miss/late 样本
    输出: LLM 重写整份 explore.md → _write_agent_explore
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

    prompt = f"""# 任务: 重写 explore.md
explore.md 是关于**用户时间相关认知**的档案, 只写:
  - 活跃时段 (他通常什么时候活跃)
  - 沉默含义 (他不回 = 在忙 / 不想聊 / 没看到?)
  - 回应速度习惯 (他通常多久会回, 早回/晚回有没有规律)
  - 追问长度偏好 (他喜欢短答/中答/长答)
  - 节奏规律 (追问后多久他会回, 哪些时间点他不回)

**不写**:
  - 单一问答过程 / 具体消息内容 / 用户偏好细节 (那些归 Crystal_memory)
  - 不要保留任何旧章节约束 (固定的三段标题已经废止)

# 当前 explore.md
{explore if explore else "(空 — 这是首次写入)"}

# 新增样本 (本批)
{samples_text}

# 输出
重写**整份** explore.md, Markdown 格式。返回**严格 JSON**:
{{"content": "重写后的 explore.md 全文"}}
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
