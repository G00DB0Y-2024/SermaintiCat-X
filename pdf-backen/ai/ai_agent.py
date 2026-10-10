"""
LangGraph 节点 + State。

本文件承担第 4 块的"节点"职责:
- 定义 PaperAIState (TypedDict)
- 所有 LangGraph 节点函数
- build_graph / run_ask / run_load 已迁到 ai_graph.py,本文件保留 LangGraph 节点实现。

依赖 (按层次自下而上):
  ai_models    数据契约
  ai_config    参数常量
  ai_utils     时间
  ai_io        IO + cache
  ai_llm       LLM 调用
  prompts_system / prompts_context  提示词
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
from typing import Any, TypedDict

from .ai_config import (
    _ai_config,
    ASK_LOCAL_LIMIT,
    CHAT_FP,
    CHAT_LOCAL_LIMIT,
    CRYSTAL_MEMORY_FILE,
    CRYSTAL_SELF_FILE,
    LOAD_LOCAL_LIMIT,
    MEMORY_COMPRESS_EVERY_M,  # 新增: 压缩触发阈值 (默认 = MEMORY_UPDATE_EVERY_N × 2)
    MEMORY_UPDATE_EVERY_N,
)
from .ai_io import (
    _agent_memory_cache,  # noqa: F401  # 由 _set_agent_memory_cache 同包维护
    _get_ask_count,
    _inc_ask_count,
    _get_agent_explore,
    _get_agent_memory,
    _get_agent_self,
    _invalidate_history_cache,
    _inc_fatigue,
    _load_paper_history,
    load_chat_messages,
    _paper_history_path,
    _read_json_safe,
    _set_agent_memory_cache,
    _set_agent_self_cache,
)
from .ai_llm import _call_llm
from .ai_models import AiAskReq, AiLoadReq
from .ai_utils import now_ms, format_dt_second
from .prompts_context import (
    buildMemoryUpdateUserPrompt,
    buildSelfUpdateUserPrompt,
    buildMemoryAndSelfUpdateUserPrompt,
    buildMemoryAndSelfCompressUserPrompt,
    build_chat_audit_block,
)
from .prompts_system import (
    MEMORY_UPDATE_SYSTEM,
    SELF_UPDATE_SYSTEM,
    MEMORY_AND_SELF_UPDATE_SYSTEM,
    MEMORY_AND_SELF_COMPRESS_SYSTEM,
    SYSTEM_LOAD,
)
from .ai_utils import get_current_time_context
from utils.log import debug


# ═══════════════════════════════════════════════════════════════════════
# 进程级单例状态
# ═══════════════════════════════════════════════════════════════════════

# 记忆更新互斥锁 (粗粒度 boolean):
# update_agent_memory_node 用 asyncio.create_task 触发 _update_crystal_memory_async,
# 不等其完成, 因此用户短时间多条消息会并发跑多次 LLM update ——
# 这会重复写 Crystal_memory.md, 也浪费 token。
# 用 _memory_updating 标志 + _update_crystal_memory_async 内的 try/finally
# 保证同一时刻只有一个 task 在跑; 后续并发的 task 直接 return 跳过。
_memory_updating: bool = False


# ═══════════════════════════════════════════════════════════════════════
# LangGraph State
# ═══════════════════════════════════════════════════════════════════════

class PaperAIState(TypedDict):
    """
    LangGraph 工作区状态。

    req:                 入口请求(AiAskReq 或 AiLoadReq)
    messages:            组装好的 OpenAI 格式 messages, 准备送给 llm_call
    agent_memory:        从 Crystal_memory.md 加载的 Markdown 全文(注入 system prompt)
    agent_self:          从 Crystal_self.md 加载的 Markdown 全文。
                         仅 chat 侧 (pdf_fp == CHAT_FP) 消费并注入 system prompt;
                         论文侧 (SYSTEM_LOAD / SYSTEM_ASK 论文分支) 刻意不注入,
                         避免哲学化内容干扰客观学术问答。
    agent_explore:       从 Crystal_explore.md 加载的 Markdown 全文 (主动外呼档案)。
                         仅 chat 侧消费并注入 system prompt —— 称呼约定/互动仪式/
                         边界忌讳已从 memory 迁出, chat 侧不读 A 就会忘记怎么称呼他。
    paper_history:       兼容字段, 论文侧即为 paper_ask_history
    paper_ask_history:   当前 pdf_fp 最近 ASK_LOCAL_LIMIT 对 ask 历史 (role 交替)
    paper_load_history:  当前 pdf_fp 最近 LOAD_LOCAL_LIMIT 对 load 历史 (role 交替)
    final_answer:        llm_call 返回的最终 content
                         (split_followup_node 之后, 若拆分成功, 会被改写为 main_body 主体,
                          原 MR 末尾问句剥离 → 进 _pending 栈)
    split_followup:      split_followup_node 输出的拆包结果。
                         None = MR 无隐含追问, 或拆解失败 (lint 异常 / lint 返回非 JSON)。
                         dict 时 keys: {"content", "predict_reply", "predict_window_sec",
                         "intent", "from_split=True"}。
                         仅 chat 侧消费; 论文侧节点全 null。
    usage:               上游 LLM 的 usage 统计
    dt:                  服务端时间字符串 (北京时区, 秒级, 来自 now_ms 单源时间),
                         通过 AiResp.dt 透传给前端, 保证前后端时间一致。
    req_fp:              save_paper_memory_node 生成的 ReqAsk msg_fp。
                         Load 模式为空字符串。透传给 AiResp.req_fp,
                         让前端 ai_res 中用户气泡的 msg_fp 与 paper_history 对齐。
    res_fp:              save_paper_memory_node 生成的 ResAsk msg_fp。
                         Load 模式为空字符串。透传给 AiResp.res_fp,
                         让前端 ai_res 中 AI 气泡的 msg_fp 与 paper_history 对齐。
    """
    req: Any  # AiAskReq | AiLoadReq
    messages: list[dict]
    agent_memory: str
    agent_self: str
    agent_explore: str
    paper_history: list[dict]
    paper_ask_history: list[dict]
    paper_load_history: list[dict]
    final_answer: str
    split_followup: dict | None
    usage: dict
    dt: str
    req_fp: str
    res_fp: str
    emotion: dict


# ═══════════════════════════════════════════════════════════════════════
# LangGraph 节点
# ═══════════════════════════════════════════════════════════════════════

async def load_agent_memory_node(state: PaperAIState) -> dict:
    """读 agent_memory / agent_self / agent_explore 缓存(首次才打磁盘)。作为 system prompt 注入。

     缓存策略: 模块级 _agent_memory_cache / _agent_self_cache / _explore_cache,
    首次调用时同步从对应 md 加载, 之后各写入回路写完文件会同步刷新缓存。

    注入范围 (本次变更):
      · agent_memory  — 论文侧 + chat 侧都注入 (维持原行为)。内容是「他是什么样的人」(U)。
      · agent_self    — **只在 chat 侧消费**。state 里始终带上 (节点不知道
        pdf_fp 语义, 判断留给 compose_messages_node), 论文侧 compose 分支
        刻意不把它拼进 system prompt —— 自我认知里的哲学化内容
        ("存在哲学/认知构建") 会干扰客观学术问答, 属于出戏风险区。
      · agent_explore — **只在 chat 侧消费** (A 元素)。与 self 同理带上,
        由 compose_messages_node 决定是否拼进 system prompt。
        为什么 chat 侧要注入: 称呼约定("XX好, 主人")、互动仪式、边界忌讳
        已从 memory 迁出, 若 chat 侧不读 A, Crystal 在日常对话里就会
        忘记自己该怎么称呼他 —— 那是人格连续性的基础, 不是外呼专属信息。

    注: 当前不做长度截断, MEMORY_MAX_CHARS 保留为占位 (后续方案处理)。
    """
    return {
        "agent_memory": _get_agent_memory(),
        "agent_self": _get_agent_self(),
        "agent_explore": _get_agent_explore(),
    }


async def load_paper_history_node(state: PaperAIState) -> dict:
    """
    一次性加载当前 pdf_fp 的 ask + load 两条历史轨。

    设计: 论文侧上下文按 fp 隔离, 数据源是 save/{fp}_ai.json (经过 _load_paper_history
    按 mode + limit 截取)。全局 track 子系统已废弃, 不再读写。

    但 ask 轨需要 load 轨作为辅助概要 (论文中的选段总结能帮 LLM 理解上下文),
    所以两个都加载, 各自按 limit 上限截取, 写入 state 供后续 compose_messages_node 拼装。

    ChatView 场景 (pdf_fp == "crystal_chat"):
      - ask_history 作为 Chat 本地上下文 (CHAT_LOCAL_LIMIT 对)
        走新统一入口 ai_io.load_chat_messages, 它从 crystal_chat_ai.json 取最近 N 轮
        返回纯 {role, content} (无元信息前缀)。
        → 时间感知由 system prompt 的 time_context 统一提供 (绝对当前时间),
          per-message 嵌入的 "[<ts> 用户说/...]" 标签已被 v3 移除 (LLM 会复读,
          污染落盘 ResAsk.content)。
        → 主动/被动通过 role=assistant + entry.active 字段的内部传递 (不入 prompt)
          区分, 跟 system prompt 的 chat_audit 块对齐。
      - load_history 暂不使用 (Chat 不会 Load)

    Vision Ask 和 Anno 在 _load_paper_history 内部已过滤。
    """
    req = state["req"]
    fp = req.pdf_fp
    is_chat = (fp == CHAT_FP)

    if isinstance(req, AiAskReq):
        # Ask 模式 (论文 + Chat): 加载本 fp ask 历史
        #   - Chat: 走新统一入口 (角色标签嵌入 content 头部, 主动/被动自描述)
        #   - 论文: 走 fp 隔离的 _load_paper_history
        if is_chat:
            ask_history = load_chat_messages(limit=CHAT_LOCAL_LIMIT)
        else:
            ask_history = _load_paper_history(fp, mode="ask", limit=ASK_LOCAL_LIMIT)
        # 同时加载本 fp load 历史 (仅论文侧使用)
        if is_chat:
            load_history: list[dict] = []
        else:
            load_history = _load_paper_history(fp, mode="load", limit=LOAD_LOCAL_LIMIT)
        debug(
            f"[paper_history] ask: fp={fp[:12]} is_chat={is_chat} "
            f"ask_rounds={len(ask_history)//2} load_rounds={len(load_history)//2}"
        )
        return {
            "paper_history": ask_history,        # 兼容
            "paper_ask_history": ask_history,
            "paper_load_history": load_history,
        }
    else:
        # Load 模式: 加载本论文 load 历史
        load_history = _load_paper_history(fp, mode="load", limit=LOAD_LOCAL_LIMIT)
        debug(f"[paper_history] load: fp={fp[:12]} load_rounds={len(load_history)//2}")
        return {
            "paper_history": load_history,
            "paper_ask_history": [],
            "paper_load_history": load_history,
        }


async def compose_messages_node(state: PaperAIState) -> dict:
    """
    按场景组装 messages:

    论文 Ask (pdf_fp != CHAT_FP):
      [system]  CrystalPersona + time + agent_mem + "在论文侧回答"
      [ask_msgs]   本论文 ask (ASK_LOCAL_LIMIT 对)
      [load_msgs]  本论文 load (LOAD_LOCAL_LIMIT 对)
      [user]    本轮提问
      -> 上下文按论文 fp 完全隔离

    Chat Ask (pdf_fp == CHAT_FP):
      [system]  CrystalPersona + time + device + agent_mem + agent_self + agent_explore
                + [近期对话感知] 块 (chat audit, 由 build_chat_audit_block 拼装)
                + emotion
      [chat_msgs]  ChatView 本地近 Z=CHAT_LOCAL_LIMIT 对
      [user]    本轮提问
      -> 上下文按 chat fp 完全隔离, 不依赖任何 track 子系统
      -> **全量注入 (explore + memory + self)**: chat 是 Crystal 的人格主场,
         三份档案一起给。explore 必须给: 称呼约定("XX好, 主人")已从 memory
         迁出, 不注入则日常对话里 Crystal 会忘记自己该怎么称呼他。

    Load 模式: 人设 + 本论文 Load 历史 (LOAD_LOCAL_LIMIT 对)
    """
    from .prompts_context import compose_chat_messages, compose_paper_ask_messages, build_load_user_content
    req = state["req"]
    agent_mem = state.get("agent_memory") or ""
    agent_self = state.get("agent_self") or ""
    agent_explore = state.get("agent_explore") or ""
    time_context = get_current_time_context()
    fp = req.pdf_fp
    is_chat = (fp == CHAT_FP)

    if isinstance(req, AiAskReq):
        # paper_ask_history: 论文走 fp 隔离的 _load_paper_history, chat 走新统一入口
        # load_chat_messages (已含角色标签嵌入 content 头部)。这里不分支调用,
        # 加载逻辑集中在 load_paper_history_node, 此处只消费结果。
        ask_history = state.get("paper_ask_history") or []
        load_history = state.get("paper_load_history") or []

        if is_chat:
            # chat 侧全量注入 explore + memory + self。
            # self / explore 只在这里进 system, 论文分支刻意不传 —— 学术问答要的是
            # 客观准确, 自我认知与外呼约定里的哲学/期待类内容会诱导表演, 反而拉低
            # 回答质量; 论文侧也没有"该不该主动开口"的问题。
            # agent_explore 当前不消费 (Crystal_explore.md 注入段在前一轮重构中已删除,
            # 后续 ai_active 重写 explore 处理时再接回), 保留入参占位。
            messages = compose_chat_messages(
                req, ask_history, agent_mem, time_context,
                agent_self=agent_self, agent_explore=agent_explore,
            )
        else:
            messages = compose_paper_ask_messages(
                req, ask_history, load_history, agent_mem, time_context
            )
    else:
        # Load: 人设 + 本论文 Load 历史 (LOAD_LOCAL_LIMIT 对)
        messages = [
            {"role": "system", "content": SYSTEM_LOAD(time_context)},
        ]
        load_history = state.get("paper_load_history") or []
        messages.extend(load_history)
        messages.append({"role": "user", "content": build_load_user_content(req)})

    debug(
        f"[compose_messages] is_chat={is_chat} "
        f"msgs={len(messages)} "
        f"first_role={messages[0]['role'] if messages else '-'}"
    )
    return {"messages": messages}


async def llm_call_node(state: PaperAIState) -> dict:
    """
    调上游 LLM。模型选择由 ai_config 决定:
      - AiAskReq + image_base64 非空 -> vision_model
      - 否则 -> model (普通 ask / load)

    同时产出服务端时间 (单源 now_ms()) → state.dt, 最终透传给前端。
    """
    req = state["req"]
    messages = state["messages"]

    is_vision = isinstance(req, AiAskReq) and req.image_base64 is not None

    content, usage = await _call_llm(
        messages=messages,
        vision_model=is_vision,
        disable_thinking=False,
    )

    return {
        "final_answer": content,
        "usage": usage,
        "dt": format_dt_second(now_ms()),
    }


async def save_paper_memory_node(state: PaperAIState) -> dict:
    """
    把本轮 user + assistant 追加写入 save/{fp}_ai.json (新 schema 格式)。

    写两条 entry: ReqAsk/ReqLoad + ResAsk/ResLoad, 统一用 AiPaperEntry 格式。
    Vision Ask 正常写盘（供前端渲染），但不在上下文加载时被 pickup（由 _load_paper_history 过滤）。

    返回字段:
      req_fp: 本轮 ReqAsk 的 msg_fp (Load 模式为空字符串), 供 AiResp 透传给前端,
              让前端 ai_res 中用户气泡的 msg_fp 与 paper_history 完全对齐。
      res_fp: 本轮 ResAsk 的 msg_fp (Load 模式为空字符串), 供 AiResp 透传给前端,
              让前端 ai_res 中 AI 气泡的 msg_fp 与 paper_history 完全对齐。
    """
    req = state["req"]
    fp = req.pdf_fp
    answer = state["final_answer"]
    usage = state.get("usage") or {}
    token_count = usage.get("total_tokens")

    ts = now_ms()
    dt_str = format_dt_second(ts)

    # --- Req entry ---
    req_fp = ""
    if isinstance(req, AiAskReq):
        req_fp = f"{secrets.token_hex(3)[:6]}_{ts}"
        req_entry = {
            "type": "ReqAsk",
            "content": req.ask,
            "ts": ts,
            "dt": dt_str,
            "msg_fp": req_fp,
            "quotes": list(req.quotes or []),
            "hl": getattr(req, "hl", None),
            "img": getattr(req, "image_filename", "") or "",
        }
    else:
        req_entry = {
            "type": "ReqLoad",
            "content": req.chosen_text,
            "ts": ts,
            "dt": dt_str,
        }

    # --- Res entry ---
    res_type = "ResAsk" if isinstance(req, AiAskReq) else "ResLoad"
    res_ts = ts + 1   # 与 Req 错开 1ms, 避免双方撞 fp (虽然 hex6 是随机,概率极低)
    res_fp = ""
    res_entry = {
        "type": res_type,
        "content": answer,
        "ts": res_ts,
        "dt": dt_str,
        "token_count": token_count,
    }
    if isinstance(req, AiAskReq):
        res_fp = f"{secrets.token_hex(3)[:6]}_{res_ts}"
        res_entry["msg_fp"] = res_fp

    # 【v3 split-origin】若本轮触发了 split, 在主答 entry 上加 origin 字段,
    # 记录"拆分前"的原始 MR 全文 (含 main_body + content)。
    # 仅 chat 侧 + 仅 ResAsk 主答上有, 主动追问 (active=True) 那条不加
    # (origin 是"主答拆分语义"的标记, 追问有 from_split 字段已够)。
    # split_followup 是 split_followup_node 写入 state 的字段, 仅 chat 侧消费。
    split_fu = state.get("split_followup")
    if (
        isinstance(split_fu, dict)
        and split_fu.get("from_split")
        and split_fu.get("origin")
        and res_type == "ResAsk"
    ):
        res_entry["origin"] = split_fu["origin"]

    # --- 读 + 追加 + 写盘 ---
    path = _paper_history_path(fp)
    history: list = _read_json_safe(path, [])
    history.append(req_entry)
    history.append(res_entry)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
        # 写盘后失效该 path 的 mtime 缓存, 保证下次请求读到的就是新内容
        _invalidate_history_cache(path)
        debug(f"[save_paper_memory] ok: fp={fp} total={len(history)}")
    except OSError as e:
        debug(f"[save_paper_memory] FAIL: {e} | fp={fp}")

    # 疲劳累加: 仅 Ask 模式 (Load 不算"回复"), chat-only 由 _inc_fatigue 内部 gate。
    # 论文 fp 进来时 _inc_fatigue 直接 return 当前 value, 不写盘。
    if isinstance(req, AiAskReq):
        _inc_fatigue(fp, source="main")

    return {"req_fp": req_fp, "res_fp": res_fp}


def _should_update_memory(fp: str, is_chat: bool) -> bool:
    """
    memory update 节流:
    - 论文 / Chat 都按 MEMORY_UPDATE_EVERY_N 次 ask 触发一次 memory LLM
      (ChatView 是与 Crystal 闲聊的主战场, memory 理应吸收 chat 内容;
       计数按 fp 分桶, chat 单独一桶, 不与论文互相干扰)
    - 触发条件 (节流在调用方对外过滤 vision 后再走到这里):
        (count % MEMORY_UPDATE_EVERY_N) == 0
      即第 N/MEMORY_UPDATE_EVERY_N 次 ask 时 (例如 N=1 表示每次) 触发。

    计数读 _get_ask_count (走 params.json 持久化), 冷启动后依然是原来的进度,
    不会因为服务器重启而在前几轮集中触发。

    注意: 函数本身不做 vision 过滤 (那在 update_agent_memory_node 入口处判断),
    这里只做 is_chat 参与下的节流逻辑 ——
    现在 is_chat 不再硬短路, 与论文走相同路径, 让 MEMORY_UPDATE_EVERY_N 在两个场景都生效。
    """
    return (_get_ask_count(fp) % MEMORY_UPDATE_EVERY_N) == 0


async def update_agent_memory_node(state: PaperAIState) -> dict:
    """
    Ask 模式:
      - 论文 fp + 非 Vision: 节流触发 _update_crystal_memory_async
        (默认每 MEMORY_UPDATE_EVERY_N 次 ask 一次)
      - Chat fp: 同样按 MCP 节流 (与论文 fp 共用同一套计数 + 节流)
      - Vision Ask (img 非空): 不进计数 (避免图片污染节流), 仍可触发 memory update

    Load 模式:
      - 论文 fp: 仅做节流触发判定, 不再写 track; load 历史已由
        _load_paper_history 通过 save/{fp}_ai.json 持久化。
      - Chat fp: 同上, 不需要再写 track。

    【v7 删除】track 子系统已废弃, 不再向 ask/load track 写记录。
    """
    req = state["req"]
    answer = state["final_answer"]
    ts = now_ms()
    dt_str = format_dt_second(ts)
    fp = req.pdf_fp
    is_chat = (fp == CHAT_FP)

    if isinstance(req, AiAskReq):
        is_vision = bool(getattr(req, "image_filename", ""))

        # 不管是论文 fp 还是 chat fp, 都不是 vision, 都做 ask 计数 + 节流判定。
        # (论文 + chat 是同一个"Crystal 与用户交互"主链, 共享相同的 memory 累积机制;
        #  计数按 fp 各自独立成桶 — "crystal_chat" 一桶, 每篇论文各一桶,
        #  互不干扰。MEMORY_UPDATE_EVERY_N 对两桶都生效, 设 1 就是每次都触发。)
        if not is_vision:
            # 走 _inc_ask_count 而不是直接改 dict —— 它会立刻落盘 params.json,
            # 服务器重启后节流进度不丢。
            count = _inc_ask_count(fp)
            trigger_mem = _should_update_memory(fp, is_chat)
        else:
            count = _get_ask_count(fp)   # vision 不计数, 仅用于日志
            trigger_mem = False

        # 【v7 删除】track 子系统已废弃, 不再向 _append_track 写入 ask 记录。
        # chat 路径改读 save/crystal_chat_ai.json (build_chat_audit_block),
        # 论文节点的 memory/self update 通过 ask_history / load_history state
        # 取窗口上下文 (见 save_paper_memory_node), 不再依赖全局 track。

        if trigger_mem:
            # 已有 memory update 在跑, 跳过 ——
            # 计数已递增, 下一轮仍会按 MEMORY_UPDATE_EVERY_N 节流再触发。
            if _memory_updating:
                debug(
                    f"[crystal_memory] SKIP (busy at node): fp={fp[:12]} "
                    f"count={count}"
                )
                return {}
            task = asyncio.create_task(
                _update_crystal_memory_async(req.ask, answer, dt_str, fp)
            )
            debug(
                f"[crystal_memory] scheduled [ask throttled]: fp={fp[:12]} "
                f"count={count} "
                f"user_len={len(req.ask)} asst_len={len(answer)} "
                f"task_id={id(task)}"
            )
        else:
            debug(
                f"[crystal_memory] SKIP: is_chat={is_chat} is_vision={is_vision} "
                f"count={_get_ask_count(fp)}/{MEMORY_UPDATE_EVERY_N}"
            )
    # else: Load 模式 — paper load 历史已由 _load_paper_history 通过
    # save/{fp}_ai.json 持久化, memory/self update 节点直接读 history。

    return {}


async def _update_crystal_memory_async(
    user_msg: str,
    assistant_msg: str,
    current_timestamp: str = "",
    fp: str = "",
) -> None:
    """
    后台任务: 读 Crystal_memory.md 与 Crystal_self.md, 加上本轮对话,
    一次 LLM 调用同时更新两份笔记 (updated_memory + updated_self), 写回磁盘。

    ════ Pipeline (2026-10-09 合并版) ════
      Phase 2 merged: 一次 LLM 调用 → JSON {updated_memory, updated_self}
        基础:    current_mem + current_self (两份 md 全文)
        上下文:  本轮对话 + chat_audit + 当前时间戳
        输出:    JSON, 两个字段均为完整 Markdown
        强约束:  self 字段不得含禁词 (「他」), 出现则作废走 fallback

      Fallback (合并版解析失败时退化): 两次独立调用 (旧 Phase 2a + 2b)
        Phase 2a memory: 基础 current_mem, 不加载 self
        Phase 2b self:   基础 current_self + 刚更新好的 new_mem (作为外部参照)

    输入边界规则 (与原有 memory 链路一致):
      · 用户问 Crystal -> system prompt 只注入 Crystal_memory (load_agent_memory_node
        已 _get_agent_memory()), 不注入 Crystal_self。✅
      · 合并 update prompt 一次性注入两份 md 全文, 让 LLM 在 JSON 输出时主动
        避免串档。✅
      · 旧的两段独立 update 函数 (buildMemoryUpdateUserPrompt /
        buildSelfUpdateUserPrompt) 保留作 fallback 路径, 解析失败时调用。✅

    并发互斥: 模块级 _memory_updating 标志。
    用户短时间多条消息触发多次 update_agent_memory_node, 互斥后第二次起直接 return 跳过。
    计数已经在调用方递增, 下一轮仍会按 MEMORY_UPDATE_EVERY_N 节流再触发。

    异常隔离: 合并路径 / fallback 路径各自 try/except, 任一失败不影响主流程。
    失败静默, 不影响用户响应。两份 md 都是锦上添花, 损坏不应阻塞主链路。

    原子写: 合并版要求两份 md **要么一起更新, 要么都不动** —
            避免出现"memory 已写新值, self 失败保留旧值"的中间态
            (该态会再次污染 self)。先 _write_text_file_atomic 写 memory,
            再写 self; 任一失败都回滚另一份到旧值 (单步原子)。
    """
    from .ai_io import _ensure_memory_dir

    # 并发互斥: 已有 task 在跑, 直接放弃本次 (计数已在调用方递增)
    global _memory_updating
    if _memory_updating:
        debug(f"[crystal_memory] SKIP (busy): fp={fp[:12]} ts={current_timestamp}")
        return
    _memory_updating = True

    # 入口摘要
    debug(
        f"[crystal_memory] >>> START: fp={fp[:12]} ts={current_timestamp} "
        f"user_len={len(user_msg)} asst_len={len(assistant_msg)} "
        f"thinking=off (memory + self)"
    )

    # 累加各 phase 的输入输出统计 (供 <<< END 统一摘要)
    stats: dict[str, int | bool] = {}

    try:
        if not _ai_config["api_key"] or not _ai_config["api_url"]:
            debug("[crystal_memory] skip: ai_config 未设置")
            return

        _ensure_memory_dir()

        # ── 读两份当前 md ──
        current_mem = _read_text_file_safe(CRYSTAL_MEMORY_FILE)
        current_self = _read_text_file_safe(CRYSTAL_SELF_FILE)
        debug(
            f"[crystal_memory] READ: "
            f"mem_len={len(current_mem)} self_len={len(current_self)}"
        )



        # 近期对话感知 (chat_audit): 读 crystal_chat_ai.json 最近 N 条
        # 非 chat fp 也调, 论文 fp 走同一函数会自然取到空串 (chat json 与 paper json
        # 是两个独立文件)。chat_audit_block() 已经做了空检测, 这里不再做 fp 判断。
        chat_audit = build_chat_audit_block()
        stats["chat_audit_len"] = len(chat_audit)
        debug(
            f"[crystal_memory] chat_audit loaded: len={len(chat_audit)}"
        )

        # ═══════════════════════════════════════════════════════════════
        # Phase 2 merged: Crystal_memory + Crystal_self 一次性更新 (2026-10-09)
        #   ⚠️ 一次 LLM 调用, 强制 JSON {updated_memory, updated_self}
        #   · 失败 → Fallback: 两次独立旧路径 (旧 Phase 2a + 2b)
        #   · 解析拒绝 (self 含禁词「他」) → 同上 fallback
        #   · 解析成功 → 原子写: memory + self 同时落盘
        # ═══════════════════════════════════════════════════════════════
        debug(
            f"[crystal_memory] PHASE 2 merged update -> LLM: "
            f"mem_input_len={len(current_mem)} self_input_len={len(current_self)} "
            f"audit_len={len(chat_audit)}"
        )

        new_mem: str | None = None
        new_self: str | None = None
        merged_ok = False

        try:
            result = await _call_memory_and_self_update_llm(
                current_mem, current_self,
                user_msg, assistant_msg,
                current_timestamp, chat_audit,
            )
            if result is not None:
                new_mem, new_self = result
                merged_ok = True
                stats["merged_delta_mem"]  = len(new_mem)  - len(current_mem)
                stats["merged_delta_self"] = len(new_self) - len(current_self)
                debug(
                    f"[crystal_memory] PHASE 2 merged update <- LLM: "
                    f"mem_out={len(new_mem)}({stats['merged_delta_mem']:+d}) "
                    f"self_out={len(new_self)}({stats['merged_delta_self']:+d})"
                )
            else:
                debug("[crystal_memory] PHASE 2 merged parse FAIL, will fallback")
        except Exception as e:
            debug(f"[crystal_memory] PHASE 2 merged CALL FAIL: {type(e).__name__}: {e}")

        # ─────────── Fallback: 旧的两段独立 update ───────────
        # 仅在合并版失败时跑, 走 buildMemoryUpdateUserPrompt / buildSelfUpdateUserPrompt。
        # 这一段完全保留 v7 的语义, 不再加载 self 进 memory, memory 加载 self 进 self。
        if not merged_ok:
            debug("[crystal_memory] PHASE 2 merged FAIL → fallback to legacy 2a+2b")

            # 旧 Phase 2a: memory update, 不加载 self
            try:
                new_mem = await _call_memory_update_llm(
                    current_mem, user_msg, assistant_msg,
                    current_timestamp, chat_audit,
                )
                stats["mem_update_delta"] = (len(new_mem) - len(current_mem)) if new_mem else 0
            except Exception as e:
                debug(f"[crystal_memory] fallback PHASE 2a FAIL: {type(e).__name__}: {e}")
                new_mem = None

            # 写 memory (这样旧 Phase 2b 才能拿到「刚更新好的 memory」)
            if new_mem:
                try:
                    _write_text_file_atomic(CRYSTAL_MEMORY_FILE, new_mem)
                    _set_agent_memory_cache(new_mem)
                    debug(
                        f"[crystal_memory] fallback PHASE 2a WRITE ok: "
                        f"new_len={len(new_mem)} delta={len(new_mem)-len(current_mem):+d}"
                    )
                except OSError as e:
                    debug(f"[crystal_memory] fallback PHASE 2a WRITE FAIL: {type(e).__name__}: {e}")
                    new_mem = None  # 写失败 → 旧 Phase 2b 拿不到新 mem, 走 current_mem

            # 旧 Phase 2b: self update, 加载刚更新好的 mem
            mem_for_self = new_mem if new_mem else current_mem
            try:
                new_self = await _call_self_update_llm(
                    current_self, mem_for_self,
                    user_msg, assistant_msg,
                    current_timestamp, chat_audit,
                )
                stats["self_update_delta"] = (len(new_self) - len(current_self)) if new_self else 0
            except Exception as e:
                debug(f"[crystal_memory] fallback PHASE 2b FAIL: {type(e).__name__}: {e}")
                new_self = None

            if new_self:
                try:
                    _write_text_file_atomic(CRYSTAL_SELF_FILE, new_self)
                    _set_agent_self_cache(new_self)
                    debug(
                        f"[crystal_memory] fallback PHASE 2b WRITE ok: "
                        f"new_len={len(new_self)} delta={len(new_self)-len(current_self):+d}"
                    )
                except OSError as e:
                    debug(f"[crystal_memory] fallback PHASE 2b WRITE FAIL: {type(e).__name__}: {e}")

            # 任一失败, 都不污染磁盘: 已经写过的回滚到旧值
            # (fallback 路径下只有一份成功另一份失败时才走回滚)
            if not (new_mem and new_self):
                # 尽力回滚, 但不要二次失败导致无限递归
                try:
                    if new_mem and not new_self:
                        _write_text_file_atomic(CRYSTAL_MEMORY_FILE, current_mem)
                        _set_agent_memory_cache(current_mem)
                        debug("[crystal_memory] fallback ROLLBACK memory")
                    if new_self and not new_mem:
                        _write_text_file_atomic(CRYSTAL_SELF_FILE, current_self)
                        _set_agent_self_cache(current_self)
                        debug("[crystal_memory] fallback ROLLBACK self")
                except OSError as e:
                    debug(f"[crystal_memory] fallback ROLLBACK FAIL: {type(e).__name__}: {e}")

            return  # fallback 完成, 跳过下方合并版写盘

        # ═══════════════════════════════════════════════════════════════
        # 合并版原子写: 两份 md 要么一起更新, 要么都不动
        #   · 顺序: 先写 memory, 再写 self
        #   · 任一失败: 把已写的那份回滚到旧值 (保证磁盘上两份是原子的)
        # ═══════════════════════════════════════════════════════════════
        assert merged_ok and new_mem is not None and new_self is not None
        try:
            # Step 1: 写 memory
            _write_text_file_atomic(CRYSTAL_MEMORY_FILE, new_mem)
            _set_agent_memory_cache(new_mem)
            # Step 2: 写 self (memory 已落盘, 若此步失败需回滚 memory)
            _write_text_file_atomic(CRYSTAL_SELF_FILE, new_self)
            _set_agent_self_cache(new_self)
            debug(
                f"[crystal_memory] PHASE 2 merged WRITE ok (atomic): "
                f"mem {len(current_mem)}→{len(new_mem)} "
                f"self {len(current_self)}→{len(new_self)}"
            )
        except OSError as e:
            # 任意一步失败: 尽力回滚, 不让磁盘进入"半新半旧"中间态
            debug(f"[crystal_memory] PHASE 2 merged WRITE FAIL: {type(e).__name__}: {e}")
            try:
                # 不区分谁失败, 一律回滚两份 (最保守: 不让任何一份被新值污染)
                _write_text_file_atomic(CRYSTAL_MEMORY_FILE, current_mem)
                _set_agent_memory_cache(current_mem)
                _write_text_file_atomic(CRYSTAL_SELF_FILE, current_self)
                _set_agent_self_cache(current_self)
                debug("[crystal_memory] PHASE 2 merged ROLLBACK both ok")
            except OSError as re:
                debug(f"[crystal_memory] PHASE 2 merged ROLLBACK FAIL: {type(re).__name__}: {re}")
                # 极端情况: 回滚也失败。此时不破坏, 由下次 update 自然覆盖。

    except Exception as e:
        debug(f"[crystal_memory] update FAIL: {type(e).__name__}: {e}")
    finally:
        _memory_updating = False
        # 取差分: 优先 merged 路径的 key (2026-10-09 新版), fallback 才用旧 key
        #   · merged 路径: stats["merged_delta_mem"] / stats["merged_delta_self"]
        #   · fallback 路径: stats["mem_update_delta"] / stats["self_update_delta"]
        # 改前 .get 永远读 fallback key, 走 merged 路径时恒为 0 (出现 +0 +0 假象)
        mem_d = stats.get("merged_delta_mem")
        if mem_d is None:
            mem_d = stats.get("mem_update_delta", 0)
        self_d = stats.get("merged_delta_self")
        if self_d is None:
            self_d = stats.get("self_update_delta", 0)
        debug(
            f"[crystal_memory] <<< END: "
            f"mem_d={mem_d:+d} "
            f"self_d={self_d:+d} "
            f"audit_len={stats.get('chat_audit_len', 0)}"
        )

        # ────────────────────────────────────────────────────────────────
        # 【v0 2026-10-10】压缩触发 — 每 MEMORY_COMPRESS_EVERY_M 次 ask 一次。
        # 位置: 必须在 update 写盘完成后, finally 块里 (保证 _memory_updating
        #        已经被 reset, compress 才能拿到锁)。
        # 频率: count % MEMORY_COMPRESS_EVERY_M == 0 (默认 6 = 3 × 2)
        #        即每 2 次 update 触发 1 次 compress, 与 update 共用 _get_ask_count 桶。
        # 隔离: compress 是 fire-and-forget, 不阻塞 update 链。
        # ────────────────────────────────────────────────────────────────
        try:
            if fp and _should_compress_memory(fp):
                task = asyncio.create_task(
                    _compress_crystal_memory_async(current_timestamp, fp)
                )
                debug(
                    f"[crystal_compress] scheduled: fp={fp[:12]} "
                    f"count={_get_ask_count(fp)}/{MEMORY_COMPRESS_EVERY_M} "
                    f"task_id={id(task)}"
                )
        except Exception as e:
            debug(f"[crystal_compress] schedule FAIL: {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Compress — "近详远略"压缩, 复用 update 的 LLM 链路
# ═══════════════════════════════════════════════════════════════════════

def _should_compress_memory(fp: str) -> bool:
    """
    判定本次 update 跑完后是否要触发 compress。

    规则: 复用 _get_ask_count 桶 (与 update 共用), 每 MEMORY_COMPRESS_EVERY_M
    次 ask 触发一次压缩 (默认 6 = 3 × 2, 即每 2 次 update 后压 1 次)。

    为什么不单独建计数:
      · compress 必然紧随 update 而来, 与 update 共享触发节流, 节奏天然一致。
      · 单独建桶会出现 "update 跑 6 次但 compress 跑 0 次" 的失同步。

    边界:
      · fp 为空 (旧路径 / load 模式) → False
      · 计数器未加载 (冷启动首轮) → _get_ask_count 返回 0, 0 % 6 == 0 但
        此时两份 md 几乎是空的, 压了也没意义, 但不会出错, 走一次空跑
        保护: 加上 len(memory)+len(self) > 500 的最小长度门槛, 跳过空文件。
    """
    if not fp:
        return False
    try:
        count = _get_ask_count(fp)
        if count <= 0 or (count % MEMORY_COMPRESS_EVERY_M) != 0:
            return False
        # 最小长度门槛: 文件太短压了没意义 (空文件 / 全新会话)
        try:
            mem_len = len(_read_text_file_safe(CRYSTAL_MEMORY_FILE))
            self_len = len(_read_text_file_safe(CRYSTAL_SELF_FILE))
        except OSError:
            return False
        if (mem_len + self_len) < 500:
            return False
        return True
    except Exception as e:
        debug(f"[crystal_compress] _should_compress_memory FAIL: {type(e).__name__}: {e}")
        return False


async def _compress_crystal_memory_async(
    current_timestamp: str = "",
    fp: str = "",
) -> None:
    """
    后台任务: 按"近详远略"压缩两份 md。

    ════ Pipeline (2026-10-10 v0) ════
      一次 LLM 调用 → JSON {updated_memory, updated_self}
        基础:    current_mem + current_self (两份 md 全文)
        上下文:  当前时间戳 (用于判"新鲜/过渡/压缩"分区) + 阈值常量
        输出:    JSON, 两个字段均为完整 Markdown (压缩后的完整版)
        强约束:  不丢关键事实 (他是谁/做了什么/我们怎么约定)

    时间分区 (按"条目时间戳距今"):
      · 新鲜区 (< COMPRESS_FRESH_HOURS, 默认 6h):  保持日记体, 一字不动
      · 过渡区 (6h ~ COMPRESS_TRANSIT_HOURS, 默认 3d): 短句/单段, 保留 1 句情境 + 1 句反应
      · 压缩区 (> 3d):                              有序/无序列表, 只留关键事实

    互斥: 复用 _memory_updating 标志位 (compress 跑时 update 不能跑, 反之亦然)。
    异常隔离: 全部 try/except, 失败静默, 损坏的两份 md 不影响主链路。
    失败语义: 失败时保留旧值, 下次 update 走完后会再次触发 (因为 count 没动)。
    """
    global _memory_updating
    if _memory_updating:
        debug(f"[crystal_compress] SKIP (busy): fp={(fp or '')[:12]} ts={current_timestamp}")
        return
    _memory_updating = True

    debug(
        f"[crystal_compress] >>> START: fp={(fp or '')[:12]} ts={current_timestamp} "
        f"thinking=off (compress memory + self)"
    )

    stats: dict[str, int] = {}

    try:
        if not _ai_config["api_key"] or not _ai_config["api_url"]:
            debug("[crystal_compress] skip: ai_config 未设置")
            return

        from .ai_io import _ensure_memory_dir
        _ensure_memory_dir()

        # 读两份当前 md
        current_mem = _read_text_file_safe(CRYSTAL_MEMORY_FILE)
        current_self = _read_text_file_safe(CRYSTAL_SELF_FILE)
        stats["mem_in"] = len(current_mem)
        stats["self_in"] = len(current_self)
        debug(
            f"[crystal_compress] READ: mem_len={len(current_mem)} self_len={len(current_self)}"
        )

        new_mem: str | None = None
        new_self: str | None = None

        try:
            result = await _call_memory_and_self_compress_llm(
                current_mem, current_self, current_timestamp,
            )
            if result is not None:
                new_mem, new_self = result
                stats["mem_out"] = len(new_mem)
                stats["self_out"] = len(new_self)
                debug(
                    f"[crystal_compress] LLM ok: mem {len(current_mem)}→{len(new_mem)} "
                    f"self {len(current_self)}→{len(new_self)}"
                )
            else:
                debug("[crystal_compress] LLM parse FAIL, skip write")
                return
        except Exception as e:
            debug(f"[crystal_compress] LLM CALL FAIL: {type(e).__name__}: {e}")
            return

        # 原子写: 与 update 同构 (先 memory, 再 self, 任一失败回滚)
        assert new_mem is not None and new_self is not None
        try:
            _write_text_file_atomic(CRYSTAL_MEMORY_FILE, new_mem)
            _set_agent_memory_cache(new_mem)
            _write_text_file_atomic(CRYSTAL_SELF_FILE, new_self)
            _set_agent_self_cache(new_self)
            debug(
                f"[crystal_compress] WRITE ok (atomic): "
                f"mem {len(current_mem)}→{len(new_mem)} "
                f"self {len(current_self)}→{len(new_self)}"
            )
        except OSError as e:
            debug(f"[crystal_compress] WRITE FAIL: {type(e).__name__}: {e}")
            # 任意一步失败: 尽力回滚
            try:
                _write_text_file_atomic(CRYSTAL_MEMORY_FILE, current_mem)
                _set_agent_memory_cache(current_mem)
                _write_text_file_atomic(CRYSTAL_SELF_FILE, current_self)
                _set_agent_self_cache(current_self)
                debug("[crystal_compress] ROLLBACK both ok")
            except OSError as re:
                debug(f"[crystal_compress] ROLLBACK FAIL: {type(re).__name__}: {re}")

    except Exception as e:
        debug(f"[crystal_compress] FAIL: {type(e).__name__}: {e}")
    finally:
        _memory_updating = False
        debug(
            f"[crystal_compress] <<< END: "
            f"mem {stats.get('mem_in', 0)}→{stats.get('mem_out', stats.get('mem_in', 0))} "
            f"self {stats.get('self_in', 0)}→{stats.get('self_out', stats.get('self_in', 0))}"
        )


async def _call_memory_and_self_compress_llm(
    current_mem: str,
    current_self: str,
    current_timestamp: str = "",
) -> tuple[str, str] | None:
    """
    一次 LLM 调用同时产出压缩后的 updated_memory + updated_self。

    · 走 main LLM (json_mode=True), 与 update 同构。
    · disable_thinking=False: 让模型有推理空间决定"哪些该压/哪些该留"。

    返回:
      (new_mem, new_self) — 成功
      None               — 解析失败 / LLM 调用失败, 调用方应保留旧值
    """
    messages = [
        {"role": "system", "content": MEMORY_AND_SELF_COMPRESS_SYSTEM()},
        {"role": "user", "content": buildMemoryAndSelfCompressUserPrompt(
            current_mem, current_self, current_timestamp,
        )},
    ]
    content, _ = await _call_llm(
        messages=messages,
        vision_model=False,
        disable_thinking=False,
        json_mode=True,
    )
    parsed = _safe_parse_update_json(content)
    if parsed is None:
        debug("[crystal_compress] parse FAIL, content head="
              f"{repr(content[:120])}")
        return None
    return parsed["updated_memory"], parsed["updated_self"]


# ═══════════════════════════════════════════════════════════════════════
# IO 辅助 — 读两份 md 都需要, 抽到模块级避免重复
# ═══════════════════════════════════════════════════════════════════════

def _read_text_file_safe(path: str) -> str:
    """读文本文件, 不存在或读失败时返回空串。"""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except OSError:
        pass
    return ""


def _write_text_file_atomic(path: str, content: str) -> int:
    """写文本文件, 返回写入字节数。出错抛 OSError。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
        return f.tell()


# ═══════════════════════════════════════════════════════════════════════
# Memory / Self 的 LLM 调用包装 (4 个)
# ═══════════════════════════════════════════════════════════════════════

async def _call_memory_update_llm(
    current_md: str,
    user_msg: str,
    assistant_msg: str,
    current_timestamp: str = "",
    chat_audit: str = "",
) -> str:
    """
    调 LLM 更新 Crystal_memory.md, 返回新的 Markdown 文本。

    参数:
      current_timestamp: 秒级可读时间字符串 (来自 now_ms + format_dt_second)
      chat_audit: 近期对话感知块 (build_chat_audit_block 的输出, 可空)

    ⚠️ 不传入 self: memory 只关心用户认知, 不载入 Crystal 关于自己的笔记。

    【v7 删除】ask_track / load_track 参数已移除 —— 全局 track 子系统废弃,
    update prompt 上下文仅含"当前笔记 + 本轮对话 + 时间戳 + chat_audit"。
    """
    messages = [
        {"role": "system", "content": MEMORY_UPDATE_SYSTEM()},
        {"role": "user", "content": buildMemoryUpdateUserPrompt(
            current_md, user_msg, assistant_msg,
            current_timestamp,
            chat_audit,
        )},
    ]
    content, _ = await _call_llm(messages=messages, vision_model=False, disable_thinking=True)
    return content.strip()  # legacy: 旧两段独立 update 的 memory 路径, 解析失败 fallback 用


async def _call_self_update_llm(
    current_self_md: str,
    updated_memory_md: str,
    user_msg: str,
    assistant_msg: str,
    current_timestamp: str = "",
    chat_audit: str = "",
) -> str:
    """
    调 LLM 更新 Crystal_self.md, 返回新的 Markdown 文本。

    与 memory update 的关键差异:
      · 基础是 current_self (不再走 compress 阶段)
      · 多喂一个「刚更新好的 memory」作为外部参照 (让 LLM 知道「我对用户的认知」,
        反向校准自我定位)
      · 多喂 chat_audit 块 (与 memory update 对称, 提升 LLM 对近期互动的感知)

    【v7 删除】ask_track / load_track 参数已移除 —— 全局 track 子系统废弃。
    """
    messages = [
        {"role": "system", "content": SELF_UPDATE_SYSTEM()},
        {"role": "user", "content": buildSelfUpdateUserPrompt(
            current_self_md,
            updated_memory_md,
            user_msg,
            assistant_msg,
            current_timestamp,
            chat_audit,
        )},
    ]
    content, _ = await _call_llm(messages=messages, vision_model=False, disable_thinking=True)
    return content.strip()  # legacy: 旧两段独立 update 的 self 路径, 解析失败 fallback 用


# ═══════════════════════════════════════════════════════════════════════
# Crystal_memory + Crystal_self 合并 update (2026-10-09 引入)
# ═══════════════════════════════════════════════════════════════════════
# 合并动机: 旧版两段独立 LLM 调用 (Phase 2a memory → Phase 2b self),
#   各自只看到"刚写好的一份", 容易把"他喜欢下厨"既写进 memory, 又在 self
#   重复一份 (主语换成"我注意到他喜欢..."), 形成串档污染。
# 新版: 一次性把 current_mem + current_self + 本轮对话 + chat_audit 全部塞给
#   main LLM (支持 thinking 推理), 强制以 JSON 输出 {updated_memory,
#   updated_self} 两个字段。LLM 在生成时就能"主动去重" — 同一事实只放一边,
#   不再二次污染。
# 同时: 一次 HTTP 调用, 省一半 token, 省一半延迟。


def _safe_parse_update_json(content: str) -> dict | None:
    """
    兜底解析 LLM 的合并 update 响应。

    LLM 输出可能在不同网关下表现不同:
      · 裸 JSON:                {"updated_memory": "...", "updated_self": "..."}
      · markdown fence 包裹:    ```json\n{...}\n```
      · 思考前缀 + JSON:        "好的, 以下是更新: {...}"
      · 解释后置 + JSON:        {...}\n注: 已按规则更新
    必须全部容忍。失败时返回 None (调用方走保留旧值路径)。

    校验: 必含 updated_memory / updated_self 两个**非空字符串**字段;
          两份长度均 < 200 KB 防异常膨胀。
    """
    if not content:
        return None
    text = content.strip()

    # 1) 尝试 markdown fence 抽取 (```json ... ``` / ``` ... ```)
    import re
    fence_match = None
    for pat in (r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            fence_match = m.group(1)
            break
    if fence_match is not None:
        text = fence_match

    # 2) 尝试直接 json.loads
    parsed: dict | None = None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            parsed = obj
    except (json.JSONDecodeError, ValueError):
        # 3) 兜底: 找第一个 { 到最后一个 } 截取再 load
        idx_first = text.find("{")
        idx_last = text.rfind("}")
        if idx_first >= 0 and idx_last > idx_first:
            try:
                obj = json.loads(text[idx_first : idx_last + 1])
                if isinstance(obj, dict):
                    parsed = obj
            except (json.JSONDecodeError, ValueError):
                return None
        else:
            return None

    if parsed is None:
        return None

    # 4) 校验字段
    mem = parsed.get("updated_memory")
    self_md = parsed.get("updated_self")
    if not isinstance(mem, str) or not isinstance(self_md, str):
        return None
    if not mem.strip() or not self_md.strip():
        return None
    # 长度上限 (200 KB 一份, 防 LLM 异常膨胀)
    if len(mem) > 200_000 or len(self_md) > 200_000:
        return None

    # 5) 长度上限校验 (200 KB 一份, 防 LLM 异常膨胀)
    if len(mem) > 200_000 or len(self_md) > 200_000:
        return None

    return parsed


async def _call_memory_and_self_update_llm(
    current_mem: str,
    current_self: str,
    user_msg: str,
    assistant_msg: str,
    current_timestamp: str = "",
    chat_audit: str = "",
) -> tuple[str, str] | None:
    """
    一次 LLM 调用同时产出 updated_memory + updated_self。

    · 走 main LLM (disable_thinking=False, json_mode=True), 支持 thinking 推理,
      质量高于旧版专用 memory LLM。

    返回:
      (new_mem, new_self) — 成功
      None               — 解析失败 / LLM 调用失败, 调用方应保留旧值

    异常隔离: LLM 调用层抛错 (网络/限流) → 也返回 None,
              外层 _update_crystal_memory_async 用 try/except 兜底。
    """
    messages = [
        {"role": "system", "content": MEMORY_AND_SELF_UPDATE_SYSTEM()},
        {"role": "user", "content": buildMemoryAndSelfUpdateUserPrompt(
            current_mem, current_self,
            user_msg, assistant_msg,
            current_timestamp, chat_audit,
        )},
    ]
    # json_mode=True → ai_llm 会加 response_format={"type":"json_object"},
    # 显著降低解析失败率。旧的两段 prompt 路径仍走默认 (json_mode=False)
    # 保留兼容。
    # disable_thinking=False: 走 main LLM (DeepSeek 等支持 thinking 的模型可以
    # 开启推理, 提升 memory/self 合并更新的质量; 用户没开 thinking 则按 main LLM
    # 的全局 deepseek_thinking 配置走。
    content, _ = await _call_llm(
        messages=messages,
        vision_model=False,
        disable_thinking=False,
        json_mode=True,
    )
    parsed = _safe_parse_update_json(content)
    if parsed is None:
        debug("[crystal_memory] merged parse FAIL, content head="
              f"{repr(content[:120])}")
        return None
    return parsed["updated_memory"], parsed["updated_self"]

