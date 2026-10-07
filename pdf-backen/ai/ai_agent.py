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
    MEMORY_UPDATE_EVERY_N,
)
from .ai_io import (
    _agent_memory_cache,  # noqa: F401  # 由 _set_agent_memory_cache 同包维护
    _append_track,
    _get_ask_count,
    _inc_ask_count,
    _ask_track_list,  # noqa: F401  # 模块单例引用保持
    _flush_track_to_disk,
    _get_agent_explore,
    _get_agent_memory,
    _get_agent_self,
    _get_track,
    _invalidate_history_cache,
    _load_chat_history_for_memory,
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
    build_track_summary_block,
    buildMemoryCompressUserPrompt,
    buildMemoryUpdateUserPrompt,
    buildSelfCompressUserPrompt,
    buildSelfUpdateUserPrompt,
)
from .prompts_system import (
    MEMORY_COMPRESS_SYSTEM,
    MEMORY_UPDATE_SYSTEM,
    SELF_COMPRESS_SYSTEM,
    SELF_UPDATE_SYSTEM,
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
    agent_explore:       从 explore.md 加载的 Markdown 全文 (主动外呼档案)。
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
                         让前端 ai_res 中用户气泡的 msg_fp 与 paper_history / track 对齐。
    res_fp:              save_paper_memory_node 生成的 ResAsk msg_fp。
                         Load 模式为空字符串。透传给 AiResp.res_fp,
                         让前端 ai_res 中 AI 气泡的 msg_fp 与 paper_history / track 对齐。
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

    设计: 论文侧上下文按 fp 隔离, 不再读全局 track。
    但 ask 轨需要 load 轨作为辅助概要 (论文中的选段总结能帮 LLM 理解上下文),
    所以两个都加载, 各自按 limit 上限截取, 写入 state 供后续 compose_messages_node 拼装。

    ChatView 场景 (pdf_fp == "crystal_chat"):
      - ask_history 作为 Chat 本地上下文 (CHAT_LOCAL_LIMIT 对)
        走新统一入口 ai_io.load_chat_messages, 它从 crystal_chat_ai.json 取最近 N 轮
        并在每条 content 头部嵌 "[<ts> 用户说 / Crystal主动说 / Crystal回复]" 标签。
        → LLM 一次性看到"谁在什么时间说的", 无需另读 track。
        → 主动/被动通过 label 区分, 与旧 entry.active 同义 (ResAsk active=True → "Crystal主动说")。
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
                + "全局闲聊" + 论文 track 摘要块 + emotion
      [chat_msgs]  ChatView 本地近 Z=CHAT_LOCAL_LIMIT 对
      [user]    本轮提问
      -> 论文全局 track 通过 _get_track 注入 (track 仅含论文, 因为 _append_track 已过滤)
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
            # agent_explore 当前不消费 (explore.md 注入段在前一轮重构中已删除,
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
              让前端 ai_res 中用户气泡的 msg_fp 与 paper_history / track 完全对齐。
      res_fp: 本轮 ResAsk 的 msg_fp (Load 模式为空字符串), 供 AiResp 透传给前端,
              让前端 ai_res 中 AI 气泡的 msg_fp 与 paper_history / track 完全对齐。
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
      - 论文 fp + 非 Vision: 追加到 ask_track (经 _append_track 自动过滤 chat fp)
      - Chat fp: 不写 track, 不写 mem (论文 track 保持纯净)
      - 论文 fp: 节流触发 _update_crystal_memory_async (默认每 5 次 ask 一次)
      - Vision Ask (img 非空): 不进 track, 但仍可节流触发 memory update
    Load 模式:
      - 论文 fp: 追加到 load_track
      - Chat fp: 不写 track
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

        # 【v3 全局化】所有 ask 场景都进同一 track (论文 + chat), 由 build_track_summary_block
        # 在 ChatView system prompt 阶段统一摘录 "近期对话感知"。chat 路径不再 skip:
        #   - 拆成功的 split: assistant 字段存 main_body (而非原 MR), 与落盘的 ResAsk 一致
        #   - 拆失败 / 无 split: assistant = answer (即整条 MR)
        # user / assistant 都 [:200] 截断, 与论文侧同尺寸约定。
        # 循环污染风险由 MAX_ASK_TRACK (15) FIFO 控制, 详见 ai_config.MAX_ASK_TRACK 注释。
        if not is_vision:
            split_fu = state.get("split_followup")
            asst_for_track = answer
            if (
                isinstance(split_fu, dict)
                and split_fu.get("from_split")
                and split_fu.get("origin")
            ):
                # split 成功时, answer 已经被 split_followup_node 改写成 main_body
                asst_for_track = answer
            _append_track("ask", {
                "ts": ts,
                "ts_str": dt_str,
                "pdf_fp": fp,
                "user": req.ask[:200],
                "assistant": asst_for_track[:200],
            })

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
    else:
        # Load 模式: 论文 fp 写 load_track (chat fp 不写)
        if not is_chat:
            _append_track("load", {
                "ts": ts,
                "ts_str": dt_str,
                "pdf_fp": fp,
                "chosen_text": req.chosen_text[:200],
                "assistant": answer[:200],
            })
            debug("[crystal_memory] load mode: appended to load_track")
        else:
            debug("[crystal_memory] load mode: skipped (chat fp)")

    return {}


async def _update_crystal_memory_async(
    user_msg: str,
    assistant_msg: str,
    current_timestamp: str = "",
    fp: str = "",
) -> None:
    """
    后台任务: 读 Crystal_memory.md 与 Crystal_self.md, 把当前 fp 窗口内的双轨 track (或 chat 历史)
    + 本轮对话一起喂 LLM, 写回两份笔记。

    ════ Pipeline (按用户设计) ════
      Phase 1:  compress (memory + self)  (LLM × 2, 并发 asyncio.gather)
      Phase 2a: Crystal_memory update      (LLM #3) — 基础:compressed_mem,
                                             上下文: ask_track + load_track + 当前对话
                                             ⚠️ 不加载 Crystal_self
      Phase 2b: Crystal_self   update      (LLM #4) — 基础:compressed_self,
                                             上下文: 更新过的 new_mem
                                             + ask_track + load_track + 当前对话

    输入边界规则 (与原有 memory 链路一致):
      · 用户问 Crystal -> system prompt 只注入 Crystal_memory (load_agent_memory_node
        已 _get_agent_memory()), 不注入 Crystal_self。✅
      · 更新 Crystal_memory -> buildMemoryUpdateUserPrompt 不传 self。✅
      · 更新 Crystal_self   -> buildSelfUpdateUserPrompt 既加载旧 self,
                              也加载刚更新好的 memory。✅

    并发互斥: 模块级 _memory_updating 标志。
    用户短时间多条消息触发多次 update_agent_memory_node, 互斥后第二次起直接 return 跳过。
    计数已经在调用方递增, 下一轮仍会按 MEMORY_UPDATE_EVERY_N 节流再触发。

    异常隔离: 每个 phase 独立 try/except, 任一失败不影响其余 phase。
    失败静默, 不影响用户响应。两份 md 都是锦上添花, 损坏不应阻塞主链路。

    track 取的是**当前论文 fp 窗口内**最近 N 条 ask + M 条 load, 与原逻辑一致。
    ChatView (fp == CHAT_FP) 场景: 直接从 crystal_chat_ai.json 读最近 CHAT_MEMORY_LIMIT 对。
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

        # ── 取上下文轨 (与原逻辑一致) ──
        is_chat = (fp == CHAT_FP)
        if is_chat:
            ask_track, load_track = _load_chat_history_for_memory()
        else:
            ask_track = [
                e for e in (_get_track("ask") or []) if e.get("pdf_fp") == fp
            ]
            load_track = [
                e for e in (_get_track("load") or []) if e.get("pdf_fp") == fp
            ]
        debug(
            f"[crystal_memory] TRACK: is_chat={is_chat} "
            f"ask_count={len(ask_track)} load_count={len(load_track)}"
        )

        # ═══════════════════════════════════════════════════════════════
        # Phase 1: 并发执行 1a (memory compress) + 1b (self compress)
        # ═══════════════════════════════════════════════════════════════
        # 1a 与 1b 完全独立:
        #   - 读: 各自的 md 文件 (不同路径, 无冲突)
        #   - 写: 各自的局部变量 (compressed_mem / compressed_self)
        #   - 共享输入: 仅 current_timestamp (只读)
        # 因此可以 asyncio.gather 并发, 节省 1 次 LLM 调用的 wall time。
        #
        # return_exceptions=True: 任一 LLM 异常不互相波及 (两个 _call_*_compress_llm
        # 内部已有 try/except 返回 fallback tuple, 这里再保险一道)。
        debug(
            f"[crystal_memory] PHASE 1 compress -> LLM [parallel]: "
            f"mem_in={len(current_mem)} self_in={len(current_self)}"
        )
        compress_results = await asyncio.gather(
            _call_memory_compress_llm(current_mem, current_timestamp),
            _call_self_compress_llm(current_self, current_timestamp),
            return_exceptions=True,
        )
        # 解构: 任一结果若是 Exception, 走 fallback (原 md + fallback=True)
        if isinstance(compress_results[0], Exception):
            debug(
                f"[crystal_memory] PHASE 1a memory compress RAISED: "
                f"{type(compress_results[0]).__name__}: {compress_results[0]}"
            )
            compressed_mem, mem_compress_fallback = current_mem, True
        else:
            compressed_mem, mem_compress_fallback = compress_results[0]
        if isinstance(compress_results[1], Exception):
            debug(
                f"[crystal_memory] PHASE 1b self compress RAISED: "
                f"{type(compress_results[1]).__name__}: {compress_results[1]}"
            )
            compressed_self, self_compress_fallback = current_self, True
        else:
            compressed_self, self_compress_fallback = compress_results[1]

        stats["mem_compress_delta"] = len(compressed_mem) - len(current_mem)
        stats["mem_compress_fallback"] = mem_compress_fallback
        stats["self_compress_delta"] = len(compressed_self) - len(current_self)
        stats["self_compress_fallback"] = self_compress_fallback
        debug(
            f"[crystal_memory] PHASE 1 compress <- LLM [parallel]: "
            f"mem=[{len(compressed_mem)}/{stats['mem_compress_delta']:+d}/"
            f"fb={mem_compress_fallback}] "
            f"self=[{len(compressed_self)}/{stats['self_compress_delta']:+d}/"
            f"fb={self_compress_fallback}]"
        )

        # ═══════════════════════════════════════════════════════════════
        # Phase 2a: Crystal_memory update
        #   ⚠️ 不加载 Crystal_self (保持 memory 独立, 避免自我认知污染用户认知)
        # ═══════════════════════════════════════════════════════════════
        debug(
            f"[crystal_memory] PHASE 2a memory update -> LLM: "
            f"input_len={len(compressed_mem)} "
            f"ask={len(ask_track)} load={len(load_track)}"
        )
        try:
            new_mem = await _call_memory_update_llm(
                compressed_mem,
                user_msg,
                assistant_msg,
                current_timestamp,
                ask_track,
                load_track,
            )
            stats["mem_update_delta"] = (len(new_mem) - len(compressed_mem)) if new_mem else 0
            debug(
                f"[crystal_memory] PHASE 2a memory update <- LLM: "
                f"output_len={len(new_mem) if new_mem else 0} "
                f"delta={stats['mem_update_delta']:+d}"
            )
        except Exception as e:
            debug(f"[crystal_memory] PHASE 2a memory update FAIL: {type(e).__name__}: {e}")
            new_mem = None
            stats["mem_update_delta"] = 0

        # 立刻写 memory (这样 Phase 2b 才能拿到「刚更新好的 memory」)
        if new_mem:
            try:
                bytes_written = _write_text_file_atomic(CRYSTAL_MEMORY_FILE, new_mem)
                _set_agent_memory_cache(new_mem)
                debug(
                    f"[crystal_memory] PHASE 2a WRITE ok: "
                    f"path={CRYSTAL_MEMORY_FILE} "
                    f"prev_len={len(current_mem)} new_len={len(new_mem)} "
                    f"delta={len(new_mem)-len(current_mem):+d} "
                    f"bytes_written={bytes_written}"
                )
            except OSError as e:
                debug(f"[crystal_memory] PHASE 2a WRITE FAIL: {type(e).__name__}: {e}")
        else:
            debug("[crystal_memory] PHASE 2a WRITE skipped: new_mem 为空, 保留旧 md")

        # ═══════════════════════════════════════════════════════════════
        # Phase 2b: Crystal_self update
        #   加载: compressed_self (基础) + new_mem (刚更新好的, 作为外部参照)
        #        + ask_track + load_track + 当前对话 + 当前时间戳
        # ═══════════════════════════════════════════════════════════════
        # 如果 Phase 2a 失败了, 用 compressed_mem 作为参照 — 至少不会因为 memory 翻车
        # 而导致 self 拿不到任何上下文。
        mem_for_self = new_mem if new_mem else compressed_mem
        debug(
            f"[crystal_memory] PHASE 2b self update -> LLM: "
            f"self_input_len={len(compressed_self)} mem_ref_len={len(mem_for_self)} "
            f"ask={len(ask_track)} load={len(load_track)}"
        )
        try:
            new_self = await _call_self_update_llm(
                compressed_self,
                mem_for_self,
                user_msg,
                assistant_msg,
                current_timestamp,
                ask_track,
                load_track,
            )
            stats["self_update_delta"] = (len(new_self) - len(compressed_self)) if new_self else 0
            debug(
                f"[crystal_memory] PHASE 2b self update <- LLM: "
                f"output_len={len(new_self) if new_self else 0} "
                f"delta={stats['self_update_delta']:+d}"
            )
        except Exception as e:
            debug(f"[crystal_memory] PHASE 2b self update FAIL: {type(e).__name__}: {e}")
            new_self = None
            stats["self_update_delta"] = 0

        if new_self:
            try:
                bytes_written = _write_text_file_atomic(CRYSTAL_SELF_FILE, new_self)
                _set_agent_self_cache(new_self)
                debug(
                    f"[crystal_memory] PHASE 2b WRITE ok: "
                    f"path={CRYSTAL_SELF_FILE} "
                    f"prev_len={len(current_self)} new_len={len(new_self)} "
                    f"delta={len(new_self)-len(current_self):+d} "
                    f"bytes_written={bytes_written}"
                )
            except OSError as e:
                debug(f"[crystal_memory] PHASE 2b WRITE FAIL: {type(e).__name__}: {e}")
        else:
            debug("[crystal_memory] PHASE 2b WRITE skipped: new_self 为空, 保留旧 md")

    except Exception as e:
        debug(f"[crystal_memory] update FAIL: {type(e).__name__}: {e}")
    finally:
        _memory_updating = False
        debug(
            f"[crystal_memory] <<< END: "
            f"mem_d=[{stats.get('mem_compress_delta', 0):+d}/"
            f"{stats.get('mem_update_delta', 0):+d}] "
            f"self_d=[{stats.get('self_compress_delta', 0):+d}/"
            f"{stats.get('self_update_delta', 0):+d}]"
        )


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
    ask_track: list[dict] | None = None,
    load_track: list[dict] | None = None,
) -> str:
    """
    调 LLM 更新 Crystal_memory.md, 返回新的 Markdown 文本。

    参数:
      current_timestamp: 秒级可读时间字符串 (来自 now_ms + format_dt_second)
      ask_track: 当前 fp 论文窗口内的最近 N 条 ask (按 pdf_fp 过滤)
      load_track: 当前 fp 论文窗口内的最近 M 条 load (按 pdf_fp 过滤)

    ⚠️ 不传入 self: memory 只关心用户认知, 不载入 Crystal 关于自己的笔记。
    """
    messages = [
        {"role": "system", "content": MEMORY_UPDATE_SYSTEM()},
        {"role": "user", "content": buildMemoryUpdateUserPrompt(
            current_md, user_msg, assistant_msg,
            current_timestamp, ask_track, load_track,
        )},
    ]
    content, _ = await _call_llm(messages=messages, vision_model=False, disable_thinking=True)
    return content.strip()


async def _call_memory_compress_llm(
    current_md: str,
    current_timestamp: str = "",
) -> tuple[str, bool]:
    """
    Phase 1 compress: 按时间分层压缩旧 Crystal_memory.md。

    返回 (compressed_md, fallback):
      - compressed_md: 压缩后的 Markdown (失败时回落到 current_md)
      - fallback: True 表示走了 fallback 路径 (LLM 失败 / 返回空), False 表示正常压缩
    """
    if not current_md:
        # 空记忆无压缩必要, 直接返回空串
        return current_md, False

    messages = [
        {"role": "system", "content": MEMORY_COMPRESS_SYSTEM()},
        {"role": "user", "content": buildMemoryCompressUserPrompt(
            current_md, current_timestamp,
        )},
    ]
    try:
        content, _ = await _call_llm(messages=messages, vision_model=False, disable_thinking=True)
        compressed = content.strip()
        if not compressed:
            debug("[crystal_memory] memory compress returned empty, fallback to original")
            return current_md, True
        return compressed, False
    except Exception as e:
        debug(f"[crystal_memory] memory compress FAIL: {type(e).__name__}: {e}")
        return current_md, True


async def _call_self_update_llm(
    current_self_md: str,
    updated_memory_md: str,
    user_msg: str,
    assistant_msg: str,
    current_timestamp: str = "",
    ask_track: list[dict] | None = None,
    load_track: list[dict] | None = None,
) -> str:
    """
    调 LLM 更新 Crystal_self.md, 返回新的 Markdown 文本。

    与 memory update 的关键差异:
      · 基础是压缩后的旧 self
      · 多喂一个「刚更新好的 memory」作为外部参照 (让 LLM 知道「我对用户的认知」,
        反向校准自我定位)
    """
    messages = [
        {"role": "system", "content": SELF_UPDATE_SYSTEM()},
        {"role": "user", "content": buildSelfUpdateUserPrompt(
            current_self_md,
            updated_memory_md,
            user_msg,
            assistant_msg,
            current_timestamp,
            ask_track,
            load_track,
        )},
    ]
    content, _ = await _call_llm(messages=messages, vision_model=False, disable_thinking=True)
    return content.strip()


async def _call_self_compress_llm(
    current_self_md: str,
    current_timestamp: str = "",
) -> tuple[str, bool]:
    """
    Phase 1 compress for Crystal_self.md: 与 memory compress 完全对称。

    返回 (compressed_md, fallback), 行为与 _call_memory_compress_llm 一致。
    """
    if not current_self_md:
        # 空自我笔记无压缩必要, 直接返回空串 (首次记录场景)
        return current_self_md, False

    messages = [
        {"role": "system", "content": SELF_COMPRESS_SYSTEM()},
        {"role": "user", "content": buildSelfCompressUserPrompt(
            current_self_md, current_timestamp,
        )},
    ]
    try:
        content, _ = await _call_llm(messages=messages, vision_model=False, disable_thinking=True)
        compressed = content.strip()
        if not compressed:
            debug("[crystal_memory] self compress returned empty, fallback to original")
            return current_self_md, True
        return compressed, False
    except Exception as e:
        debug(f"[crystal_memory] self compress FAIL: {type(e).__name__}: {e}")
        return current_self_md, True


async def flush_track_node(state: PaperAIState) -> dict:
    """
    对话结束后的最后一个节点: 把 ask + load 两条 track 缓存一次性写盘 (覆盖式)。

    之前 update_agent_memory_node 用 _append_track 累积到内存 cache + 标 dirty;
    这里统一做一次 flush, 避免高频小 I/O。
    """
    _flush_track_to_disk("ask")
    _flush_track_to_disk("load")
    return {}
