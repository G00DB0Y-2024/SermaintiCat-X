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
    LOAD_LOCAL_LIMIT,
    MEMORY_UPDATE_EVERY_N,
)
from .ai_io import (
    _agent_memory_cache,  # noqa: F401  # 由 _set_agent_memory_cache 同包维护
    _append_track,
    _ask_count_by_fp,
    _ask_track_list,  # noqa: F401  # 模块单例引用保持
    _flush_track_to_disk,
    _get_agent_memory,
    _get_track,
    _invalidate_history_cache,
    _load_chat_history_for_memory,
    _load_paper_history,
    _paper_history_path,
    _read_json_safe,
    _set_agent_memory_cache,
)
from .ai_llm import _call_llm
from .ai_models import AiAskReq, AiLoadReq
from .ai_utils import now_ms, format_dt_second
from .prompts_context import (
    build_track_summary_block,
    buildMemoryCompressUserPrompt,
    buildMemoryUpdateUserPrompt,
)
from .prompts_system import MEMORY_COMPRESS_SYSTEM, MEMORY_UPDATE_SYSTEM, SYSTEM_LOAD
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
    paper_history:       兼容字段, 论文侧即为 paper_ask_history
    paper_ask_history:   当前 pdf_fp 最近 ASK_LOCAL_LIMIT 对 ask 历史 (role 交替)
    paper_load_history:  当前 pdf_fp 最近 LOAD_LOCAL_LIMIT 对 load 历史 (role 交替)
    final_answer:        llm_call 返回的最终 content
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
    paper_history: list[dict]
    paper_ask_history: list[dict]
    paper_load_history: list[dict]
    final_answer: str
    usage: dict
    dt: str
    req_fp: str
    res_fp: str


# ═══════════════════════════════════════════════════════════════════════
# LangGraph 节点
# ═══════════════════════════════════════════════════════════════════════

async def load_agent_memory_node(state: PaperAIState) -> dict:
    """读 agent_memory 缓存(首次才打磁盘)。作为 system prompt 注入。

    缓存策略: 模块级 _agent_memory_cache, 首次调用时同步从 Crystal_memory.md 加载,
    之后 _update_crystal_memory_async 写完文件会同步刷新缓存。

    注: 当前不做长度截断, MEMORY_MAX_CHARS 保留为占位 (后续方案处理)。
    """
    return {"agent_memory": _get_agent_memory()}


async def load_paper_history_node(state: PaperAIState) -> dict:
    """
    一次性加载当前 pdf_fp 的 ask + load 两条历史轨。

    设计: 论文侧上下文按 fp 隔离, 不再读全局 track。
    但 ask 轨需要 load 轨作为辅助概要 (论文中的选段总结能帮 LLM 理解上下文),
    所以两个都加载, 各自按 limit 上限截取, 写入 state 供后续 compose_messages_node 拼装。

    ChatView 场景 (pdf_fp == "crystal_chat"):
      - ask_history 作为 Chat 本地上下文 (CHAT_LOCAL_LIMIT 对)
      - load_history 暂不使用 (Chat 不会 Load)

    Vision Ask 和 Anno 在 _load_paper_history 内部已过滤。
    """
    req = state["req"]
    fp = req.pdf_fp
    is_chat = (fp == CHAT_FP)

    if isinstance(req, AiAskReq):
        # Ask 模式 (论文 + Chat): 加载本 fp ask 历史
        ask_limit = CHAT_LOCAL_LIMIT if is_chat else ASK_LOCAL_LIMIT
        ask_history = _load_paper_history(fp, mode="ask", limit=ask_limit)
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
      [system]  CrystalPersona + time + agent_mem + "全局闲聊" + 论文 track 摘要块
      [chat_msgs]  ChatView 本地近 Z=CHAT_LOCAL_LIMIT 对
      [user]    本轮提问
      -> 论文全局 track 通过 _get_track 注入 (track 仅含论文, 因为 _append_track 已过滤)

    Load 模式: 人设 + 本论文 Load 历史 (LOAD_LOCAL_LIMIT 对)
    """
    from .prompts_context import compose_chat_messages, compose_paper_ask_messages, build_load_user_content
    req = state["req"]
    agent_mem = state.get("agent_memory") or ""
    time_context = get_current_time_context()
    fp = req.pdf_fp
    is_chat = (fp == CHAT_FP)

    if isinstance(req, AiAskReq):
        ask_history = state.get("paper_ask_history") or []
        load_history = state.get("paper_load_history") or []

        if is_chat:
            messages = compose_chat_messages(
                req, ask_history, agent_mem, time_context
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
        (_ask_count_by_fp[fp] % MEMORY_UPDATE_EVERY_N) == 0
      即第 N/MEMORY_UPDATE_EVERY_N 次 ask 时 (例如 N=1 表示每次) 触发。

    注意: 函数本身不做 vision 过滤 (那在 update_agent_memory_node 入口处判断),
    这里只做 is_chat 参与下的节流逻辑 ——
    现在 is_chat 不再硬短路, 与论文走相同路径, 让 MEMORY_UPDATE_EVERY_N 在两个场景都生效。
    """
    return (_ask_count_by_fp.get(fp, 0) % MEMORY_UPDATE_EVERY_N) == 0


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
            _ask_count_by_fp[fp] = _ask_count_by_fp.get(fp, 0) + 1
            trigger_mem = _should_update_memory(fp, is_chat)
        else:
            trigger_mem = False

        # 非 chat 场景下, 把 ask 追加到论文全局 track (track_summary_block 喂 ChatView)。
        # chat 不写 track, 是为了避免 chat 自己的对话回灌到 chat system prompt 引起循环污染。
        if not is_chat and not is_vision:
            _append_track("ask", {
                "ts": ts,
                "ts_str": dt_str,
                "pdf_fp": fp,
                "user": req.ask[:200],
                "assistant": answer[:200],
            })

        if trigger_mem:
            # 已有 memory update 在跑, 跳过 ——
            # 计数已递增, 下一轮仍会按 MEMORY_UPDATE_EVERY_N 节流再触发。
            if _memory_updating:
                debug(
                    f"[crystal_memory] SKIP (busy at node): fp={fp[:12]} "
                    f"count={_ask_count_by_fp[fp]}"
                )
                return {}
            task = asyncio.create_task(
                _update_crystal_memory_async(req.ask, answer, dt_str, fp)
            )
            debug(
                f"[crystal_memory] scheduled [ask throttled]: fp={fp[:12]} "
                f"count={_ask_count_by_fp[fp]} "
                f"user_len={len(req.ask)} asst_len={len(answer)} "
                f"task_id={id(task)}"
            )
        else:
            debug(
                f"[crystal_memory] SKIP: is_chat={is_chat} is_vision={is_vision} "
                f"count={_ask_count_by_fp.get(fp, 0)}/{MEMORY_UPDATE_EVERY_N}"
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
    后台任务: 读 Crystal_memory.md, 把当前 fp 窗口内的双轨 track (或 chat 历史) +
    本轮对话一起喂 LLM, 写回。

    复用 ai_config 中的 api_key / api_url / model。

    并发互斥: 模块级 _memory_updating 标志 + _memory_lock。
    用户短时间内多条消息会触发多次 update_agent_memory_node, 每次都
    asyncio.create_task 本函数。互斥后, 第二次起会立即跳过 —
    跳过的 task 因为没持有 _ask_count_by_fp 的递增副作用, 也不改 Crystal_memory.md,
    不会与正在运行的 update 产生 race。计数已经递增, 下一轮仍会按
    MEMORY_UPDATE_EVERY_N 节流再触发。

    异常静默, 不影响用户响应。Crystal_mem.md 是锦上添花, 损坏不应阻塞主链路。

    track 取的是**当前论文 fp 窗口内**最近 N 条 ask + M 条 load (跨论文 track 已被
    按 pdf_fp 过滤), 这样 memory update 只反映当前论文的对话, 不会混入其他论文内容。

    ChatView (fp == CHAT_FP) 场景:
      - 论文 track 里没有 chat fp 记录 (_append_track 故意过滤 chat, 避免污染
        chat 自己的 system prompt), 所以 chat 直接从 crystal_chat_ai.json
        读最近 CHAT_MEMORY_LIMIT 条 ReqAsk/ResAsk 对, 转成 {user, assistant} 形式
        喂给 prompt。这样 memory 既能吸收 chat 真实脉络, 又不绕回污染 ChatView 上下文。
      - chat 没有 load 轨, load_track 为空。
    """
    from .ai_io import _ensure_memory_dir
    from .ai_config import CRYSTAL_MEMORY_FILE

    # 并发互斥: 已有 task 在跑, 直接放弃本次 (计数已在调用方递增, 下一轮仍会触发)
    global _memory_updating
    if _memory_updating:
        debug(f"[crystal_memory] SKIP (busy): fp={fp[:12]} ts={current_timestamp}")
        return
    _memory_updating = True

    # 入口摘要
    debug(
        f"[crystal_memory] >>> START: fp={fp[:12]} ts={current_timestamp} "
        f"user_len={len(user_msg)} asst_len={len(assistant_msg)} "
        f"thinking=off"
    )

    try:
        if not _ai_config["api_key"] or not _ai_config["api_url"]:
            debug("[crystal_memory] skip: ai_config 未设置")
            return

        _ensure_memory_dir()

        # 读当前 Markdown
        current = ""
        if os.path.exists(CRYSTAL_MEMORY_FILE):
            try:
                with open(CRYSTAL_MEMORY_FILE, "r", encoding="utf-8") as f:
                    current = f.read()
            except OSError:
                current = ""
        debug(
            f"[crystal_memory] READ ok: path={CRYSTAL_MEMORY_FILE} "
            f"current_len={len(current)} "
            f"current_lines={current.count(chr(10)) + (1 if current else 0)}"
        )

        # 取"上下文轨" ——
        #   论文 fp: 双轨 track (按 pdf_fp 过滤), 反映当前论文最近 N+M 对
        #   chat fp: 直接从 crystal_chat_ai.json 读最近 CHAT_MEMORY_LIMIT 对 ReqAsk/ResAsk
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

        # ── Phase 1: compress ──
        # 按时间分层压缩旧记忆 (LLM 调用 1)。失败时回落到原 md, 不阻塞 update。
        debug(
            f"[crystal_memory] PHASE1 compress -> LLM: "
            f"input_len={len(current)} now={current_timestamp}"
        )
        compressed_md, compress_fallback = await _call_memory_compress_llm(
            current, current_timestamp
        )
        compress_delta = len(compressed_md) - len(current)
        debug(
            f"[crystal_memory] PHASE1 compress <- LLM: "
            f"output_len={len(compressed_md)} delta={compress_delta:+d} "
            f"fallback={compress_fallback} "
            f"output_lines={compressed_md.count(chr(10)) + (1 if compressed_md else 0)}"
        )

        # ── Phase 2: update ──
        # 在压缩后的 md 基础上, 融入本轮对话 + 双轨 track, 写出最终 md (LLM 调用 2)。
        debug(
            f"[crystal_memory] PHASE2 update -> LLM: "
            f"input_len={len(compressed_md)} "
            f"ask={len(ask_track)} load={len(load_track)} "
            f"user_len={len(user_msg)} asst_len={len(assistant_msg)}"
        )
        new_md = await _call_memory_update_llm(
            compressed_md,
            user_msg,
            assistant_msg,
            current_timestamp,
            ask_track,
            load_track,
        )
        update_delta = (len(new_md) - len(compressed_md)) if new_md else 0
        debug(
            f"[crystal_memory] PHASE2 update <- LLM: "
            f"output_len={len(new_md) if new_md else 0} delta={update_delta:+d} "
            f"output_lines={(new_md or '').count(chr(10)) + (1 if new_md else 0)}"
        )

        if new_md:
            with open(CRYSTAL_MEMORY_FILE, "w", encoding="utf-8") as f:
                f.write(new_md)
                bytes_written = f.tell()
            _set_agent_memory_cache(new_md)
            debug(
                f"[crystal_memory] WRITE ok: "
                f"path={CRYSTAL_MEMORY_FILE} "
                f"prev_len={len(current)} new_len={len(new_md)} "
                f"delta={len(new_md)-len(current):+d} "
                f"compress_delta={compress_delta:+d} update_delta={update_delta:+d} "
                f"compress_fallback={compress_fallback} "
                f"bytes_written={bytes_written} "
                f"prev_lines={current.count(chr(10)) + (1 if current else 0)} "
                f"new_lines={new_md.count(chr(10)) + 1}"
            )
        else:
            debug("[crystal_memory] WRITE skipped: new_md 为空, 保留旧 md")
    except Exception as e:
        debug(f"[crystal_memory] update FAIL: {type(e).__name__}: {e}")
    finally:
        _memory_updating = False
        debug("[crystal_memory] <<< END")


async def _call_memory_update_llm(
    current_md: str,
    user_msg: str,
    assistant_msg: str,
    current_timestamp: str = "",
    ask_track: list[dict] | None = None,
    load_track: list[dict] | None = None,
) -> str:
    """
    调 LLM 更新 Crystal_mem.md, 返回新的 Markdown 文本。

    参数:
      current_timestamp: 秒级可读时间字符串 (来自 now_ms + format_dt_second)
      ask_track: 当前 fp 论文窗口内的最近 N 条 ask (按 pdf_fp 过滤)
      load_track: 当前 fp 论文窗口内的最近 M 条 load (按 pdf_fp 过滤)
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
            debug("[crystal_memory] compress returned empty, fallback to original")
            return current_md, True
        return compressed, False
    except Exception as e:
        debug(f"[crystal_memory] compress FAIL: {type(e).__name__}: {e}")
        return current_md, True


async def flush_track_node(state: PaperAIState) -> dict:
    """
    对话结束后的最后一个节点: 把 ask + load 两条 track 缓存一次性写盘 (覆盖式)。

    之前 update_agent_memory_node 用 _append_track 累积到内存 cache + 标 dirty;
    这里统一做一次 flush, 避免高频小 I/O。
    """
    _flush_track_to_disk("ask")
    _flush_track_to_disk("load")
    return {}
