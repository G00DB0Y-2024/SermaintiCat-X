"""
AI HTTP 端点 — /ai/config、/ai/ask、/ai/load、/ai/history。

设计:
- /ai/config (POST): 前端 SET_MODEL 时调用, 更新 ai_agent.ai_config 模块变量。
                    请求体: {api_key, api_url, model, vision_model, deepseek_thinking}
- /ai/config (GET):  返回当前 ai_config 快照。
- /ai/ask (POST):    用户自由提问, 读 ai_config 调 LLM。
- /ai/load (POST):   用户选中文本要求总结, 读 ai_config 调 LLM。
- /ai/history (POST): 读取 save/{fp}_ai.json 原始 entry 列表 (供 ChatView 加载历史)。
"""
from __future__ import annotations

import os
import json

from fastapi import APIRouter, HTTPException

from .ai_models import (
    AiConfigReq, AiConfigResp,
    AiAskReq, AiLoadReq, AiResp,
    AiHistoryReq, AiHistoryResp,
)
from . import ai_agent
from utils.log import debug


router = APIRouter(prefix="/ai", tags=["ai"])


# ── /ai/config ──────────────────────────────────────────────────────────

@router.post("/config")
async def ai_config(req: AiConfigReq) -> AiConfigResp:
    """
    前端 SET_MODEL 时调用。
    将 api_key / api_url / model / vision_model / deepseek_thinking
    存入 ai_agent.ai_config 模块变量,后续 /ai/ask 和 /ai/load 直接读取。
    """
    ai_agent.update_ai_config(req.model_dump())
    debug(
        f"[/ai/config] api_url={req.api_url!r}  "
        f"model={req.model!r}  vision_model={req.vision_model!r}  "
        f"deepseek_thinking={req.deepseek_thinking}"
    )
    return AiConfigResp(**ai_agent.get_current_config())


@router.get("/config", response_model=AiConfigResp)
async def ai_config_get() -> AiConfigResp:
    """返回当前 ai_config 快照, 用于前端调试。"""
    return AiConfigResp(**ai_agent.get_current_config())


# ── /ai/ask ────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AiResp)
async def ai_ask(req: AiAskReq) -> AiResp:
    """
    Ask 模式 — 用户在论文阅读过程中自由提问。
    支持 vision(图文多模态, 通过 req.image_base64 非空判定)。
    api_key / api_url / model / vision_model 来自 ai_config。
    """
    try:
        result = await ai_agent.run_ask(req)
        return AiResp(
            content=result.get("final_answer", ""),
            usage=result.get("usage") or None,
            dt=result.get("dt", ""),
        )
    except HTTPException:
        raise
    except Exception as e:
        debug(f"[/ai/ask] ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# ── /ai/load ───────────────────────────────────────────────────────────

@router.post("/load", response_model=AiResp)
async def ai_load(req: AiLoadReq) -> AiResp:
    """
    Load 模式 — 用户选中文本, 要求 Crystal 总结 / 解释。
    api_key / api_url / model 来自 ai_config。
    """
    try:
        result = await ai_agent.run_load(req)
        return AiResp(
            content=result.get("final_answer", ""),
            usage=result.get("usage") or None,
            dt=result.get("dt", ""),
        )
    except HTTPException:
        raise
    except Exception as e:
        debug(f"[/ai/load] ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# ── /ai/history ────────────────────────────────────────────────────────

@router.post("/history", response_model=AiHistoryResp)
async def ai_history(req: AiHistoryReq) -> AiHistoryResp:
    """
    读取 save/{pdf_fp}_ai.json 原始 entry 列表。

    用途: ChatView 进入页面时拉取历史, 把 ReqAsk / ResAsk 还原成 UI 消息列表。
    物理位置由 ai_agent.SAVE_DIR 统一维护, 避免在 routes 层再算一遍路径。

    返回:
        AiHistoryResp.entries = list[dict], 每条至少含 {type, content, ts, dt}。
        缺文件 / 解析失败 / 文件为空 → 返回空列表 (不报错)。
    """
    save_path = os.path.join(ai_agent.SAVE_DIR, f"{req.pdf_fp}_ai.json")
    debug(f"[/ai/history] pdf_fp={req.pdf_fp!r} path={save_path!r}")

    if not os.path.exists(save_path):
        return AiHistoryResp(entries=[])

    try:
        with open(save_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        debug(f"[/ai/history] read failed: {type(e).__name__}: {e}")
        return AiHistoryResp(entries=[])

    # 防御: 文件存在但不是列表(被外部覆盖), 兜底返回空
    if not isinstance(raw, list):
        debug(f"[/ai/history] unexpected top-level type: {type(raw).__name__}")
        return AiHistoryResp(entries=[])

    return AiHistoryResp(entries=raw)
