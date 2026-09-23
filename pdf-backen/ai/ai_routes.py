"""
AI HTTP 端点 — /ai/config、/ai/ask、/ai/load。

设计:
- /ai/config (POST): 前端 SET_MODEL 时调用, 更新 ai_agent.ai_config 模块变量。
                    请求体: {api_key, api_url, model, vision_model, deepseek_thinking}
- /ai/config (GET):  返回当前 ai_config 快照。
- /ai/ask (POST):    用户自由提问, 读 ai_config 调 LLM。
- /ai/load (POST):   用户选中文本要求总结, 读 ai_config 调 LLM。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .ai_models import (
    AiConfigReq, AiConfigResp,
    AiAskReq, AiLoadReq, AiResp,
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
