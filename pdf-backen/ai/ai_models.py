"""
AI 模块的 Pydantic 请求 / 响应模型。

设计变更 (简化版):
- 前端 SET_MODEL 时 POST /ai/config, 后端维护 ai_config 模块变量;
  ask / load 请求不再每次透传 api_key / api_url / model。
- ask / load 请求只包含业务参数 (pdf_fp, ask, quotes …),
  LLM 调用直接从 ai_config 读取 api_key / api_url / model / vision_model。
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── /ai/config ──────────────────────────────────────────────────────────

class AiConfigReq(BaseModel):
    """前端 SET_MODEL 时 POST 此请求, 更新后端 ai_config。"""
    api_key: str = Field(default="", description="上游 LLM API Key")
    api_url: str = Field(default="", description="上游 LLM API 地址")
    model: str = Field(default="", description="普通文本模型 (ask / load)")
    vision_model: str = Field(
        default="",
        description="视觉模型 (ask + image_base64)。空则 fallback 到 model",
    )
    deepseek_thinking: bool = Field(
        default=False,
        description="DeepSeek 思考模式开关 (仅 DeepSeek 非视觉模型生效)",
    )


class AiConfigResp(BaseModel):
    """GET /ai/config 返回当前快照。"""
    api_key: str
    api_url: str
    model: str
    vision_model: str
    deepseek_thinking: bool


# ── /ai/ask ────────────────────────────────────────────────────────────

class AiAskReq(BaseModel):
    """Ask 模式 — 用户在论文阅读过程中自由提问。"""
    pdf_fp: str = Field(..., description="论文指纹, 用于隔离 save/{fp}_ai.json")
    ask: str = Field(..., description="用户的提问内容")
    quotes: list[dict] = Field(
        default_factory=list,
        description="用户引用的原文选段 [{quote_gid, quote_msg}]",
    )
    quote_content: str = Field(
        default="",
        description="getQuoteContent() 展开的完整引用文本",
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="图片 base64 (含 data:image/...;base64, 前缀), null 表示非视觉模式",
    )


# ── /ai/load ──────────────────────────────────────────────────────────

class AiLoadReq(BaseModel):
    """Load 模式 — 用户选中论文文本, 要求 Crystal 总结 / 解释。"""
    pdf_fp: str = Field(..., description="论文指纹")
    chosen_text: str = Field(..., description="用户选中的论文文本")
    added_prompt: str = Field(
        default="",
        description="用户额外指令 (空则用默认: 用中文准确概括)",
    )


# ── 统一响应 ───────────────────────────────────────────────────────────

class AiResp(BaseModel):
    """统一响应: content 是 Crystal 的最终回复, usage 是上游 LLM usage 统计 (可选)。"""
    content: str
    usage: Optional[dict] = None
