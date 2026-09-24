"""
AI 模块的 Pydantic 请求 / 响应模型。

设计变更 (简化版):
- 前端 SET_MODEL 时 POST /ai/config, 后端维护 ai_config 模块变量;
  ask / load 请求不再每次透传 api_key / api_url / model。
- ask / load 请求只包含业务参数 (pdf_fp, ask, quotes …),
  LLM 调用直接从 ai_config 读取 api_key / api_url / model / vision_model。

Schema 变更 (v2):
- save/{fp}_ai.json 从 [{role, content, ts}] 改为 [{type, content, ts, dt, ...}]
  type 取值: ReqLoad | ResLoad | ReqAsk | ResAsk | Anno
"""
from enum import Enum
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
    image_filename: Optional[str] = Field(
        default="",
        description="Vision 图片文件名(写盘用), 空字符串表示非视觉模式",
    )
    hl: Optional[dict] = Field(
        default=None,
        description="chosen_hldata, 用户选中的高亮上下文",
    )
    device: Optional[str] = Field(
        default=None,
        description="发送设备类型: 'desktop' | 'mobile' | 'tablet' | None(未知)",
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
    """统一响应: content 是 Crystal 的最终回复, usage 是上游 LLM usage 统计 (可选), dt 是服务端时间。"""
    content: str
    usage: Optional[dict] = None
    dt: str = Field(
        default="",
        description="服务端时间戳字符串 (北京时间, 格式: YYYY-MM-DD HH:MM:SS)。"
                    "前端应优先使用此字段而非本地时间, 确保前后端时区一致。"
                    "空字符串表示后端未提供 (兼容老接口)。",
    )


# ── 新 Schema Entry 类型 ──────────────────────────────────────────────
# 用于 save/{fp}_ai.json v2 格式: [{type, content, ts, dt, ...}]


class AiMsgType(str, Enum):
    """_ai.json entry 的 type 枚举。"""
    REQ_LOAD = "ReqLoad"
    RES_LOAD = "ResLoad"
    REQ_ASK  = "ReqAsk"
    RES_ASK  = "ResAsk"
    ANNO     = "Anno"


class AiHistoryReq(BaseModel):
    """ChatView 进入页面时调用, 拉取某 fp 的历史条目列表。"""
    pdf_fp: str = Field(..., description="会话 fp, 聊天场景固定为 crystal_chat")


class AiHistoryResp(BaseModel):
    """save/{fp}_ai.json 里的全部 entry 列表 (含 ReqAsk/ResAsk)。"""
    entries: list[dict] = Field(
        default_factory=list,
        description="原始 entry 列表, 每条至少有 {type, content, ts, dt}",
    )


class AiPaperEntry(BaseModel):
    """
    save/{fp}_ai.json 的单条 entry。

    字段说明:
      type:        "ReqLoad"|"ResLoad"|"ReqAsk"|"ResAsk"|"Anno"
      content:     文本内容
      ts:          UTC ms, now_ms() 单源
      dt:          "YYYY-MM-DD HH:MM:SS" 北京时间
      hl:          论文高亮上下文 (仅 Req/Res 有)
      quote_gids:  Ask 引用段 ID 列表
      img:         图片文件名 (Vision 模式); Anno 模式也可能有
      token_count: 仅 Res 有, LLM token 消耗
      anno_id:     仅 Anno 有, "ANNO_{fp}_{ts}"
    """
    type: str
    content: str
    ts: int
    dt: str
    hl: Optional[dict] = None
    quote_gids: Optional[list[str]] = None
    img: Optional[str] = ""
    token_count: Optional[int] = None
    anno_id: Optional[str] = None

