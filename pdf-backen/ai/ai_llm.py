"""
LLM 调用层：URL 解析 + httpx 封装。

把 ai_agent 中的 _is_deepseek / _is_openai_compat / _resolve_api_url / _call_llm
整体迁出,ai_llm 唯一职责是"怎么和上游 LLM 对话"。
"""
from __future__ import annotations

import json

import httpx

from .ai_config import _ai_config, DEEPSEEK_MARKER
from utils.log import debug


# ═══════════════════════════════════════════════════════════════════════
# URL 解析
# ═══════════════════════════════════════════════════════════════════════

def is_deepseek(api_url: str, model: str) -> bool:
    """
    统一 DeepSeek 检测: url 或 model 名称中任一含 deepseek 即视为 DeepSeek。

    理由:
      · 官方 deepseek.com 端点 → URL 含 DEEPSEEK_MARKER
      · aihubmix / 第三方代理若转发 DeepSeek 模型 (如 deepseek-flash),模型名含 deepseek
      · 两者都支持 DeepSeek thinking 协议, 都走同一套 request_body 参数注入逻辑。
    """
    url_clean = (api_url or "").lower()
    model_clean = (model or "").lower()
    return DEEPSEEK_MARKER in url_clean or "deepseek" in model_clean


def _is_openai_compat(api_url: str) -> bool:
    """判断 base URL 是否已带 API 路径(如 /v1)。"""
    return bool(api_url) and ("/v1" in api_url or "/v1/" in api_url)


def _resolve_api_url(api_url: str, model: str) -> str:
    base = api_url.rstrip("/")
    if is_deepseek(api_url, model):
        return f"{base}/chat/completions"
    # OpenAI 兼容 API(aihubmix、Azure、第三方代理等):
    # 如果 URL 已带路径(如 /v1)则保持原样,否则追加 /chat/completions
    if _is_openai_compat(base):
        return base
    return f"{base}/chat/completions"


# ═══════════════════════════════════════════════════════════════════════
# LLM 调用
# ═══════════════════════════════════════════════════════════════════════

async def _call_llm(
    messages: list[dict],
    vision_model: bool = False,
    disable_thinking: bool | None = None,  # ⚠️ 改为 None: 跟随全局 deepseek_thinking (None=未指定)
    *,
    thinking_effort: int | None = None,    # DeepSeek 思考强度, 仅 DeepSeek 模型生效
                                             # None  → 跟随 cfg["deepseek_thinking"] 全局开关
                                             # 0     → 强制关闭 thinking (reasoning.effort="none")
                                             # 1/2/3 → low/high/max (仅 enable_thinking 时有效)
    emotion_config: dict | None = None,
    json_mode: bool = False,
) -> tuple[str, dict]:
    """
    直接调上游 LLM, 配置从 ai_config 读取 (或 emotion_config 覆盖)。
    vision_model 参数仅用于判断 DeepSeek thinking 开关(仅非视觉模式有效)。

    emotion_config:
      可选 dict {"api_key", "api_url", "model"} 用于覆盖 ai_config。
      当前仅 emotion 模块调此参数, 让 Lint LLM 走独立配置。

    disable_thinking:
      None  (默认) → 跟随 cfg["deepseek_thinking"] 全局开关 (True=开思考, False=关)
      True  → 强制关闭 DeepSeek thinking 模式 (无视全局开关)
      False → 强制启用 thinking (无视全局开关)
      用于结构化输出任务 (memory compress / update), 这些任务不需要 reasoning,
      开 thinking 会白白烧 token + 拖慢响应 — 这些任务显式传 True 关掉。
      Chat / ask 路径传 None 即可, 行为由前端用户开关 (params.json deepseek_thinking) 决定。

    thinking_effort (2026-10-16 新增):
      DeepSeek 思考强度, 由 thinking_effort_node 提前判断注入。
      仅 DeepSeek 模型生效; 非 DeepSeek 模型此参数被忽略。
      映射:
        0 → reasoning.effort="none" + thinking.type="disabled" (完全关闭)
        1 → reasoning.effort="low"
        2 → reasoning.effort="high"   (默认值,兼容旧逻辑)
        3 → reasoning.effort="max"
      注意: thinking_effort=0 时 disable_thinking=True 语义相同,但 thinking_effort
      优先级更高 (由 lint 决定), disable_thinking 仅作为结构化任务的强制覆盖手段。
      双重保险: 同时塞 thinking.type + reasoning.effort 两套字段, 跨 provider 兼容。

    json_mode:
      True 时在 request_body 里加 "response_format": {"type": "json_object"},
      让 LLM 走 OpenAI 兼容的 json 模式 (DeepSeek / aihubmix / Gemini 兼容层都支持)。
      用于结构化输出 (如 memory_and_self 合并 update 的 JSON {updated_memory,
      updated_self}), 显著降低解析失败率。

      注意: json_mode 不会自动重试; 解析失败由调用方决定是否 fallback。
      弱模型 (gemini flash) 在 json_object 下偶发空字符串, 调用方
      _safe_parse_update_json 已有兜底, 这里不重复处理。

    返回: (content, usage, reasoning_content)
      - content           : LLM 回复正文; 解析失败时回落到完整 JSON dump
      - usage             : 上游 LLM usage 统计 (可能为空 dict)
      - reasoning_content : DeepSeek 等支持 thinking 的模型返回的思考过程
                            (OpenAI 标准的 reasoning 字段, 兼容 aihubmix / 第三方代理)
                            无 thinking 时为 None — 调用方不应在落库 / 返前端时
                            写入 None 字段, 仅在模型真的返回了非空思考时才保存。
    """
    # 优先用 emotion_config 覆盖; 否则回落到全局 ai_config
    cfg = emotion_config if emotion_config else _ai_config
    api_url = cfg["api_url"]
    api_key = cfg["api_key"]
    model = cfg["vision_model"] if vision_model else cfg["model"]

    if not api_url or not api_key or not model:
        raise ValueError(
            f"ai_config 未完整配置: api_url={bool(api_url)} "
            f"api_key={bool(api_key)} model={bool(model)}"
        )

    request_url = _resolve_api_url(api_url, model)
    ds = is_deepseek(api_url, model)

    request_body: dict = {"model": model, "messages": messages}

    if json_mode:
        # OpenAI 兼容的 json_object 模式 (DeepSeek / aihubmix / Gemini 兼容层都支持)。
        # 强制 LLM 输出合法 JSON 对象, 大幅降低后端解析失败率。
        request_body["response_format"] = {"type": "json_object"}

    # ---------------------------------------------------------------
    # DeepSeek thinking 注入逻辑 (2026-10-16 重构: thinking_effort 主导)
    # ---------------------------------------------------------------
    if ds and not vision_model:
        # 决策优先级: thinking_effort > disable_thinking > cfg 全局开关
        if thinking_effort == 0:
            # lint 判断为关闭 → 强制关闭
            request_body["thinking"] = {"type": "disabled"}
            request_body["reasoning"] = {"effort": "none"}
        elif thinking_effort in (1, 2, 3):
            # lint 判断为低/中/高 → 按强度注入
            effort_map = {1: "low", 2: "high", 3: "max"}
            request_body["thinking"] = {"type": "enabled"}
            request_body["reasoning"] = {"effort": effort_map[thinking_effort]}
        else:
            # thinking_effort is None → 回退到旧 disable_thinking / cfg 全局逻辑
            if disable_thinking is None:
                enable_thinking = bool(cfg.get("deepseek_thinking", False))
            else:
                enable_thinking = not disable_thinking
            if enable_thinking:
                request_body["thinking"] = {"type": "enabled"}
                request_body["reasoning"] = {"effort": "high"}
            else:
                request_body["thinking"] = {"type": "disabled"}
                request_body["reasoning"] = {"effort": "none"}
    elif ds:
        # vision 模式: thinking 强制关闭 (避免 reasoning_tokens 算视觉输出, 拖慢首字)
        request_body["thinking"] = {"type": "disabled"}
        request_body["reasoning"] = {"effort": "none"}
    # 非 DeepSeek 模型: thinking_effort / disable_thinking 均被忽略 (无副作用)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # APP-Code 是 DeepSeek 专属 header,其他 API 不需要
    if ds:
        headers["APP-Code"] = "DMQU5622"

    async with httpx.AsyncClient(timeout=180.0) as client:
        res = await client.post(request_url, json=request_body, headers=headers)
        if not (200 <= res.status_code < 300):
            # 非 2xx:打完整响应体,便于定位 aihubmix / 第三方 API 错误原因
            debug(f"[_call_llm] non-2xx {res.status_code} body={res.text[:500]}")
            res.raise_for_status()
        data = res.json()
        # JSON 解析失败时打原始响应体
        if not data:
            debug(f"[_call_llm] empty JSON body, raw={res.text[:500]}")

    content = (
        data.get("choices", [{}])[0].get("message", {}).get("content")
        or data.get("choices", [{}])[0].get("text")
        or json.dumps(data, ensure_ascii=False)
    )
    usage = data.get("usage") or {}
    # reasoning_content 提取: DeepSeek 原生返回 message.reasoning_content;
    # OpenAI 标准 / 第三方代理 (aihubmix) 用 reasoning 字段 (字符串或 dict 都可能)。
    # 仅在非空时返回字符串, 空字符串 / 缺失 → None (调用方据此决定是否落库)。
    reasoning_content = (
        data.get("choices", [{}])[0].get("message", {}).get("reasoning_content")
        or data.get("choices", [{}])[0].get("reasoning")
    )
    if isinstance(reasoning_content, dict):
        # aihubmix / OpenAI reasoning 字段可能是 {"content": "..."} 形态
        reasoning_content = reasoning_content.get("content") or reasoning_content.get("text")
    if not isinstance(reasoning_content, str) or not reasoning_content.strip():
        reasoning_content = None
    return content, usage, reasoning_content
