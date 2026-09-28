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

def _is_deepseek(api_url: str) -> bool:
    return DEEPSEEK_MARKER in api_url


def _is_openai_compat(api_url: str) -> bool:
    """判断 base URL 是否已带 API 路径(如 /v1)。"""
    return bool(api_url) and ("/v1" in api_url or "/v1/" in api_url)


def _resolve_api_url(api_url: str) -> str:
    base = api_url.rstrip("/")
    if _is_deepseek(api_url):
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
    disable_thinking: bool = True,
) -> tuple[str, dict]:
    """
    直接调上游 LLM, 配置全部从 ai_config 读取。
    vision_model 参数仅用于判断 DeepSeek thinking 开关(仅非视觉模式有效)。

    disable_thinking:
      True 时强制关闭 DeepSeek thinking 模式 (即便 deepseek_thinking 全局开关为 on)。
      用于结构化输出任务 (memory compress / update), 这些任务不需要 reasoning,
      开 thinking 会白白烧 token + 拖慢响应。

      双重保险: DeepSeek 同时认两套 reasoning 字段, 任意一套生效即可关闭思考:
        · thinking.type              (DeepSeek 原生别名)
        · reasoning.effort="none"    (OpenAI 标准字段, 跨 provider 通用)
      本函数同时塞这两个字段, 不管 model 是 DeepSeek 默认开 reasoning, 还是
      aihubmix / 第三方代理把 reasoning.effort 转成对应字段, 都能确保 thinking 被关掉。
      默认全局基线 effort="none" 也作为兜底 — 即便 model 默认行为是开 reasoning,
      全局这一条也能压下去。

    返回: (content, usage)
      - content: LLM 回复正文; 解析失败时回落到完整 JSON dump
      - usage  : 上游 LLM usage 统计 (可能为空 dict)
    """
    api_url = _ai_config["api_url"]
    api_key = _ai_config["api_key"]
    model = _ai_config["vision_model"] if vision_model else _ai_config["model"]

    if not api_url or not api_key or not model:
        raise ValueError(
            f"ai_config 未完整配置: api_url={bool(api_url)} "
            f"api_key={bool(api_key)} model={bool(model)}"
        )

    request_url = _resolve_api_url(api_url)
    is_ds = _is_deepseek(api_url)

    request_body: dict = {"model": model, "messages": messages}


    if is_ds and not vision_model and not disable_thinking and _ai_config["deepseek_thinking"]:
        # ask 路径 + 用户主动开了 thinking → 启用思考,顺便给个 low 强度兜底
        request_body["thinking"] = {"type": "enabled"}
        request_body["reasoning"] = {"effort": "low"}
    else:
        # 任意"应关思考"的路径 → 用 reasoning.effort="none" 强制关闭
        request_body["thinking"] = {"type": "disabled"}
        request_body["reasoning"] = {"effort": "none"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # APP-Code 是 DeepSeek 专属 header,其他 API 不需要
    if is_ds:
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
    return content, usage
