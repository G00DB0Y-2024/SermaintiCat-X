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
    *,
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

    json_mode:
      True 时在 request_body 里加 "response_format": {"type": "json_object"},
      让 LLM 走 OpenAI 兼容的 json 模式 (DeepSeek / aihubmix / Gemini 兼容层都支持)。
      用于结构化输出 (如 memory_and_self 合并 update 的 JSON {updated_memory,
      updated_self}), 显著降低解析失败率。

      注意: json_mode 不会自动重试; 解析失败由调用方决定是否 fallback。
      弱模型 (gemini flash) 在 json_object 下偶发空字符串, 调用方
      _safe_parse_update_json 已有兜底, 这里不重复处理。

    返回: (content, usage)
      - content: LLM 回复正文; 解析失败时回落到完整 JSON dump
      - usage  : 上游 LLM usage 统计 (可能为空 dict)
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

    request_url = _resolve_api_url(api_url)
    is_ds = _is_deepseek(api_url)

    request_body: dict = {"model": model, "messages": messages}

    if json_mode:
        # OpenAI 兼容的 json_object 模式 (DeepSeek / aihubmix / Gemini 兼容层都支持)。
        # 强制 LLM 输出合法 JSON 对象, 大幅降低后端解析失败率。
        request_body["response_format"] = {"type": "json_object"}

    if is_ds and not vision_model and not disable_thinking:
        # ask 路径 + 用户主动开了 thinking → 启用思考,顺便给个 high 强度兜底
        request_body["thinking"] = {"type": "enabled"}
        request_body["reasoning"] = {"effort": "high"}
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

    msg = data.get("choices", [{}])[0].get("message") or {}
    # 【v2 修复 2026-10-10】DeepSeek thinking 模式下, 上游把"答案正文"
    # 放在 reasoning_content, content 字段常为空字符串。直接读 content 会
    # 拿到 "", 然后被 `or` 短路落进 json.dumps(data) 的兜底分支, 把整个
    # 上游响应 (含 reasoning + metadata) 写进 ResAsk.content, 进而污染
    # split 拆解 → 用户看到一坨原始 JSON 推理串。
    #
    # 优先级: content > reasoning_content > text > JSON dump
    #  - content 优先: 兼容 aihubmix / OpenAI 等不走 thinking 的链路
    #  - reasoning_content 兜底: 仅当 content 为空时取 (thinking 模式)
    #  - text 兜底: 兼容老式 completions 接口
    #  - json.dumps: 最后一根稻草, 便于排查
    content = (
        msg.get("content")
        or msg.get("reasoning_content")
        or data.get("choices", [{}])[0].get("text")
        or json.dumps(data, ensure_ascii=False)
    )
    if msg.get("content") == "" and msg.get("reasoning_content"):
        # 标记一次, 方便后端日志 / 上线后回溯
        debug(
            f"[_call_llm] thinking-mode fallback: content='', "
            f"len(reasoning_content)={len(msg.get('reasoning_content') or '')}"
        )
    usage = data.get("usage") or {}
    return content, usage
