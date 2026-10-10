"""
AI HTTP 端点 — /ai/config、/ai/ask、/ai/load、/ai/history、/ai/memory、/ai/emotion。

设计:
- /ai/config (POST): 前端 SET_MODEL 时调用, 更新 ai_agent.ai_config 模块变量。
                    请求体: {api_key, api_url, model, vision_model, deepseek_thinking}
- /ai/config (GET):  返回当前 ai_config 快照。
- /ai/ask (POST):    用户自由提问, 读 ai_config 调 LLM。
- /ai/load (POST):   用户选中文本要求总结, 读 ai_config 调 LLM。
- /ai/history (POST): 读取 save/{fp}_ai.json 原始 entry 列表 (供 ChatView 加载历史)。
- /ai/memory  (GET):  读取 Crystal_memory.md 原文供 ChatMem.vue 渲染。
- /ai/emotion (GET):  返回当前 emotion_vector, 供前端 HomeView 轮询拉取。
"""
from __future__ import annotations

import os
import json
import time

from fastapi import APIRouter, HTTPException
from .ai_io import _load_emotion
from .ai_emotion import EMOTION_LLM_CONFIG, update_emotion_llm_config

from .ai_config import (
    SAVE_DIR,
    CRYSTAL_MEMORY_FILE,
    CRYSTAL_SELF_FILE,
    EXPLORE_FILE,
    PARAMS_FILE,
)
from .ai_graph import run_ask, run_load
from .ai_models import (
    AiConfigReq, AiConfigResp,
    AiAskReq, AiLoadReq, AiResp,
    AiHistoryReq, AiHistoryResp,
)
from pydantic import BaseModel
from .ai_config import get_current_config, update_ai_config
from utils.log import debug


router = APIRouter(prefix="/ai", tags=["ai"])

# ── /ai/config ──────────────────────────────────────────────────────────

@router.post("/config")
async def ai_config(req: AiConfigReq) -> AiConfigResp:
    """
    前端 SET_MODEL 时调用。
    将 api_key / api_url / model / vision_model / deepseek_thinking
    存入 ai_config 模块变量,后续 /ai/ask 和 /ai/load 直接读取。
    """
    update_ai_config(req.model_dump())
    debug(
        f"[/ai/config] api_url={req.api_url!r}  "
        f"model={req.model!r}  vision_model={req.vision_model!r}  "
        f"deepseek_thinking={req.deepseek_thinking}"
    )
    return AiConfigResp(**get_current_config())


@router.get("/config", response_model=AiConfigResp)
async def ai_config_get() -> AiConfigResp:
    """返回当前 ai_config 快照, 用于前端调试。"""
    return AiConfigResp(**get_current_config())


# ── /ai/emotion ─────────────────────────────────────────────────────────

@router.get("/emotion")
async def ai_emotion() -> dict:
    """返回当前情绪向量, 供前端 HomeView 轮询拉取。"""
    emo = _load_emotion()
    debug(
        f"[/ai/emotion] hit → "
        f"valence={emo.get('valence'):+.4f} "
        f"arousal={emo.get('arousal'):+.4f} "
        f"novelty={emo.get('novelty'):+.4f} "
        f"clarity={emo.get('clarity'):+.4f} "
        f"dt={emo.get('last_update_dt')!r} "
        f"v={emo.get('version')} "
        f"|max|={max(abs(emo.get(axis, 0.0)) for axis in ('valence','arousal','novelty','clarity')):.4f}"
    )
    return emo


# ── /ai/emotion/config ─────────────────────────────────────────────────

class LintLLMConfigReq(BaseModel):
    api_key: str = ""
    api_url: str = ""
    model: str = "deepseek-flash"


@router.get("/emotion/config")
async def ai_emotion_config_get() -> LintLLMConfigReq:
    """返回当前 Lint LLM 配置快照。"""
    return LintLLMConfigReq(**EMOTION_LLM_CONFIG)


@router.post("/emotion/config")
async def ai_emotion_config_post(req: LintLLMConfigReq) -> LintLLMConfigReq:
    """
    前端设置 Lint LLM (弱模型) 配置。
    存入 ai_emotion.EMOTION_LLM_CONFIG 模块变量, emotion_llm_node 读取此配置。
    """
    update_emotion_llm_config(req.model_dump())
    debug(
        f"[/ai/emotion/config] api_url={req.api_url!r}  "
        f"model={req.model!r}  has_key={bool(req.api_key)}"
    )
    return LintLLMConfigReq(**EMOTION_LLM_CONFIG)


# ── /ai/ask ────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AiResp)
async def ai_ask(req: AiAskReq) -> AiResp:
    """
    Ask 模式 — 用户在论文阅读过程中自由提问。
    支持 vision(图文多模态, 通过 req.image_base64 非空判定)。
    api_key / api_url / model / vision_model 来自 ai_config。

    响应中 req_fp / res_fp 是 save_paper_memory_node 生成的 msg_fp,
    供前端写回 ai_res 气泡, 保证 ai_res.msg_fp 与 paper_history 完全对齐。
    """
    try:
        result = await run_ask(req)
        return AiResp(
            content=result.get("final_answer", ""),
            usage=result.get("usage") or None,
            dt=result.get("dt", ""),
            req_fp=result.get("req_fp", "") or "",
            res_fp=result.get("res_fp", "") or "",
            emotion=result.get("emotion") or {},
            split_followup=result.get("split_followup"),
        )
    except HTTPException:
        raise
    except Exception as e:
        debug(f"[/ai/ask] ERROR: {type(e).__name__}: {e}")
        # 把服务端原始错误 detail 透传给前端 (非 HTTPException 才打 detail)
        # 避免直接 expose 内部堆栈给外部,但打印出来便于本地调试
        raise HTTPException(status_code=500, detail=str(e))


# ── /ai/load ───────────────────────────────────────────────────────────

@router.post("/load", response_model=AiResp)
async def ai_load(req: AiLoadReq) -> AiResp:
    """
    Load 模式 — 用户选中文本, 要求 Crystal 总结 / 解释。
    api_key / api_url / model 来自 ai_config。

    Load 模式不写 msg_fp (ReqLoad/ResLoad entry 历史上就没有 msg_fp),
    因此 req_fp / res_fp 在此端点固定为空字符串。
    """
    try:
        result = await run_load(req)
        return AiResp(
            content=result.get("final_answer", ""),
            usage=result.get("usage") or None,
            dt=result.get("dt", ""),
            req_fp=result.get("req_fp", "") or "",
            res_fp=result.get("res_fp", "") or "",
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
    物理位置由 ai_config.SAVE_DIR 统一维护, 避免在 routes 层再算一遍路径。

    返回:
        AiHistoryResp.entries = list[dict], 每条至少含 {type, content, ts, dt}。
        缺文件 / 解析失败 / 文件为空 → 返回空列表 (不报错)。
    """
    save_path = os.path.join(SAVE_DIR, f"{req.pdf_fp}_ai.json")
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


# ── /ai/memory ─────────────────────────────────────────────────────────

# 记忆库文件清单 — ChatMem.vue 通过 ?file= 切换查看
#
#   memory  — Crystal_memory.md (U): 关于「他」的认知
#   self    — Crystal_self.md   (S): Crystal 的自我认知
#   explore — Crystal_explore.md (A): 主动外呼档案 (主动判断 / 节奏规律 / 外呼记录)
#   params  — params.json: 运行期技术状态 (情绪向量 / LLM 配置 / λ / 外呼队列与计数)
#   plans   — plans.json: 计划外呼队列 (Layer3 硬计划, 人工审计用)
#
# explore 暴露出来的意义: 这些判断完全由 LLM 写, 一开始必然不准, 需要人能看能改。
# 给前端开这个入口就是让人能直接修正, 而不是每次都靠 LLM 重新推断。
#
# plans.json 的定位变化: 最初注释写的是「**不**在这里暴露, 它是代码执行的
# 数据源, 不是 LLM 读的档案」。这个理由对 LLM 成立 (它确实不进任何 prompt),
# 但**人工**有正当理由要看: 计划由 LLM 从自然语言里提取, 提取错了 (时刻解析错、
# repeat 猜错、日期算错) 只有人能发现, 而 plan 走的是**不过 judge 的硬承诺**路径
# —— 一条解析错的计划会准时发出错误的话。故按「只读审计」开放。
#
# 安全性: 与 params 同一条脱敏链路 (字段名匹配, 见 _redact_secrets)。目前
# plans.json 无密钥字段, 走脱敏是零成本的保险 —— 将来若有人往 plan 里塞
# 备注/token, 不会因为多开了一个入口就泄漏。
MEMORY_FILES: dict[str, str] = {
    "memory":  CRYSTAL_MEMORY_FILE,
    "self":    CRYSTAL_SELF_FILE,
    "explore": EXPLORE_FILE,
    "params":  PARAMS_FILE,
}

# 需要走「解析 → 脱敏 → 回序列化」的文件 (而不是原样吐文本)。
_REDACT_JSON_FILES: frozenset[str] = frozenset({"params"})

# params.json 里**永远不能外泄**的字段 (明文 api_key)。
# /ai/memory?file=params 走这里做递归脱敏, 而不是干脆禁掉整个文件 ——
# 排查"状态没存上"仍需要看到情绪向量 / λ / 队列, 只是不该看见密钥。
_REDACT_KEYS: frozenset[str] = frozenset({
    "api_key", "apikey", "key", "token", "access_token", "secret",
})


def _redact_secrets(obj):
    """
    递归把 dict / list 里的敏感 key 替换为 "***"。

    只按 key 名匹配, 不按位置 —— params.json 未来新增任何 *_api_key 之类的
    字段都会自动被脱敏, 不依赖调用方记得加进黑名单。
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.strip().lower() in _REDACT_KEYS:
                out[k] = "***"
            else:
                out[k] = _redact_secrets(v)
        return out
    if isinstance(obj, list):
        return [_redact_secrets(x) for x in obj]
    return obj


@router.get("/memory")
async def ai_memory(file: str = "memory") -> dict:
    """
    返回 Crystal 记忆库原文, 供 ChatMem.vue 渲染。

    Query 参数:
        file (str): "memory"  (默认, Crystal_memory.md — 关于用户, U 元素)
                    "self"    (Crystal_self.md   — Crystal 自我认知, S 元素)
                    "explore" (Crystal_explore.md — 主动外呼档案, A 元素)
                    "params"  (params.json       — 运行期技术状态, 密钥已脱敏)
                    "plans"   (plans.json        — 计划外呼队列, 密钥已脱敏)
                    其它值视为非法, 返回 400 (旧的 "appoint" 已随改名失效)。

    返回: { content: str, updated: str, file: str, filename: str }
        content  — markdown 原文 (params/plans 为脱敏后的 JSON 文本)
        updated  — 文件最后修改时间字符串 (北京时间)
        file     — 回显 file 参数 (供前端校准)
        filename — 原始文件名 (供 ChatMem 状态栏显示 e.g. "Crystal_memory.md")
    失败 (文件不存在 / 读失败) → HTTP 404 / 500。
    """
    fp = (file or "memory").strip().lower()
    path = MEMORY_FILES.get(fp)
    if path is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown file={file!r}, expected one of: "
                   f"{sorted(MEMORY_FILES.keys())}",
        )

    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"Memory file not found: {os.path.basename(path)}",
        )

    try:
        mtime = os.path.getmtime(path)
        # UTC mtime → 北京时间字符串
        updated = time.strftime("%Y-%m-%d %H:%M:%S",
                                time.localtime(mtime))
    except Exception:
        updated = ""

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        debug(f"[/ai/memory] read failed: {e}")
        raise HTTPException(status_code=500, detail=f"Read error: {e}")

    # params.json / plans.json 走「解析 → 递归脱敏 → 回序列化」。
    # 脱敏失败 (文件被外部写坏成非 JSON) 时退回空对象, 绝不把原文吐出去。
    if fp in _REDACT_JSON_FILES:
        try:
            content = json.dumps(
                _redact_secrets(json.loads(content)),
                ensure_ascii=False, indent=2,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            debug(f"[/ai/memory] {fp} parse/redact failed: {e}")
            content = "{}\n// 解析失败, 已隐藏原文以防泄漏密钥"

    return {
        "content": content,
        "updated": updated,
        "file": fp,
        "filename": os.path.basename(path),
    }
