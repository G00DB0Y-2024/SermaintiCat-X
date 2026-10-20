"""

AI 模块所有可调参数 + LLM 配置。



分为三块:

- 路径:  BASE_DIR / SAVE_DIR / MEMORY_DIR / CRYSTAL_MEMORY_FILE / EXPLORE_FILE /

- 窗口:  ASK_LOCAL_LIMIT / LOAD_LOCAL_LIMIT / CHAT_LOCAL_LIMIT / CHAT_AUDIT_LIMIT / ...

- LLM:   DEEPSEEK_MARKER + ai_config 字典 (由 /ai/config 端点维护)



ai_config 字典 (api_key / api_url / model / vision_model / deepseek_thinking)

由 ai_routes 写入, _call_llm 读取 — 单一可变状态只有这一份。



Crystal 三份认知档案的物理落点 (隔离, 互不串档):

    U = Crystal_memory.md   LLM 读, 关于「他」的认知

    S = Crystal_self.md     LLM 读, Crystal 对自己的认知

    A = Crystal_explore.md  LLM 读, 主动开口的自然语言判断 (软)

"""

from __future__ import annotations



import os

import sys

from typing import Final





# ═══════════════════════════════════════════════════════════════════════

# 路径

# ═══════════════════════════════════════════════════════════════════════



def _get_base_dir() -> str:

    """

    打包环境兼容: PyInstaller 临时目录 vs 开发目录。



    ai/* 子模块的 BASE_DIR 应当回到 pdf-backen 项目根目录,

    这样 save / static 等目录与 PdfBacken.py 一致。

    """

    if getattr(sys, "frozen", False):

        return sys._MEIPASS  # type: ignore[attr-defined]

    # 本文件位于 pdf-backen/ai/<name>.py, 向上一层就是 pdf-backen 项目根

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))





BASE_DIR: Final[str] = _get_base_dir()

SAVE_DIR: Final[str] = os.path.join(BASE_DIR, "save")

MEMORY_DIR: Final[str] = os.path.join(BASE_DIR, "ai", "memory")

CRYSTAL_MEMORY_FILE: Final[str] = os.path.join(MEMORY_DIR, "Crystal_memory.md")

CRYSTAL_SELF_FILE:   Final[str] = os.path.join(MEMORY_DIR, "Crystal_self.md")

# appoint.md → Crystal_explore.md — 主动外呼档案。

#

# 与 memory / self 的分工 (三者物理隔离, 互不串档):

#   - Crystal_memory.md   (U): 关于「他」的一切认知 —— 身份、偏好、忌讳、项目、环境。

#   - Crystal_self.md     (S): Crystal 对自己的认知 —— 性格、语气、愿望。

#   - Crystal_explore.md  (A): 关于「主动开口」的档案 —— LLM 自由管理,
#     不再有固定的三段标题 (三段标题已废止);
#     整份重写, prompt 仅约束"只写时间相关认知" (活跃时段/沉默含义/回应速度等)。

#

# 为什么叫 explore 而不是 appoint:

#   旧版有一段「## 约定」承载硬承诺(每天早上问好), 靠正则猜时间窗做硬门控。

#   那套已退役。Crystal_explore.md 只留「自然语言判断」, 归 LLM 读。

#   判据是**谁读数据**: LLM 读 md, 代码读 json。

EXPLORE_FILE:       Final[str] = os.path.join(MEMORY_DIR, "Crystal_explore.md")

# 运行期可变参数的持久化文件 ("跨进程要留住"的状态)。

# 目前存 _ask_count_by_fp —— memory update 的节流计数。

# 放 ai/memory/ 而不是 save/: 它跟 memory 生命周期待在一起, 便于打包时一起收集。

PARAMS_FILE: Final[str] = os.path.join(MEMORY_DIR, "params.json")





# ═══════════════════════════════════════════════════════════════════════

# 上下文窗口参数

# ═══════════════════════════════════════════════════════════════════════

#

# 论文侧 (按 fp 隔离):

#   ASK_LOCAL_LIMIT  : 本论文 ask 最近 10 对 (user+assistant)

#   LOAD_LOCAL_LIMIT : 本论文 load 最近 5 对

#

# Chat 侧:

#   CHAT_LOCAL_LIMIT : ChatView 近 10 对

#   CHAT_MEMORY_LIMIT: 喂给 memory update 的 Chat 历史窗口 (5 对)

#

# Chat 路径近期对话感知:

#   CHAT_AUDIT_LIMIT: build_chat_audit_block 从 save/crystal_chat_ai.json
#                     读取的最近条目上限

# 节流:

#   MEMORY_UPDATE_EVERY_N: 每 N 次论文 ask 触发一次 memory LLM



ASK_LOCAL_LIMIT:   Final[int] = 10

LOAD_LOCAL_LIMIT:  Final[int] = 5

CHAT_LOCAL_LIMIT:  Final[int] = 20

CHAT_MEMORY_LIMIT: Final[int] = 10

CHAT_AUDIT_LIMIT:  Final[int] = CHAT_LOCAL_LIMIT

MEMORY_UPDATE_EVERY_N: Final[int] = 3  # = ASK_LOCAL_LIMIT // 2

# Crystal_memory / Crystal_self 压缩档位阈值 (2026-10-11 引入)
#  档位依据: 笔记里"最旧"一条 [更新时间: YYYY-MM-DD HH:MM] 距今多久。
#  第一档 (< COMPRESS_L1_HOURS 小时): 日记体, 不压缩
#  第二档 (COMPRESS_L1_HOURS ~ COMPRESS_L2_HOURS 小时): 简化措辞, 合并相似条目, 仍按时间
#  第三档 (> COMPRESS_L2_HOURS 小时): 维护 Hippocampus Area, 按主题分类, 无序列表, 时间顺序抹除
#  注意: 三档并非互斥, 越旧的内容被压得越深
#        (同一份 md 里可同时有 24h 内日记 + 24~72h 简化 + >72h Hippocampus)
COMPRESS_L1_HOURS: Final[int] = 24
COMPRESS_L2_HOURS: Final[int] = 72


# ═══════════════════════════════════════════════════════════════════════

# Chat 标识 (前端通过 pdf_fp=="crystal_chat" 调用 Chat 上下文)

# ═══════════════════════════════════════════════════════════════════════



CHAT_FP: Final[str] = "crystal_chat"





# ═══════════════════════════════════════════════════════════════════════

# 情绪状态机参数

# ═══════════════════════════════════════════════════════════════════════

#

# 情绪衰减 (词池范式, 2026-10-14):
#   EMOTION_DECAY_HALF_LIFE_MIN : 单一 intensity 半衰期 (分钟)。
#
#       衰减公式: new_intensity = old_intensity * 0.5 ^ (elapsed_min / HALF_LIFE_MIN)
#       label 不参与衰减 — 强度降到 WRITE_THRESHOLD 以下时强制改为 "中性"。
#
#   EMOTION_DECAY_INTERVAL_SEC : 后台守护协程每 N 秒跑一次衰减判定。
#
#       跑的时候读 last_update_dt, 算 elapsed_min, 算衰减后的 intensity;
#       若 |Δintensity| > EMOTION_DECAY_WRITE_THRESHOLD 才落盘,
#       避免每 30s 都刷一次 git 噪音。
#
#   EMOTION_DECAY_WRITE_THRESHOLD : 落盘阈值 + 中性强制切换阈值 (二合一)。

EMOTION_DECAY_HALF_LIFE_MIN:  Final[int]   = 240   # 4 小时基础半衰

EMOTION_DECAY_INTERVAL_SEC:   Final[int]   = 30    # 后台守护协程周期

EMOTION_DECAY_WRITE_THRESHOLD: Final[float] = 0.01  # 强度变化超过此值才写盘

# 降级阈值: 强度衰减到当前 label 基准的多少倍以下, 沿同象限降一级
#   · 0.5 = 衰减一半就降级 (推荐: 4h 半衰期下, 大约 4h 后从"狂喜"自然降到"欣喜")
#   · 调整历史: 初版 0.3 太严苛, label 长时间不切换; 0.7 太松, 切换太频繁
EMOTION_DECAY_DEMOTE_RATIO: Final[float] = 0.5

# 降级链 (2026-10-15 引入) — 12 词按 (v, a) 象限内"强度从高到低"组织成 4 条链。
#   衰减机制走降级链, 不再硬切到中性, 让"狂喜→欣喜→愉悦→中性"自然冷却。
#   链的"末梢"中性化条件:
#     · 走到链最轻词 (e.g. 愉悦), 且新强度仍 < 该词基准*0.5 → 转中性
#     · 一旦 label=中性, 不再沿任何链走 (中性是终点)
#
# 注意: 链的顺序是"强→中→轻" (从高 intensity 到低 intensity)。
#   衰减时沿链向下走, 升级 (LLM update 时) 不走链, 由 LLM 直接给目标 label。
EMOTION_FALLBACK_CHAINS: Final[dict[str, tuple[str, ...]]] = {
    # +V +A 兴奋家系
    "狂喜": ("欣喜", "愉悦"),
    "欣喜": ("愉悦",),
    "愉悦": (),  # 链最轻, 衰减再低转中性
    # +V -A 平和家系
    "沉静": ("安心", "满足"),
    "安心": ("满足",),
    "满足": (),  # 链最轻
    # -V +A 紧张家系
    "愤怒": ("焦虑", "烦躁"),
    "焦虑": ("烦躁",),
    "烦躁": (),  # 链最轻
    # -V -A 低落家系
    "绝望": ("落寞", "惆怅"),
    "落寞": ("惆怅",),
    "惆怅": (),  # 链最轻
}

# 跨象限防直跳: 当前 label 不在任一链 (e.g. "中性") 或目标 label 不在同象限,
# apply_decay_once 拒绝切换 — 这是兜底, 正常链结构下不该触发。


# ═══════════════════════════════════════════════════════════════════════
# 情绪词池 (词池范式, 2026-10-14 引入)
# ═══════════════════════════════════════════════════════════════════════
#
# 设计:
#   · 12 词 (4 象限 × 3 强度档) + 1 中性 (原点 0,0,0) = 13 词
#   · 每词 4 字段: label (显示名) / v (valence ∈ [-1, 1])
#                / a (arousal ∈ [-1, 1]) / intensity (强度档 ∈ {0.3, 0.6, 1.0})
#   · 中性 "中性" 强度 0, 坐标 (0, 0) — 是衰减终点, 不是 None
#   · LLM 在 prompt 中看到的是这 12 个 label, 后端按 label 查表得到 (v, a)
#     用于构建情绪上下文 (build_emotion_context_block 兼容旧 4 family 文本)
#
# 坐标布局 (伏安象限, 4 象限 × 3 强度):
#
#   +V +A 兴奋 ─ 愉悦(0.3)  欣喜(0.6)  狂喜(1.0)
#   +V -A 平和 ─ 满足(0.3)  安心(0.6)  沉静(1.0)
#   -V +A 紧张 ─ 烦躁(0.3)  焦虑(0.6)  愤怒(1.0)
#   -V -A 低落 ─ 惆怅(0.3)  落寞(0.6)  绝望(1.0)
#   (0, 0)     ─                    中性(0.0)
#
# 坐标值选自原各轴典型量级 (各轴 0.3 / 0.6 / 1.0),
# 落到 [-1, 1] 后再按象限符号取号。
EMOTION_POOL: Final[dict[str, dict]] = {
    "中性": {"v":  0.0, "a":  0.0, "intensity": 0.0},

    # +V +A 兴奋
    "愉悦": {"v":  0.3, "a":  0.3, "intensity": 0.3},
    "欣喜": {"v":  0.6, "a":  0.6, "intensity": 0.6},
    "狂喜": {"v":  1.0, "a":  1.0, "intensity": 1.0},

    # +V -A 平和
    "满足": {"v":  0.3, "a": -0.3, "intensity": 0.3},
    "安心": {"v":  0.6, "a": -0.6, "intensity": 0.6},
    "沉静": {"v":  1.0, "a": -1.0, "intensity": 1.0},

    # -V +A 紧张
    "烦躁": {"v": -0.3, "a":  0.3, "intensity": 0.3},
    "焦虑": {"v": -0.6, "a":  0.6, "intensity": 0.6},
    "愤怒": {"v": -1.0, "a":  1.0, "intensity": 1.0},

    # -V -A 低落
    "惆怅": {"v": -0.3, "a": -0.3, "intensity": 0.3},
    "落寞": {"v": -0.6, "a": -0.6, "intensity": 0.6},
    "绝望": {"v": -1.0, "a": -1.0, "intensity": 1.0},
}

# LLM prompt 用的词表 (顺序固定, 给 LLM 看的列表)
EMOTION_POOL_LABELS: Final[tuple[str, ...]] = (
    "愉悦", "欣喜", "狂喜",
    "满足", "安心", "沉静",
    "烦躁", "焦虑", "愤怒",
    "惆怅", "落寞", "绝望",
)

# 衰减 / 更新用阈值常量 (集中定义, 避免散落 magic number)
EMOTION_POOL_FALLBACK_LABEL: Final[str] = "中性"
EMOTION_LLM_DELTA_MAX: Final[float] = 0.30   # LLM 单轮 delta_intensity 上限
EMOTION_INTENSITY_MIN: Final[float] = 0.0
EMOTION_INTENSITY_MAX: Final[float] = 1.0





# ═══════════════════════════════════════════════════════════════════════


# LLM 路由

# ═══════════════════════════════════════════════════════════════════════



# DeepSeek API 需要在 base url 后拼 /chat/completions

DEEPSEEK_MARKER: Final[str] = "deepseek.com"





# ═══════════════════════════════════════════════════════════════════════

# ai_config 模块状态 (由 /ai/config 端点写入)

# ═══════════════════════════════════════════════════════════════════════



_ai_config: dict = {

    "api_key": "",

    "api_url": "",

    "model": "",          # 普通文本模型 (ask / load)

    "vision_model": "",   # 视觉模型 (ask + image_base64)

    "deepseek_thinking": False,

}





def update_ai_config(config: dict) -> None:

    """

    由 /ai/config 端点调用, 更新模块级 _ai_config。

    视觉模型缺省回落到普通 model。



    同步把配置持久化到 ai/memory/params.json.llm_configs.main, 这样:

      · 后端重启后配置不丢

      · 多端 (电脑/手机) 任何一端更新了配置, 都能从磁盘读到最新值

    """

    global _ai_config

    _ai_config.update({

        "api_key": config.get("api_key") or "",

        "api_url": config.get("api_url") or "",

        "model": config.get("model") or "",

        "vision_model": config.get("vision_model") or config.get("model") or "",

        "deepseek_thinking": bool(config.get("deepseek_thinking")),

    })

    # 落盘 (静默失败, 内存已更新, 持久化只是 bonus)

    try:

        from .ai_io import _save_llm_configs

        _save_llm_configs({"main": dict(_ai_config)})

    except Exception:

        # 落盘失败不影响本次请求的处理 (模块级状态已经是最新的)

        pass





def get_current_config() -> dict:

    """返回当前 _ai_config 快照 (供 /ai/config GET 使用)。"""

    return dict(_ai_config)





def load_ai_config_from_disk() -> bool:

    """

    进程启动时从 params.json.llm_configs.main 把配置读回到 _ai_config 模块变量。

    由 PdfBacken.py 的 lifespan 调用。



    这样做的好处:

      · 后端重启 → 配置不丢 (原先只在模块 dict, 重启即丢)

      · 多端 (电脑 + 手机) 任何一端更新了 → 另一端下次访问后端时,

        看到的就是最新的 (前端 GET /ai/config 拉一次即可同步)



    Returns:

        bool — True 表示从磁盘读到非空配置, False 表示磁盘无配置或读取失败

               (False 时 _ai_config 保持默认空值, 由前端推送兜底)

    """

    try:

        from .ai_io import _load_llm_configs

        cfg = _load_llm_configs().get("main") or {}

        if not cfg or not cfg.get("api_key"):

            return False

        global _ai_config

        _ai_config.update({

            "api_key": cfg.get("api_key") or "",

            "api_url": cfg.get("api_url") or "",

            "model": cfg.get("model") or "",

            "vision_model": cfg.get("vision_model") or cfg.get("model") or "",

            "deepseek_thinking": bool(cfg.get("deepseek_thinking")),

        })

        return True

    except Exception:

        return False

