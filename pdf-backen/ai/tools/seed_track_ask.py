"""
一次性脚本: 从 crystal_chat_ai.json 抽最近 N 对 ReqAsk + 主答 ResAsk,
写入 Crystal_track_ask.json, 用于手动验证 v3 全局 track 改动。

【v3 全局化测试用】原 track 此前只存论文场景, 现已放开 chat fp。
此脚本把历史 chat 对话回填到 track, 让 build_track_summary_block
能立刻看到"近期对话感知"里出现 chat 条目, 验证 label / 时间戳 / scene 标注。

用法 (在 pdf-backen 目录下任一):
    python -m ai.tools.seed_track_ask
    python ai/tools/seed_track_ask.py

可调常量:
    N_PAIRS        取最近几对 (默认 5, 与 MAX_ASK_TRACK=15 留出余量)
    USER_MAX       user 字段最大字符数 (默认 200, 与 _append_track 一致)
    ASST_MAX       assistant 字段最大字符数 (默认 200, 与 _append_track 一致)
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════
N_PAIRS = 5            # 取最近 N 对
USER_MAX = 200         # user 截断 (与 _append_track 同源)
ASST_MAX = 200         # assistant 截断 (与 _append_track 同源)
CHAT_FP = "crystal_chat"  # chat fp (与 ai_config.CR 同步)
                        #    YSTAL_FP 一致; 直接写字面量避免额外依赖

# ═══════════════════════════════════════════════════════════════
# 路径 (基于脚本自身位置, 不依赖 cwd)
# ═══════════════════════════════════════════════════════════════
_HERE = Path(__file__).resolve().parent           # .../pdf-backen/ai/tools
AI_DIR = _HERE.parent                              # .../pdf-backen/ai
BACKEN_DIR = AI_DIR.parent                         # .../pdf-backen
SRC = BACKEN_DIR / "save" / "crystal_chat_ai.json"
DST = AI_DIR / "memory" / "Crystal_track_ask.json"
BAK = DST.with_suffix(".json.bak")                 # 备份后缀


def collect_pairs(entries: list[dict], n: int) -> list[dict]:
    """
    反向扫描 entries, 收集最近 n 对 (ReqAsk, 主答 ResAsk)。

    配对规则:
      · 找到 ReqAsk → 紧邻的下一个 entry 必须是 ResAsk
      · 跳过 active=True 的 ResAsk (那是 layer1 主动追问, 不是主答)
      · 顺序: 时间正序, 最新一条在列表末尾 (与 _append_track 追加一致)

    Returns:
        list of {ts, ts_str, pdf_fp, user, assistant}
    """
    # 反向遍历, 配对成功的 (req, asst) 对先按反序收集, 最后整体倒序
    pairs_rev: list[dict] = []
    i = len(entries) - 1
    while i >= 0 and len(pairs_rev) < n:
        e = entries[i]
        et = e.get("type", "")
        if et == "ResAsk":
            # 主答: 必须 active 不为 True (排除主动追问)
            if not e.get("active"):
                # 找前面最近的 ReqAsk 配对
                j = i - 1
                while j >= 0:
                    pj = entries[j]
                    if pj.get("type") == "ReqAsk":
                        # 拼成 track entry
                        user = (pj.get("content") or "")[:USER_MAX]
                        asst = (e.get("content") or "")[:ASST_MAX]
                        ts = e.get("ts", pj.get("ts", 0))
                        ts_str = e.get("dt", pj.get("dt", ""))
                        pairs_rev.append({
                            "ts": ts,
                            "ts_str": ts_str,
                            "pdf_fp": CHAT_FP,
                            "user": user,
                            "assistant": asst,
                        })
                        i = j - 1  # 跳过 ReqAsk 本身, 继续向前
                        break
                    j -= 1
                else:
                    # 没找到配对 ReqAsk, 跳过这条 ResAsk
                    i -= 1
            else:
                # active=True (主动追问), 跳过
                i -= 1
        else:
            # 非 ResAsk (ReqAsk / ReqLoad / ResLoad), 继续向前
            i -= 1

    pairs_rev.reverse()  # 整体倒序 → 最新在末尾
    return pairs_rev


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"[seed_track_ask] 源文件不存在: {SRC}")

    # 读源
    with open(SRC, "r", encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise SystemExit(f"[seed_track_ask] 源文件不是 list: {type(entries).__name__}")
    print(f"[seed_track_ask] 源 entry 总数: {len(entries)}")

    # 配对
    pairs = collect_pairs(entries, N_PAIRS)
    if not pairs:
        raise SystemExit("[seed_track_ask] 未配对到任何 ReqAsk+ResAsk, 退出")
    print(f"[seed_track_ask] 配对成功: {len(pairs)} 对")
    for idx, p in enumerate(pairs, 1):
        print(
            f"  [{idx}] ts={p['ts']} ts_str={p['ts_str']} "
            f"user={p['user'][:30]!r} asst={p['assistant'][:30]!r}"
        )

    # 备份原文件
    DST.parent.mkdir(parents=True, exist_ok=True)
    if DST.exists():
        shutil.copy2(DST, BAK)
        print(f"[seed_track_ask] 已备份: {DST} -> {BAK}")

    # 写目标
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"[seed_track_ask] 已写入: {DST} (共 {len(pairs)} 条)")


if __name__ == "__main__":
    main()
