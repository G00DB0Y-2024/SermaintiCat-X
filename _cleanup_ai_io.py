#!/usr/bin/env python3
"""Final cleanup pass."""
from pathlib import Path
import re

p = Path('F:/MyProjects/PDFAI/pdf-backen/ai/ai_io.py')
content = p.read_text(encoding='utf-8')

BOX_LINE_BODY = "# " + ("\u2550" * 71)
BOX_LINE = BOX_LINE_BODY + "\n"

def delete_block(text, header_substr, end_substr):
    s = text.find(header_substr)
    assert s != -1, f"can't find header: {header_substr!r}"
    # Use rfind of end_substr starting AFTER s
    e = text.find(end_substr, s + 1)
    assert e != -1, f"can't find end after header: {end_substr!r}"
    # Find leading box line BEFORE s
    box_start = text.rfind(BOX_LINE_BODY, 0, s)
    assert box_start != -1, f"no leading box line"
    # backtrack to start of line
    if box_start > 0 and text[box_start - 1] == "\n":
        box_start -= 1
    return text[:box_start] + text[e:]

# Delete plans block
content = delete_block(
    content,
    "# plans.json (Layer3 计划队列) — 缓存 + 读写",
    "# Crystal_track (跨论文 Ask+Load 全局追踪) — 双轨缓存 + 读写",
)

# Clean imports (PLANS_FILE already gone from lambda pass)
# But PLANS_FILE might still be imported
old_imp = "from .ai_config import (\n    ASK_TRACK_FILE,\n    CHAT_FP,\n    CHAT_MEMORY_LIMIT,\n    CRYSTAL_MEMORY_FILE,\n    CRYSTAL_SELF_FILE,\n    EXPLORE_FILE,\n    LOAD_TRACK_FILE,\n    MAX_ASK_TRACK,\n    MAX_LOAD_TRACK,\n    MEMORY_DIR,\n    PARAMS_FILE,\n    PLANS_FILE,\n    SAVE_DIR,\n)"
new_imp = "from .ai_config import (\n    ASK_TRACK_FILE,\n    CHAT_FP,\n    CHAT_MEMORY_LIMIT,\n    CRYSTAL_MEMORY_FILE,\n    CRYSTAL_SELF_FILE,\n    EXPLORE_FILE,\n    LOAD_TRACK_FILE,\n    MAX_ASK_TRACK,\n    MAX_LOAD_TRACK,\n    MEMORY_DIR,\n    PARAMS_FILE,\n    SAVE_DIR,\n)"
assert old_imp in content, "import block not found"
content = content.replace(old_imp, new_imp)

p.write_text(content, encoding='utf-8')
print(f"OK: {len(content)} bytes")

for k in ["PLANS_FILE", "OUTREACH_LAMBDA", "_load_plans", "_load_lambda",
          "_clamp_lambda", "_save_lambda", "PLANS_KEEP_MAX", "_plans_cache",
          "_plans_lock", "_lambda_cache", "_lambda_clamp_hits",
          "from .ai_outreach", "ACUS", "_upsert_section",
          "_outbound_fp", "outbound_fp", "proactive"]:
    cnt = content.count(k)
    flag = "OK" if cnt == 0 else "REMAINING"
    print(f"  [{flag}] {k}: {cnt}")