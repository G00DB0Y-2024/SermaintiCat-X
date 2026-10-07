"""Smoke test for Active Layer1 refactor — verifies all modules import cleanly."""
import sys
sys.path.insert(0, r"F:\MyProjects\PDFAI\pdf-backen")

# Direct imports of all touched modules
import ai.ai_io
import ai.ai_config
import ai.ai_active
import ai.ai_graph
import ai.ai_routes
import ai.ai_emotion
import ai.ai_agent
import ai.prompts_context
import ai.prompts_system

print("=" * 60)
print("PHASE 1: All module imports OK")
print("=" * 60)

# Verify deleted modules are NOT importable
import importlib
try:
    importlib.import_module("ai.ai_outreach")
    print("FAIL: ai.ai_outreach should not be importable (was deleted)")
    sys.exit(1)
except ModuleNotFoundError:
    print("OK: ai.ai_outreach is gone (deleted)")

# Verify removed config constants are gone
from ai import ai_config
removed_constants = [
    "PLANS_FILE",
    "OUTREACH_DAILY_LIMIT_LONG",
    "OUTREACH_DAILY_LIMIT_PLAN",
    "OUTREACH_LEARN_EVERY_N_LARGE",
    "OUTREACH_LAMBDA_DEFAULT",
    "OUTREACH_LAMBDA_MIN",
    "OUTREACH_LAMBDA_MAX",
    "OUTREACH_LAMBDA_CLAMP_LOG",
    "OUTREACH_LAMBDA_MAX_GAP_SEC",
]
for c in removed_constants:
    if hasattr(ai_config, c):
        print(f"FAIL: ai_config.{c} should be removed")
        sys.exit(1)
print(f"OK: removed {len(removed_constants)} Layer2/3/Lambda constants from ai_config")

# Verify removed prompts_context functions are gone
from ai import prompts_context
removed_funcs = [
    "OUTREACH_INTENT_USER",
    "OUTREACH_DECIDE_USER",
    "OUTREACH_PREDICT_USER",
    "OUTREACH_JUDGE_USER",
    "OUTREACH_RHYTHM_SECTION",
]
for f in removed_funcs:
    if hasattr(prompts_context, f):
        print(f"FAIL: prompts_context.{f} should be removed")
        sys.exit(1)
print(f"OK: removed {len(removed_funcs)} OUTREACH_* prompt builders from prompts_context")

# Verify removed ai_io functions are gone
from ai import ai_io
removed_io_funcs = [
    "_load_plans",
    "_save_plans",
    "_load_lambda",
    "_save_lambda",
    "_clamp_lambda",
]
for f in removed_io_funcs:
    if hasattr(ai_io, f):
        print(f"FAIL: ai_io.{f} should be removed")
        sys.exit(1)
print(f"OK: removed {len(removed_io_funcs)} plans/lambda functions from ai_io")

# Verify shared helpers moved to ai_io
expected_io_helpers = [
    "_write_text_file_atomic",
    "_read_text_file_safe",
    "_extract_section",
    "_heading_level",
    "_heading_text",
    "_upsert_section",
]
for f in expected_io_helpers:
    if not hasattr(ai_io, f):
        print(f"FAIL: ai_io.{f} should be present (moved from ai_outreach)")
        sys.exit(1)
print(f"OK: all {len(expected_io_helpers)} shared helpers present in ai_io")

# Verify ai_active has the public API expected by plan §2.4
from ai import ai_active
expected_active_api = [
    "active_layer1_node",
    "on_user_msg",
    "on_read_receipt",
    "register_ws_client",
    "unregister_ws_client",
    "start_explore_writer_thread",
    "check_guard_now",
    "predict_intent",
    "predict_reply",
    "gen_draft",
    "judge_prediction",
    "_format_chat_local_block",
    "_outreach_context_for_llm",
]
for f in expected_active_api:
    if not hasattr(ai_active, f):
        print(f"FAIL: ai_active.{f} should be present")
        sys.exit(1)
print(f"OK: all {len(expected_active_api)} public APIs present in ai_active")

# Verify graph wiring
import ai.ai_graph
g = ai.ai_graph.build_graph()
print(f"OK: graph compiled (nodes: {sorted(g.nodes.keys()) if hasattr(g, 'nodes') else 'compiled'})")

# Verify _format_chat_local_block contract
sample_entries = [
    {"role": "user", "content": "你好", "ts": 1700000000000, "msg_fp": "a1"},
    {"role": "assistant", "content": "我好", "ts": 1700000060000, "msg_fp": "a2"},
    {"role": "assistant", "content": "我主动的", "ts": 1700000120000, "msg_fp": "a3", "active": True},
]
block = ai_active._format_chat_local_block(sample_entries)
assert "[20" in block, f"timestamp prefix missing: {block[:200]}"
assert "用户说" in block, f"用户说 label missing"
assert "Crystal回复说" in block, f"Crystal回复说 label missing"
assert "Crystal主动说" in block, f"Crystal主动说 label missing"
assert "上下文元数据" in block, f"段首声明 missing"
assert "严禁在你的实际输出中复刻" in block, f"段首禁令 missing"
print("OK: _format_chat_local_block 标签契约正确 (含 [ts 用户说/Crystal回复说/Crystal主动说] 标签 + 段首禁用声明)")

print("=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
