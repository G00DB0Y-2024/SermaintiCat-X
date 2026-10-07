"""Deeper integration test: instantiate the FastAPI app + lifespan, verify graph runs."""
import sys
import asyncio

sys.path.insert(0, r"F:\MyProjects\PDFAI\pdf-backen")


# 1) App import (this triggers PdfBacken import chain)
print("=" * 60)
print("PHASE 2: App + lifespan instantiation")
print("=" * 60)

try:
    from PdfBacken import app, lifespan
    print("OK: FastAPI app imported")
except Exception as e:
    print(f"FAIL: PdfBacken import: {e}")
    sys.exit(1)


# 2) Run the lifespan startup (which calls ai_active.start_explore_writer_thread)
async def run_lifespan():
    async with lifespan(app):
        print("OK: lifespan startup completed (ai_active.start_explore_writer_thread ran)")

try:
    asyncio.run(run_lifespan())
    print("OK: lifespan startup + teardown clean")
except Exception as e:
    print(f"FAIL: lifespan: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# 3) Verify a paper-mode ask (non-chat) doesn't trigger layer1
print("=" * 60)
print("PHASE 3: Paper-mode ask does not trigger layer1")
print("=" * 60)

from ai.ai_models import AiAskReq

async def test_paper_mode():
    from ai.ai_graph import run_ask
    req = AiAskReq(
        ask="test ask",
        pdf_fp="test_paper_fp",
        device="desktop",
        img="",
    )
    # Don't actually invoke LLM (no api_key). Just verify active_layer1_node returns {} for paper fp.
    from ai.ai_active import active_layer1_node
    state = {"req": req, "final_answer": ""}
    result = await active_layer1_node(state)
    # paper fp != CHAT_FP → should return {}
    assert result == {}, f"Expected {{}} for paper fp, got {result}"
    print("OK: paper-mode fp returns empty dict from active_layer1_node (chat gate works)")

asyncio.run(test_paper_mode())


# 4) Verify chat-mode (without LLM call) — only the gate runs, no LLM invocation
print("=" * 60)
print("PHASE 4: Chat-mode layer1 gate check")
print("=" * 60)

async def test_chat_mode_gate():
    from ai.ai_active import active_layer1_node
    req = AiAskReq(
        ask="test",
        pdf_fp="crystal_chat",
        device="desktop",
        img="",
    )
    state = {"req": req, "final_answer": "x", "req_fp": "r1", "res_fp": "s1"}
    result = await active_layer1_node(state)
    # chat + no Lint LLM configured (test env) → skip silently
    # Just verify the gate doesn't crash
    print(f"OK: chat-mode result = {result} (lint LLM may be unconfigured in test env)")

asyncio.run(test_chat_mode_gate())


print("=" * 60)
print("ALL INTEGRATION TESTS PASSED")
print("=" * 60)
