"""
ai 子包 — Crystal 论文问答 / Agent Memory 核心逻辑。

- ai_models     Pydantic 请求/响应模型
- prompts       Crystal 人设 + Prompt 模板 + Crystal_mem.md 更新 prompt
- ai_agent      LangGraph StateGraph(load / compose / llm / save / memory)
- ai_routes     FastAPI router(/ai/ask, /ai/load)

入口(由 PdfBacken.py 注册):
    from ai.ai_routes import router as ai_router
    app.include_router(ai_router)
"""
