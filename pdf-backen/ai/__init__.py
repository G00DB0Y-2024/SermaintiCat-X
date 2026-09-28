"""
ai 子包入口 — 公开 API:

模块:
- ai_models        Pydantic 请求/响应模型
- ai_utils         工具函数 (时间 + 设备)
- ai_config        参数 + LLM 配置
- ai_io            文件 IO + cache
- prompts_system   上层系统提示词
- prompts_context  下层上下文与 prompt 拼装
- ai_llm           LLM 调用层
- ai_agent         LangGraph 节点 + State
- ai_graph         Graph 构建 + 入口 (run_ask / run_load)
- ai_routes        FastAPI router

入口 (由 PdfBacken.py 注册):
    from ai.ai_routes import router as ai_router
    app.include_router(ai_router)
"""
