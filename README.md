T0: 用户点发送
     │
T1: 前端 POST /ai/ask (chat fp)
     │
T2: 后端 run_ask → LangGraph 启动
     │
T3: load_agent_memory_node     ─→ 读 Crystal_memory.md (U)
T4: load_paper_history_node    ─→ 读 save/crystal_chat_ai.json (C) 最近 10 对
T5: compose_messages_node      ─→ 拼 messages (注入 U, 不注入 S)
T6: llm_call_node              ─→ LLM 出 final_answer (阻塞,1~10s)
     │
T7: emotion_l1_node            ─→ 启发式打分 → apply_emotion_delta → params.json ✏️
T8: save_paper_memory_node     ─→ 写 C (save/crystal_chat_ai.json) ✏️
T9: update_agent_memory_node   ─→ 触发 _update_crystal_memory_async (后台) ✏️
                                   · Phase 1: 并发 compress U + S
                                   · Phase 2a: LLM 更新 U →写 Crystal_memory.md
                                   · Phase 2b: LLM 更新 S → 写 Crystal_self.md
T10: emotion_l2_node           ─→ 节流: 第 3 次才执行, 调 LLM 拿 delta → apply_emotion_delta ✏️
T11: flush_track_node          ─→ track 缓存写盘 (Crystal_track_ask/load.json)
     │
T12: AiResp(含 emotion={...})  ─→ 回前端
T13: 前端解析, 更新 profile-cover 图

(并发) 后台守护线程一直跑:
  每 30s → apply_decay_once → (各轴超过 0.01) 才写盘 ✏️
