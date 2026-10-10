import uvicorn
import base64
import os, json, sys

import xml.etree.ElementTree as ET

from utils.log import debug

from datetime import datetime
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Request
from fastapi import APIRouter,File, UploadFile, Form, Header
from typing import List, Dict, Any

from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from ai.ai_routes import router as ai_router
from fastapi import WebSocket, WebSocketDisconnect

'''
******************************************
'''
GREEN = "\033[32m"
PURPLE = "\033[35m"  # 紫色
RED = "\033[31m"   # 红色
YELLOW = "\x1b[33m" 
RESET = "\033[0m"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动情绪衰减后台守护线程
    from ai.ai_emotion import start_emotion_decay_thread
    start_emotion_decay_thread()

    # 启动疲劳衰减后台守护线程 (与 emotion decay 独立, 周期 60s)
    from ai.ai_emotion import start_fatigue_decay_thread
    start_fatigue_decay_thread()

    # 启动时从 ai/memory/params.json.llm_configs 把 Main + Lint 两份 LLM
    # 配置读回到 ai_config._ai_config / ai_emotion.EMOTION_LLM_CONFIG。
    # 这样:
    #   · 后端重启 → 用户在前端设过的配置不会丢
    #   · 电脑/手机任意一端更新过配置 → 另一端 GET /ai/config 就能看到最新
    # 失败静默 — 读不到就用默认空值, 由前端推送兜底 (旧行为不变)。
    try:
        from ai.ai_config import load_ai_config_from_disk
        from ai.ai_emotion import load_emotion_llm_config_from_disk
        if load_ai_config_from_disk():
            print("[lifespan] Main LLM config restored from params.json")
        if load_emotion_llm_config_from_disk():
            print("[lifespan] Lint LLM config restored from params.json")
    except Exception as e:
        print(f"[lifespan] LLM config restore skipped: {type(e).__name__}: {e}")

    yield
    # 这样:
    #   · 后端重启 → 用户在前端设过的配置不会丢
    #   · 电脑/手机任意一端更新过配置 → 另一端 GET /ai/config 就能看到最新
    # 失败静默 — 读不到就用默认空值, 由前端推送兜底 (旧行为不变)。
    try:
        from ai.ai_config import load_ai_config_from_disk
        from ai.ai_emotion import load_emotion_llm_config_from_disk
        if load_ai_config_from_disk():
            print("[lifespan] Main LLM config restored from params.json")
        if load_emotion_llm_config_from_disk():
            print("[lifespan] Lint LLM config restored from params.json")
    except Exception as e:
        print(f"[lifespan] LLM config restore skipped: {type(e).__name__}: {e}")

    yield

app = FastAPI(lifespan=lifespan)

'''
******************************************
'''
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # 允许的源，可以是单个或多个
    allow_credentials=False,
    allow_methods=["*"],  # 允许的方法，例如 GET, POST 等
    allow_headers=["*"],  # 允许的请求头
)

if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS  # PyInstaller临时目录
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

static_file_abspath = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_file_abspath, exist_ok=True)  # 如果目录不存在则创建
app.mount("/static", StaticFiles(directory=static_file_abspath), name="static")

app.include_router(ai_router)

class AiLoadReq(BaseModel):
    fp:str

class AiSaveReq(BaseModel):
    fp:str
    mode:str
    index:int
    cont:Any

class OpacityArray(BaseModel):
    ops: List[float]  # 接收一个浮点数数组
    fp:str

class ImageReq(BaseModel):
    imgname:str
    base64:str
    mode:str
    # 论文/会话指纹, 决定图片落在 static/ 下的哪个子目录。
    # 为空时回落到 static/ 根目录 (兼容旧调用方 / 手工 curl 调试)。
    fp:str = ""

class HlInfo(BaseModel):
    fp:str
    key:str
    hl:Any

@app.put("/reqHighlightSet")
async def req_highlight_save(req_info: HlInfo):
    # 确保save目录存在
    save_dir = os.path.join(base_dir, 'save')
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建
    file_path = os.path.join(save_dir, f'{req_info.fp}_hls.json')

    if not os.path.exists(file_path):
        res = {}
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            res = json.load(f)  # 读取JSON文件内容并赋值给load
    if req_info.key in res:
        del res[req_info.key]
    else:
        res[req_info.key] = req_info.hl

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(res, f)

@app.post("/reqHighlightGet")
async def req_highlight_load(req_info: HlInfo):
    # 确保save目录存在
    save_dir = os.path.join(base_dir, 'save')
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建
    file_path = os.path.join(save_dir, f'{req_info.fp}_hls.json')

    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        res = json.load(f)  # 读取JSON文件内容并赋值给load
        return res

@app.post("/reqLoadAIPre")
async def func(req_info:AiLoadReq):
    """检查该论文是否有 _ai.json 历史（新版 schema）"""
    file_path = os.path.join(base_dir, 'save', f'{req_info.fp}_ai.json')
    return os.path.exists(file_path)

@app.post("/reqLoadAI")
async def func(req_info: AiLoadReq):
    """
    读 _ai.json 并透传给前端。

    输出 entry 字段与后端持久化格式完全一致:
      {
        "type": "ReqLoad" | "ReqAsk" | "ResLoad" | "ResAsk" | "Anno",
        "content": str,
        "ts": int,
        "dt": str,
        "msg_fp": str,
        "quotes": [...],
        "img": str,
        "hl": ...,
        "token_count": int | null,
        "flag": str | null,        # Anno 专属
      }

    旧数据(role/user/assistant 格式)透传兼容。

    注意: 此接口不参与 LLM 上下文加载，上下文过滤由 _load_paper_history 负责。
    """
    file_path = os.path.join(base_dir, 'save', f'{req_info.fp}_ai.json')
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    viewmodel = []
    for e in entries:
        t = e.get("type", "")
        if t in ("ReqLoad", "ReqAsk", "ResLoad", "ResAsk"):
            viewmodel.append({
                "type": t,  # 透传:ReqLoad | ReqAsk | ResLoad | ResAsk
                "content": e.get("content", ""),
                "ts": e.get("ts"),
                "dt": e.get("dt", ""),
                "msg_fp": e.get("msg_fp", ""),
                "quotes": e.get("quotes", []),
                "hl": e.get("hl"),
                "img": e.get("img", ""),
                "token_count": e.get("token_count"),
            })
        elif t == "Anno":
            viewmodel.append({
                "type": "Anno",
                "content": e.get("content", ""),
                "ts": e.get("ts"),
                "dt": e.get("dt", ""),
                "hl": e.get("hl"),
                "img": e.get("img", ""),
                "flag": e.get("flag", e.get("anno_id", "Anno")),
            })
        else:
            # 旧数据兼容 (role/user/assistant 格式): 仍按旧字段透传
            viewmodel.append(e)

    debug(f'PDF[{req_info.fp}] has loaded {len(viewmodel)} entries!')
    return viewmodel

@app.put("/reqSaveAI")
async def func(req_info:AiSaveReq):
    # 确保save目录存在
    save_dir = os.path.join(base_dir, 'save')
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建

    file_path = os.path.join(save_dir, f'{req_info.fp}.json')

    try:
        # 如果文件不存在，创建一个空列表
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            data:list = json.load(f)

        # 根据mode处理数据
        if req_info.mode == 'ADD':
            data.append(req_info.cont)
        elif req_info.mode == 'DEL':
            # 越界保护:前端 ai_res 与后端 file 在并发/失败场景下长度可能不一致;
            # 越界直接静默跳过(等同于删除一个不存在的元素),不再 500。
            idx = req_info.index
            if idx is None or idx < 0 or idx >= len(data):
                debug(f'[reqSaveAI] DEL out-of-range idx={idx} len={len(data)} | fp={req_info.fp}')
            else:
                data.pop(idx)
        elif req_info.mode == 'SET':
            # 越界保护:SET 越界时降级为 append,保住前端的对话内容,
            # 后续 handelAiLoad 会按新长度重新对齐。
            idx = req_info.index
            if idx is None or idx < 0 or idx >= len(data):
                debug(f'[reqSaveAI] SET out-of-range idx={idx} len={len(data)} -> append | fp={req_info.fp}')
                data.append(req_info.cont)
            else:
                data[idx] = req_info.cont
        else:
            debug(f'[reqSaveAI] unknown mode: {req_info.mode}, fp={req_info.fp}')
            raise ValueError(f"Unknown mode: {req_info.mode}")

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        debug(f'[reqSaveAI] saved ok: mode={req_info.mode} fp={req_info.fp} index={req_info.index}')

    except json.JSONDecodeError as e:
        debug(f'[reqSaveAI] JSONDecodeError: {e} | fp={req_info.fp}')
        raise HTTPException(status_code=500, detail=f"Invalid JSON format in file: {e}")
    except Exception as e:
        debug(f'[reqSaveAI] ERROR: {type(e).__name__}: {e} | fp={req_info.fp}')
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.put("/reqSaveAnno")
async def func(req_info: AiSaveReq):
    """
    ANNO 批注专用路由，写入 _ai.json。

    不复用 /reqSaveAI（该路由专管前端 Load/Ask 写盘，已废弃）。
    """
    save_dir = os.path.join(base_dir, 'save')
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'{req_info.fp}_ai.json')

    try:
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

        with open(file_path, 'r', encoding='utf-8') as f:
            data: list = json.load(f)

        if req_info.mode == 'ADD':
            data.append(req_info.cont)
        elif req_info.mode == 'DEL':
            idx = req_info.index
            if idx is None or idx < 0 or idx >= len(data):
                debug(f'[reqSaveAnno] DEL out-of-range idx={idx} len={len(data)} | fp={req_info.fp}')
            else:
                data.pop(idx)
        elif req_info.mode == 'SET':
            idx = req_info.index
            if idx is None or idx < 0 or idx >= len(data):
                debug(f'[reqSaveAnno] SET out-of-range idx={idx} len={len(data)} -> append | fp={req_info.fp}')
                data.append(req_info.cont)
            else:
                data[idx] = req_info.cont
        else:
            debug(f'[reqSaveAnno] unknown mode: {req_info.mode}, fp={req_info.fp}')
            raise ValueError(f"Unknown mode: {req_info.mode}")

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        debug(f'[reqSaveAnno] saved ok: mode={req_info.mode} fp={req_info.fp} index={req_info.index}')

    except json.JSONDecodeError as e:
        debug(f'[reqSaveAnno] JSONDecodeError: {e} | fp={req_info.fp}')
        raise HTTPException(status_code=500, detail=f"Invalid JSON format in file: {e}")
    except Exception as e:
        debug(f'[reqSaveAnno] ERROR: {type(e).__name__}: {e} | fp={req_info.fp}')
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/saveOps")
async def func(op_info: OpacityArray): 
    # 确保save目录存在
    save_dir = os.path.join(base_dir, 'save')
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建
    file_path = os.path.join(save_dir, f'{op_info.fp}_ops.json')
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(op_info.ops, f)

    return True

@app.post("/loadOps")
async def func(op_info: AiLoadReq): 
    file_path = os.path.join(base_dir, 'save', f'{op_info.fp}_ops.json')
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            res = json.load(f)  # 读取JSON文件内容并赋值给load
            return res
            
    return [0]*100  # 返回完整的响应


def _fp_static_dir(fp: str) -> str:
    """
    把 fp 映射成 static/ 下的子目录, 并保证结果被限制在该子目录内。

    fp 来自前端 (PDF 的 hash / crystal_chat), 理论上可控, 但这里仍做一次
    白名单清洗 —— os.path.join(static, "../../etc") 会直接逃出 static 根目录,
    配合 ADD 模式就能往任意路径写文件。
    """
    static_root = os.path.realpath(os.path.join(base_dir, 'static'))
    if not fp:
        return static_root
    # 只保留字母数字下划线横线, 其余一律丢弃; 清洗后为空则回落到根目录。
    safe = "".join(ch for ch in str(fp) if ch.isalnum() or ch in ("_", "-"))
    if not safe:
        return static_root
    d = os.path.realpath(os.path.join(static_root, safe))
    # 双保险: 确认没有跳出 static 根目录
    if os.path.commonpath([static_root, d]) != static_root:
        return static_root
    return d


def _img_path(req: "ImageReq") -> str:
    """图片的完整落盘路径 = static/{fp}/{imgname}。"""
    d = _fp_static_dir(req.fp)
    # imgname 只取 basename, 挡掉 "a/b.png" / "../x.png" 这类带路径的输入
    name = os.path.basename(str(req.imgname))
    if not name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="imgname 无效")
    return os.path.join(d, name)


@app.put("/reqImg")
async def handle_image(req: ImageReq):
    static_dir = os.path.join(base_dir, 'static')
    os.makedirs(static_dir, exist_ok=True)  # 如果目录不存在则创建

    # 构建文件完整路径: static/{fp}/{imgname}
    img_path = _img_path(req)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    save_dir = os.path.join(base_dir, 'save')
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建
    file_path = os.path.join(save_dir, f'imgname_maps.json')

    # 处理ADD模式：解码Base64并保存为PNG
    if req.mode == "ADD":
        # 移除Base64前缀（如 "data:image/png;base64,"）
        base64_data = req.base64.split(",")[-1]
        image_data = base64.b64decode(base64_data)
        with open(img_path, "wb") as f:
            f.write(image_data)

    # 处理DEL模式：删除指定图片
    elif req.mode == "DEL":
        if os.path.exists(img_path):
            os.remove(img_path)
            # 目录空了就顺手删掉, 避免 static/{fp}/ 留一堆空壳。
            # 只回收"正好是 static/{fp}"这一层, 绝不碰 static/ 根目录本身。
            parent = os.path.dirname(img_path)
            root = _fp_static_dir("")
            try:
                if os.path.realpath(parent) != root and not os.listdir(parent):
                    os.rmdir(parent)
            except OSError:
                pass

        #删除名字映射
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                res = json.load(f)
                if req.imgname in res:
                    del res[req.imgname]

            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(res, f)

    
    # 处理无效模式
    else:
        raise HTTPException(
            status_code=400,
            detail="mode参数无效，仅支持 'ADD' 或 'DEL'"
        )

@app.post("/reqImgInfoGet")
async def handle_image(req: ImageReq):
    save_dir = os.path.join(base_dir, 'save')
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建
    file_path = os.path.join(save_dir, f'imgname_maps.json')

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            res = json.load(f)  
            if req.imgname in res:
                return res[req.imgname]
    
    return ""


@app.post("/reqImgInfoSet")
async def handle_image(req: ImageReq):
    save_dir = os.path.join(base_dir, 'save')
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建
    file_path = os.path.join(save_dir, f'imgname_maps.json')

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            res = json.load(f)  
            res[req.imgname] = req.base64   #base64位置暂代图片名字
            if req.base64 == "":
                del res[req.imgname]
                
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(res, f)

    else:
        # 写回文件
        if req.base64 != "":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    req.imgname:req.base64
                }, f)

def delete_dir_fp(dir, fp):
    """
    清空与该指纹相关的文件。

    现在图片落在 static/{fp}/ 子目录下, 而删除是按"文件名包含 fp"匹配的
    (论文侧 imgname 形如 {fp}_{ts}.png), 所以必须递归进子目录才能删到。
    """
    if not os.path.exists(dir):
        return
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)

        if os.path.isdir(file_path):
            # 子目录: 命中 fp 就整目录删 (static/{fp}/ 正是这种结构),
            # 否则递归下去继续找文件名里带 fp 的散图。
            if fp in filename:
                import shutil
                shutil.rmtree(file_path, ignore_errors=True)
            else:
                delete_dir_fp(file_path, fp)
            continue

        # 只处理文件
        if fp in filename:
            os.remove(file_path)
                
@app.put("/clear_all")
async def handle_image(req: Dict):
    debug(f"Request for delete {req['fp']} all!")
    # 清空所有和该指纹相关的记录
    save_dir = os.path.join(base_dir, 'save')
    static_dir = os.path.join(base_dir, 'static')

    delete_dir_fp(save_dir, req['fp'])
    delete_dir_fp(static_dir, req['fp'])
     
               
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """
    Layer1 WebSocket 端点 (Crystal 主动追问的推送通道)。

    上行 (客户端 → 服务端):
      {"type": "user_msg", "pdf_fp": str, "content": str, "ts_ms": float}
      (Layer1 不需要已读回执 — deadline 是软标记)

    下行 (服务端 → 客户端):
      {"type": "push", "entry": ResAsk entry dict}            ← ai_emotion 等
      {"type": "ResActive", "pdf_fp": str, "entry": entry}    ← ai_active 主动追问
        (前端 ChatView 用 type 区分主动/被动; active=True 时打视觉徽章)

    连接维护:
      · FastAPI/WebSocket 原生处理重连, 前端 ws.onclose / onerror 中重连即可。
      · 应用层 ping/pong 保活 (chat 场景用户可能沉默 > 1h, 任何中间环节
        路由器/NAT/nginx 都会在 60-300s idle 后清连接):
          客户端每 30s 发 {"type":"ping"} → 服务端回 {"type":"pong"}。
        不更新业务状态, 仅用于刷新中间链路 idle 计数器。
    """
    from ai.ai_active import (
        on_user_msg,
        register_ws_client,
        unregister_ws_client,
    )

    await websocket.accept()
    register_ws_client(websocket)
    debug("[ws] client connected")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")
            if msg_type == "user_msg":
                on_user_msg(
                    pdf_fp=msg.get("pdf_fp", ""),
                    content=msg.get("content", ""),
                    ts_ms=float(msg.get("ts_ms", 0)),
                )
            elif msg_type == "ping":
                # 应用层保活: 静默回 pong, 不动业务状态
                try:
                    await websocket.send_text(json.dumps({"type": "pong"}))
                except Exception as e:
                    debug(f"[ws] pong send fail: {type(e).__name__}: {e}")
    except WebSocketDisconnect:
        debug("[ws] client disconnected")
    finally:
        unregister_ws_client(websocket)


# if __name__ == '__main__':
#     uvicorn.run(app='Main:app', host="127.0.0.1", port=8225, reload=True)
    
if __name__ == '__main__':
    # 打包环境下需要绝对导入
    from PdfBacken import app  # 明确从Main模块导入app对象
    
    uvicorn.run(
        app,  # 直接使用app对象而不是字符串
        host="127.0.0.1",
        port=8225,
        reload=False,
        log_config=None,
    )