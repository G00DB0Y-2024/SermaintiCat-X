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
    读 _ai.json (新 schema entry) 并转换为前端 LIST_TYPE_* viewmodel。

    返回 _ai.json 全部 entry（含 Anno、含 Vision Ask）供前端渲染。
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

    _MSG_TYPE_TO_LIST_TYPE = {
        "ReqLoad": "LIST_TYPE_ASK",
        "ReqAsk":  "LIST_TYPE_ASK",
        "ResLoad": "LIST_TYPE_AI",
        "ResAsk":  "LIST_TYPE_AI",
    }

    viewmodel = []
    for e in entries:
        t = e.get("type", "")
        if t in _MSG_TYPE_TO_LIST_TYPE:
            viewmodel.append({
                "type": _MSG_TYPE_TO_LIST_TYPE[t],
                "text": e.get("content", ""),
                "dt":   e.get("dt", ""),
                "hl":   e.get("hl"),
                "img":  e.get("img", ""),
                "token_count": e.get("token_count"),
            })
        elif t == "Anno":
            viewmodel.append({
                "type": f"LIST_TYPE_{e.get('anno_id', 'ANNO_' + str(e.get('ts', '')))}",
                "text": e.get("content", ""),
                "dt":   e.get("dt", ""),
                "hl":   e.get("hl"),
                "img":  e.get("img", ""),
            })
        else:
            # 旧数据兼容 (role/user/assistant 格式)
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


@app.put("/reqImg")
async def handle_image(req: ImageReq):
    static_dir = os.path.join(base_dir, 'static')
    os.makedirs(static_dir, exist_ok=True)  # 如果目录不存在则创建

    # 构建文件完整路径
    img_path = os.path.join(static_dir, req.imgname)  #名字自带.png
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
    if not os.path.exists(dir):
        return
    # 遍历目录中的所有文件
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        
        # 只处理文件，跳过子目录
        if os.path.isfile(file_path):
            # 检查文件名是否包含fp
            if fp in filename:
                # 删除文件
                os.remove(file_path)
                
@app.put("/clear_all")
async def handle_image(req: Dict):
    debug(f"Request for delete {req['fp']} all!")
    # 清空所有和该指纹相关的记录
    save_dir = os.path.join(base_dir, 'save')
    static_dir = os.path.join(base_dir, 'static')

    delete_dir_fp(save_dir, req['fp'])
    delete_dir_fp(static_dir, req['fp'])
     
               
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