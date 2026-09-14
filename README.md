# PDFAI 技术文档

## 目录
1. [项目概述](#1-项目概述)
2. [技术架构](#2-技术架构)
3. [项目结构](#3-项目结构)
4. [前端模块详解](#4-前端模块详解)
5. [后端API接口](#5-后端api接口)
6. [组件通信机制](#6-组件通信机制)
7. [数据模型](#7-数据模型)
8. [配置管理](#8-配置管理)
9. [样式系统](#9-样式系统)
10. [部署说明](#10-部署说明)

---

## 1. 项目概述

### 1.1 项目简介
**PDFAI** (项目代号: SemantiCat) 是一个 Chrome 浏览器插件，为用户在浏览器中阅读 PDF 文档时提供 AI 辅助功能。该插件支持划词翻译、视觉问答、注释标注等核心功能，极大地提升了学术论文和技术文档的阅读效率。

### 1.2 核心功能
| 功能 | 描述 |
|------|------|
| 划词翻译 | 选中 PDF 文本后自动翻译，支持自定义提示词 |
| 视觉问答 | 上传图片并针对图片内容提问，获取 AI 分析 |
| 注释标注 | 为文档添加个人注释，带原文引用 |
| 文本高亮 | Alt+左键快速高亮，支持持久化存储 |
| Markdown 渲染 | 支持 KaTeX 数学公式、代码高亮 |
| 智能引用识别 | 自动识别并美化图、表、算法编号引用 |

### 1.3 技术栈

| 层级 | 技术选型 |
|------|----------|
| **浏览器插件框架** | Chrome Extension Manifest V3 |
| **前端框架** | Vue 3 (Composition API) |
| **构建工具** | Vite 4.x |
| **UI 样式** | 自定义 CSS + GitHub 主题 |
| **Markdown 渲染** | markdown-it + markdown-it-katex-gpt |
| **数学公式** | KaTeX |
| **代码高亮** | highlight.js |
| **HTTP 客户端** | Axios |
| **后端框架** | FastAPI (Python) |
| **异步服务器** | Uvicorn |
| **API 集成** | OpenAI SDK |

---

## 2. 技术架构

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Chrome Browser                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Chrome Extension                            │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │                                                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │   │
│  │  │ ai-settings │  │ pdf-iframe  │  │   crx-viewer-new    │   │   │
│  │  │ (设置面板)   │  │ (主功能)    │  │   (路由脚本)         │   │   │
│  │  │   popup     │  │   iframe    │  │ content_script      │   │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘   │   │
│  │         │                │                     │               │   │
│  │         │    chrome.storage.local            │               │   │
│  │         │    ┌──────────┴──────────┐         │               │   │
│  │         └────┤  配置数据中转中心    ├─────────┘               │   │
│  │              └──────────┬──────────┘                         │   │
│  │                         │                                    │   │
│  │              chrome.runtime.sendMessage                      │   │
│  │                         │                                    │   │
│  │              ┌──────────┴──────────┐                        │   │
│  │              │   routerscript.js   │                        │   │
│  │              │   (组件通信中枢)     │                        │   │
│  │              └──────────┬──────────┘                        │   │
│  │                         │                                   │   │
│  └─────────────────────────┼───────────────────────────────────┘   │
│                            │                                      │
│                            │ postMessage / iframe                 │
│                            ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    PDF Viewer (PDF.js)                      │   │
│  │              contentscript.js (PDF 内容脚本)                  │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             │ HTTP / WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Backend Server                                │
│                   FastAPI (Python) Port: 8225                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │ AI 对话接口 │  │ 数据持久化   │  │ 静态资源    │                  │
│  │ /api/chat  │  │ /save/*.json│  │ /static/    │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                     Local File Storage                        │    │
│  │  ├── save/              (AI 对话、标注、高亮数据)            │    │
│  │  └── static/            (用户上传的图片)                      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心设计模式

#### 观察者模式 (配置同步)
- `ai-settings` 保存配置 → `chrome.storage.local` 触发 `onChanged` 事件
- `routerscript.js` 监听变化 → 广播 `SET_MODEL` 消息给所有组件

#### 发布-订阅模式 (组件通信)
- 组件通过 `chrome.runtime.sendMessage` 发布消息
- 目标组件通过 `chrome.runtime.onMessage.addListener` 订阅消息

#### MVVM 模式 (前端组件)
- Vue 3 组件实现数据响应式绑定
- `home.vue` 作为主容器，管理 `ai_res` 状态数组

---

## 3. 项目结构

```
PDFAI/
│
├── ai-settings/                    # 设置面板模块
│   ├── src/
│   │   ├── main.js               # Vue 应用入口
│   │   └── App.vue               # 设置面板主组件
│   ├── index.html                # popup 页面入口
│   ├── package.json              # 依赖配置
│   ├── vite.config.js            # Vite 构建配置
│   └── assets/                   # 静态资源
│
├── pdf-iframe/                    # 主功能模块 (PDF 辅助阅读)
│   ├── src/
│   │   ├── main.js              # Vue 应用入口 + Axios 配置
│   │   ├── App.vue             # 根组件
│   │   ├── components/
│   │   │   ├── home.vue        # 主容器组件 (核心业务逻辑)
│   │   │   ├── gptRenderUnit.vue  # AI 响应渲染单元
│   │   │   ├── imageRender.vue    # 图片渲染组件
│   │   │   ├── waiting.vue        # 加载等待组件
│   │   │   ├── range-serializer.js # DOM Range 序列化
│   │   │   └── selector-generator.js # CSS 选择器生成
│   │   ├── assets/
│   │   │   ├── base.css         # 基础样式
│   │   │   ├── main.css        # 主样式入口
│   │   │   ├── github.css      # GitHub 风格主题
│   │   │   ├── images/         # 资源图片
│   │   │   └── github/         # Open Sans 字体文件
│   │   └── router/             # 路由配置
│   ├── public/
│   │   └── test.pdf            # 测试 PDF 文件
│   ├── index.html              # iframe 页面入口
│   ├── package.json            # 依赖配置
│   ├── vite.config.js         # 构建配置 (输出到 crx-viewer-new/gptcore)
│   └── prompt.txt              # AI 翻译提示词模板
│
├── pdf-backen/                  # 后端服务模块
│   ├── PdfBacken.py            # FastAPI 主应用
│   ├── utils/
│   │   └── log.py              # 日志工具
│   ├── static/                 # 静态文件目录 (运行时创建)
│   ├── save/                   # 数据存储目录 (运行时创建)
│   ├── Pipfile                # Python 依赖管理
│   └── requirements.txt       # pip 依赖列表
│
├── crx-viewer-new/              # Chrome 扩展核心
│   ├── manifest.json           # 扩展清单 (Manifest V3)
│   ├── background.js          # Service Worker
│   ├── contentscript.js       # PDF 内容脚本
│   ├── routerscript.js        # 消息路由脚本
│   ├── contentstyle.css       # 内容脚本样式
│   ├── extension-router.js    # 扩展路由
│   ├── preferences_schema.json # 托管配置 schema
│   ├── options/               # 选项页面
│   ├── settings/              # 设置面板构建输出
│   ├── gptcore/               # pdf-iframe 构建输出
│   ├── translator-mini/       # 轻量翻译模块
│   ├── pdfbar/                # PDF 工具栏
│   └── images/                # 扩展图标
│
└── README.md                   # 项目说明文档
```

---

## 4. 前端模块详解

### 4.1 ai-settings (设置面板)

**功能**: 管理 API 配置和模型选择

**配置项**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | string | `gemini-2.0-flash-lite` | 主模型 (用于翻译和文本处理) |
| `visionModel` | string | `gpt-4o-mini` | 视觉模型 (用于图片问答) |
| `apiUrl` | string | `""` | API 端点地址 |
| `apiKey` | string | `""` | API 密钥 |
| `prompt` | string | `""` | 自定义提示词 |

**关键代码逻辑**:

```javascript
// 保存配置
chrome.storage.local.set({
  model: this.model,
  visionModel: this.visionModel,
  apiKey: this.apiKey,
  apiUrl: this.apiUrl,
  prompt: this.prompt,
})
```

**构建输出**: `crx-viewer-new/settings/`

---

### 4.2 pdf-iframe (主功能模块)

#### 4.2.1 home.vue - 主容器组件

**职责**: 核心业务逻辑处理中心

**核心状态 (data)**:

```javascript
{
  // 选中文本相关
  chosen_text: "",           // 当前选中的文本
  chosen_add_text: [],       // 追加选中的文本列表
  shown_text: "",           // 显示的原文
  
  // AI 响应相关
  ai_res: [],               // AI 响应数组 (核心状态)
  memorylist: [],           // 记忆列表
  
  // 模型配置
  gptModel: "gemini-2.0-flash",
  askModel: "gemini-2.5-flash",
  gptModelVision: "",       // 视觉模型
  
  // UI 状态
  showOrigin: false,        // 显示原文开关
  askFocus: false,         // 追问框焦点
  waiting: false,          // 加载状态
  
  // 注释相关
  annotation_mode: false,  // 注释模式
  annotation_hldata: null, // 注释高亮数据
  
  // 高亮相关
  hl_arr: {},              // 高亮位置映射 {key: {range, position}}
  
  // 滚动相关
  childrenHeightArr: [],    // 子元素高度数组
  classifyRateArr: [0,0,0], // 分类占比 [Input, Feedback, Output]
  scrollTopRate: "0%",     // 滚动位置百分比
}
```

**消息类型 (message.type)**:

| 类型 | 来源 | 说明 |
|------|------|------|
| `SET_TEXT` | contentscript.js | 设置选中文本 |
| `SET_MODEL` | routerscript.js | 更新模型配置 |
| `PDF_FP` | routerscript.js | 设置 PDF 指纹 |
| `ADD_ANNOTATION` | contentscript.js | 请求添加注释 |
| `RESET_MORE` | contentscript.js | 清除追加文本 |
| `VUE_INIT` | home.vue | 组件初始化完成 |

**响应类型 (ai_res[].type)**:

| 类型 | 说明 | 颜色标识 |
|------|------|----------|
| `LIST_TYPE_AI` | AI 翻译/响应 | 蓝色 #4870AC |
| `LIST_TYPE_ASK` | 用户追问 | 绿色 #559145 |
| `LIST_TYPE_ANNO_*` | 注释 (带时间戳) | 红色 #AC4848 |

**核心方法**:

| 方法 | 说明 |
|------|------|
| `onTextChange(text, add, hldata)` | 处理选中文本变化 |
| `handleAsk()` | 发送追问请求 |
| `handleAiChange(mode, index, content)` | 修改 ai_res 状态 |
| `setRangeHighlight(range, key)` | 设置文本高亮 |
| `handelSelectedHighlight(e)` | 鼠标选择高亮处理 |
| `UpdatePointerPosition()` | 更新滚动指针位置 |

#### 4.2.2 gptRenderUnit.vue - 渲染单元组件

**功能**: 单个 AI 响应的渲染组件

**Props**:

```javascript
{
  content: String,     // Markdown 内容
  datetime: String,   // 时间戳
  hldata: Any,        // 高亮数据
  gid: Number,        // 组 ID
  headtype: String,   // 响应类型
  vid: String,        // 视图 ID
  img_name: String,    // 图片名称
  editting: Boolean   // 编辑状态
}
```

**Emits**:

| 事件 | 说明 | 参数 |
|------|------|------|
| `onQuote` | 引用文本 | `(gid, content, mode)` |
| `onClose` | 关闭响应 | `()` |
| `onHlClick` | 高亮点击 | `()` |
| `onANNOClick` | 注释点击 | `()` |

**渲染流程**:

1. `parseMessage(content)` - 解析 `>>原文<<` 格式提取注释引用
2. `md.render(parsedContent)` - Markdown 转 HTML
3. `updateMathSpan()` - span.katex → p.katex (块级公式)
4. `checkMathOverflow()` - 检测公式溢出，动态调整对齐
5. `styleFigureText()` - 美化图/表/算法引用标签

**交互处理**:

| 交互 | 行为 |
|------|------|
| 左键点击公式 | 复制 LaTeX 源码到剪贴板 |
| 左键点击代码 | 复制代码文本到剪贴板 |
| 中键点击段落 | 引用文本到追问框 |
| 双击文本 | 复制处理后的文本 |

#### 4.2.3 imageRender.vue - 图片渲染组件

**功能**: 渲染用户上传的注释图片

**API 调用**:

| 端点 | 方法 | 说明 |
|------|------|------|
| `/reqImgInfoGet` | POST | 获取图片描述 |
| `/reqImgInfoSet` | POST | 设置图片描述 |

**图片 URL**: `http://localhost:8225/static/{img_name}`

#### 4.2.4 range-serializer.js - DOM Range 序列化

**功能**: 将浏览器选区 (Selection/Range) 序列化为 JSON，用于高亮数据持久化

**序列化格式**:

```javascript
{
  start: {
    selector: "div > p > span.katex",
    childNodeIndex: 0,
    offset: 5
  },
  end: {
    selector: "div > p > span.katex",
    childNodeIndex: 0,
    offset: 10
  }
}
```

**导出函数**:

| 函数 | 说明 |
|------|------|
| `serialize(range)` | Range → JSON 对象 |
| `deserialize(result)` | JSON 对象 → Range |

#### 4.2.5 selector-generator.js - CSS 选择器生成

**功能**: 根据 DOM 节点生成稳定的 CSS 选择器

**生成规则**:

1. 从当前节点向上遍历到根元素
2. 每层记录标签名和 `:nth-of-type()` 索引
3. 使用 `>` 连接生成完整路径

**示例**:

```
DOM: <body><div><p>text</p></div></body>
节点: text 文本节点
选择器: "div > p" + childNodeIndex: 0
```

---

## 5. 后端 API 接口

**基础 URL**: `http://localhost:8225`

**CORS**: 允许所有来源 (`allow_origins=['*']`)

### 5.1 AI 对话接口

> 注意: AI 对话接口未在 `PdfBacken.py` 中定义，需要在 `home.vue` 中直接调用外部 API

**调用方式** (home.vue):

```javascript
this.$axios.post(this.apiUrl, {
  model: this.askModel,
  messages: [
    { role: "system", content: systemPrompt },
    { role: "user", content: fullContent }
  ],
  stream: true
}, {
  cancelToken: this.cancelTokenSource.token,
  responseType: 'stream'
})
```

### 5.2 数据持久化接口

#### 5.2.1 高亮数据

| 接口 | 方法 | 说明 |
|------|------|------|
| `/reqHighlightSet` | PUT | 保存/删除高亮 |
| `/reqHighlightGet` | POST | 获取高亮列表 |

**请求体 (HlInfo)**:

```json
{
  "fp": "pdf_fingerprint",
  "key": "HL_1234567890_000001",
  "hl": { "start": {...}, "end": {...} } | null
}
```

**存储位置**: `save/{fp}_hls.json`

#### 5.2.2 AI 对话存储

| 接口 | 方法 | 说明 |
|------|------|------|
| `/reqLoadAIPre` | POST | 检查是否存在记录 |
| `/reqLoadAI` | POST | 加载所有对话 |
| `/reqSaveAI` | PUT | 保存对话 (ADD/DEL/SET) |

**请求体 (AiSaveReq)**:

```json
{
  "fp": "pdf_fingerprint",
  "mode": "ADD" | "DEL" | "SET",
  "index": 0,
  "cont": { "type": "LIST_TYPE_AI", "text": "...", "dt": "..." }
}
```

**存储位置**: `save/{fp}.json`

#### 5.2.3 透明度数据

| 接口 | 方法 | 说明 |
|------|------|------|
| `/saveOps` | POST | 保存透明度数组 |
| `/loadOps` | POST | 加载透明度数组 |

**存储位置**: `save/{fp}_ops.json`

#### 5.2.4 图片管理

| 接口 | 方法 | 说明 |
|------|------|------|
| `/reqImg` | PUT | 上传/删除图片 |
| `/reqImgInfoGet` | POST | 获取图片描述 |
| `/reqImgInfoSet` | POST | 设置图片描述 |

**图片存储位置**: `static/{pdf_fp}_{timestamp}.png`

**描述映射存储**: `save/imgname_maps.json`

#### 5.2.5 清理数据

| 接口 | 方法 | 说明 |
|------|------|------|
| `/clear_all` | PUT | 清空指定 PDF 的所有数据 |

**请求体**:

```json
{ "fp": "pdf_fingerprint" }
```

---

## 6. 组件通信机制

### 6.1 通信架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Chrome Extension Runtime                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐                                                   │
│  │ ai-settings  │                                                   │
│  │  (popup)     │  chrome.storage.local.set()                      │
│  └──────┬───────┘                                                   │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              chrome.storage.local                              │  │
│  │  { model, visionModel, apiKey, apiUrl, prompt }                │  │
│  └──────────────────────────┬─────────────────────────────────────┘  │
│                             │ onChanged 监听                          │
│                             ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    routerscript.js                             │  │
│  │                                                               │  │
│  │  chrome.runtime.sendMessage({ type: 'SET_MODEL', ... })      │  │
│  └──────────────────────────┬─────────────────────────────────────┘  │
│                             │                                        │
│         ┌────────────────────┼────────────────────┐                   │
│         │                    │                    │                   │
│         ▼                    ▼                    ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│  │  pdf-iframe  │    │ contentscript │    │   其他组件    │         │
│  │  (home.vue)  │    │     .js       │    │              │         │
│  │              │    │              │    │              │         │
│  │ onMessage    │    │ onMessage    │    │ onMessage    │         │
│  │ 监听器       │    │ 监听器        │    │ 监听器        │         │
│  └──────────────┘    └──────────────┘    └──────────────┘         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 消息流详解

#### 流程 1: 配置更新

```
用户修改设置
    ↓
ai-settings 保存到 chrome.storage.local
    ↓
storage.onChanged 触发
    ↓
routerscript.js 监听器收到变化
    ↓
chrome.runtime.sendMessage({ type: 'SET_MODEL', ... })
    ↓
home.vue 监听器接收并更新状态
```

#### 流程 2: 选中文本翻译

```
用户在 PDF 中选中文本
    ↓
contentscript.js 获取选区文本
    ↓
chrome.runtime.sendMessage({ type: 'SET_TEXT', text: '...' })
    ↓
routerscript.js 路由转发
    ↓
home.vue 接收文本，调用 AI 接口
    ↓
AI 响应追加到 ai_res 数组
    ↓
UI 自动更新渲染
```

#### 流程 3: PDF 指纹传递

```
PDF.js 解析 PDF 完成
    ↓
contentscript.js 计算 PDF 指纹
    ↓
window.postMessage({ type: 'PDF_FP', fp: '...' })
    ↓
routerscript.js 监听并转发
    ↓
chrome.runtime.sendMessage({ type: 'PDF_FP', fp: '...' })
    ↓
home.vue 接收指纹，加载历史数据
```

### 6.3 vid (视图 ID) 机制

**目的**: 支持多标签页/多 PDF 同时使用

**生成方式**: URL 参数 `?ts={timestamp}`

**匹配逻辑**:

```javascript
// 消息发送时
chrome.runtime.sendMessage({ vid: ts, type: 'SET_TEXT', ... })

// 消息接收时
if (message.vid === '' || message.vid === this.vid) {
  // 处理消息
}
```

- `vid === ''`: 广播消息 (如 SET_MODEL)
- `vid === this.vid`: 单播消息 (精确匹配)

---

## 7. 数据模型

### 7.1 AI 响应项 (ai_res 元素)

```typescript
interface AIResponse {
  type: 'LIST_TYPE_AI' | 'LIST_TYPE_ASK' | 'LIST_TYPE_ANNO_{timestamp}';
  text: string;           // Markdown 内容
  dt: string;            // 时间戳 "YYYY-MM-DD HH:mm:ss"
  hl: any;               // 高亮数据 (序列化 Range)
  img: string;           // 图片文件名 (可选)
}
```

### 7.2 高亮数据

```typescript
interface Highlight {
  key: string;           // "HL_{timestamp}_{randomCode}"
  hl: SerializedRange | null;
}

interface SerializedRange {
  start: Selector;
  end: Selector;
}

interface Selector {
  selector: string;       // CSS 选择器路径
  childNodeIndex: number; // 子节点索引
  offset: number;        // 文本偏移
}
```

### 7.3 用户选中文本

```typescript
interface SelectedText {
  text: string;          // 选中文本内容
  add: string[];         // 追加的文本数组
  hldata: SerializedRange; // 选区 Range 序列化
}
```

### 7.4 存储文件结构

```
pdf-backen/
├── save/
│   ├── {fp}.json           // AI 对话记录
│   ├── {fp}_hls.json       // 高亮数据
│   ├── {fp}_ops.json       // 透明度数组
│   └── imgname_maps.json   // 图片名称映射
│
└── static/
    └── {fp}_{timestamp}.png  // 用户上传的图片
```

---

## 8. 配置管理

### 8.1 扩展配置存储

**存储位置**: `chrome.storage.local`

**配置项**:

| 键 | 类型 | 默认值 | 描述 |
|----|------|--------|------|
| `model` | string | `gemini-2.0-flash-lite` | 主语言模型 |
| `visionModel` | string | `gpt-4o-mini` | 视觉模型 |
| `apiUrl` | string | `""` | API 端点 |
| `apiKey` | string | `""` | API 密钥 |
| `prompt` | string | `""` | 自定义提示词 |

### 8.2 提示词模板 (prompt.txt)

```markdown
- 翻译与总结：
  - 保持学术严谨性，语言简洁明了，合理分段，适当总结
  - 调整原文语序结构，进行适当分条列点

- 术语处理：
  - 首次出现专业术语：**中文(English)**
  - 术语缩写：**中文(Full English Name, 缩写)**

- 公式处理：
  - 行内公式：$公式$
  - 独立公式：$$ 单独一行 $$

- 格式要求：
  - 使用 Markdown 格式
  - 代码用 ``` 包裹
```

### 8.3 环境变量

**前端构建** (`pdf-iframe/vite.config.js`):

| 变量 | 值 | 说明 |
|------|-----|------|
| `build.outDir` | `crx-viewer-new/gptcore` | 构建输出目录 |
| `axios.defaults.baseURL` | `http://localhost:8225` | 后端 API 地址 |

**后端服务** (`pdf-backen/PdfBacken.py`):

| 变量 | 值 | 说明 |
|------|-----|------|
| `host` | `127.0.0.1` | 服务监听地址 |
| `port` | `8225` | 服务监听端口 |

---

## 9. 样式系统

### 9.1 样式文件结构

```
pdf-iframe/src/assets/
├── base.css      # 基础样式重置
├── main.css      # 主样式入口 (@import base.css)
└── github.css    # GitHub 风格主题
```

### 9.2 主题色彩系统 (github.css)

```css
:root {
  --primary-color: #4870ac;      /* 主色调 - 蓝色 */
  --text-color: #40464f;         /* 文本颜色 */
  --bg-color: #ffffff;           /* 背景色 */
  --side-bar-bg-color: #fafafa;  /* 侧边栏背景 */
  --marker-color: #a2b6d4;       /* 列表标记 */
  --source-color: #a8a8a9;       /* 引用/链接源 */
  --highlight-color: #ffffb5c2;  /* 高亮背景 */
  --block-bg-color: #f6f8fa;     /* 块级元素背景 */
}
```

### 9.3 组件样式

#### 响应类型颜色映射

| 类型 | 头部颜色 | 头部背景 | 索引条颜色 |
|------|----------|----------|------------|
| AI 响应 | `#4870AC` | `rgba(72,112,172,0.15)` | `rgb(122,155,205)` |
| 用户追问 | `#559145` | `rgba(203,226,201,0.5)` | `rgba(86,145,69,0.741)` |
| 注释 | `#AC4848` | `rgba(172,72,72,0.15)` | `#AC4848` |

#### KaTeX 数学公式样式

```css
/* 行内公式 */
.katex {
  margin: 0.3em 0;
  cursor: pointer;
}

/* 行间公式 */
math[display="block"] {
  display: flex;
  justify-content: center;  /* 溢出时改为 flex-start */
  overflow-x: auto;
}

/* 悬浮效果 */
semantics:hover {
  text-shadow: 1px 1px 1.5px rgba(0,89,255,0.534);
}
```

### 9.4 交互状态样式

```css
/* 段落悬浮 */
p:hover {
  background-color: rgb(232, 242, 255);
}

/* 引用高亮 */
code:hover {
  text-shadow: 2px 2px 2px rgba(0, 0, 0, 0.25);
}

/* 关闭按钮悬浮 */
.lb-close:hover {
  filter: brightness(0.75);
  scale: 1.02;
}
```

---

## 10. 部署说明

### 10.1 开发环境搭建

#### 前置条件

- Node.js 18+
- Python 3.9+
- Chrome 浏览器

#### 步骤 1: 安装前端依赖

```bash
# 设置面板
cd ai-settings
npm install

# 主功能模块
cd ../pdf-iframe
npm install
```

#### 步骤 2: 安装后端依赖

```bash
cd pdf-backen
pip install fastapi uvicorn pydantic python-multipart
# 或使用 Pipfile
pipenv install
```

#### 步骤 3: 启动后端服务

```bash
cd pdf-backen
python PdfBacken.py
# 服务运行在 http://127.0.0.1:8225
```

#### 步骤 4: 构建前端

```bash
# 构建设置面板
cd ai-settings
npm run build  # 输出到 ../crx-viewer-new/settings/

# 构建主功能模块
cd ../pdf-iframe
npm run build  # 输出到 ../crx-viewer-new/gptcore/
```

#### 步骤 5: 加载扩展

1. 打开 Chrome `chrome://extensions/`
2. 开启「开发者模式」
3. 点击「加载已解压的扩展程序」
4. 选择 `crx-viewer-new` 目录

### 10.2 生产环境构建

```bash
#!/bin/bash
# build.sh

# 清理并构建
rm -rf crx-viewer-new/settings crx-viewer-new/gptcore

cd ai-settings && npm run build
cd ../pdf-iframe && npm run build

echo "构建完成!"
```

### 10.3 目录权限

确保以下目录存在且有写入权限:

```bash
pdf-backen/
├── save/        # 777 权限
└── static/     # 777 权限
```

---

## 附录 A: 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl + Enter` | 发送追问 |
| `Alt + 左键选择` | 添加高亮 |
| `Alt + 左键点击高亮` | 删除高亮 |
| `鼠标中键点击段落` | 引用到追问框 |

## 附录 B: 调试日志

在 `home.vue` 中查看以下日志:

| 日志前缀 | 说明 |
|----------|------|
| `VUE Confirm VID` | 组件初始化，确认视图 ID |
| `VUE[vid] SELECT TEXT` | 接收到选中文本 |
| `VUE PARAMETERS SET` | 接收到模型配置 |

## 附录 C: 文件清单

| 文件路径 | 行数 | 功能描述 |
|----------|------|----------|
| `pdf-iframe/src/components/home.vue` | 1734 | 主容器组件，核心业务逻辑 |
| `pdf-iframe/src/components/gptRenderUnit.vue` | 630 | AI 响应渲染单元 |
| `pdf-backen/PdfBacken.py` | 310 | FastAPI 后端应用 |
| `crx-viewer-new/routerscript.js` | 60 | 消息路由中枢 |
| `crx-viewer-new/manifest.json` | 80 | Chrome 扩展清单 |

---

*文档版本: 1.0*
*最后更新: 2026-09-08*
*项目代号: SemantiCat*
