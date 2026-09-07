# PDFAI - 智能 PDF 阅读辅助工具

一个 Chrome 浏览器插件，用于在浏览器中打开 PDF 时提供 AI 辅助功能，包括划词翻译、视觉问答、注释等功能。

## 📁 项目结构

```
PDFAI/
├── ai-settings/          # 设置面板 - 管理 API 配置和模型选择
├── pdf-iframe/           # 主功能模块 - PDF 阅读辅助界面
├── crx-viewer-new/       # 路由脚本 - 组件间通信中枢
└── pdf-translate-mini/   # 轻量翻译模块
```

## 🔧 功能模块说明

### 1. ai-settings (设置面板)

负责存储用户配置到 `chrome.storage.local`：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| model | **Main Model** - 主要语言模型，用于翻译和文本处理 | gemini-2.0-flash-lite |
| visionModel | **Vision Model** - 视觉模型，用于图片问答和视觉理解 | gpt-4o-mini |
| apiUrl | API 端点地址 | - |
| apiKey | API 密钥 | - |
| prompt | 自定义提示词 | - |

### 2. pdf-iframe (主功能模块)

核心功能组件，包含：
- 划词翻译和解释
- 视觉问答（上传图片提问）
- 注释功能
- PDF 内容高亮和定位

### 3. crx-viewer-new (路由脚本)

组件间通信的中枢，负责：
- 从 `chrome.storage.local` 读取配置
- 通过 `chrome.runtime.sendMessage` 向 iframe 发送 `SET_MODEL` 消息
- 监听存储变化并实时更新所有组件

## 🔄 组件通信流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         Chrome Extension                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │  ai-settings    │  用户修改设置                               │
│  │  (设置面板)      │──────────┐                                │
│  └────────┬────────┘           │                                │
│           │ 保存到               │ chrome.storage.local.set()   │
│           ▼                      ▼                                │
│  ┌────────────────────────────────────────┐                     │
│  │          chrome.storage.local          │  (持久化存储)        │
│  │  - model: Main Model                    │                     │
│  │  - visionModel: Vision Model            │                     │
│  │  - apiUrl, apiKey, prompt               │                     │
│  └────────────────────┬───────────────────┘                     │
│                       │ 变化监听                                 │
│                       ▼                                          │
│  ┌─────────────────┐                                            │
│  │  routerscript   │  读取新配置                                 │
│  │  (路由脚本)      │────────┐                                   │
│  └────────┬────────┘         │                                   │
│           │                  │ chrome.runtime.sendMessage()     │
│           │                  │ type: "SET_MODEL"                 │
│           └────────┬─────────┘                                   │
│                    │                                             │
│                    ▼                                             │
│  ┌─────────────────────────────┐                                │
│  │        pdf-iframe           │  接收配置并更新                 │
│  │        (主功能模块)          │◄───────────────────────────────┘
│  │                             │                                 │
│  │  • askModel = model        │  Main Model 用于语言翻译        │
│  │  • gptModelVision = visionModel  Vision Model 用于视觉问答   │
│  └─────────────────────────────┘                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 🔑 关键配置项

### Main Model vs Vision Model

| 模型类型 | 用途 | 示例模型 |
|----------|------|----------|
| **Main Model** | 文本翻译、解释、总结 | gemini-2.0-flash-lite, gpt-4o |
| **Vision Model** | 图片问答、视觉理解 | gpt-4o-mini, claude-3-haiku |

## 🚀 开发说明

### 重新构建项目

```bash
# 设置面板
cd ai-settings
npm install
npm run build

# 主功能模块
cd pdf-iframe
npm install
npm run build
```

### 调试技巧

在 `home.vue` 中查看控制台日志：
- `VUE PARAMETERS SET` - 显示接收到的模型配置
- `VUE[vid] SELECT TEXT` - 显示选中的文本

## 📝 最近更新

### 2026-09-07
- ✨ 新增 Vision Model 配置项
  - 将原有 Model 字段改名为 Main Model
  - 新增 Vision Model 独立配置项
  - Main Model 负责语言输出
  - Vision Model 负责视觉问答
