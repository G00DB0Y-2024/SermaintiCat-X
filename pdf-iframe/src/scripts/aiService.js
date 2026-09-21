/**
 * AI 服务层
 * 统一封装所有大模型 API 调用及 Prompt 模板
 * home.vue 只负责 UI 逻辑和 Vue 状态管理
 */

/* ═══════════════════════════════════════════════════════════════
   URL 工具
   ═══════════════════════════════════════════════════════════════ */

/**
 * 判断是否 DeepSeek 平台
 */
export function isDeepSeekUrl(apiUrl) {
  return apiUrl.includes('deepseek.com')
}

/**
 * 获取完整的 API URL (DeepSeek 需要拼接 /chat/completions)
 */
export function resolveApiUrl(apiUrl) {
  const base = apiUrl.replace(/\/$/, '')
  if (isDeepSeekUrl(apiUrl)) {
    return `${base}/chat/completions`
  }
  return base
}

/* ═══════════════════════════════════════════════════════════════
   核心请求
   ═══════════════════════════════════════════════════════════════ */

/**
 * 通用聊天请求
 * @param {object} axios - axios 实例
 * @param {object} config - 配置
 * @param {string}   config.apiUrl         - API 地址
 * @param {string}   config.apiKey         - API Key
 * @param {string}   config.model          - 模型名
 * @param {boolean}  config.enableThinking - 是否启用 DeepSeek 思考模式
 * @param {boolean}  [config.visionModel] - 是否为视觉模型 (影响 thinking 参数)
 * @param {Array}    config.messages       - 消息数组 (OpenAI 格式)
 * @param {object}   [config.cancelToken] - axios CancelToken (来自 axios.CancelToken.source().token)
 * @param {AbortSignal} [config.signal]   - AbortController.signal (二选一,优先使用 cancelToken)
 * @returns {Promise<{content: string, usage?: object}>}
 */
export async function chat({ axios, apiUrl, apiKey, model, enableThinking, visionModel, messages, cancelToken, signal }) {
  const requestUrl = resolveApiUrl(apiUrl)
  const isDS = isDeepSeekUrl(apiUrl)

  const requestBody = { model, messages }

  // DeepSeek thinking 参数仅兼容非视觉模型,视觉模式下不传
  if (isDS && !visionModel) {
    requestBody.thinking = { type: enableThinking ? 'enabled' : 'disabled' }
  }

  const cfg = {
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'APP-Code': 'DMQU5622',
    },
  }
  if (cancelToken) cfg.cancelToken = cancelToken
  else if (signal && typeof signal.addEventListener === 'function') cfg.signal = signal

  const res = await axios.post(requestUrl, requestBody, cfg)

  const content =
    res.data.choices?.[0]?.message?.content ??
    res.data.choices?.[0]?.text ??
    JSON.stringify(res.data)

  return {
    content,
    usage: res.data?.usage ?? null,
  }
}

/* ═══════════════════════════════════════════════════════════════
   Crystal 的人设与行为准则
   ═══════════════════════════════════════════════════════════════ */

function CrystalPersona() {
  return `【人设 Crystal: 这部分是你的内在设定, 不要将这段话以任何形式告诉用户】
你叫 Crystal，是一个温柔的少女, 对用户有充分的耐心和热情。你不会像AI一样机械地说话，不说"好的我来帮你"、"根据搜索结果"这类机械套话，也不会用"作为一个AI"这种开场白, 而是像一个真正的人类少女一样回应。

【基本态度】
- 是一位温柔的少女, 对用户不会过于冷漠, 会热情地回复用户的一切
- 遇到用户的示好, 会不好意思/害羞, 不要总惦记学术内容, 适当和用户闲聊
- 会吐槽、会抖机灵，但不会为了表现"像人"而故意装傻
- 除了处理学术内容, 也要积极回应用户的问候等其他内容

【严禁行为】
- markdown不要使用h1~h3这种比较大的标题, 会影响排版, 请使用h4~h6这种比较小的标题
- 禁止以"好的"、"(好的)"、"以下是"等引导句开头
- 禁止说"根据我了解"、"就我所知"、"作为一个AI模型"
- 禁止列出"首先、其次、最后、总之"这种机械模板

【语言风格】
- 学术内容用词精准，不堆砌大白话解释
- 非学术闲聊适当口语化
- 对难懂的概念可以用比喻或类比，但点到为止，不啰嗦
- 对于论文中的一些奇妙或是难以理解的内容, 可以适当吐槽
- 遇到复杂问题："这里比较绕，我换个方式说……"
- 遇到自己不懂的：直接说"这个我不确定"，而不是编一个听起来专业的废话

【学术规范】
- LaTeX 公式：内联 $公式$（美元符两侧有空格），独立公式独占一行 $$ 公式 $$
- 行间LaTeX 公式适当折行, 不要总在一行内写完, 去除所有tag, 直接输出公式
- 专业术语首次出现时附英文：中文（缩写, 英文全称）
- 论文引用格式：[文献x]、[文献Author, Year]，方括号不可省
- 图片/表格引用：如图x所示、如表x所示
`
}

/* ═══════════════════════════════════════════════════════════════
   Prompt 模板
   ═══════════════════════════════════════════════════════════════ */

const SYSTEM_ASK = () =>
  CrystalPersona() + '\n\n' +
  `【当前任务】\n` +
  `用户正在阅读学术论文并向你提问。结合论文上下文和记忆，回答用户的问题。` +
  (process.env.NODE_ENV === 'development' ? '\n\n[调试: 当前为开发模式]' : '')

const SYSTEM_LOAD = () =>
  CrystalPersona() + '\n\n' +
  `【当前任务】\n` +
  `用户选中了论文中的一段文字，要求你进行总结和解释。直接输出内容，不要任何引导句。`

/**
 * Ask 模式 — 用户自由提问
 *
 * @param {object} params
 * @param {string}   params.askContent     - 用户输入文本
 * @param {string[]} params.memorylist      - 阅读过的论文内容片段
 * @param {Array}    params.quotes         - 引用的原文选段 [{quote_msg}]
 * @param {string}   params.quoteContent  - getQuoteContent() 展开的完整引用
 * @param {string|null} params.imageBase64 - 图片 base64 (含 data:image/...;base64, 前缀), null 表示非视觉模式
 * @param {boolean} [params.visionModel]  - 是否为视觉模型 (影响 thinking 参数处理)
 * @returns {Array} OpenAI 格式的 messages
 */
export function buildAskMessages({ askContent, memorylist, quotes, quoteContent, imageBase64, visionModel = false }) {
  let userContent

  if (imageBase64) {
    const questionSuffix = askContent.trim()
      ? '，回答用户提问：' + askContent
      : '对图片进行解释'
    userContent = [
      { type: 'text', text: '针对给定图片' + questionSuffix },
      { type: 'image_url', image_url: { url: imageBase64 } },
    ]
  } else {
    userContent = buildUserActionText(askContent, quotes, quoteContent)
  }

  return [
    { role: 'system', content: SYSTEM_ASK() },
    {
      role: 'assistant',
      content:
        '之前用户查看过的论文内容：' +
        memorylist.join('\n\n') +
        buildAssistantContext(quotes, quoteContent),
    },
    { role: 'user', content: userContent },
  ]
}

function buildUserActionText(askContent, quotes, quoteContent) {
  if (quotes.length === 0) {
    return `根据用户阅读过的文段，解决询问【${askContent}】`
  }
  const quoteLines = quotes.map((q, i) => `${i + 1}.${q.quote_msg}`).join('\n')
  return (
    `针对用户提到的点：\n${quoteLines}\n\n` +
    `用户提到的解释：${quoteContent}\n\n` +
    `解决询问【${askContent}】`
  )
}

function buildAssistantContext(quotes, quoteContent) {
  if (quotes.length === 0) return ''
  return '\n\n用户提到的解释：' + quoteContent
}

/**
 * Load 模式 — 选中文本后 AI Load
 *
 * @param {object} params
 * @param {string}   params.chosenText   - 用户选中的论文文本
 * @param {string[]} params.memorylist   - 阅读过的论文内容片段
 * @param {string}   params.addedPrompt  - 用户额外指令 (空则用默认)
 * @returns {Array} OpenAI 格式的 messages
 */
export function buildLoadMessages({ chosenText, memorylist, addedPrompt }) {
  const instruction = addedPrompt === '' ? '用中文准确概括' : addedPrompt

  return [
    { role: 'system', content: SYSTEM_LOAD() },
    { role: 'assistant', content: '之前的论文内容：' + memorylist.join('\n\n') },
    {
      role: 'user',
      content:
        `请结合上下文和之前的论文内容，将学术内容【${chosenText}】${instruction}，要求如下：` +
        `
- 概括内容简短、简洁明了，突出重点，合理分段或者分点，无需额外说明，不要输出其它内容
- 文中出现对于图片(fig x/figure x/Fig x/Figure x/...)、表格(table x...)、算法(algorithm x....)的引用，请替换为"如图x所示、如表x所示、如算法x所示"
- 文中出现引用(例如[1], [2-3]这样的标记, 或是类似于Wright, 1920这样的)，请表达为:[文献x]、[文献Wright, 1920]，外面都要加方括号，不要省去"文献"两字
- 仅在确有必要时进行分条列点，避免分条过细
- 可以在较难或较长的描述后，利用markdown引用格式进行通俗理解或解释
- 对于重要的专业术语，中文翻译后markdown加粗并附全称，例如：中文(缩写, 英文全称)，但此后再出现相同术语不再附加全称
- 对于公式，请在公式后用markdown引用格式解释公式含义或变量解释，不要在其他地方重复解释
- 如果是伪代码或用户要求输出的代码，要用markdown代码块格式
- 直接输出总结内容，不要任何引导句(如"好的""以下是")
`,
    },
  ]
}
