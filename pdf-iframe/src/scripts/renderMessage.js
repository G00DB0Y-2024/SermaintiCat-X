// ───────────────────────────────────────────────────────────────────────
//  renderMessage.js
// ───────────────────────────────────────────────────────────────────────
//  单一渲染管道,把"一段文本"转成"安全的 HTML + mentions 列表":
//
//      1. extractMath  — 抽出 $..$ / $$..$$ / \(..\) / \[..\] 换成占位符
//                         这样 markdown-it 的 breaks 配置不会再切到公式
//      2. markdown-it  — 渲染 Markdown 语法(标题/列表/引用/链接/代码块/...)
//      3. injectMath   — 把占位符换回 KaTeX 渲染结果
//      4. @ 提及扫描   — 在最终 HTML 的文本节点里扫描 @xxx,xxx 命中
//                         agentMap 的项替换为 chip span,同时把 agent_id
//                         加入返回的 mentions 数组(去重,保持首次顺序)
//
//  与之前"markdown-it 之后再正则替换 $..$"的实现相比:
//      · 三阶段法彻底绕开 markdown-it breaks:true 对换行的处理,公式前后
//        不会再被 <br> 包裹
//      · 与 ViewerView 走的是同一套生产验证过的代码路径(从 math.js 复用)
//
//  安全:
//      · markdown-it 已配置 html:false,外部 HTML 不会被信任解析
//      · KaTeX 的 throwOnError:false,出错仅降级显示原 $..$
//      · chip 替换发生在 markdown 渲染之后,因此 chip 的可见文本不会
//        被 markdown 当成 inline markup 处理
// ───────────────────────────────────────────────────────────────────────

import MarkdownIt from 'markdown-it'
// 与 ViewerView 复用同一套 LaTeX 处理代码 — 已生产验证
import { extractMath, injectMath } from './math.js'

// 单例 markdown-it 配置。breaks:true 保留(LLM 输出依赖单换行变 <br>),
// html:false 防 XSS,typographer:false 避免智能引号搞乱 LaTeX。
const md = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
  typographer: false,
})

// ── 默认 agent 映射表 ──────────────────────────────────────────────────
//
// name → id 的单向映射。同一 agent 可能有多个 name 变体(大小写、希腊字母),
// 全部列在这里。键集合同时被用作"合法 @ 提及白名单"。
//
// 调用方可以传入自定义 agentMap 覆盖默认值;缺省走下表。
//
export const DEFAULT_AGENT_MAP = {
  // crystal
  Crystal: 'crystal',
  crystal: 'crystal',
  // nahco3 (N + Lambda + HCO3)
  'NΛHCO3': 'nahco3',
  'N\u039B\u0048\u0043\u004F\u0033': 'nahco3',
  nahco3:  'nahco3',
  // soda
  Soda:    'soda',
  soda:    'soda',
}

// ── id → 规范显示名 ────────────────────────────────────────────────────
//
// 渲染 chip 时,@xxx 里的 xxx 会被替换为这里的规范名(避免同一个 agent 出
// 现多种大小写/希腊字母写法)。键集合应与 DEFAULT_AGENT_MAP 的 value 集合
// 一一对应。
//
export const DEFAULT_DISPLAY_NAME = {
  crystal: 'Crystal',
  nahco3:  'NΛHCO3',
  soda:    'Soda',
}

// ── 工具:HTML 实体转义 ─────────────────────────────────────────────────
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

// ── 主函数 ─────────────────────────────────────────────────────────────
/**
 * 渲染一段消息文本为 HTML,并提取其中提及的 agent id。
 *
 * @param {object}   opts
 * @param {string}   opts.text              原始纯文本消息
 * @param {object=}  opts.agentMap          name → id 映射;缺省走 DEFAULT_AGENT_MAP
 * @param {object=}  opts.displayName       id   → 显示名;缺省走 DEFAULT_DISPLAY_NAME
 * @returns {{ html: string, mentions: string[] }}
 */
export function renderMessage(opts) {
  const text       = opts?.text ?? ''
  const agentMap   = opts?.agentMap   || DEFAULT_AGENT_MAP
  const displayName = opts?.displayName || DEFAULT_DISPLAY_NAME

  if (!text) return { html: '', mentions: [] }

  // 1) 抽出数学公式,markdown-it 看不到 $ 字符,breaks 不会再切到公式
  let html
  let inlineMath, blockMath
  try {
    const extracted = extractMath(text)
    inlineMath = extracted.inlineMath
    blockMath  = extracted.blockMath
    // 2) markdown 渲染(此时公式已被占位符 @@MATH_INLINE_N@@ 替代)
    html = md.render(extracted.text)
    // 3) 把占位符换回 KaTeX 输出
    html = injectMath(html, inlineMath, blockMath)
  } catch (e) {
    console.warn('[renderMessage] 渲染失败,降级原文:', e)
    return { html: escapeHtml(text), mentions: [] }
  }

  // 4) @ 提及扫描与替换
  //    按 agentMap 的键动态生成一个正则。所有 name 用 | 连接,并要求
  //    后跟非 word 字符(/b/)以避免匹配 @Crystalize 这类词。
  const names = Object.keys(agentMap)
  if (names.length === 0) return { html, mentions: [] }

  // 按 name 长度倒序,长名优先匹配(防止 "NΛHCO3" 被 "N" 或 "Na" 之类
  // 截走)。这里通过把名字按长度降序后用 | 连接实现。
  const sortedNames = [...names].sort((a, b) => b.length - a.length)
  const escaped = sortedNames.map(escapeRegex).join('|')
  const mentionRe = new RegExp(`@(${escaped})\\b`, 'g')

  const mentions = []
  const seen = new Set()
  let displayFor = (id) => displayName[id] || id

  // 替换发生在渲染后的 HTML 字符串上。要避免破坏 KaTeX / markdown 输出的
  // 标签 — 因此我们用 matchAll 找出所有标签边界,只对"文本段"做正则替换。
  const out = replaceInTextNodes(html, mentionRe, (m, rawName) => {
    const id = agentMap[rawName]
    if (!id) return m[0]
    if (!seen.has(id)) { seen.add(id); mentions.push(id) }
    const label = `@${displayFor(id)}`
    return `<span class="mention-chip" data-agent-id="${escapeHtml(id)}" contenteditable="false">${escapeHtml(label)}</span>`
  })

  return { html: out, mentions }
}

// ── 工具:仅在"标签之间的文本"里做正则替换 ──────────────────────────────
//
// 用 String.matchAll 找出所有 <...> 标签的起止位置,把整段 HTML 切成
// "文本 + 标签 + 文本 + 标签 + ..."的交错序列,只对文本段跑替换。
//
// 这个分词器在 markdown-it / KaTeX 输出上安全,因为两者都不会产生未配对
// 的 < 或裸 &。`breaks:true` 产生的 <br> 也是良构的 <...> 标签。
//
const TAG_RE = /<[^>]+>/g

function replaceInTextNodes(html, re, replacer) {
  let result = ''
  let cursor = 0
  for (const m of html.matchAll(TAG_RE)) {
    const tagStart = m.index
    const tagEnd = tagStart + m[0].length
    // 把 [cursor, tagStart) 这一段作为文本段替换
    const textSegment = html.slice(cursor, tagStart)
    result += textSegment.replace(re, replacer)
    // 标签段原样保留
    result += m[0]
    cursor = tagEnd
  }
  // 收尾:把最后一段文本段也跑一遍替换
  result += html.slice(cursor).replace(re, replacer)
  return result
}

// ── 工具:正则元字符转义 ───────────────────────────────────────────────
function escapeRegex(s) {
  return String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

// 默认导出,方便 Vue 组件 import。
export default {
  renderMessage,
  DEFAULT_AGENT_MAP,
  DEFAULT_DISPLAY_NAME,
}
