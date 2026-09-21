// renderMarkdown.js
// 1:1 抽取自 gptRenderUnit.vue 的渲染管线:
//   - 原 data().md  (第 35-46 行)         → 模块级单例
//   - 原 methods.parseMessage  (第 419-441 行)
//   - 原 computed.renderedContent (第 444-446 行)
//
// 调用方只需:
//   import { parseMessage, renderMarkdown } from './renderMarkdown.js'

import MarkdownIt from 'markdown-it'
import markdownItKatexGpt from 'markdown-it-katex-gpt'

// 与原 gptRenderUnit.vue 一字不差的 delimiters 配置
const md = new MarkdownIt().use(markdownItKatexGpt, {
  delimiters: [
    { left: '\\[', right: '\\]', display: true },
    { left: '\\(', right: '\\)', display: false },
    { left: '$$', right: '$$', display: true },
    { left: '$', right: '$', display: false }
  ]
})

export function parseMessage(message) {
  // 原 gptRenderUnit.vue 第 419-441 行,只多加了 String(...) 防 null/undefined
  const separatorIndex = String(message ?? '').indexOf(">>")

  if (separatorIndex === -1) {
    // 如果没有找到分隔符,返回原始消息和空字符串
    return [message, ""]
  }

  const firstPart = message.substring(0, separatorIndex)

  // 查找结束标记 << 的位置
  const endMarkerIndex = message.indexOf("<<", separatorIndex + 2)

  if (endMarkerIndex === -1) {
    // 如果没有找到结束标记,返回第一部分和剩余部分(不含开始标记)
    return [firstPart, message.substring(separatorIndex + 2)]
  }

  const secondPart = message.substring(separatorIndex + 2, endMarkerIndex)

  return [firstPart, secondPart]
}

export function renderMarkdown(content) {
  // 原 gptRenderUnit.vue 第 444-446 行的 computed.renderedContent
  return md.render(parseMessage(content)[0])
}
