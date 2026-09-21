<!--
  MsgUnitComponent (PDFAI 版)
  从 CrystalBlog open-scholar/MsgUnitComponent.vue 1:1 拷过来,
  改动:
    · 加 thinking prop(思考中气泡)
    · 加 .msg-avatar-wrap--thinking 旋转光晕 + .msg-thinking 三点动画
    · user 角色的 isSelection 兼容保留
    · 气泡背景改为淡色(rgba)
    · 气泡和边缘加了 margin
    · ANNO 类型通过 anno-text prop 显示选区引用副标题
    · thinking 时自动滚动置顶(父组件 home.vue 负责调用 scroll 方法)
-->
<template>
  <!-- user 消息:整行右对齐,气泡占右侧,时间在气泡外下方 -->
  <div ref="msgRow" :class="rootClass" @copy="$emit('annoClick') /* 占位,真正 copy 走 capture */">
    <template v-if="isUser">
      <div class="msg-user-stack">
        <div class="msg-bubble msg-bubble--user msg-bubble--slide-user" :class="{ 'msg-bubble--selection': isSelection }">
          <div class="msg-content msg-content--user" v-html="displayedContent"></div>
          <!-- 气泡内嵌额外内容(如图片/注释),左右各 15px 内边距 -->
          <div v-if="$slots.extra" class="msg-bubble-extra">
            <slot name="extra" />
          </div>
        </div>
        <div class="msg-meta msg-meta--user">
          <span class="msg-time">{{ timestamp }}</span>
        </div>
      </div>
    </template>

    <!-- assistant 消息:头像+名字行 / 副标题(ANNO选区引用) / 气泡 / 时间元数据行 -->
    <template v-else>
      <!-- 头像列 -->
      <div class="msg-avatar-col">
        <div class="msg-avatar-wrap msg-avatar-wrap--fadein">
          <!-- 常驻光晕层(thinking 时通过 opacity 渐显,不在 thinking 时 opacity 0 渐隐) -->
          <div class="msg-avatar-halo" :class="{ 'msg-avatar-halo--active': thinking }"></div>
          <img v-if="agentAvatar" class="msg-avatar" :src="agentAvatar" :alt="agentName || 'agent'" />
          <div v-else class="msg-avatar msg-avatar--placeholder">{{ initial }}</div>
        </div>
      </div>

      <!-- 内容列:名字 / 副标题(ANNO选区) / 气泡 / 元数据 -->
      <div class="msg-content-col">
        <div class="msg-name-row msg-name-row--fadein">
          <span class="msg-agent-name">{{ agentName || 'Assistant' }}</span>
        </div>

        <!-- ANNO 选区引用副标题 -->
        <div v-if="annoText" class="msg-anno-quote" @click="$emit('annoClick')">
          <span class="msg-anno-quote-mark">"</span>{{ annoText }}
        </div>

        <!-- 气泡 -->
        <div class="msg-bubble msg-bubble--assistant msg-bubble--slide-assistant">
          <span v-if="thinking" class="msg-thinking">
            <span class="msg-thinking-dot"></span>
            <span class="msg-thinking-dot"></span>
            <span class="msg-thinking-dot"></span>
            思考中
          </span>
          <div v-else class="msg-content msg-content--assistant" v-html="renderedContent"></div>
          <!-- 气泡内嵌额外内容(如图片/注释),左右各 15px 内边距 -->
          <div v-if="$slots.extra" class="msg-bubble-extra">
            <slot name="extra" />
          </div>
        </div>

        <!-- 元数据行 -->
        <div class="msg-meta msg-meta--assistant">
          <span class="msg-time">{{ timestamp }}</span>
          <template v-if="tokenCount != null">
            <span class="msg-sep">·</span>
            <span class="msg-tokens">{{ tokenCount }} {{ tokenCount === 1 ? 'token' : 'tokens' }}</span>
          </template>
        </div>
      </div>
    </template>
  </div>
</template>

<script>
// 单一渲染管道: markdown + LaTeX + @mention chip + mentions 列表,
// 三件事在一次渲染中完成,互不干扰。
// 详见 ./renderMessage.js
import { renderMessage } from './renderMessage.js'

// ─── 公式横滚转发 (独立模块级函数,可被单元测试) ──────────────
//
// 鼠标悬浮在 KaTeX/MathML 渲染出的公式上时,把竖向滚轮
// deltaY → 转换为该公式所在容器的横向 scrollLeft。
// 其它位置的 wheel 事件继续正常冒泡(消息列表纵向滚动)。
//
// 策略:
//   1) 从 event.target 向上遍历,找到"公式标记元素"
//      (.katex-display / .math-display / .katex / <math>)
//   2) 再向上找到"能横向滚动的祖先" (overflow-x:auto/scroll 且确实有溢出)
//   3) 找到则 preventDefault 并横滚;否则放行

/** 沿 DOM 树向上找最近的"公式容器顶层"元素(.katex / .katex-display /
 * .math-display / math),跳过内部细节节点(.katex-html / .katex-mathml)
 *
 * 之所以把 .katex-html 从"命中"列表里剔除:它是 .katex 的子节点,
 * 用户点击 .katex-html 时如果命中它,后续 extractLatex 的向下 querySelector
 * 找不到 annotation(annotation 在 .katex-mathml 子树,与 .katex-html 是兄弟),
 * 会 fallback 到 textContent,得到的是 KaTeX 渲染出来的"纯文本可视字符"
 * (例如 x² → "x²"),而不是真正的 LaTeX 源码。 */
const FORMULA_SELECTOR = [
  '.katex-display',
  '.math-display',
  '.katex',
]

/** 判断节点是否为公式节点(KaTeX 行内/行间、math-display、math)
 * 用于 rangeToTextWithLatex 在递归时检测公式节点,整段用 LaTeX 替换。 */
function isFormulaNode(node) {
  if (!node || node.nodeType !== Node.ELEMENT_NODE) return false
  if (node.tagName === 'MATH') return true
  if (node.classList) {
    for (const cls of node.classList) {
      if (FORMULA_SELECTOR.some((s) => s === '.' + cls)) return true
    }
  }
  return false
}

function findFormulaAncestor(start, root) {
  let el = start
  while (el && el !== root) {
    if (el.tagName === 'MATH') return el
    if (el.classList) {
      for (const cls of el.classList) {
        if (FORMULA_SELECTOR.some((s) => s === '.' + cls)) return el
      }
    }
    el = el.parentElement
  }
  return null
}

/** 沿 DOM 树向上找最近的"确实有横向溢出"且 overflow-x 为 auto/scroll 的祖先 */
function findScrollableAncestor(start, root) {
  let el = start.parentElement
  while (el && el !== root) {
    if (el.scrollWidth > el.clientWidth + 1) {
      const style = window.getComputedStyle(el)
      if (style.overflowX === 'auto' || style.overflowX === 'scroll') {
        return el
      }
    }
    el = el.parentElement
  }
  return start // 退而求其次:公式本身
}

/** 构造 wheel 处理器,捕获给定 root 内公式上的竖向滚轮并转为横滚 */
function makeWheelHandler(root) {
  return (event) => {
    if (!root || !root.contains(event.target)) return

    const formulaEl = findFormulaAncestor(event.target, root)
    if (!formulaEl) return

    const scrollEl = findScrollableAncestor(formulaEl, root)
    if (scrollEl.scrollWidth <= scrollEl.clientWidth + 1) return

    event.preventDefault()
    event.stopPropagation()
    scrollEl.scrollLeft += event.deltaY
  }
}

/* ═══════════════════════════════════════════════════════════════
   公式交互:左键单击复制 LaTeX 源码(纯 LaTeX,不带 $ 包裹)
   ───────────────────────────────────────────────────────────────
   KaTeX 输出结构:
     <span class="katex">                ← 行内公式
       <span class="katex-mathml">       ← MathML 备用渲染
         <math><annotation encoding="application/x-tex">x^2</annotation></math>
       </span>
       <span class="katex-html">…</span>  ← HTML 主渲染
     </span>
   行间公式则被包在 <span class="katex-display"> 中。
   <annotation> 里就是纯 LaTeX 源码,直接复制即可。

   .math-display 是后端 pipeline 注入的另一种行间公式容器(见
   renderMessage.js),里面通常是 KaTeX 渲染产物,所以内部仍能找到
   annotation;少数情况下只有纯文本或没有 annotation,所以这里用
   多级 fallback 提取 LaTeX。
*/
function extractLatex(formulaEl) {
  // 1) 直接的 <annotation>
  const direct = formulaEl.querySelector('annotation[encoding="application/x-tex"], annotation')
  if (direct?.textContent?.trim()) return direct.textContent.trim()

  // 2) data-latex / data-tex 属性(部分自定义渲染器会标记)
  const dataAttr = formulaEl.getAttribute('data-latex')
    || formulaEl.getAttribute('data-tex')
    || formulaEl.querySelector('[data-latex]')?.getAttribute('data-latex')
    || formulaEl.querySelector('[data-tex]')?.getAttribute('data-tex')
  if (dataAttr?.trim()) return dataAttr.trim()

  // 3) 嵌套的 KaTeX 子树里的 annotation(常见于 .math-display)
  const innerKatex = formulaEl.querySelector('.katex')
  if (innerKatex) {
    const innerAnn = innerKatex.querySelector('annotation')
    if (innerAnn?.textContent?.trim()) return innerAnn.textContent.trim()
  }

  // 4) 兜底:取 textContent(纯文本/未渲染公式),可能含 $$/$ 包裹,尝试剥掉
  const raw = formulaEl.textContent?.trim()
  if (!raw) return ''
  return raw
    .replace(/^\$\$?/, '')
    .replace(/\$\$?$/, '')
    .trim()
}

async function copyFormulaLatex(formulaEl) {
  const latex = extractLatex(formulaEl)
  if (!latex) return { ok: false }

  try {
    await navigator.clipboard.writeText(latex)
    return { ok: true, latex }
  } catch (err) {
    // 兜底:execCommand('copy')
    try {
      const ta = document.createElement('textarea')
      ta.value = latex
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      const ok = document.execCommand('copy')
      document.body.removeChild(ta)
      return ok ? { ok: true, latex } : { ok: false, latex }
    } catch {
      return { ok: false, latex }
    }
  }
}


/* ═══════════════════════════════════════════════════════════════
   划词复制:含公式时,把 KaTeX 节点替换为对应 LaTeX 字符串
   ───────────────────────────────────────────────────────────────
   策略:
     1) 监听气泡 root 的 copy 事件(capture)
     2) 取当前 selection 的所有 Range
     3) 对每个 Range,把其中所有 .katex / .katex-display / .math-display
        / math 节点抽出,用 <annotation> 文本替换:
          · 行内(.katex / math[display=inline])→ " $latex$ "
          · 行间(.katex-display / .math-display / math[display=block])
            → "\n$$\nlatex\n$$\n"
     4) 把 Range 的纯文本 + 替换块拼成新字符串
     5) clipboardData.setData 覆盖默认行为 + preventDefault
   注意:Selection API 默认会把选区内的 innerText 拷出,所以即使不主动
   拦截,普通文字也能正常复制。我们只在"含公式"时改写,确保拿到的是
   markdown 友好的 LaTeX 字符串。
   */

/** 判断一个公式节点是行间(display)还是行内 */
function isBlockFormulaNode(node) {
  if (!node) return false
  if (node.classList?.contains('katex-display')) return true
  if (node.classList?.contains('math-display')) return true
  if (node.tagName === 'MATH' && node.getAttribute('display') === 'block') return true
  return false
}

/** 把一个公式节点序列化成 LaTeX 字符串(带 $ / $$ 包裹)
 *
 * 注意:annotation 位于 .katex-mathml > math > annotation,而
 * .katex-mathml 与 .katex-html 都是 .katex 的子节点;对 node
 * (.katex / .katex-display / .math-display / math) 做向下
 * querySelector('annotation') 可命中,因为 annotation 在其子树里。
 */
function formulaNodeToLatex(node) {
  const ann = node.querySelector('annotation[encoding="application/x-tex"], annotation')
  if (ann?.textContent?.trim()) {
    const tex = ann.textContent.trim()
    return isBlockFormulaNode(node) ? `\n$$\n${tex}\n$$\n` : `$${tex}$`
  }
  // 兜底:纯文本(理论上走不到)
  return node.textContent?.trim() || ''
}

/**
 * 把一个 Range 的内容序列化为可粘贴的字符串。
 * 公式节点用 LaTeX 字符串替换(带 $ / $$ 包裹),其它文本节点用 textContent。
 *
 * 关键:walker 必须从"公式顶层祖先"开始,否则当 selection 起点在
 * .katex-html / .katex-mathml 内部某个 textNode 时,递归会绕过
 * 顶层的 .katex,直接取到纯文本字符(plain text)。
 *
 * 同时递归时跳过非顶层的公式节点(.katex-display 内部的 .katex 会被
 * 父级 .katex-display 处理,递归进去会重复 / 拿到 plain text)。
 */
function rangeToTextWithLatex(range, root) {
  if (range.collapsed) return ''

  // 起点容器:元素节点走自身,文本节点走 parentElement
  let walker = range.commonAncestorContainer
  if (walker && walker.nodeType === Node.TEXT_NODE) walker = walker.parentNode
  if (!walker) return range.toString()

  // ↑ 关键修复:如果 walker 落在某个公式节点 *内部*,把 walker 提到
  //   该公式的顶层 —— 这样递归时能正确命中"公式节点 → 用 LaTeX 替换"。
  //   同时记录这个公式节点,递归时其子树整体跳过(避免重复处理)。
  const rootFormula = findFormulaAncestor(walker, root) || null

  let out = ''
  let first = true
  let lastWasFormula = false

  function isInRange(node) {
    return range.intersectsNode(node)
  }

  function visit(node, insideFormulaSubtree) {
    if (!isInRange(node)) return
    if (node.nodeType === Node.TEXT_NODE) {
      // 公式内部的文本节点不输出(由父级公式节点统一替换为 LaTeX)
      if (insideFormulaSubtree) return
      const text = node.textContent
      if (!text) return
      let startOffset = 0
      let endOffset = text.length
      if (node === range.startContainer && range.startOffset > 0) {
        startOffset = Math.min(range.startOffset, text.length)
      }
      if (node === range.endContainer && range.endOffset < text.length) {
        endOffset = Math.max(range.endOffset, 0)
      }
      const slice = text.slice(startOffset, endOffset)
      if (!slice) return
      if (!first && !lastWasFormula && !/^\s/.test(slice) && !/\s$/.test(out)) out += ' '
      out += slice
      first = false
      lastWasFormula = false
      return
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return

    // 公式节点 → 用 LaTeX 替换(不递归子节点)
    if (isFormulaNode(node)) {
      const tex = formulaNodeToLatex(node)
      if (!first && !lastWasFormula) out += ' '
      out += tex
      first = false
      lastWasFormula = true
      return
    }

    // 块级元素 → 在前后补换行
    const tag = node.tagName.toLowerCase()
    const display = window.getComputedStyle(node).display
    const isBlock = display === 'block' || display === 'list-item' || display === 'flex' ||
      /^p|div|li|h\d|pre/.test(tag)
    if (isBlock && !first) out += '\n'

    let child = node.firstChild
    while (child) {
      const next = child.nextSibling
      visit(child, insideFormulaSubtree)
      child = next
    }

    if (isBlock) out += '\n'
  }

  // 若 walker 本身就是某个公式节点 → 直接以它为根(并标记其子树为公式子树)
  if (rootFormula && rootFormula.contains(walker)) {
    visit(rootFormula, true)
  } else {
    visit(walker, false)
  }
  return out
}

/** 构造 copy 处理器:含公式时改写剪贴板内容 */
function makeCopyHandler(root) {
  return (event) => {
    if (!root || !root.contains(event.target)) return

    const sel = window.getSelection()
    if (!sel || sel.rangeCount === 0 || sel.isCollapsed) return

    // 检测是否含有公式节点(任何一个 Range 内)
    let hasFormula = false
    const formulas = root.querySelectorAll('.katex, .katex-display, .math-display, math')
    for (let i = 0; i < sel.rangeCount; i++) {
      const r = sel.getRangeAt(i)
      if (r.collapsed) continue
      for (const f of formulas) {
        if (r.intersectsNode(f)) { hasFormula = true; break }
      }
      if (hasFormula) break
    }

    if (!hasFormula) return  // 普通文字复制 → 走浏览器默认行为

    // 含公式 → 改写剪贴板
    let text = ''
    for (let i = 0; i < sel.rangeCount; i++) {
      const r = sel.getRangeAt(i)
      if (r.collapsed) continue
      text += rangeToTextWithLatex(r, root)
    }

    event.preventDefault()
    try {
      event.clipboardData.setData('text/plain', text)
      // 同步触发一次视觉反馈(闪一下根容器)
      const prev = root.style.outline
      root.style.outline = '2px solid #22c55e'
      root.style.outlineOffset = '0px'
      setTimeout(() => {
        root.style.outline = prev
        root.style.outlineOffset = ''
      }, 180)
    } catch {
      // clipboardData 不可写时静默放行
    }
  }
}

export default {
  name: 'MsgUnitComponent',
  props: {
    role: {
      type: String,
      default: 'assistant',
      validator: (v) => ['user', 'assistant'].includes(v),
    },
    content: { type: String, default: '' },
    // 历史消息里可能携带 mentions(老消息没有时由渲染函数回扫补齐)
    mentions: { type: Array, default: () => [] },
    timestamp: { type: String, default: '' },
    agentName: { type: String, default: '' },
    agentAvatar: { type: String, default: '' },
    tokenCount: { type: Number, default: null },
    // 来自划词操作的用户气泡(而非输入框)
    isSelection: { type: Boolean, default: false },
    // 思考中状态(显示三点动画 + 旋转光晕)
    thinking: { type: Boolean, default: false },
    // ANNO 选区原文(用于显示在气泡上方的副标题引用)
    annoText: { type: String, default: '' },
    // 原始 headtype(用于标记高亮触发区)
    headtype: { type: String, default: '' },
  },
  emits: ['annoClick'],
  computed: {
    isUser() {
      return this.role === 'user'
    },
    /** 根节点的 class —— 多根分支用 template 包到一个 div 上后,
     *  这里根据 isUser 切换 row 样式 */
    rootClass() {
      return this.isUser ? 'msg-row msg-row--user' : 'msg-row msg-row--assistant'
    },
    /**
     * 渲染后的 HTML。user / assistant 走同一管道,LaTeX 与 chip 互不干扰。
     * - PDFAI 没有 agent 体系,所以 agentMap 强制空对象 → 跳过 @mention 扫描
     */
    renderedContent() {
      const { html } = renderMessage({ text: this.content || '', agentMap: {} })
      return html
    },
    /**
     * 实际显示给 v-html 的内容。thinking 状态下 user 不渲染,assistant 渲染空
     * (assistant 模板已通过 v-if="thinking" 改为显示三点气泡,不走到这里)
     */
    displayedContent() {
      if (this.thinking) return ''
      return this.renderedContent
    },
    initial() {
      const name = (this.agentName || 'A').trim()
      return name.charAt(0).toUpperCase()
    },
  },
  mounted() {
    // 多根分支 → 已统一到一个根 div (ref="msgRow")。
    // 之前用 this.$el 是错的:Vue 多根时 this.$el 指向第一个根,
    // 第二个分支渲染时事件根本不会冒到这个 div 上,导致点击复制失效。
    const root = this.$refs.msgRow
    if (!root) return

    this._onWheel = makeWheelHandler(root)
    root.addEventListener('wheel', this._onWheel, { passive: false })

    // 公式点击复制:用 capture 拦截左键单击,避免误中选区 / 其它 click 处理器
    this._onClick = (event) => {
      if (event.button !== 0) return
      // 跳过 a / button / 块级 code(行内 code 不在公式容器内,自然不会命中)
      if (event.target.closest('a, button, pre')) return
      // 划词状态下的 user 气泡里也可能含公式,但 user 气泡几乎不会渲染 LaTeX,
      // 这里统一放开(让 user 也能复制)
      const formulaEl = findFormulaAncestor(event.target, root)
      if (!formulaEl) return
      event.preventDefault()
      event.stopPropagation()
      const r = copyFormulaLatex(formulaEl)

    }
    root.addEventListener('click', this._onClick, true)

    // 划词复制含公式 → 用 LaTeX 字符串覆盖剪贴板
    this._onCopy = makeCopyHandler(root)
    root.addEventListener('copy', this._onCopy, true)
  },
  beforeUnmount() {
    const root = this.$refs.msgRow
    if (this._onWheel && root) {
      root.removeEventListener('wheel', this._onWheel)
    }
    if (this._onClick && root) {
      root.removeEventListener('click', this._onClick, true)
    }
    if (this._onCopy && root) {
      root.removeEventListener('copy', this._onCopy, true)
    }
    this._onWheel = null
    this._onClick = null
    this._onCopy = null
  },
}
</script>

<style scoped>
/* ═══════════════════════════════════════════════════════════════
   Mention chip (display only — same look as the editor's chip)
   ═══════════════════════════════════════════════════════════════ */
.msg-content :deep(.mention-chip) {
  display: inline-block;
  padding: 1px 6px;
  margin: 0 1px;
  border-radius: 4px;
  background: #dbeafe;
  color: #1d4ed8;
  font-size: 13px;
  font-weight: 500;
  line-height: 1.4;
  white-space: nowrap;
  user-select: none;
  -webkit-user-select: none;
  cursor: default;
}

/* ═══════════════════════════════════════════════════════════════
   Layout
   ═══════════════════════════════════════════════════════════════ */
.msg-row {
  display: flex;
  width: 100%;
  /* 上下间距改为 6px,左右加 8px margin 与边缘留白 */
  margin: 6px 8px;
  transition: opacity 0.2s ease, transform 0.2s ease;
}
.msg-row--user {
  justify-content: flex-end;
}
.msg-row--assistant {
  justify-content: flex-start;
  gap: 10px;
  align-items: flex-start;
}

/* ═══════════════════════════════════════════════════════════════
   User
   ═══════════════════════════════════════════════════════════════ */
.msg-user-stack {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  max-width: 100%;
  margin-left: 45px;
  margin-right: 20px;
}
/* 气泡改为淡淡的浅蓝背景 */
.msg-bubble--user {
  background: rgba(219, 234, 254, 0.55);  /* 淡淡的浅蓝 */
  color: #0f172a;
  border: 1px solid rgba(191, 219, 254, 0.6);
  border-radius: 12px;
  padding: 8px 10px;
  word-wrap: break-word;
  overflow-wrap: anywhere;
  max-width: 100%;
  box-sizing: border-box;
  /* 气泡和边缘留 6px margin */
  margin: 4px 0;
  transition: background-color 0.15s ease, border-color 0.15s ease;
}
/* 划词产生的用户气泡:淡蓝色边框 + 虚线,区分于输入框文字 */
.msg-bubble--selection {
  background: #eff6ff;
  border-color: #93c5fd;
  border-style: dashed;
}
.msg-meta--user {
  text-align: right;
  font-size: 11px;
  color: #9ca3af;
  margin-top: 4px;
  padding-right: 4px;
  line-height: 1.4;
}

/* ═══════════════════════════════════════════════════════════════
   Assistant
   ═══════════════════════════════════════════════════════════════ */
.msg-avatar-col {
  flex-shrink: 0;
  width: 36px;
  display: flex;
  justify-content: center;
}
.msg-avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  object-fit: cover;
  background: #f3f4f6;
  border: 1px solid #e5e7eb;
  display: block;
}
/* 头像外层包装:旋转光晕画布 */
.msg-avatar-wrap {
  position: relative;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: transparent;
}
.msg-avatar-wrap .msg-avatar {
  width: 36px;
  height: 36px;
}
.msg-avatar--placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 600;
  color: #6b7280;
  background: #e5e7eb;
}

.msg-content-col {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  flex: 1;
  min-width: 0;
  margin-right: 20px;
}

.msg-name-row {
  height: 14px;
  display: flex;
  align-items: center;
  line-height: 1;
  margin-top: 10px;
  margin-bottom: 4px;
}
.msg-agent-name {
  font-size: 13px;
  font-weight: 600;
  color: #374151;
  white-space: nowrap;
}

/* ANNO 选区引用副标题:淡淡的灰底 + 引号 */
.msg-anno-quote {
  display: flex;
  align-items: flex-start;
  gap: 2px;
  font-size: 11.5px;
  color: rgba(0, 0, 0, 0.5);
  background: rgba(0, 0, 0, 0.04);
  border-left: 2px solid rgba(0, 0, 0, 0.12);
  border-radius: 2px;
  padding: 3px 8px;
  margin-bottom: 4px;
  max-width: 100%;
  word-break: break-all;
  overflow-wrap: anywhere;
  cursor: pointer;
  transition: background-color 0.15s;
  font-family: 'Times New Roman', serif;
  /* 最多显示两行 */
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.msg-anno-quote:hover {
  background: rgba(0, 0, 0, 0.07);
}
.msg-anno-quote-mark {
  font-size: 16px;
  line-height: 1;
  color: rgba(0, 0, 0, 0.35);
  flex-shrink: 0;
}

/* 气泡改为淡淡的浅灰背景 */
.msg-bubble--assistant {
  background: rgba(243, 244, 246, 0.6);  /* 淡淡的浅灰 */
  color: #111827;
  border: 1px solid rgba(229, 231, 235, 0.5);
  border-radius: 12px;
  padding: 8px 10px;
  padding-left: 13px;
  word-wrap: break-word;
  overflow-wrap: anywhere;
  max-width: 100%;
  box-sizing: border-box;
}

/* 气泡内嵌额外内容(如图片/注释):左右各 15px 内边距,
   与 msg-content 文字保持视觉间距 */
.msg-bubble-extra {
  padding: 8px 15px 2px 15px;
  margin-top: 4px;
  width: 100%;
  box-sizing: border-box;
}
.msg-bubble-extra:empty {
  display: none;
}

/* User 气泡: 从右向左滑入 */
.msg-bubble--slide-user {
  animation: slide-from-right 0.15s ease-out forwards;
}

/* Assistant 头像/名字: 渐显 */
.msg-avatar-wrap--fadein {
  opacity: 0;
  animation: fade-in 0.15s ease-out forwards;
}

.msg-name-row--fadein {
  opacity: 0;
  animation: fade-in 0.15s ease-out forwards;
}

/* Assistant 气泡: 从左向右滑入 */
.msg-bubble--slide-assistant {
  transform: translateX(-20px);
  opacity: 0;
  animation: slide-from-left 0.15s ease-out forwards;
}

@keyframes slide-from-right {
  from {
    transform: translateX(20px);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

@keyframes slide-from-left {
  from {
    transform: translateX(-20px);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

@keyframes fade-in {
  from { opacity: 0; }
  to   { opacity: 1; }
}

.msg-meta--assistant {
  text-align: left;
  font-size: 11px;
  color: #9ca3af;
  margin-top: 4px;
  padding-left: 4px;
  line-height: 1.4;
}
.msg-sep {
  margin: 0 4px;
  color: #d1d5db;
}

/* ═══════════════════════════════════════════════════════════════
   Content (markdown)
   ═══════════════════════════════════════════════════════════════ */
.msg-content {
  font-size: 14px;
  line-height: 1.55;
  /* flex item 的默认 min-width:auto 会按内容撑出宽度,
     设为 0 让气泡按父容器宽度收缩 — 否则长公式/chip 会撑出页面 */
  min-width: 0;
  max-width: 100%;
  /* 给右侧一点缓冲,避免字符紧贴 padding */
  overflow-wrap: anywhere;
}
/* ═══════════════════════════════════════════════════════════════
   KaTeX — 长公式横向滚动 (仿照 PDFAI pdf-iframe/gptRenderUnit.vue)
   ───────────────────────────────────────────────────────────────
   KaTeX 输出的 <math display="block"> 是一个 inline-block(不是真正的
   块级),当内容宽度超过父气泡时会让 layout 横向溢出。给 <math> 设
   overflow-x:auto + max-width:100% 让它在自己容器内滚,不撑出页面。
   */
.msg-content :deep(.katex-display),
.msg-content :deep(.math-display) {
  /* 确保公式居中 */
  text-align: center;
  /* 允许横向滚动,但不默认 nowrap(否则小公式也撑满一行) */
  overflow-x: auto;
  overflow-y: hidden;
  box-sizing: border-box;
  /* 隐藏滚动条但仍可滚动 */
  scrollbar-width: none;
  -ms-overflow-style: none;
  /* KaTeX 0.16.22 内部是 table 结构,强制居中 */
  display: block;
  margin-block: 5px;
}
.msg-content :deep(.katex-display) > .katex,
.msg-content :deep(.katex-display) > .katex-html,
.msg-content :deep(.math-display) > .katex,
.msg-content :deep(.math-display) > .katex-html {
  /* 内部真正渲染元素横向居中 */
  display: block;
  text-align: center !important;
  
}
.msg-content :deep(.katex-display)::-webkit-scrollbar,
.msg-content :deep(.math-display)::-webkit-scrollbar {
  display: none;
}
.msg-content :deep(math[display="block"]) {
  /* MathML 标签(0.16.22 htmlAndMathml 输出) */
  display: block;
  text-align: center;
  max-width: 100%;
  overflow-x: auto;
  overflow-y: hidden;
  white-space: nowrap;
  scrollbar-width: none;
  -ms-overflow-style: none;
}
.msg-content :deep(math[display="block"])::-webkit-scrollbar {
  display: none;
}

/* ───────────────────────────────────────────────────────────────
   公式交互样式:hover 高亮 + cursor pointer(左键单击复制 LaTeX)
   ───────────────────────────────────────────────────────────────
   简化设计:只有 .katex:hover 触发高亮,行间公式外层(.katex-display /
   .math-display)只设 cursor,不画高亮。

   行间公式 DOM:
     <div class="math-display">              ← 后端包的最外层(某些场景)
       <span class="katex-display">          ← KaTeX 行间容器
         <span class="katex">                ← 真正的公式
           <span class="katex-html">…</span>
         </span>
       </span>
     </div>

行内公式只有 .katex 一层。无论行内还是行间,鼠标停在 .katex 上时只有
.katex:hover 一条规则画高亮,外层没有任何 bg/box-shadow 干扰 → 单层。
*/
.msg-content :deep(.katex-display),
.msg-content :deep(.math-display),
.msg-content :deep(.katex),
.msg-content :deep(math) {
  cursor: pointer;
  transition: background-color 0.15s ease, box-shadow 0.15s ease;
  border-radius: 4px;
}

.msg-content :deep(.katex-display),
.msg-content :deep(.math-display) {
  display: block;
}

/* 唯一高亮规则:所有公式核心 .katex:hover(覆盖行内 + 行间内部)。
   行间公式的外层 .katex-display / .math-display 没有 hover 样式,
   鼠标停在 .katex 上时只有这一条规则画高亮 → 单层高亮,无重复。
   仅保留淡蓝色背景作为视觉反馈,不再画外发光边框。 */
.msg-content :deep(.katex):hover {
  background-color: rgba(99, 145, 255, 0.12);
}

/* 行内公式:保留 inline 行为,但当单个公式超长时也要能 horizontal scroll.
   inline 元素 overflow 需要 display 切换 — 用 inline-block 让 max-width
   可以生效。 */
.msg-content :deep(math:not([display="block"])),
.msg-content :deep(.katex),
.msg-content :deep(.math) {
  display: inline-block;
  max-width: 100%;
  vertical-align: middle;
  overflow-x: auto;
  overflow-y: hidden;
  white-space: nowrap;
  scrollbar-width: none;
  -ms-overflow-style: none;
  font-size: 16px;
}
.msg-content :deep(math:not([display="block"]))::-webkit-scrollbar,
.msg-content :deep(.katex)::-webkit-scrollbar,
.msg-content :deep(.math)::-webkit-scrollbar {
  display: none;
}

.msg-content :deep(p) { margin: 0 0 6px 0; }
.msg-content :deep(p:last-child) { margin-bottom: 0; }
/* 引用块(blockquote):类似 .msg-anno-quote 的形态(左侧细边条 + 浅色背景 +
   圆角),灰色风格,文字直立(非斜体),颜色继承 .msg-content 与段落一致。
   重写浏览器默认的 blockquote(默认 margin: 1em 40px,过宽过大),
   收紧上下边距到与段落一致的 4px。 */
.msg-content :deep(blockquote) {
  display: block;
  margin: 4px 0;
  padding: 4px 8px;
  background: rgba(0, 0, 0, 0.04);
  border-left: 3px solid rgba(0, 0, 0, 0.12);
  border-radius: 2px;
  color: inherit;
  font-style: normal;
  line-height: 1.55;
  word-break: break-word;
  overflow-wrap: anywhere;
  transition: background-color 0.15s ease, border-left-color 0.15s ease;
}
/* 引用块内部 p 不再加额外下边距,本身的 :deep(p:last-child) 已收敛 */
.msg-content :deep(blockquote p) {
  margin: 0;
}
.msg-content :deep(blockquote p:last-child) {
  margin-bottom: 0;
}
/* 引用块内部允许嵌套的 blockquote 也走同一套样式,缩进一点 */
.msg-content :deep(blockquote blockquote) {
  margin: 4px 0 4px 8px;
}
.msg-content :deep(pre) {
  background: #1f2937;
  color: #f9fafb;
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 12px;
  overflow-x: auto;
  margin: 6px 0;
}
.msg-content :deep(code) {
  background: rgba(0, 0, 0, 0.06);
  padding: 1px 4px;
  border-radius: 4px;
  font-size: 12.8px;
}
.msg-content :deep(pre code) {
  background: transparent;
  padding: 0;
}
.msg-content :deep(a) {
  color: #2563eb;
  text-decoration: underline;
}

/* ═══════════════════════════════════════════════════════════════
   Thinking 状态(第三步新增)
   - 头像外层:旋转光晕
   - 气泡内:三dot 跳动 + "思考中"
   ═══════════════════════════════════════════════════════════════ */
/* 常驻光晕层:始终在 DOM,通过 opacity 切换渐显渐隐,
   避免 thinking 切换时元素挂载/卸载导致 transition 失效 */
.msg-avatar-halo {
  position: absolute;
  inset: -4px;
  border-radius: 50%;
  background: conic-gradient(from 0deg, #3b82f6, #06b6d4, #8b5cf6, #3b82f6);
  animation: msg-avatar-spin 1.4s linear infinite;
  z-index: -1;
  filter: blur(2px);
  opacity: 0;
  transition: opacity 0.4s ease;
  pointer-events: none;
}
.msg-avatar-halo--active {
  opacity: 0.75;
}
@keyframes msg-avatar-spin {
  to { transform: rotate(360deg); }
}

.msg-thinking {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: #6b7280;
  font-size: 14px;
}
.msg-thinking-dot {
  display: inline-block;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #6b7280;
  animation: msg-thinking-bounce 1.2s infinite ease-in-out;
}
.msg-thinking-dot:nth-child(2) { animation-delay: 0.15s; }
.msg-thinking-dot:nth-child(3) { animation-delay: 0.30s; }
@keyframes msg-thinking-bounce {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
  30%           { transform: translateY(-3px); opacity: 1; }
}
</style>
