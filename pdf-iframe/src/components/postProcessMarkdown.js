// postProcessMarkdown.js
// 1:1 抽取自 gptRenderUnit.vue 的 5 个 DOM 后处理方法:
//   - boldFirstStrongInContainer (108-133)  → 参数化 container
//   - styleFigureText            (164-184)  → 参数化 container,内部调 module-level normalizeLabels
//   - processDivText             (186-302)  → 递归 helper (processTopLevelElements/processLists) 内化为 module-level
//   - updateMathSpan             (334-354)  → 参数 root,默认 document(保留全局扫,纯照抄零行为变更)
//   - checkMathOverflow          (357-369)  → 参数 root,默认 document(同上)
//
// 调用方:
//   import { boldFirstStrongInContainer, styleFigureText,
//            updateMathSpan, checkMathOverflow, processDivText } from './postProcessMarkdown.js'

// ────────────────────────────────────────────────────────────────────
// normalizeLabels:原 gptRenderUnit.vue 第 149-160 行
// 从 styleFigureText 内化为模块级函数,行为完全等价
// ────────────────────────────────────────────────────────────────────
function normalizeLabels(text) {
  // 替换 fig/FIG/Figure/FIGURE 等为 "图 x"
  text = text.replace(/(?:fig|FIG|Figure|FIGURE)\s+(\d+)/gi, "图 $1")

  // 替换 table/Table/TABLE 等为 "表 x"
  text = text.replace(/(?:table|Table|TABLE)\s+(\d+)/gi, "表 $1")

  // 替换 algorithm/Algorithm/ALGORITHM 等为 "算法 x"
  text = text.replace(/(?:algorithm|Algorithm|ALGORITHM)\s+(\d+)/gi, "算法 $1")

  return text
}

// ────────────────────────────────────────────────────────────────────
// boldFirstStrongInContainer:原 gptRenderUnit.vue 第 108-133 行
// ────────────────────────────────────────────────────────────────────
export function boldFirstStrongInContainer(container) {
  // 使用 ref 直接获取容器元素
  if (!container) return

  // 创建一个临时 div 来安全地操作 HTML 内容
  const tempDiv = document.createElement('div')
  tempDiv.innerHTML = container.innerHTML

  // 查找所有直接子段落
  const paragraphs = tempDiv.querySelectorAll('p')

  // 遍历每个段落
  paragraphs.forEach(p => {
    // 获取第一个子节点(包括文本节点)
    let firstNode = p.firstChild

    // 检查第一个节点是否存在且是 element 节点,并且是 strong 标签
    if (firstNode && firstNode.nodeType === Node.ELEMENT_NODE &&
        firstNode.tagName.toLowerCase() === 'strong') {
      // 应用加粗样式
      firstNode.textContent = `${firstNode.textContent.trim()}` // 去除多余空格
    }
  })

  // 更新容器内容
  container.innerHTML = tempDiv.innerHTML
}

// ────────────────────────────────────────────────────────────────────
// styleFigureText:原 gptRenderUnit.vue 第 164-184 行
// 注意:原代码用 this.normalizeLabels,这里改为调模块级 normalizeLabels
// ────────────────────────────────────────────────────────────────────
export function styleFigureText(container) {
  if (!container) return

  // 获取容器中的 HTML 内容
  let content = container.innerHTML

  content = normalizeLabels(content) // 预先替换

  // 使用正则表达式匹配 "图 x"、"算法 x" 和 "表 x"
  const regex = /图\s*(\d+)|算法\s*(\d+)|表\s*(\d+)|\[文献[^\]]*\]/g

  // 替换为指定的 HTML 代码
  content = content.replace(regex, (match) => {
    if (match.startsWith('[文献')) {
      return `<span class="styled-reference" style="color: white; background-color: rgb(72, 112, 172); padding-inline: 3px; margin-inline: 2px; border-radius: 2px;">${match}</span>`
    } else {
      return `<span class="styled-figure" style="color: white; background-color: rgb(72, 112, 172); padding-inline: 3px; margin-inline: 2px; border-radius: 2px;">${match.trim().replace(/(\S)\s*(\d+)/, '\$1 \$2')}</span>`
    }
  })

  // 更新容器的内容
  container.innerHTML = content
}

// ────────────────────────────────────────────────────────────────────
// processDivText:原 gptRenderUnit.vue 第 186-302 行
// 内部两个递归 helper 内化为模块级函数,行为等价
// ────────────────────────────────────────────────────────────────────

// 处理选区中的一级元素 —— 原内嵌函数,行为不变
function processTopLevelElements(container) {
  // 先处理独立的 <li> 元素
  Array.from(container.children).forEach(child => {
    if (child.tagName === 'LI') {
      const listParent = child.parentNode
      const isOrdered = listParent.tagName === 'OL'
      const index = isOrdered ?
        Array.from(listParent.children).indexOf(child) + 1 : null
      const prefix = isOrdered ? `${index}. ` : '- '

      const textNode = document.createTextNode(`[NEWLINE]${prefix}${child.textContent.trim()}`)
      container.replaceChild(textNode, child)
    }
  })
}

// 处理 ul 和 ol 列表(递归处理嵌套)—— 原内嵌函数,行为不变
function processLists(element, indentLevel = 0) {
  const lists = element.querySelectorAll('ul, ol')

  lists.forEach(list => {
    const isOrdered = list.tagName === 'OL'
    const items = list.querySelectorAll('li')
    const listContainer = document.createElement('p')

    items.forEach((item, index) => {
      // 添加缩进空格
      const indent = ' '.repeat(indentLevel * 2)

      // 处理列表标记
      const prefix = isOrdered ? `[NEWLINE]${indent}${index + 1}. ` : `[NEWLINE]${indent}- `

      // 处理子节点(可能包含嵌套列表)
      const itemContent = document.createElement('span')
      itemContent.innerHTML = item.innerHTML

      // 递归处理嵌套列表
      processLists(itemContent, indentLevel + 1)

      const itemPara = document.createTextNode(prefix + itemContent.textContent.trim())
      listContainer.appendChild(itemPara)
    })

    // 替换原列表
    list.parentNode.replaceChild(listContainer, list)
  })
}

export function processDivText(fragment) {
  // 创建临时容器
  const tempDiv = document.createElement('div')
  tempDiv.appendChild(fragment.cloneNode(true))

  // 处理所有 katex 元素
  const processedKatex = tempDiv.querySelectorAll('.katex')
  processedKatex.forEach(katex => {
    const annotation = katex.querySelector('annotation')
    if (annotation) {
      // 使用特殊标记代替换行符,稍后再恢复
      let katex_flag_l = katex.tagName === 'P' ? '[NEWLINE]$$[NEWLINE]' : ' $'
      let katex_flag_r = katex.tagName === 'P' ? '[NEWLINE]$$[NEWLINE]' : '$ '
      katex.textContent = katex_flag_l + annotation.textContent + katex_flag_r
    }
  })

  // 处理 strong, code, em
  let mdIdentify = {
    'strong': '**',
    'em': '*',
    'code': '`'
  }
  for (const [key, value] of Object.entries(mdIdentify)) {
    const processed = tempDiv.querySelectorAll(key)
    processed.forEach(el => {
      if (key === 'code' && el.parentNode.tagName === 'PRE') {
        return // 跳过代码块
      }
      let text = document.createTextNode(` ${value}${el.textContent}${value} `)
      el.parentNode.replaceChild(text, el)
    })
  }

  // 处理选区中的一级元素
  processTopLevelElements(tempDiv)

  // 处理 ul 和 ol 列表(递归处理嵌套)
  processLists(tempDiv)

  // 处理 <p> 标签:除了第一个 p 标签,其他 p 标签开头添加 [NEWLINE]
  const paragraphs = tempDiv.querySelectorAll('p')
  if (paragraphs.length > 1) {
    paragraphs.forEach((p, index) => {
      if (index > 0) { // 跳过第一个 p 标签
        const firstChild = p.firstChild
        if (firstChild && firstChild.nodeType === Node.TEXT_NODE) {
          firstChild.textContent = '[NEWLINE]' + firstChild.textContent
        } else {
          const textNode = document.createTextNode('[NEWLINE]')
          p.insertBefore(textNode, p.firstChild)
        }
      }
    })
  }

  // 获取处理后的文本内容
  let textToCopy = tempDiv.textContent
  // 清理文本
  textToCopy = textToCopy
    .replace(/\s+/g, ' ')      // 合并多个空格
    .replace(/\$ /g, '$ ')      // 清理公式后的空格
    .replace(/ \$/g, ' $')      // 清理公式前的空格
    .replace(/\[NEWLINE\]/g, '\n')  // 恢复换行符
    .trim()

  return textToCopy
}

// ────────────────────────────────────────────────────────────────────
// updateMathSpan:原 gptRenderUnit.vue 第 334-354 行
// 保留全局 document.querySelectorAll(纯照抄,零行为变更)
// 改为接受可选 root 参数,默认 document 等价于原行为
// ────────────────────────────────────────────────────────────────────
export function updateMathSpan(root = document) {
  // 获取所有 class 为 katex 的 span 元素
  const katexSpans = root.querySelectorAll('span.katex')

  // 遍历每个 span.katex 元素
  katexSpans.forEach(span => {
    // 检查是否包含 math 子元素且该子元素有 display="block" 属性
    const mathElement = span.querySelector('math[display="block"]')

    if (mathElement) {
      // 创建新的 p 元素
      const pElement = document.createElement('p')

      // 将 span 的所有属性和子元素转移到 p 元素
      pElement.className = span.className
      pElement.innerHTML = span.innerHTML

      // 用 p 元素替换 span 元素
      span.parentNode.replaceChild(pElement, span)
    }
  })
}

// ────────────────────────────────────────────────────────────────────
// checkMathOverflow:原 gptRenderUnit.vue 第 357-369 行
// 保留全局 document.querySelectorAll(纯照抄,零行为变更)
// 改为接受可选 root 参数,默认 document 等价于原行为
// ────────────────────────────────────────────────────────────────────
export function checkMathOverflow(root = document) {
  // 计算元素是否溢出,修改 justify-content
  root.querySelectorAll('.markdown-container math[display="block"]').forEach(mathElement => {
    // 检测是否溢出
    const isOverflowing = mathElement.scrollWidth > mathElement.clientWidth
    // 动态设置 justify-content
    mathElement.style.setProperty(
      '--justify-content',
      isOverflowing ? 'flex-start' : 'center'
    )
  })
}
