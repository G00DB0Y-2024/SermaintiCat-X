// KaTeX-related helpers ported from BlogMain/viewer.html.
// These functions operate on plain strings (no DOM access) and are used
// by `markdown/index.js` to extract and re-inject math fragments around
// a markdown-it pass.

import katex from 'katex'

/**
 * Normalise a few common LaTeX shorthands used by Typora authors so that
 * things like `_\max` or `_{ \min }` render as `_{\text{max}}` instead of
 * producing KaTeX errors.
 */
export function normalizeLatexContent(content) {
  return String(content)
    .replace(/_\s*\{\s*\\?((?:max|min|argmax|argmin|sup|inf))\s*\}/g, '_{\\text{$1}}')
    .replace(/_\s*\\((?:max|min|argmax|argmin|sup|inf)\b)/g, (_, symbol) => `_{\\text{${symbol.slice(1)}}}`)
    .replace(/_\s*((?:max|min|argmax|argmin|sup|inf)\b)/g, (_, symbol) => `_{\\text{${symbol}}}`)
}

/**
 * Render a TeX fragment to HTML via KaTeX.
 *
 * @param {string} content       The TeX source.
 * @param {boolean} displayMode  True for $$..$$ / \[..\] blocks, false for $..$ / \(..\).
 */
export function renderKatex(content, displayMode) {
  return katex.renderToString(normalizeLatexContent(content), {
    displayMode,
    throwOnError: false,
    strict: 'ignore',
    trust: false,
    output: 'htmlAndMathml',
  })
}

/**
 * Hide fenced and inline code blocks from downstream regex passes by
 * replacing them with opaque tokens that survive `@@CODE_N@@`-style
 * stashing.
 */
export function protectCode(source) {
  const stash = []
  const replaceWithToken = (match) => {
    const token = `@@CODE_${stash.length}@@`
    stash.push(match)
    return token
  }

  const text = String(source)
    .replace(/```[\s\S]*?```|~~~[\s\S]*?~~~/g, replaceWithToken)
    .replace(/`[^`\n]+`/g, replaceWithToken)

  return {
    text,
    restore(value) {
      return String(value).replace(/@@CODE_(\d+)@@/g, (_, index) => stash[Number(index)] ?? '')
    },
  }
}

/**
 * Trim stray whitespace inside paired emphasis/strong markers, e.g.
 * `** text **` becomes `**text**`. The original implementation
 * stashed every `@@TOKEN@@` placeholder behind `\u0001` indices so the
 * regexes only saw literal text.
 */
export function normalizeWritingFormat(text) {
  const trim = (s) => s.replace(/^\s+|\s+$/g, '')
  const stash = []
  const masked = String(text).replace(/@@[A-Z_0-9]+@@/g, (m) => {
    stash.push(m)
    return `\u0001${stash.length - 1}`
  })
  const restored = masked
    .replace(/(?<![\*])\*\*([^*\n]+?)\*\*(?![\*])/g, (_, inner) => `**${trim(inner)}**`)
    .replace(/(?<![\*])\*([^*\n]+?)\*(?![\*])/g, (_, inner) => `*${trim(inner)}*`)
  return restored.replace(/\u0001(\d+)/g, (_, i) => stash[Number(i)])
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/**
 * Extract all math fragments from *source* and replace them with opaque
 * tokens. The caller can then pass the tokenised text through markdown-it
 * and re-inject the rendered KaTeX HTML afterwards (see `injectMath`).
 *
 * Returns `{ text, inlineMath, blockMath }`.
 */
export function extractMath(source) {
  const protectedCode = protectCode(source)
  let text = normalizeWritingFormat(protectedCode.text)
  text = markHtmlImages(text)
  const inlineMath = []
  const blockMath = []

  const makeInlineToken = (html) => {
    const token = `@@MATH_INLINE_${inlineMath.length}@@`
    inlineMath.push(html)
    return token
  }

  const makeBlockToken = (html) => {
    const token = `@@MATH_BLOCK_${blockMath.length}@@`
    blockMath.push(html)
    return token
  }

  const replaceDisplayMath = (pattern) => {
    text = text.replace(pattern, (_, prefix, content) => {
      // Keep the Markdown container prefix so lists and block quotes remain open.
      const containerPrefix = new RegExp(`^${escapeRegExp(prefix)}`, 'gm')
      const latex = content.replace(containerPrefix, '').trim()
      const html = `<div class="math-display">${renderKatex(latex, true)}</div>`
      const token = makeBlockToken(html)
      return `${prefix}\n${prefix}${token}\n${prefix}`
    })
  }

  // A display formula may be nested in a list or block quote. Replacing it
  // with root-level blank lines closes that container and turns following
  // indented text into a Markdown code block.
  replaceDisplayMath(/^((?:[ \t]*>[ \t]?)+[ \t]*|[ \t]+)\$\$\s*\r?\n([\s\S]*?)\r?\n\1\$\$\s*$/gm)
  replaceDisplayMath(/^((?:[ \t]*>[ \t]?)+[ \t]*|[ \t]+)\\\[\s*\r?\n([\s\S]*?)\r?\n\1\\\]\s*$/gm)

  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, content) => {
    const html = `<div class="math-display">${renderKatex(content.trim(), true)}</div>`
    return `\n\n${makeBlockToken(html)}\n\n`
  })

  text = text.replace(/\\\[([\s\S]+?)\\\]/g, (_, content) => {
    const html = `<div class="math-display">${renderKatex(content.trim(), true)}</div>`
    return `\n\n${makeBlockToken(html)}\n\n`
  })

  text = text.replace(/\\\(([\s\S]+?)\\\)/g, (_, content) => {
    return makeInlineToken(renderKatex(content.trim(), false))
  })

  text = text.replace(/(?<!\\)\$([^\n$]+?)\$/g, (_, content) => {
    return makeInlineToken(renderKatex(content.trim(), false))
  })

  return {
    text: protectedCode.restore(text),
    inlineMath,
    blockMath,
  }
}

/**
 * Re-inject math HTML emitted by `extractMath` into the markdown-it
 * output. Block math tokens may end up wrapped in `<p>` tags; this
 * function unwraps them so KaTeX stays on its own line.
 */
export function injectMath(html, inlineMath, blockMath) {
  let output = html

  for (let index = 0; index < blockMath.length; index += 1) {
    const token = `@@MATH_BLOCK_${index}@@`
    const replacement = blockMath[index]
    const tokenEscaped = token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    output = output
      .replace(new RegExp(`<p>\\s*${tokenEscaped}\\s*<\\/p>`, 'g'), replacement)
      .replace(new RegExp(tokenEscaped, 'g'), replacement)
  }

  output = output.replace(
    /<p>\s*<div class="math-display">([\s\S]*?)<\/div>\s*<\/p>/g,
    '<div class="math-display">$1</div>',
  )

  for (let index = 0; index < inlineMath.length; index += 1) {
    const token = `@@MATH_INLINE_${index}@@`
    const replacement = inlineMath[index]
    const tokenEscaped = token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    output = output.replace(new RegExp(tokenEscaped, 'g'), replacement)
  }

  return output
}

/**
 * Tag any `<img>` tags inside the markdown source so the markdown-it
 * image parser doesn't try to re-process them.
 */
export function markHtmlImages(source) {
  return String(source).replace(/<img\b([^>]*?)>/gi, (match, attributes) => {
    if (/\bdata-html-image\b/i.test(attributes)) return match
    return `<img data-html-image${attributes}>`
  })
}
