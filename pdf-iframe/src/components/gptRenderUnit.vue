<template>
  <!--
    gptRenderUnit:转发器外壳
    - 顶部时间条已删除(颜色现在由 MsgUnitComponent 气泡背景承载,淡淡的 rgba)
    - ANNO 选区文本保留在 MsgUnitComponent 内部作为气泡副标题
    - 头像用 agent_reason.png(Crystal 专属)
  -->
  <div style="display: flex; flex-flow: column; transition: all 0.2s ease"
       :style="{opacity:editting?0.5:1}">

    <MsgUnitComponent
      :role="resolvedRole"
      :content="resolvedContent"
      :timestamp="datetime"
      :agent-name="resolvedAgentName"
      :agent-avatar="resolvedAgentAvatar"
      :is-selection="false"
      :thinking="!!_thinking"
      :anno-text="annoText"
      :headtype="headtype"
      :token-count="tokenCount ?? null"
      :gid="gid"
      @dblclick.native="handleMiddleClick"
      @annoClick="$emit('annoClick')"
      @onQuote="gid => $emit('onQuote', gid)"
    >
      <!-- 把图片搬进气泡内,左右各 15px 由 MsgUnitComponent 的
           .msg-bubble-extra 容器提供 -->
      <template v-if="img_name" #extra>
        <image-render :img_name="img_name" />
      </template>
    </MsgUnitComponent>
  </div>
</template>

<script>
// gptRenderUnit:转发器
// 渲染管线已在 scripts/renderMessage.js(单一管道:markdown + LaTeX + @mention)
// 本组件职责:
//   · 路由 role / content / annoText 到 MsgUnitComponent
//   · 处理中键引用(onQuote)、ANNO 高亮监听(chrome.runtime.onMessage)、双击复制
import MsgUnitComponent from './MsgUnitComponent.vue'
import imageRender from './imageRender.vue'
import AgentAvatar from '../assets/images/agent_reason.png'

export default {
  props: ['content', 'datetime', 'hldata', 'gid', 'headtype', 'vid', 'img_name', 'editting', '_thinking', 'tokenCount', 'entryFlag'],
  emits: ['onQuote', 'onClose', 'onHlClick', 'onANNOClick', 'annoClick'],
  components: { MsgUnitComponent, imageRender },
  data() { return {} },
  beforeMount() {
    window.addEventListener('resize', () => {})
  },
  mounted() {
    this.highlightActive = false
    this.changeAllowed = true

    // ANNO 点击高亮监听
    chrome.runtime.onMessage.addListener((message, sender) => {
      if (message.vid === '' || message.vid === this.vid) {
        // Anno type 识别: 'Anno' / 'Anno_*' / 'LIST_TYPE_ANNO_*'
        const isAnno = this.headtype && (
          this.headtype === 'Anno' ||
          this.headtype.includes('Anno_') ||
          this.headtype.includes('LIST_TYPE_ANNO')
        )
        if (message.type === 'ACTIVE_CLICK_ANNO' && isAnno) {
          // flag 来自 entry.flag(新格式)或 type 末尾(旧格式)
          const flag = this.entryFlag ?? this.headtype.split('_').at(-1)
          if (flag === message.flag && this.changeAllowed) {
            this.changeAllowed = false
            this.highlightActive = true
            this.applyHighlightStyles()
            setTimeout(() => {
              this.resetHighlightStyles()
              this.highlightActive = false
              setTimeout(() => { this.changeAllowed = true }, 150)
            }, 250)
          }
        }
      }
    })
  },
  methods: {
    applyHighlightStyles() {
      const el = this.$el
      if (!el) return
      el.style.borderColor = '#AC4848'
      el.style.scale = '1.005'
    },
    resetHighlightStyles() {
      const el = this.$el
      if (!el) return
      el.style.borderColor = 'transparent'
      el.style.scale = ''
    },

    // ─── 解析:从 >>...<< 提取 ANNO 选区文本 ──────────────────────
    parseAnnoSelection(message) {
      if (!message) return ''
      const sep = String(message).indexOf('>>')
      if (sep === -1) return ''
      const end = message.indexOf('<<', sep + 2)
      if (end === -1) return ''
      return message.substring(sep + 2, end)
    },

    // ─── 鼠标交互(中键引用、划词复制) ──────────────────
    async handleMiddleClick(event) {
      window.focus()
      const katexEl = event.target.closest('.katex')

      if (event.button === 0) { // 左键
        // 单击复制公式 LaTeX 已移交给 MsgUnitComponent 的 capture-click
        // 此处不再处理 katex / code 复制,避免重复
      } else if (event.button === 1) { // 中键
        event.preventDefault()
        if (['P', 'LI', 'PRE', 'CODE'].includes(event.target.tagName) || katexEl) {
          this.$emit('onQuote', this.gid)
        }
        event.stopPropagation()
      }
    },
  },
  computed: {
    // Req* → user 气泡; Res* / Anno* / 旧数据 → assistant 气泡
    resolvedRole() {
      if (this.headtype?.startsWith('Req')) return 'user'
      return 'assistant'
    },
    // 实际内容: Anno 类型显示注释内容(在 >> 之前),非 Anno 显示全文
    resolvedContent() {
      const isAnno = this.headtype === 'Anno'
        || this.headtype?.includes('Anno_')
        || this.headtype?.includes('LIST_TYPE_ANNO')
      if (isAnno) {
        const sep = String(this.content).indexOf('>>')
        return sep === -1 ? this.content : this.content.substring(0, sep)
      }
      return this.content || ''
    },
    // Anno 选区原文(用于 MsgUnitComponent 显示在气泡上方的副标题引用)
    annoText() {
      const isAnno = this.headtype === 'Anno'
        || this.headtype?.includes('Anno_')
        || this.headtype?.includes('LIST_TYPE_ANNO')
      if (!isAnno) return ''
      return this.parseAnnoSelection(this.content)
    },
    resolvedAgentName() {
      return this.resolvedRole === 'user' ? '' : 'Crystal'
    },
    resolvedAgentAvatar() {
      return this.resolvedRole === 'user' ? '' : AgentAvatar
    },
  },
}
</script>

<style scoped>
/* ANNO 副标题(选区原文引用)样式 —— 在 MsgUnitComponent 内通过 anno-text prop 渲染 */
</style>
