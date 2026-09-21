import { createApp } from 'vue'
import App from './App.vue'
import axios from 'axios'
import 'katex/dist/katex.min.css'  // KaTeX 渲染必需,Crystal 版有,PDFAI 漏了

import hljs from 'highlight.js'
import 'highlight.js/styles/atom-one-dark.css'

// 聊天气泡内 markdown + LaTeX 渲染样式(.msg-content 下所有元素)
// 必须在 katex.min.css 之后,样式覆盖才能生效。
import './styles/note_style.css'

const app = createApp(App)

app.config.globalProperties.$axios = axios;
axios.defaults.withCredentials = false;
axios.defaults.baseURL = 'http://localhost:8225'

app.directive('highlight', {
    mounted(el) {
        // console.log('el', el)
        const blocks = el.querySelectorAll('pre code')
        for (let i = 0; i < blocks.length; i++) {
            hljs.highlightElement(blocks[i])
        }
    }
})

app.mount('#app')