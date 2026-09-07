<template>
  <div 
    class="container" 
    :style="{backgroundColor: aiResponseJson === null ? 'rgba(255, 255, 255, 0.1)' : 'rgba(255, 255, 255, 1)'}"
  >
    <!-- 加载状态 -->
    <div v-if="aiResponseJson === null" class="loading-container">
      <div class="loading-text">请求中</div>
      <div class="loading-dots">
        <div class="dot" :style="{ backgroundColor: '#4e7bbf' }"></div>
        <div class="dot" :style="{ backgroundColor: '#4e7bbf', animationDelay: '0.2s' }"></div>
        <div class="dot" :style="{ backgroundColor: '#4e7bbf', animationDelay: '0.4s' }"></div>
      </div>
    </div>
    
    <!-- 结果显示 -->
    <div v-else class="result-container">
      <div class="word-section">
        <h2 class="word-title">{{ aiResponseJson.word }}</h2>
      </div>
      
      <div class="explain-section">
        <h3>释义</h3>
        <div v-if="aiResponseJson.explain.n" class="explain-item">
          <strong>名词 (n):</strong> {{ aiResponseJson.explain.n }}
        </div>
        <div v-if="aiResponseJson.explain.v" class="explain-item">
          <strong>动词 (v):</strong> {{ aiResponseJson.explain.v }}
        </div>
        <div v-if="aiResponseJson.explain.adj" class="explain-item">
          <strong>形容词 (adj):</strong> {{ aiResponseJson.explain.adj }}
        </div>
        <div v-if="aiResponseJson.explain.adv" class="explain-item">
          <strong>副词 (adv):</strong> {{ aiResponseJson.explain.adv }}
        </div>
        <div v-if="aiResponseJson.explain.sci" class="explain-item">
          <strong>学术释义:</strong> {{ aiResponseJson.explain.sci }}
        </div>
      </div>
      
      <div v-if="aiResponseJson.utilize && aiResponseJson.utilize.length > 0" class="utilize-section">
        <h3>用法示例</h3>
        <ul class="utilize-list">
          <li v-for="(example, index) in aiResponseJson.utilize" :key="index" class="utilize-item">
            {{ example }}
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      vid: "",
      gptModel:"gpt-4o-mini",
      apiKey: "", 
      apiUrl: "",
      aiResponseJson:null,
    }
  },
  mounted() { 
    const urlParams = new URLSearchParams(window.location.search);
    const ts = urlParams.get('ts');
    console.log(`Translator Confirm VID: ${ts}`)
    this.vid = ts;
    
    chrome.runtime.onMessage.addListener((message, sender, senderResponse) => {
      if (message.vid === '' || message.vid === this.vid) {
        if (message.type === 'SET_TEXT_MINI') {
          console.log("接受单词: "+message.text)
          this.doTranslate(message.text)
          window.top.postMessage({
              type: 'SET_TRANSPARENCY',
              pointNone: false
          }, '*');
        }
        if(message.type === 'VOID_MOUSEUP') {
          this.showContainer = false;
          window.top.postMessage({
              type: 'SET_TRANSPARENCY',
              pointNone: true
          }, '*');
        }
        if(message.type === 'SET_MODEL'){
          this.apiKey = message.apiKey
          this.apiUrl = message.apiUrl
        }
      }
    });
  },
  methods: {
    doTranslate(input_text){
      this.aiResponseJson = null

      this.$axios.post(this.apiUrl, {
        model: this.gptModel,
        messages: [
          {
            role:"developer",
            content: `
你是一个词典, 请将用户给出的英文进行翻译, 给出json格式的回答, 键"word"显示被查询单词, 键"explain"是词典部分, 给出"v, n, adj, adv"之类的释义, 以及"sci"部分表示学术释义, 键值对形式, 输出中文, 不要嵌套; "utilize"是用法, 给出三个学术例句, 输出英文, 数组形式;
一个合法的示例为:
{
  "word": "Transformer",
  "explain": {
    "n": "变压器；变形金刚；转换器",
    "sci": "一种深度学习模型，主要用于自然语言处理任务，也可应用于其他领域，如图像识别和语音识别。"
  },
  "utilize": [
    "Transformers have revolutionized natural language processing by enabling models to capture long-range dependencies in text.",
    "The attention mechanism in transformers allows the model to focus on the most relevant parts of the input sequence.",
    "Transformers have been successfully utilized in various applications, including machine translation, text summarization, and question answering."
  ]
}
`,
          },
          {
            role: "user",
            content: input_text
          }
        ]
      }, 
      {
        headers: {
        "Authorization": `Bearer ${this.apiKey}`,
        'APP-Code': 'DMQU5622'
      },
      }).then(res=>{
        const aiResponse = res.data.choices?.[0]?.message?.content || 
                          res.data.choices?.[0]?.text || 
                          JSON.stringify(res.data);
        this.aiResponseJson = this.extractJsonFromResponse(aiResponse)
      }).catch(error => {
        console.error('翻译请求失败:', error);
        this.aiResponseJson = {
          word: input_text,
          explain: {
            error: "请求失败，请检查API配置"
          },
          utilize: []
        }
      })
    },
    extractJsonFromResponse(response) {
      try {
        const jsonData = JSON.parse(response);
        return jsonData;
      } catch (e) {
        const jsonMatch = response.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          try {
            return JSON.parse(jsonMatch[0]);
          } catch (parseError) {
            console.error('提取的JSON解析失败:', parseError);
            return {
              word: "解析错误",
              explain: {
                error: "无法解析AI响应"
              },
              utilize: []
            };
          }
        }
        console.error('未找到JSON内容');
        return {
          word: response.substring(0, 20) + "...",
          explain: {
            raw: "原始响应: " + response.substring(0, 100)
          },
          utilize: []
        };
      }
    }
  }
}
</script>

<style scoped>
.container {
  display: flex;
  flex-flow: column;
  width: 100%;
  max-height: 300px;
  min-height: 300px;
  background-color: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(5px);
  overflow: hidden;
  position: relative;
  transition: all 0.2s ease;
  z-index: 1000;
}

/* 加载样式 */
.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding: 20px;
}

.loading-text {
  font-size: 20px;
  font-weight: bold;
  margin-bottom: 20px;
  margin-top: 70px;
  color: #4e7bbf;
}

.loading-dots {
  display: flex;
  gap: 10px;
}

.dot {
  width: 13px;
  height: 13px;
  border-radius: 50%;
  animation: pulse 1.4s infinite ease-in-out;
}

@keyframes pulse {
  0%, 100% {
    transform: scale(0.8);
    opacity: 0.6;
  }
  50% {
    transform: scale(1.2);
    opacity: 1;
  }
}

/* 结果样式 */
.result-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow-y: auto;
  padding: 13px;
  margin-bottom: 10px;
}

.word-section {
  margin-bottom: 20px;
  border-bottom: 2px solid #4e7bbf;
  padding-bottom: 10px;
}

.word-title {
  margin: 0;
  font-size: 20px;
  color: #333;
  font-weight: bold;
}

.explain-section, .utilize-section {
  margin-bottom: 26px;
}

.explain-section h3, .utilize-section h3 {
  margin: 0 0 12px 0;
  font-size: 16px;
  color: #4e7bbf;
  font-weight: 600;
}

.explain-item {
  margin-bottom: 8px;
  line-height: 1.5;
  font-size: 12px;
}

.explain-item strong {
  color: #333;
  margin-right: 8px;
}

.utilize-list {
  margin: 0;
  padding-left: 16px;
}

.utilize-item {
  margin-bottom: 10px;
  line-height: 1.5;
  font-size: 13px;
  color: #555;
  text-align: justify;
}

/* 滚动条样式 */
.result-container::-webkit-scrollbar {
  width: 8px;
}

.result-container::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

.result-container::-webkit-scrollbar-thumb {
  background: #4e7bbf;
  border-radius: 4px;
}

.result-container::-webkit-scrollbar-thumb:hover {
  background: #3a5f9a;
}
</style>

<style>
html, body {
  height: 100% !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}
</style>