<template>
  <!-- @mouseenter="console.log('Mouse Entered')" @mouseleave="console.log('Mouse Left')" -->
  <div class="content" @mouseup="handelSelectedHighlight">
    <div class="sliderbox">

      <!-- 顶部栏 -->
      <div
        style="display: flex; width: 100%; background-color: #F9F9FA; min-height: 32px; border-bottom: 1px solid #B8B8B8;">
        <div style="display: flex; height: 100%; align-items: center; margin-left: 10px; gap: 10px;">
          <label class="btn-show-ori" @click="showOrigin=!showOrigin"
            :style="{color:showOrigin?'#4e7bbf':''}">ORIGIN</label>
          <label class="btn-adding" @click="chosen_add_text.length=0; chosen_text=''"
            :style="{width:chosen_add_text.length===0?0:'52px', opacity:chosen_add_text.length===0?0:1}">+{{
            chosen_add_text.length }} MORE</label>
          <div style="
            display: flex;
            position: absolute;
            font-size: 10px;
            right: 2px;
            column-gap: 10px;
            align-items: center;
            transform: scaleX(0.95);
            transform-origin: right;
          ">
            <label style="color: black;">Input:<span
                style="color: #4870AC;">{{classifyRateArr[0].toFixed(1)}}%</span></label>
            <label style="color: black;">Feedback:<span
                style="color: #82AD75;">{{classifyRateArr[1].toFixed(1)}}%</span></label>
            <label style="color: black;">Output:<span
                style="color: #AC4848;">{{classifyRateArr[2].toFixed(1)}}%</span></label>
          </div>
        </div>

      </div>

      <!-- 左侧功能区 -->
      <div style="
        display: flex;
        flex-flow: column;
        overflow: hidden;
        width: 100%;
        height: 100%;
        align-items: center;
      ">

        <!-- 原文 -->
        <div style="transition: all 0.2s ease; display: flex; flex-flow: column; width: 100%; height: 500px;"
          :style="{height: showOrigin? '500px':'0'}">
          <textarea class="textarea-ori"
            :style="{height: showOrigin? '100%':'0', opacity:showOrigin?'1':'0', pointerEvents:showOrigin?'all':'none'}">{{ shown_text }}</textarea>
        </div>

        <!-- GPT显示区域 -->
        <div style="display: flex; width: 100%; height: 100%; overflow: hidden; position: relative;">

<!-- 索引部分(按"问答组"合并显示,鼠标侧键也以组为单位跳转) -->
        <div style="display: flex; flex-flow: column; height: 100%; width: 7px; background-color: white;">
          <div v-for="(group, gi) in groupedChildrenHeightArr"
            :key="`g-${group.start}-${group.end}`"
            :class="current_index>=group.start && current_index<group.end && !can_current_index_change?'div-index-block-active':'div-index-block'"
            :style="{
            height:`${group.hrate}%`,
            backgroundColor:MapGroupColor(group),
          }" @click="adjustScroll(group.start); highlightScroll(group.start);">
          </div>

            <!-- 位置指针 -->
            <div v-if="ai_res.length!==0" style="
                left: 7px;
                background-color: transparent;
                position: absolute;
                display: flex;
                align-items: start;
                justify-content: start;
                height: 14px;
                margin-top: 0;
                transform: translateY(-50%);
                font-size: 10px;
                line-height: 14px;
                color: black;
                font-family: cursive;
                text-shadow: 2px 2px 2px rgba(0,0,0,0.3);
                transition: all 0.15s ease;
              " :style="{top: scrollTopRate}">◀</div>

            <!-- 高亮位置指针 -->
            <div v-for="(item, key, index) in hl_arr" :key="key" style="
                left: 5px;
                background-color: transparent;
                position: absolute;
                display: flex;
                align-items: start;
                justify-content: start;
                height: 0;
                margin-top: -7.5px;
                font-size: 10px;
                color: #B05151;
                font-family: cursive;
                scale: 0.5;
                text-shadow: 2px 2px 2px rgba(0,0,0,0.3);
              " :style="{top: item.position}">◀</div>

          </div>

          <!-- GPT显示内容 -->
          <div ref="render_container" class="gpt-render" @scroll="UpdatePointerPosition"
            @auxclick="handleMouseSideButton"
            @mouseup="handleSideButtonCapture"> <!-- Chrome 把 back/forward 行为绑在 mouseup 鼠标抬起时捕获 -->
            <div v-for="(item, index) in ai_res" ref="render" :key="index"
              style="display: block; background-color: transparent;">
              <gptRenderUnit style="width: 100%;" :content="item.text" :datetime="item.dt" :hldata="item.hl"
                :headtype="item.type" :gid="index" :vid="vid" :img_name="item.img"
                :editting="index === annotation_edit_index"
                :_thinking="index === placeholderIndex && placeholderIndex !== -1"
                :token-count="item.token_count ?? null"
                :key="`${refreshKey}-${index}`" @onQuote="handleQuote"
                @onClose="handleAiChange('DEL', index, null)" @onHlClick="handleHlClick(item.hl, item.type, index)"
                @onANNOClick="handleANNOClick(item.hl, item.type)"
                @annoClick="handleANNOClick(item.hl, item.type)" />

            </div>

            <div ref="last_blank"
              style="min-height: 0; color: transparent; background-color: transparent; flex-shrink: 0;"></div>
          </div>

          <!-- 精灵图片 -->
          <img src="../assets/images/Cyrstal.png" style="
            width: 75%;
            position: absolute;
            bottom: 0;
            left: 0;
            object-fit: contain;
            -webkit-user-select:none;
            -moz-user-select:none;
            -ms-user-select:none;
            user-select:none;
            pointer-events: none;
            transition: all 0.2s ease, opacity 0.3s ease, filter 0.3s ease;
          " :style="{
            opacity: `${imageShowFlag?0.25:0}`,
            filter: `blur(${lastBottomRate < 0.05? 0 : 4*lastBottomRate*lastBottomRate*100 }px)`
          }" />

        </div>


        <!-- 追问 — 移到 .content 级别定位，固定在底部 -->
        <div ref="askMask"
          style="position: fixed; bottom: 0; left: 0; right: 0; z-index: 10; background: transparent; padding: 0 12px 12px; box-sizing: border-box; pointer-events: none;">
          <div class="ask-area" style="pointer-events: all;">
            <label class="lb-quote-num"
              :style="{opacity:askQuote.length===0?'0':'1', pointerEvents:askQuote.length===0?'none':'all', borderColor:askQuote.length>1?'darkred':'#4870AC', color:askQuote.length>1?'darkred':'#4870AC'}"
              @click="askQuote.length=0">quote +{{ askQuote.length }}</label>
            <label class="lb-quote-num"
              :style="{opacity:askImage.img===null?'0':'1', pointerEvents:askImage.img===null?'none':'all', borderColor:'#6812b0', color:'#6812b0'}"
              @mouseover="can_blur=false" @mouseleave="can_blur=true"
              @click="handleAnnoationImgDel();resetImgInfo();">image {{ askImage.size }} kB</label>

            <div class="ask-input-wrap"
              :class="{ 'ask-input-wrap--focused': askFocus, 'ask-input-wrap--disabled': waiting }">
              <textarea
                ref="askTextrea"
                class="ask-input"
                v-model="askContent"
                :placeholder="askFocus ? '' : '输入问题,Enter 发送,Shift+Enter 换行'"
                @focus="askFocus=true; showOrigin=false"
                @blur="handleAskAreaBlur()"
                @keydown.enter.exact.prevent="onAskEnter"
                @keydown.shift.enter.stop
                @paste="handlePaste"
              ></textarea>
            </div>
          </div>
        </div>

        <!-- 等待中气泡已改为 MsgUnitComponent + thinking prop,不再需要全屏遮罩 -->
      </div>

      <textarea ref="blurTextarea" style="position: absolute; opacity: 0; height: 0; width: 0;"></textarea>
    </div>


  </div>



</template>

<script >
import gptRenderUnit from './gptRenderUnit.vue'
// waiting.vue 已在第三步删除(改用 MsgUnitComponent + thinking prop)
// import waiting from './waiting.vue'
import { serialize, deserialize } from '../scripts/range-serializer.js';

// RulesOfMarkdown / RulesOfLatex 已迁移到 scripts/aiService.js

export default{
  components:{
    gptRenderUnit,
    // waiting 已在第三步删除

  },
  data(){
    return{
      chosen_text:"",
      chosen_text_history:"",
      chosen_add_text:[],
      chosen_hldata:null,
      shown_text:"",

      ai_res:[],  //控制侧边显示的内容viewmodel，由于早期开发的因素，此处命名为ai_res，其实包含ai、询问以及注释
      isDragOver: false,
      showOrigin: false, // 新增状态控制
      askFocus: false,
      askContent:"",
      askQuote:[],
      askImage:{img:null, size:0},
      

      memorylist:[],

      pdf_fp:"", //pdf指纹，用于访问AI的储存

      waiting:false,
      waiting_cancel:false,
      waiting_mouse:false,
      wait_text:'WAIT',
      wait_color:'#4e7bbf',  //#4e7bbf #AC4848
      cancelTokenSource:null,

      opacityArr:[],

      annotation_mode:false,
      annotation_flag:false, //判断是否确认注释
      annotation_hldata:null,
      annotaiton_selected:"",  //注释选中文本
      annotation_edit_index:-1, //准备修改的item
      annotation_edit_texts:[],
      can_blur:true,
      refreshKey:0,
      can_render_click:true,
      placeholderIndex: -1,  // 当前 thinking 占位消息在 ai_res 中的索引,-1 表示无

      childrenHeightArr:[],  //索引显示model(单条粒度,保留向后兼容)
      groupedChildrenHeightArr:[],  //索引显示model(ASK+AI 合并为一组的版本,侧条渲染用这个)
      classifyRateArr:[0,0,0], //显示每一种卡片百分比
      scrollTopRate:"0%",  //指针位置
      scrollHeightSum:1,

      containerRect:null,
      current_index:0,  //追踪索引(指向 ai_res 中的某一条消息)
      group_cursor:0,   //鼠标侧键用的"组"游标(指向 groupedChildrenHeightArr 的下标)
      can_current_index_change:true, //是否允许当前索引改变
      lastBottomRate:1, //最后一个元素的底部距离容器底部占容器百分比
      imageShowFlag: false, //图片显示标志
    
      hl_arr:{},  //高亮位置
    }

    
  },
  mounted() {
    const container = this.$refs.render_container;
    this.containerRect = container.getBoundingClientRect();

    window.addEventListener('resize', () => {
      const container = this.$refs.render_container;
      if (container) {
        this.containerRect = container.getBoundingClientRect();
        this.UpdateWindowUI().then(()=>{
          this.adjustScroll(this.current_index);
        });
      }
    });

    const urlParams = new URLSearchParams(window.location.search);
    const ts = urlParams.get('ts');
    console.log(`VUE Confirm VID: ${ts}`)
    this.vid = ts

    chrome.runtime.onMessage.addListener((message, sender, senderResponse) => {
      if(message.vid==='' ||  message.vid === this.vid){

        if(message.type === 'SET_TEXT'){
          if(message.text.length > 50){
            console.log(`VUE[${this.vid}] SELECT TEXT: ${message.text.length} words`)
            this.onTextChange(message.text, message.add, message.hldata)
            window.focus()
            this.$refs.blurTextarea.focus()
          }
          else{
            chrome.runtime.sendMessage({
              type: "SET_TEXT_MINI",
              vid: this.vid,
              text: message.text,
            });

          }

        }
        if(message.type === 'SET_MODEL'){
          console.log("VUE PARAMETERS SET: "+`${JSON.stringify(message)}`)
          // 每次 SET_MODEL 都 POST 到后端,后端 ai_agent.ai_config 统一维护。
          // ask / load 请求不再需要透传 api_key / api_url / model。
          this.$axios.post('/ai/config', {
            api_key: message.apiKey || '',
            api_url: message.apiUrl || '',
            model: message.model || '',
            vision_model: message.visionModel || message.model || '',
            deepseek_thinking: !!message.deepseekThinking,
          }).catch(err => {
            console.error('[/ai/config] failed:', err)
          })
        }
        if(message.type === 'RESET_MORE'){
          //MORE 清除请求
          if(this.chosen_add_text.length !== 0){
            this.chosen_add_text.length = 0
            this.chosen_text = ''
          }
        }
        if(message.type === 'PDF_FP'){
          //获得PDF_FP以请求AI储存资料
          console.log("VUE Get FP:"+message.fp)
          this.pdf_fp = message.fp
          this.$axios.post('/reqLoadAIPre', {fp:this.pdf_fp}).then(res=>{
            if(res.data){
              this.wait_text = 'LOAD'
              this.waiting = true
              this.handelAiLoad()

            }
          })

        }
        if(message.type === 'ADD_ANNOTATION'){
          //请求添加注释，此时输入框获得焦点
          this.annotation_flag = false 
          this.annotation_mode = true
          this.annotation_hldata = message.hldata
          this.askFocus = true
          this.$refs.askTextrea.focus()
          this.annotaiton_selected = message.text

        }
        if(message.type === 'ACTIVE_CANCEL_ANNOATION'){
          //pdf主动失去焦点
          if(this.annotation_mode){
            this.annotation_mode = false
            this.askContent = ""
            this.askQuote.length = 0
            this.askFocus = false
            this.$refs.blurTextarea.focus()
            this.resetImgInfo()
          }


        }
        if(message.type === 'ACTIVE_CLICK_ANNO'){
          //pdf主动点击注释，vue跳转
          let index = -1
          this.ai_res.forEach((item, i)=>{
            if(item.type==='LIST_TYPE_ANNO_'+message.flag){
              index = i
              return
            }
          })
          this.adjustScroll(index)
        }
        if(message.type === 'API_FORCE_STOP'){
          //主动请求取消
          this.handleForceStop()

        }
      }

    });

    window.top.postMessage({type:'VUE_INIT', vid:ts}, '*')

  },
  methods:{
    getHighlightPos(rangeRect){
      
      //控制计算
      const container = this.$refs.render_container;
      const targetElement = container.children[0];

      if (targetElement) {
        const targetRect = targetElement.getBoundingClientRect();
        const offsetTop = Math.abs(rangeRect.top - targetRect.top);
        const percentage = offsetTop/this.scrollHeightSum*100
        return `${percentage}%`;
      }

      return '0%';
    },
    setRangeHighlight(selectedRange, key) {
      // 检查选中范围是否包含数学公式节点(KaTeX 行内/行间 + 原生 <math>)
      // 注意:不要直接用 extractContents 拆掉公式 DOM,会让公式消失
      const mathNodes = document.querySelectorAll(
        '.katex:has(math), .katex-display:has(math), .math-display:has(math), math'
      )
      let containsMath = false

      // 检查选中范围是否与任何数学公式节点重叠
      mathNodes.forEach(node => {
        if (selectedRange.intersectsNode(node)) {
          containsMath = true
        }
      })

      // 如果包含数学公式节点,则不执行高亮
      if (containsMath) {
        console.log('选中范围包含数学公式,已跳过高亮')
        return
      }

      // 创建高亮span
      const highlightSpan = document.createElement('span');
      highlightSpan.style.cssText = `
        background-color: transparent;
        color: inherit;
        font-style: inherit;
        font-weight: inherit;
        text-decoration: inherit;
        border-inline: 2px solid transparent;
        padding: 1px;
        margin-inline: 1px;
        transition: all 0.2s ease;
      `;

      // 添加鼠标悬停事件监听器来控制光标样式
      highlightSpan.addEventListener('mouseover', (event) => {
        if (event.altKey) {
          highlightSpan.style.cursor = 'pointer';
        } else {
          highlightSpan.style.cursor = '';
        }
      });

      // 添加鼠标移出事件监听器清除光标样式
      highlightSpan.addEventListener('mouseout', () => {
        highlightSpan.style.cursor = '';
      });

      // 注册按住 Alt 再左键删除高亮
      highlightSpan.addEventListener('click', (event) => {
        if (!(event.altKey && event.button === 0)) return;
        event.stopPropagation();
        const parent = highlightSpan.parentNode;
        // 提取高亮节点的所有子节点（支持非文本节点）
        const fragment = document.createDocumentFragment();
        while (highlightSpan.firstChild) {
          fragment.appendChild(highlightSpan.firstChild);
        }
        // 用子节点替换高亮span，恢复原始结构
        parent.replaceChild(fragment, highlightSpan);

        if (this.hl_arr.hasOwnProperty(key)) {
          delete this.hl_arr[key];
        }
        this.$axios.put('/reqHighlightSet', { fp: this.pdf_fp, key: key, hl: null });
      });

      // 提取选中范围的内容（支持非文本节点）
      const extractedContent = selectedRange.extractContents();
      // 将提取的内容放入高亮span
      highlightSpan.appendChild(extractedContent);
      // 将高亮span插入到原选中范围的位置
      selectedRange.insertNode(highlightSpan);

      // 延迟添加高亮样式，产生过渡效果
      setTimeout(() => {
        highlightSpan.style.backgroundColor = '#B0515135';
        highlightSpan.style.borderColor = '#B05151';
      }, 50);

    },
    loadRangeHighlight(){
      this.hl_arr = {};  // 清空高亮位置对象
      this.$axios.post('/reqHighlightGet', {fp:this.pdf_fp, key:'', hl:null}).then(res=>{
        if (res.data) {
          Object.entries(res.data).forEach(([key, hl]) => {
            if (hl) {
              const range = deserialize(hl);
              if(range){
                this.setRangeHighlight(range, key);
                this.hl_arr[key] = {
                  range: range,
                  position: this.getHighlightPos(range.getBoundingClientRect()),  // 获取高亮位置
                }
              }

            }
          });
        }

      })
    },
    handelSelectedHighlight(e) {
      if (e.button === 0) {
        // 只处理来自当前 iframe 内的 mouseup,忽略跨 iframe 冒泡来的。
        // document.contains(e.target) 对跨 iframe 事件返回 false,
        // 因为 PDF iframe 和 Vue iframe 各是不同的 document。
        // 这样在 PDF 区域选中文本复制时就不会触发 SET_TEXT → 绿框高亮。
        if (!document.contains(e.target)) return

        const selection = window.getSelection();
        if (!selection.isCollapsed) {
          const selectedRange = selection.getRangeAt(0);
          const selectedText = selection.toString();

          if (e.altKey && selectedText) {
            const key = `HL_${Date.now()}_${this.generateRandomCode()}`
            const hl_range_sel = serialize(selectedRange);  //序列化range

            const hl_range_des = deserialize(hl_range_sel);  //反序列化range
            this.setRangeHighlight(hl_range_des, key);  // 设置高亮

            this.hl_arr[key] = {
              range: hl_range_des,
              position: this.getHighlightPos(hl_range_des.getBoundingClientRect()),  // 获取高亮位置
            }
            this.$axios.put('/reqHighlightSet', {fp:this.pdf_fp, key:key, hl:hl_range_sel})  //发送高亮数据
            
            selection.removeAllRanges();
          }
        }
      }
    },
    handleMouseSideButton(event){
      if(!this.can_current_index_change)
        return
      if (this.groupedChildrenHeightArr.length === 0) return

      if (event.button === 3) {  // 后退侧键(向下走)
        this.group_cursor++
      } else if (event.button === 4) {  // 前进侧键(向上走)
        this.group_cursor--
      } else {
        return
      }
      // 边界修正
      const len = this.groupedChildrenHeightArr.length
      if (this.group_cursor < 0) this.group_cursor = 0
      if (this.group_cursor >= len) this.group_cursor = len - 1

      const targetGroup = this.groupedChildrenHeightArr[this.group_cursor]
      // 滚到组的起始消息(整段向上推,ASK 在上、AI 在下,看起来跟聊天一致)
      this.adjustScroll(targetGroup.start)
      this.highlightScroll(targetGroup.start)
    },
    handleSideButtonCapture(event) {
      if (event.button === 3 || event.button === 4) {
        event.preventDefault()
        event.stopPropagation()
        // stopImmediatePropagation 防止同一节点上其他监听器也被触发
        if (event.stopImmediatePropagation) event.stopImmediatePropagation()
      }
    },
    UpdateWindowUI(){
      return new Promise((resolve, reject) => {
        //控制更新UI
        this.$nextTick(()=>{
          let container = this.$refs.render_container
          this.containerRect = container.getBoundingClientRect();

          this.UpdateChildrenHeightArr()
          this.UpdatePointerPosition()

          let last_ai_el = container.children[this.ai_res.length-1]  //最后一个ai元素
          if(last_ai_el){
            this.$refs.last_blank.style.height = `${this.containerRect.height}px`
          }

          // 等容器内所有 <img> 加载完(否则最后一条 div 高度偏小,
          // adjustScroll 算出的 offsetTop 不准,最后一条置顶会差一截)。
          // 2s 兜底超时,避免被某个 404 图永久卡住。
          this.waitImagesLoaded(container, 2000).then(() => {
            this.$nextTick(resolve)
          })
        })
      });


    },
    waitImagesLoaded(container, timeout = 2000) {
      // 收集容器里所有 <img>,等待全部 complete。
      // 已加载的(decode 后)直接 resolve,未加载的挂 onload/onerror。
      return new Promise(resolve => {
        if (!container) return resolve()
        const imgs = Array.from(container.querySelectorAll('img'))
        if (imgs.length === 0) return resolve()
        let pending = 0
        const done = () => { if (pending === 0) resolve() }
        const timer = setTimeout(done, timeout)
        imgs.forEach(img => {
          if (img.complete && img.naturalWidth > 0) return
          pending++
          const onEnd = () => { pending--; if (pending === 0) { clearTimeout(timer); resolve() } }
          img.addEventListener('load', onEnd, { once: true })
          img.addEventListener('error', onEnd, { once: true })
        })
        if (pending === 0) { clearTimeout(timer); resolve() }
      })
    },
    UpdatePointerPosition() {
      const container = this.$refs.render_container;

      // ── 侧条游标定位 ──
      // 修复: 原写法用 container.children[0] 的位置算"距顶偏移",但 children[0]
      // 永远是同一个节点,滚动越深它的 top 越负,公式就越怪;
      // 且容器是自己(不是父),不能拿 children[0] 当锚点。
      // 正确表达:"滚动条已滚动的距离 占 可滚动距离 的百分比"。
      // 内容不足一屏时 scrollHeight <= clientHeight,不可滚动,游标置 0%。
      const scrollable = container.scrollHeight - container.clientHeight
      if (scrollable > 0) {
        const ratio = (container.scrollTop / scrollable) * 100
        this.scrollTopRate = `${Math.max(0, Math.min(100, ratio))}%`
      } else {
        this.scrollTopRate = '0%'
      }

      let last_ai_el = container.children[this.ai_res.length - 1]; // 最后一个ai元素
      if (last_ai_el) {
        // 语义修正(原 bug:用 last_ai_el_rect 计算的"最后元素在容器内位置",
        // 短消息时该值接近 0,误判为"在底部",导致未滚到底就清晰渲染):
        // 改为"滚动条距底部的比例": 0=在底部,1=距底部 1 屏
        // 新公式: distanceFromBottom / clientHeight
        // clientHeight 代表一屏高度,所以 ratio=1 表示距底部刚好一屏
        const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
        // Guard: if content fits in viewport (scrollHeight <= clientHeight), image should stay hidden
        // because there's no meaningful "scrolled to bottom" state to trigger it
        if (container.scrollHeight <= container.clientHeight) {
          this.imageShowFlag = false;
        } else {
          this.lastBottomRate = Math.max(0, Math.min(1, distanceFromBottom / container.clientHeight));
          this.imageShowFlag = distanceFromBottom / container.clientHeight < 0.5;
        }
      }

      //更新高亮位置
      Object.entries(this.hl_arr).forEach(([key, hl]) => {
        if (hl.range) {
          const rangeRect = hl.range.getBoundingClientRect();
          if (rangeRect.top === 0 && rangeRect.left === 0)  //位置有效性判断
          {
            delete this.hl_arr[key];
            this.$axios.put('/reqHighlightSet', { fp: this.pdf_fp, key: key, hl: null }); 
          }
          else{
              hl.position = this.getHighlightPos(rangeRect);
          }
          
        }
      });

    },
    UpdateChildrenHeightArr(){
      this.childrenHeightArr.length = 0
      this.groupedChildrenHeightArr.length = 0
      const container = this.$refs.render_container;

      let height_arr = []
      let height_sum = 0

      this.ai_res.forEach((item, index)=>{
        // 确保索引有效
        if (index < -1 || index >= container.children.length) {
          console.error('Invalid index:', index);
          return;
        }
        const targetElement = container.children[index];

        if (targetElement) {
          // 获取目标元素的边界矩形
          const targetRect = targetElement.getBoundingClientRect();
          let h = targetRect.height
          height_arr.push(h)
          height_sum += h
        }
      })
      this.scrollHeightSum = height_sum  //更新求和高度
      this.classifyRateArr = [0,0,0]

      height_arr.forEach((h, index)=>{
        const ai_type = this.ai_res[index].type
        const ai_hrate = h/height_sum*100
        this.childrenHeightArr.push({
          hindex:index,
          hrate:ai_hrate,
          htype:ai_type
        })
        //classifyRateArr [input  feedback   ouput]
        if(ai_type.includes('LIST_TYPE_AI')){
          this.classifyRateArr[0] += ai_hrate
        }
        if(ai_type.includes('LIST_TYPE_ASK')){
          this.classifyRateArr[1] += ai_hrate
        }
        if(ai_type.includes('LIST_TYPE_ANNO')){
          this.classifyRateArr[2] += ai_hrate
        }
      })

      // ── 分组:ASK + 紧邻的 AI 合为一组;其他各自单组 ──
      // 分组规则在 ai_res 索引层面跑,高度比例按组聚合求和。
      // 单一 ASR 或 ANNO 也作为一个独立组(便于点击/侧键跳转)。
      let i = 0
      while (i < this.ai_res.length) {
        const cur = this.ai_res[i]
        const curType = cur && cur.type ? cur.type : ''
        let start = i
        let end = i + 1  // 半开区间 [start, end)
        let groupType = 'OTHER'

        if (curType.includes('LIST_TYPE_ASK')) {
          // 一组最多 = ASK + 紧邻的 1 条 AI。
          // 关键:不能无脑吞掉后续所有 AI — 后续 AI 很可能是另一次划词回复
          // (callAIResponse 触发,前面没 ASK),必须留给下一轮扫描自己独立成组。
          groupType = 'ASK_AI'
          if (i + 1 < this.ai_res.length
              && this.ai_res[i + 1].type
              && this.ai_res[i + 1].type.includes('LIST_TYPE_AI')) {
            end = i + 2
          } else {
            end = i + 1
          }
        } else if (curType.includes('LIST_TYPE_AI')) {
          groupType = 'AI'
          end = i + 1
        } else if (curType.includes('LIST_TYPE_ANNO')) {
          groupType = 'ANNO'
          end = i + 1
        }

        let hrate = 0
        for (let k = start; k < end; k++) hrate += this.childrenHeightArr[k].hrate
        this.groupedChildrenHeightArr.push({
          start,
          end,
          hrate,
          groupType,
        })
        i = end
      }
    },
    MapGroupColor(group){
      // 整段 ASK + AI 在侧条上一律显示为绿色(统一一个问答组)
      if (group.groupType === 'ASK_AI') {
        return 'rgba(86, 145, 69, 0.741)'
      }
      if (group.groupType === 'AI') {
        return 'rgb(122, 155, 205)'
      }
      if (group.groupType === 'ANNO') {
        return '#AC4848'
      }
      return '#cccccc'
    },
    MapHeadBarBackgroundColor(headtype){
      if(headtype.includes('LIST_TYPE_ASK')){
        return 'rgba(86, 145, 69, 0.741)'
      }
      if(headtype.includes('LIST_TYPE_AI')){
        return 'rgb(122, 155, 205)'
      }
      if(headtype.includes('LIST_TYPE_ANNO')){
        return '#AC4848'
      }
    },
    handleForceStop(){
      if(this.wait_text === 'LOAD' || !this.waiting)  //加载操作不可取消
        return
      
      this.waiting = false //强制停止
      this.waiting_cancel = true //强制遮罩
      this.cancelTokenSource?this.cancelTokenSource.cancel('请求被用户取消'):null //强制停止请求
      this.wait_text = 'CANCEL'
      this.wait_color = '#AC4848'

      const delay_time = 500
      setTimeout(() => {
        this.waiting_cancel = false
      }, delay_time);
      setTimeout(() => {
        this.wait_text = 'WAIT'
        this.wait_color = '#4e7bbf'
      }, delay_time + 200);  //附加动画时间 0.2s

    },
    generateRandomCode() {
      // 生成0-999999的随机数，不足六位时前补0
      return Math.floor(Math.random() * 1000000).toString().padStart(6, '0');
    },
    ConfirmAnnotation(){
      if(!this.annotation_mode && this.annotation_edit_index===-1)
        return

      if(this.askContent.trim() !== ""){

        if(this.annotation_edit_index !== -1){
          //修改模式
          let history_res = {...this.ai_res[this.annotation_edit_index]}
          history_res.text = `${this.askContent}>>${this.annotation_edit_texts[1]}<<`
          
          if(this.askImage.img !== null){
            let img_name = `${this.pdf_fp}_${Date.now()}.png`
            //启动图片覆盖
            this.$axios.put('/reqImg', {
              imgname: history_res.img? history_res.img:img_name,
              base64:this.askImage.img,
              mode:'ADD'
            })
            history_res.img = history_res.img? history_res.img:img_name
            
          }
          this.handleAiChange('SET', this.annotation_edit_index, history_res)  //如果同时填写index和content，表示修改
          this.handleAskAreaBlur()
          this.refreshKey++

        }
        else{
          const timestamp = Date.now();  //确认标记
          let img_name = ""

          if(this.askImage.img !== null){
            img_name = `${this.pdf_fp}_${Date.now()}.png`
            //启动图片上传
            this.$axios.put('/reqImg', {
              imgname: img_name,
              base64:this.askImage.img,
              mode:'ADD'
            })
          }

          //确认注释储存内容
          this.handleAiChange('ADD', -1, {
            type:'LIST_TYPE_ANNO_'+timestamp,
            text:this.askContent + `>>${this.annotaiton_selected}<<`,
            dt:this.getFormattedDate(),
            hl:this.annotation_hldata,
            img:img_name,
          })
          
          this.annotation_flag = true  //确认注释，不再发送CANCEL消息
          chrome.runtime.sendMessage({  //发送确认消息
            type:'CONFIRM_ANNOATION', 
            vid:this.vid,
            flag:timestamp,
          })

        }
      }

      // 焦点失去
      this.askContent = ""
      this.askQuote.length = 0
      this.askFocus = false
      this.$refs.blurTextarea.focus()
      this.annotation_hldata = null
      this.resetImgInfo()

    },
    handleAskAreaBlur(clear=false){
      if(!this.can_blur)
        return

      //询问区域失去焦点
      this.annotation_mode = false
      this.annotation_edit_index = -1

      //判断是否确认注释
      if(!this.annotation_flag){
        chrome.runtime.sendMessage({
          type:'CANCEL_ANNOATION', 
          vid:this.vid,
        })
      }

      // 焦点失去
      if (clear){
        this.askContent = "" 
        this.askQuote.length = 0
        this.resetImgInfo()
      }

      this.askFocus = false
      this.$refs.blurTextarea.focus()

    },
    parseMessage(message) {
      const separatorIndex = message.indexOf(">>");
      
      if (separatorIndex === -1) {
        // 如果没有找到分隔符，返回原始消息和空字符串
        return [message, ""];
      }
      
      const firstPart = message.substring(0, separatorIndex);
      
      // 查找结束标记 << 的位置
      const endMarkerIndex = message.indexOf("<<", separatorIndex + 2);
      
      if (endMarkerIndex === -1) {
        // 如果没有找到结束标记，返回第一部分和剩余部分（不含开始标记）
        return [firstPart, message.substring(separatorIndex + 2)];
      }
      
      const secondPart = message.substring(separatorIndex + 2, endMarkerIndex);
      
      return [firstPart, secondPart];
    },
    handleHlClick(hl, htype, index){
      if(!this.can_render_click)
        return
      if(htype === 'LIST_TYPE_AI'){
        if(hl){
          chrome.runtime.sendMessage({
            type:'PDF_HIGHLIGHT', 
            vid:this.vid,
            hl:JSON.parse(JSON.stringify(hl)),
            hl_type:htype,
          })
        }
      }
      if(htype.includes('LIST_TYPE_ANNO')){
        if(this.annotation_edit_index!==index){
          this.annotation_edit_index = index
          //主动获得焦点，进入编辑模式
          this.askFocus = true
          this.$refs.askTextrea.focus()
          this.annotation_edit_texts = this.parseMessage(this.ai_res[index].text)
          this.askContent = this.annotation_edit_texts[0]
          if(this.ai_res[index].img!==''){
            this.readImageData('http://localhost:8225/static/'+this.ai_res[index].img)
          }

        }
        else{
          this.annotation_edit_index = -1
          this.handleAskAreaBlur()
        }


      }

    },
    handleANNOClick(hl, htype){
      chrome.runtime.sendMessage({
        type:'PDF_HIGHLIGHT', 
        vid:this.vid,
        hl:JSON.parse(JSON.stringify(hl)),
        hl_type:htype,
      })

    },
    async handleAiChange(mode, index, content, {skipScroll = false} = {}){
      // 1) 先更新本地数组(乐观更新),保证 UI 即时响应
      if(mode==='ADD'){
        this.ai_res.push(content)
      } else if(mode==='DEL'){
        this.can_render_click = false
        const target = this.ai_res[index]
        if(target && target.type && target.type.includes("LIST_TYPE_ANNO")){
          let flag = target.type.split('_').at(-1)
          chrome.runtime.sendMessage({
            type:'DEL_ANNOATION',
            vid:this.vid,
            flag:flag,
          })
        }
        if(target && target.img){
          //启动图片删除
          this.$axios.put('/reqImg', {
            imgname: target.img,
            base64:"",
            mode:'DEL'
          })
        }
        this.ai_res.splice(index, 1)
      } else if(mode === 'SET'){
        this.ai_res[index]=content
      }

      // 2) 同步到后端;失败则回滚本地状态,并提示用户
      try {
        await this.$axios.put('/reqSaveAI', {fp:this.pdf_fp, mode:mode, index:index, cont:content})
      } catch (err) {
        console.error('[reqSaveAI] 失败,回滚本地:', err)
        if (mode === 'ADD') {
          this.ai_res.pop()
        } else if (mode === 'DEL') {
          this.ai_res.splice(index, 0, content)
        } else if (mode === 'SET') {
          // SET 没有"旧版本快照",只提示,不动本地
          this.$message?.error?.('保存会话失败,刷新页面后可能丢失')
        }
      }

      let scrollToIndex;
      if (mode === 'ADD') {
        scrollToIndex = this.ai_res.length - 1;
      } else if (mode === 'DEL') {
        scrollToIndex = Math.max(index - 1, 0);
      } else if (mode === 'SET') {
        scrollToIndex = index;
      } else {
        scrollToIndex = this.ai_res.length - 1;
      }

      this.UpdateWindowUI().then(()=>{
        if (skipScroll) return
        if (
          (mode === 'ADD' && content && !content.type.includes("LIST_TYPE_ANNO")) ||
          (mode === 'SET' && content && !content.type.includes("LIST_TYPE_ANNO")) ||
          (mode === 'DEL')
        ) {
          this.adjustScroll(scrollToIndex);  //跳转
        }

        setTimeout(() => {
          this.can_render_click = true
        }, 200);
      })
    },
    handelAiLoad(){
      this.$axios.post('/reqLoadAI', {fp:this.pdf_fp}).then(res=>{

          setTimeout(()=>{
            this.ai_res = res.data
            this.wait_text = 'WAIT'
            this.waiting = false

            //自动跳转
            if(this.ai_res){

              this.UpdateWindowUI().then(()=>{
                // 跳转到最后一组的开头(而不是最后一条消息)
                // 原因:最后一组通常是 ASK + AI 的问答,用户期望看到的是用户气泡置顶。
                // 之前 adjustScroll() 不传参默认跳到 ai_res 最后一条,会把 AI 气泡顶到屏幕顶部,
                // 用户得再往上滚才能看到自己的提问,体验割裂。
                const lastGroup = this.groupedChildrenHeightArr[this.groupedChildrenHeightArr.length - 1]
                const jumpTo = lastGroup ? lastGroup.start : this.ai_res.length - 1
                this.adjustScroll(jumpTo)
                this.loadRangeHighlight()
                this.current_index = jumpTo
                // 初始化组游标到末尾(侧键向上跳才合理)
                this.group_cursor = Math.max(0, this.groupedChildrenHeightArr.length - 1)
                this.$refs.blurTextarea.focus()
              })

            }

            //检出其中所有的ANNO，发送给pdfjs
            let anno_list = []
            this.ai_res.forEach(item=>{

              if(item.type.includes('LIST_TYPE_ANNO')){
                anno_list.push({
                  flag: item.type.split('_').at(-1),
                  hldata:item.hl,
                })                
                // console.log("VUE 已经读取flag= "+item.type.split('_').at(-1) +"的数据")
              }
            })
            chrome.runtime.sendMessage({
              type:'LOAD_ANNOATION', 
              vid:this.vid,
              notify_list:anno_list,
            })

          }, 1000)
      })

    },
    generateDateTimeString() {
      const now = new Date();

      // 获取年、月、日、小时、分钟、秒
      const year = now.getFullYear().toString().slice(-2); // 取后两位年份
      const month = String(now.getMonth() + 1).padStart(2, '0'); // 月份从0开始，补零
      const day = String(now.getDate()).padStart(2, '0'); // 补零
      const hours = String(now.getHours()).padStart(2, '0'); // 补零
      const minutes = String(now.getMinutes()).padStart(2, '0'); // 补零

      // 拼接字符串
      return `${year}${month}${day}-${hours}${minutes}`;
    },
    handleAnnoationImgDel(){
      if(this.annotation_edit_index !== -1 && this.ai_res[this.annotation_edit_index].img!==''){
        let history_res = this.ai_res[this.annotation_edit_index]
        //启动图片删除
        this.$axios.put('/reqImg', {
          imgname: history_res.img,
          base64:"",
          mode:'DEL'
        })
        history_res.img = ''
        this.handleAiChange('SET', this.annotation_edit_index, history_res)  //如果同时填写index和content，表示修改
      }
      //恢复blur操作
      this.can_blur = true
      this.handleAskAreaBlur()

    },
    resetImgInfo(){
      this.askImage.img = null
      this.askImage.size = 0
      
    },
    handlePaste(e) {
      e.preventDefault();

      // 1. 检查粘贴内容是否为图片
      const clipboardItems = e.clipboardData?.items || [];
      for (const item of clipboardItems) {
        if (item.type.startsWith('image/')) {
          this.handleAddImg();
          return;
        }
      }

      // 2. 获取粘贴文本
      const pastedText = e.clipboardData?.getData('text') || '';
      if (!pastedText) return;

      // 3. 获取当前光标位置
      const textarea = e.target;
      const startPos = textarea.selectionStart;
      const endPos = textarea.selectionEnd;

      // 4. 在光标位置插入文本
      const currentValue = textarea.value;
      const newValue =
        currentValue.substring(0, startPos) +
        pastedText +
        currentValue.substring(endPos);

      // 5. 同步更新 Vue 数据模型:
      //    这里 home.vue 是 Vue 根实例(this.$emit 不会触发 v-model),
      //    必须直接赋值,否则粘贴后 textarea.value 有内容,
      //    但 askContent 还是空 → onAskEnter 里
      //    `!this.askContent.trim()` 直接 return,导致按 Enter 不发送,
      //    只有用户再敲一个字触发 input 事件同步一次才恢复正常。
      textarea.value = newValue;
      this.askContent = newValue;

      // 6. 恢复光标位置(在粘贴文本之后)
      this.$nextTick(() => {
        textarea.selectionStart = textarea.selectionEnd = startPos + pastedText.length;
      });
    },
    async readImageData(url) {
      // 1. 获取图片为 ArrayBuffer
      const response = await fetch(url);
      const buffer = await response.arrayBuffer();

      // 2. 计算文件大小（字节）
      const fileSizeInBytes = buffer.byteLength;
      const fileSizeInKB = (fileSizeInBytes / 1024).toFixed(2);

      // 3. 转换为 Base64
      const base64 = btoa(
        new Uint8Array(buffer).reduce(
          (data, byte) => data + String.fromCharCode(byte),
          ''
        )
      );

      this.askImage.img = `data:${response.headers.get('content-type')};base64,${base64}`;
      this.askImage.size = fileSizeInKB;

    },
    async handleAddImg(){
      const clipboardItems = await navigator.clipboard.read();

      for (const clipboardItem of clipboardItems) {
        for (const type of clipboardItem.types) {
          if (type.startsWith('image/')) {
            const blob = await clipboardItem.getType(type);
            
            // 获取图片文件大小（字节）
            const fileSize = blob.size;
            const fileSizeKB = Math.round(fileSize / 1024 * 100) / 100; // 转换为KB
            
            // 创建临时URL来获取图片尺寸
            const img = new Image();
            img.onload = function() {
              // 转换为Base64
              const reader = new FileReader();
              reader.onload = function(e) {
                const base64data = e.target.result;
                // 这里的 this 现在指向外层 this
                this.askImage.img = base64data;
                this.askImage.size = fileSizeKB;
              }.bind(this); // 绑定外层 this 到 FileReader 的回调
              
              reader.readAsDataURL(blob);
            }.bind(this); // 绑定外层 this 到 Image 的 onload 回调
            
            img.src = URL.createObjectURL(blob);

            this.askQuote.length = 0
            return;
          }
        }
      }

    },
    getQuoteContent(){
      let res = ""
      let gid_set = new Set();
      this.askQuote.forEach(node=>{
        gid_set.add(node.gid)
      })
      gid_set.forEach(i=>{
        res += this.ai_res[i] + '\n\n'
      })
      return res
    },
    onTextChange(t, add, hl){
      let text = this.selectionProcess(t)
      text = this.AIinputProcess(text)
      if(text.length >=3500){
        const isConfirmed = confirm(`选中字数异常 ${text.length} words，确认吗？`);
        if (!isConfirmed) {
          return
        }

      }
      this.resetImgInfo()
      if(!this.showOrigin && !this.waiting && text != "" && text != this.chosen_text_history){
        this.chosen_hldata = hl
        if(add){
          this.chosen_text_history = this.chosen_text
          this.chosen_add_text.push(text)
          this.chosen_text = this.chosen_add_text.join('')

        }else{
          this.chosen_text_history = this.chosen_text
          this.chosen_text += text  //这一步是为了上面add
          this.addMemoryList(this.chosen_text)
          this.callAIResponse()
          this.chosen_text = ""
          this.chosen_add_text.length = 0
        }


      }
      this.shown_text = text

    },
    addMemoryList(text){
      const mem_len = 20  //记忆长度
      if(this.memorylist.length < mem_len){
        if(!this.memorylist.includes(text)){
          this.memorylist.push(text)
        }
      }else{
        let index = this.memorylist.indexOf(text)
        if(index === -1 || index === 0){
          this.memorylist.shift()
          this.memorylist.push(text)
        }
      }
    },
    handleQuote(gid, msg, method){
      let new_quote = {quote_gid:gid, quote_msg:msg}
      let index = this.askQuote.findIndex(item => item.quote_gid===new_quote.quote_gid && item.quote_msg===new_quote.quote_msg);

      if(method === 'set'){
        if (index !== -1) {
          // 如果存在则删除
          this.askQuote.length = 0
        } 
        else{
          this.askQuote.length = 0
          this.askQuote.push(new_quote);
        }
      }
      else if(method === 'add'){
        if (index !== -1) {
          // 如果存在则删除
          this.askQuote.splice(index, 1);
        } else {
          // 如果不存在则添加
          this.askQuote.push(new_quote);
        }
        
      }
      this.resetImgInfo()
      this.askFocus = false

    },
    checkAndRemoveTest(str) {
        if (str.includes("@Test")) {
            return {
                found: true,
                newString: str.replace("@Test", "")
            };
        } else {
            return {
                found: false,
                newString: str
            };
        }
    },
    onAskEnter(){
      // Enter 发送逻辑:
      // 1. 注释模式 / 编辑模式下,需要同时 handleAsk + ConfirmAnnotation
      // 2. 普通追问模式下,直接 handleAsk
      if(this.annotation_mode || this.annotation_edit_index !== -1){
        this.handleAsk()
        this.ConfirmAnnotation()
        return
      }
      this.handleAsk()
    },
    onAskShiftEnter(){
      // Shift+Enter 插入换行 —— 浏览器默认行为已经插入了 \n
      // 这里作为锚点 method 存在,方便后续扩展(例如自动调整高度)
    },
    handleAsk(){
      if(this.annotation_mode)
        return
      if(this.annotation_edit_index!==-1)
        return

      if(this.askContent.toLowerCase().includes("@clea")){
        const isConfirmed = confirm(`你确定要删除这 ${this.ai_res.length} 条记录吗？`);
        if (isConfirmed) {
          this.$axios.put('/clear_all', {
            fp: this.pdf_fp,
          })
          this.ai_res.length = 0
          this.hl_arr.length = 0
          this.UpdateWindowUI().then(() => {
            this.adjustScroll()
            this.loadRangeHighlight()
            this.current_index = this.ai_res.length - 1
            this.$refs.blurTextarea.focus()
          })

          chrome.runtime.sendMessage({  //发送确认消息
            type: 'CONFIRM_CLEARALL',
            vid: this.vid,
          })
        }

        // 焦点失去
        this.handleAskAreaBlur(true)

        return
      }

      // 取消空内容请求(非视觉模式才检查,视觉模式有图片)
      if (!this.askImage.img && !this.askContent.trim()) return

      // ── 准备 cancel token ──
      this.cancelTokenSource?.cancel('请求被用户取消')
      this.cancelTokenSource = this.$axios.CancelToken.source()
      this.controller = new AbortController()

      // ── 判断是否视觉模式 ──
      const isVision = this.askImage.img !== null

      // ── Push 用户气泡 ──
      this.handleAiChange('ADD', -1, {
        type: 'LIST_TYPE_ASK',
        text: this.askContent,
        dt: this.getFormattedDate(),
        hl: null,
      }, {skipScroll: true})

      // ── Push AI 占位 ──
      // token_count 必须在初始化时声明,后续 stream 结束时才能可靠地
      // 通过 placeholder.token_count = usage.total_tokens 触发响应式更新。
      const placeholder = {
        type: 'LIST_TYPE_AI',
        text: '',
        dt: this.getFormattedDate(),
        hl: this.chosen_hldata,
        img: '',
        token_count: null,
      }
      this.ai_res.push(placeholder)
      this.placeholderIndex = this.ai_res.length - 1
      this.waiting = true
      this.$nextTick(() => {
        this.UpdateWindowUI().then(() => this.adjustScroll(this.placeholderIndex - 1))
      })

      // ── 发送请求(后端 LangGraph /ai/ask) ──
      // memorylist 是中文片段, header 只能 ISO-8859-1, 必须 base64
      const headers = {
        'x-pdf-memorylist': btoa(unescape(encodeURIComponent(JSON.stringify(this.memorylist || [])))),
      }
      this.$axios.post('/ai/ask', {
        pdf_fp: this.pdf_fp,
        ask: this.askContent,
        quotes: this.askQuote,
        quote_content: this.getQuoteContent(),
        image_base64: isVision ? this.askImage.img : null,
      }, {
        cancelToken: this.cancelTokenSource.token,
        headers,
      })
        .then(({ data }) => {
          const finalText = this.AIoutputProcess(data.content || '')
          const tokenCount = data.usage?.total_tokens ?? null
          // skipScroll:true ── 占位 AI 在 push 时已经滚到位(adjustScroll(this.placeholderIndex - 1)),
          // stream 完成的 SET 不应再触发 adjustScroll 把 AI 顶到顶上去,
          // 否则会再次强行滚动覆盖用户当前查看位置。
          this.handleAiChange('SET', this.placeholderIndex, {
            type: 'LIST_TYPE_AI',
            text: finalText,
            dt: this.getFormattedDate(),
            hl: this.chosen_hldata,
            img: '',
            token_count: tokenCount,
          }, { skipScroll: true })
          this.waiting = false
          this.placeholderIndex = -1
          window.focus()
        })
        .catch(err => {
          if (this.placeholderIndex !== -1) {
            this.ai_res.splice(this.placeholderIndex, 1)
            this.placeholderIndex = -1
          }
          this.waiting = false
          console.error('AI 请求失败:', err)
        })

      this.handleAskAreaBlur(true)
    },
    callAIResponse() {
      //请求控制
      this.cancelTokenSource?this.cancelTokenSource.cancel('请求被用户取消'):null //强制停止请求
      this.cancelTokenSource = this.$axios.CancelToken.source();
      this.controller = new AbortController();

      // 【第三步】先 push 占位 assistant 气泡(text 留空,_thinking 由 placeholderIndex 触发)
      // token_count 必须在初始化时声明(stream 结束时 aiService 返回的
      // usage.total_tokens 才会正确写入并触发响应式更新)。
      const placeholder = {
        type:'LIST_TYPE_AI',
        text: '',
        dt: this.getFormattedDate(),
        hl: this.chosen_hldata,
        img: '',
        token_count: null,
      }
      this.ai_res.push(placeholder)
      this.placeholderIndex = this.ai_res.length - 1
      this.waiting = true

      // 【改动 4】push 占位后立即滚动置顶(头像浮现时就开始跳转)
      this.$nextTick(() => {
        this.UpdateWindowUI().then(() => {
          this.adjustScroll(this.placeholderIndex)
        })
      })

      // ── 发送请求(后端 LangGraph /ai/load) ──
      this.$axios.post('/ai/load', {
        pdf_fp: this.pdf_fp,
        chosen_text: this.chosen_text,
        added_prompt: this.chosen_add_text && this.chosen_add_text.length > 0
          ? this.chosen_add_text.join('\n')
          : '',
      }, {
        cancelToken: this.cancelTokenSource.token,
        headers: {
          'x-pdf-memorylist': btoa(unescape(encodeURIComponent(JSON.stringify(this.memorylist || [])))),
        },
      })
        .then(({ data }) => {
          const finalText = this.AIoutputProcess(data.content || '')
          placeholder.text = finalText
          placeholder.dt = this.getFormattedDate()
          placeholder.hl = this.chosen_hldata
          if (data.usage?.total_tokens != null) placeholder.token_count = data.usage.total_tokens
          // skipScroll:true ── 占位 AI 在 push 时已经 adjustScroll(placeholderIndex)
          // 滚到位,stream 完成的 SET 不应再触发滚动把 AI 顶上去。
          this.handleAiChange('SET', this.placeholderIndex, placeholder, { skipScroll: true })
          this.waiting = false
          this.placeholderIndex = -1
          this.cancelTokenSource = null
          window.focus()
        })
        .catch(err => {
          if (this.placeholderIndex !== -1) {
            this.ai_res.splice(this.placeholderIndex, 1)
            this.placeholderIndex = -1
            this.waiting = false
            this.UpdateWindowUI()
          }
          console.error('[AI Response] 失败:', err)
        })

      // 焦点失去
      this.handleAskAreaBlur(true)

    },
    getFormattedDate() {
      const now = new Date();
      
      const year = now.getFullYear(); // 2025
      const month = now.getMonth() + 1; // 月份从0开始，所以要加1
      const day = now.getDate(); // 30
      const hours = now.getHours(); // 14
      const minutes = now.getMinutes(); // 52
      
      return `${year}/${month}/${day} ${hours}:${minutes.toString().padStart(2, '0')}`;
    },
    adjustScroll(index=this.ai_res.length -1) {
      const container = this.$refs.render_container;
      const targetElement = container.children[index];

      if (targetElement) {
        this.current_index = index
        // 同步更新 group_cursor:找到 index 所在的组
        const gi = this.groupedChildrenHeightArr.findIndex(g => index >= g.start && index < g.end)
        if (gi !== -1) this.group_cursor = gi
        const targetRect = targetElement.getBoundingClientRect();
        const offsetTop = targetRect.top - this.containerRect.top;
        this.can_current_index_change = false
        this.smoothScrollBy(container, offsetTop, 250, index)
        setTimeout(() => {
          this.can_current_index_change=true
        }, 250);
      }

    },
    smoothScrollBy(container, offsetTop, duration = 500, targetIndex = -1) {
      const start = container.scrollTop;
      const startTime = performance.now();
      const easeOutQuad = (t) => t * (2 - t);
      let resolveFinal = null;
      const done = new Promise(r => { resolveFinal = r });

      function scrollStep(timestamp) {
        const elapsed = timestamp - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const currentOffset = start + offsetTop *easeOutQuad(progress) ;
        container.scrollTop = currentOffset;

        if (progress < 1) {
          requestAnimationFrame(scrollStep);
        } else {
          // 末帧再校准:异步资源(layout/<img>/字体)可能在动画期间
          // 撑高了目标 div,使真实 offsetTop 变大。补一次,确保置顶精准。
          // targetIndex: 滚动目标在 children 里的索引;默认 -1 表示
          // "最后一条" = children.length - 2(最后是 last_blank)。
          const idx = targetIndex >= 0 ? targetIndex : (container.children.length - 2)
          const targetEl = container.children[idx];
          if (targetEl) {
            const containerRectNow = container.getBoundingClientRect();
            const drift = targetEl.getBoundingClientRect().top - containerRectNow.top;
            if (Math.abs(drift) > 0.5) {
              container.scrollTop = container.scrollTop + drift;
            }
          }
          resolveFinal();
        }
      }

      requestAnimationFrame(scrollStep);
      return done;
    },
    highlightScroll(index) {
      const container = this.$refs.render_container;
      // 确保索引有效
      if (index < 0 || index >= container.children.length) {
        console.error('Invalid index:', index);
        return;
      }
      const targetElement = container.children[index];
      if (targetElement) {
        targetElement.style.scale = '1.01'
        targetElement.style.transition = 'scale 0.2s ease'
        setTimeout(() => {
          targetElement.style.scale = '1.00'
        }, 200);
      }
    },
    selectionProcess(text){
      // 第一步：处理英文换行连字符（"-"后跟换行符的情况）
      let processedText = text.replace(/-\r?\n|\r/g, '');
      // 第二步：将所有换行符替换为空格
      processedText = processedText.replace(/\r?\n|\r/g, ' ');
      // 第三步：将双斜杠 // 替换为空格
      processedText = processedText.replace(/\/\//g, ' ');
      // 第四步：删除文献标识符，如[1]、[2,3]、[4-6]等
      //processedText = processedText.replace(/\[\d+(?:,\s?\d+|-\d+)*\]/g, '');
      // 可选：将多个连续空格替换为单个空格
      processedText = processedText.replace(/\s+/g, ' ').trim();
      return processedText;
    },
    AIinputProcess(text) {
      return text
        // 替换A*和A\*为A-star
        .replace(/A\*/g, 'A-star')
        .replace(/A\\\*/g, 'A-star')

    },
    AIoutputProcess(text) {
      return text
        // // 处理加粗格式，确保前后有空格
        .replace(/\*\*([^*]+)\*\*/g, function (match, p1) {
          return ' **' + p1.trim() + '** ';
        })
        // // 处理缩进，将4个空格替换为2个空格
        // .replace(/^ {4}/gm, '  ');
    },


  },


}

</script>
<style scoped>
.search-input{
  width: 100%;
  height: 100%;
  padding-left: 20px;
  padding-right: 75px;
  font-size: 15px;
  outline: none;
  box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.25);
  border-radius: 25px;
  border: 1px solid lightgray;
  transition: all 0.2s ease;
}
.search-input:hover{
  border: 1px solid transparent;
}
.search-btn{
  right: 15px;
  
  font-size: 18px; 
  font-family:'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif; 
  color: #4870AC; 
  border: none; 
  background-color: transparent;
  cursor: pointer;
  position: absolute;

  transition: all 0.2s ease;
}
.search-btn:hover{
  color: #554acc;

}
.wait-outer-continer{
  display: flex; 
  flex-flow: column; 
  align-items: center; 
  justify-content: center; 
  background-color: transparent;
  cursor: pointer;

  position: absolute;
  width: 100%;
  height: 100%;
  
  background: rgba(255, 255, 255, 0.0.25);
  backdrop-filter: blur(8px);

  transition: all 0.2s ease;
}

.content{
  display: flex; 
  flex-flow: row;  
  height: 100vh;
  width: 100%;
  overflow: hidden;
  background-color: white;

}
.iframebox {
  flex: 1;
}

.div-index-block{
  display: block; 
  width: 100%;
  border-bottom: 1px solid white;
  cursor: pointer;
  transition: all 0.2s ease;

}
.div-index-block-active{
  display: block; 
  width: 100%;
  border-bottom: 1px solid white;
  cursor: pointer;
  transition: all 0.2s ease;
  scale: 1.05;
  box-shadow: 2px 2px 2px rgba(0, 0, 0, 0.5);
  border-color: transparent;

}
.div-index-block:hover{
  scale: 1.05;
  box-shadow: 2px 2px 2px rgba(0, 0, 0, 0.5);
  border-color: transparent;
}

.gpt-render{
  display: flex;
  flex-flow: column;
  overflow-y:scroll;
  overflow-x: hidden;
  height: 100%;
  flex: 1;  /*撑满右侧空间*/

  /* 底部渐变遮罩，配合悬浮输入框 */
  /* mask 已移除，输入框完整显示在底部 */

  /* WebKit浏览器滚动条样式 */
  &::-webkit-scrollbar {
    height: 4px !important; /* 水平滚动条高度 */
    width: 4px !important;  /* 垂直滚动条宽度（虽然这里用不到） */
  }
  &::-webkit-scrollbar-thumb {
    background: #8a8a8a7b; 
    border-radius: 2px;
  }
  &::-webkit-scrollbar-track {
    background: #f1f1f1;
  }
}
.sliderbox{
  display: flex;
  flex-flow: column;
  overflow: hidden;
  width: 100%;
  align-items: center;
  position: relative;
}

.btn-show-ori{
  font-size: 12px;
  background-color: transparent;
  cursor: pointer;
  /* 不要被选中文字 */
  caret-color: transparent;
  user-select: none;
  transition: all 0.2s ease;
}
.btn-show-ori:hover{
  color: #4e7bbf;
  text-shadow: 0.5px 0.5px 1px rgba(0, 0, 0, 0.25);
}

.btn-adding{
  font-size: 12px;
  text-align: center;
  color: #4870AC;
  border: 1px solid #4870AC;
  font-weight: bold;
  cursor: pointer;
  /* 不要被选中文字 */
  caret-color: transparent;
  user-select: none;
  padding-inline: 10px;
  border-radius: 15px;
  overflow: hidden; /* 隐藏溢出内容 */
  white-space: nowrap; /* 防止文字换行 */
  min-width: 0;

  transition: all 0.2s ease;

}
.btn-adding:hover{
  border: 1px solid transparent;
  background-color: lightgray;
  color: rgb(71, 71, 71);

}

.btn-show-img{
  font-size: 12px;
  background-color: transparent;
  cursor: pointer;
  /* 不要被选中文字 */
  caret-color: transparent;
  user-select: none;
  transition: all 0.2s ease;
}
.btn-show-img:hover{
  color: #6812b0;
  text-shadow: 0.5px 0.5px 1px rgba(0, 0, 0, 0.25);
}


.textarea-ori{
  font-size: 13px;
  padding: 5px;
  margin-bottom: 5px;
  border: 2px dashed #4870acb2;
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  resize: none;
  transition: all 0.2s ease;
}
.textarea-ask{
  width: 100%; 
  margin: 2px; 
  resize: none;
  outline: none;
  border: 1.5px solid #4870AC;
  border-radius: 5px;
  transition: all 0.2s ease;

  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  padding-left: 5px;
  padding-right: 5px;
}
.textarea-ask:hover{
  background-color: #eff6ff;
}
.lb-quote-num{
  position: absolute;
  left: 4px;
  top: -22px;
  color: #4870AC;
  font-weight: bold;
  border-left: #4870AC solid 5px;
  
  padding-inline: 5px;
  padding-block: 1px;
  font-size: 10px;
  text-align: center;

  background: rgba(255, 255, 255, 0.25);
  backdrop-filter: blur(1px);

  cursor: pointer;
  transition: all 0.2s ease;
  
}
.lb-quote-num:hover{
  color: grey !important;
  border-color: grey !important;
}

/* ─────────────────────────────────────────────────────────────────
   输入框视觉对齐 (第二步)
   - 默认 dashed 边框 + 灰底
   - focus → solid 蓝边框 + halo + 高度增加
   - waiting → 灰底禁用态
   ───────────────────────────────────────────────────────────────── */
.ask-area {
  position: absolute;
  bottom: 0;
  left: 12px;
  right: 12px;
  width: auto;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
}
.ask-input-wrap {
  position: relative;
  border: 1.5px dashed rgba(72, 112, 172, 0.5);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(4px);
  padding: 5px;
  height: 24px;
  bottom: 5px;
  max-height: 72px;
  overflow: hidden;
  transition: border-color 0.2s, background-color 0.2s,
              box-shadow 0.2s, height 0.2s, max-height 0.2s,
              overflow 0.2s;
  transform-origin: bottom center;
}
.ask-input-wrap--focused {
  border-color: #3b82f6;
  border-style: solid;
  background: rgba(255, 255, 255, 0.95);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
  height: 72px;
  max-height: 200px;
  overflow: visible;
  padding: 5px;
}
.ask-input-wrap--disabled {
  background: rgba(243, 244, 246, 0.85);
  border-color: #e5e7eb;
  opacity: 0.7;
}
.ask-input {
  width: 100%;
  border: 0;
  outline: 0;
  resize: none;
  background: transparent;
  font: inherit;
  font-size: 14px;
  line-height: 1.5;
  color: #111827;
  caret-color: #3b82f6;
  display: block;
  height: 24px;
  max-height: 180px;
  overflow-y: auto;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  box-sizing: border-box;
  scrollbar-width: thin;
  scrollbar-color: #d1d5db transparent;
}
/* 聚焦时 wrap 撑高到 72px,textarea 跟着把 height 撑满 wrap,
   否则 .ask-input 仍只有 24px,看上去"还是一行"。 */
.ask-input-wrap--focused .ask-input {
  height: 100%;
  max-height: 180px;
}
.ask-input::-webkit-scrollbar {
  width: 6px;
}
.ask-input::-webkit-scrollbar-thumb {
  background: #d1d5db;
  border-radius: 3px;
}
.ask-input::placeholder {
  color: #9ca3af;
}
</style>
