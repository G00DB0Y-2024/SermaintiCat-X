<template>
<div class="main-container">
    <div class="img-container" @click="handleImgClick">
        <img :src="loadImg" style="height: 100%; width: 100%; object-fit: contain;"/>
    </div>
    <input
    v-model="img_desc"
    style="
        border: none;
        text-align: center;
        margin-top: 4px;
        font-size: 13px;
        color: rgba(0, 0, 0, 0.75);
        outline: none;
        width: 100%;
        background: transparent;
    "
    :placeholder="`Fig. ${img_name.split('.')[0].split('_')[1]}`"
    @change="handleNameChange"
    >

    
</div>


</template>

<script>

export default{
    props:['img_name'],
    data(){
        return{
            img_desc:'',
        }


    },
    mounted(){
        this.$axios.post("/reqImgInfoGet", {imgname:this.img_name, base64:"", mode:""}).then(res=>{
            if(res.data){
                this.img_desc = res.data
            }
        })

    },
    methods:{
        handleImgClick(){
            window.open('http://localhost:8225/static/'+this.img_name, "_blank");

        },
        handleNameChange(){
            this.$axios.post("/reqImgInfoSet", {imgname:this.img_name, base64:this.img_desc, mode:""})
        }


    },
    computed:{

        loadImg(){
            return 'http://localhost:8225/static/'+this.img_name+`?t=${Date.now()}`
        },

    }

}

</script>

<style scoped>
/* 嵌入气泡内的图片:无边框、无固定高度,与气泡融为一体。
   注释输入框在图片下方,文字与气泡风格一致。 */
.img-container{
    display: block;
    width: 100%;
    border-radius: 5px;
    padding: 0;
    cursor: pointer;
    overflow: hidden;
    transition: box-shadow 0.2s ease;
}
.img-container:hover{
    box-shadow: 0 1px 6px rgba(0, 0, 0, 0.18);
}

.main-container{
    display: flex;
    flex-flow: column;
    align-items: center;
    justify-content: flex-start;
    /* 不再强制 150px 高度,让图片按原始比例自适应撑开,
       避免在气泡中留下多余空白 */
    padding-inline: 8px;
    margin-bottom: 6px;
    width: 100%;
}
</style>