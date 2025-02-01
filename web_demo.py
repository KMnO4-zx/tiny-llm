import json
import random
import numpy as np
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.generation.utils import GenerationConfig

st.set_page_config(page_title="K-Model-215M LLM")
st.title("K-Model-215M LLM")
st.caption("🚀 A streamlit chatbot powered by Self-LLM")
st.markdown("*K-Model-215M 是一个拿来练手的基于 Pytorch 实现的中文 Tiny-LLM*")
st.markdown("*Pretrain 阶段在 Seq-Monkey 10B token的中文语料，在 512 长度，4×A100 训练 24 小时*")
st.markdown("*SFT 阶段在 BelleGroup 350万条中文指令,4×A100 训练4小时，在此感谢 InternStudio 提供的算力支持！*")
st.markdown("*之前就对大模型的模型结构做过细致的剖析，但从没有实际上手从零训练过 LLM*")
st.markdown("*这次从零训练LLM，算是对自己的一个小小的突破（遇到了很多意料之外的问题，幸好都解决了）*")
st.markdown("***纸上得来终觉浅，绝知此事要躬行***")
st.markdown("*注：详细代码与试验记录保存在 Tiny-LLM: https://github.com/KMnO4-zx/tiny-llm*")

def clear_chat_messages():
    del st.session_state.messages

with st.sidebar:
    st.markdown("# K-Model-215M LLM")
    "[开源大模型食用指南 self-llm](https://github.com/datawhalechina/self-llm.git)"
    # 创建一个滑块，用于选择最大长度，范围在 0 到 8192 之间，默认值为 512（Qwen2.5 支持 128K 上下文，并能生成最多 8K tokens）
    st.sidebar.title("Settting")
    st.session_state.max_new_tokens = st.sidebar.slider("最大输入/生成长度", 128, 512, 512, step=1)
    st.session_state.temperature = st.sidebar.slider("temperature", 0.1, 1.2, 0.75, step=0.01)

    st.button("清空对话", on_click=clear_chat_messages)


model_id = "./k-model-215M/"

# 定义一个函数，用于获取模型和 tokenizer
@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0").eval()
    return tokenizer, model


tokenizer, model = get_model()

# 如果 session_state 中没有 "messages"，则创建一个包含默认消息的列表
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": "你是一个AI助手"}]

# 遍历 session_state 中的所有消息，并显示在聊天界面上
for i in range(1, len(st.session_state.messages)):
    msg = st.session_state.messages[i]
    st.chat_message(msg["role"]).write(msg["content"])

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():

    # 在聊天界面上显示用户的输入
    st.chat_message("user", avatar='🧑‍💻').write(prompt)

    # 将用户输入添加到 session_state 中的 messages 列表中
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 将对话输入模型，获得返回
    input_ids = tokenizer.apply_chat_template(st.session_state.messages,tokenize=False,add_generation_prompt=True)
    input_ids = tokenizer(input_ids).data['input_ids']
    x = (torch.tensor(input_ids, dtype=torch.long)[None, ...]).to(model.device)

    with torch.no_grad():
        y = model.generate(x, stop_id=tokenizer.eos_token_id, max_new_tokens=st.session_state.max_new_tokens, temperature=st.session_state.temperature)
        response = tokenizer.decode(y[0].tolist())

    # 将模型的输出添加到 session_state 中的 messages 列表中
    st.session_state.messages.append({"role": "assistant", "content": response})
    # 在聊天界面上显示模型的输出
    st.chat_message("assistant", avatar='🤖').write(response)
    # print(st.session_state) # 打印 session_state 调试

