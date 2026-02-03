import streamlit as st
import os
from huggingface_hub import InferenceClient

# 设置页面配置
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="wide"
)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 侧边栏配置
with st.sidebar:
    st.title("⚙️ 配置")
    
    # Token 输入
    hf_token = st.text_input(
        "Hugging Face Token",
        type="password",
        value=os.environ.get("HF_TOKEN", ""),
        help="输入你的 Hugging Face Token"
    )
    
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    # 模型选择
    model_name = st.selectbox(
        "选择模型",
        ["gpt2", "microsoft/DialoGPT-medium", "meta-llama/Llama-2-7b-chat-hf"],
        index=0
    )
    
    # 清除历史记录按钮
    if st.button("🗑️ 清除对话历史"):
        st.session_state.messages = []
        st.rerun()

# 主界面
st.title("🤖 Chatbot")

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("输入你的消息..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 生成回复
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # 初始化客户端
                client = InferenceClient(token=hf_token if hf_token else None)
                
                # 构建消息历史
                messages_for_api = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ]
                
                # 调用 API
                if model_name in ["microsoft/DialoGPT-medium"]:
                    # 对话模型
                    response = client.chat_completion(
                        model=model_name,
                        messages=messages_for_api,
                        max_tokens=150
                    )
                    reply = response.choices[0].message.content
                else:
                    # 文本生成模型
                    # 构建提示词
                    prompt_text = "\n".join([
                        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                        for msg in messages_for_api[-5:]  # 只使用最近5条消息
                    ])
                    prompt_text += "\nAssistant:"
                    
                    response = client.text_generation(
                        prompt_text,
                        model=model_name,
                        max_new_tokens=150,
                        temperature=0.7
                    )
                    reply = response
                
                st.markdown(reply)
                
                # 添加助手回复到历史
                st.session_state.messages.append({"role": "assistant", "content": reply})
                
            except Exception as e:
                error_msg = f"❌ 错误: {str(e)}"
                st.error(error_msg)
                st.info("💡 提示：请检查你的 Token 是否正确，或者尝试更换模型")

