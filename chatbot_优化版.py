import streamlit as st
import os
import hashlib
import time
from huggingface_hub import InferenceClient
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
import json

# 设置页面配置
st.set_page_config(
    page_title="Chatbot (优化版)",
    page_icon="🤖",
    layout="wide"
)

# ============================================================================
# 优化1: 初始化所有会话状态（包括缓存和客户端）
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# 缓存：存储相同问题的回答，避免重复调用 API
if "cache" not in st.session_state:
    st.session_state.cache = {}

# 客户端：复用 InferenceClient，避免重复创建
if "client" not in st.session_state:
    st.session_state.client = None

# 客户端配置：用于检测 Token 或模型是否改变
if "client_config" not in st.session_state:
    st.session_state.client_config = {"token": None, "model": None}

# 上一次选择的模型：用于检测模型切换
if "previous_model" not in st.session_state:
    st.session_state.previous_model = None

# 速率限制：记录最后一次请求时间
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0

# 请求计数：统计请求次数（可选，用于监控）
if "request_count" not in st.session_state:
    st.session_state.request_count = 0

# 消息历史最大长度
MAX_MESSAGE_HISTORY = 20  # 最多保留 20 条消息
RATE_LIMIT_SECONDS = 2    # 速率限制：每 2 秒最多一次请求

# ============================================================================
# 辅助函数：获取或创建客户端（优化1: 客户端复用）
# ============================================================================

def get_client(token):
    """
    获取或创建 InferenceClient 客户端
    
    优化点：
    - 如果客户端已存在且 Token 未改变，直接复用
    - 如果 Token 改变，创建新客户端
    - 避免每次请求都创建新客户端，节省资源
    """
    # 检查是否需要创建新客户端
    if (st.session_state.client is None or 
        st.session_state.client_config["token"] != token):
        # Token 改变或客户端不存在，创建新客户端
        st.session_state.client = InferenceClient(token=token if token else None)
        st.session_state.client_config["token"] = token
        st.session_state.client_config["model"] = None  # 模型改变不影响客户端
    
    return st.session_state.client

# ============================================================================
# 辅助函数：生成缓存键（优化2: 添加缓存）
# ============================================================================

def get_cache_key(prompt, model_name, recent_messages):
    """
    生成缓存键
    
    参数：
    - prompt: 用户当前输入
    - model_name: 模型名称
    - recent_messages: 最近的对话历史（用于上下文）
    
    返回：MD5 哈希值作为缓存键
    """
    # 将提示词、模型名称和最近消息组合成字符串
    cache_string = f"{model_name}:{prompt}:{str(recent_messages)}"
    # 生成 MD5 哈希值
    return hashlib.md5(cache_string.encode()).hexdigest()

# ============================================================================
# 辅助函数：检查缓存（优化2: 添加缓存）
# ============================================================================

def get_cached_reply(cache_key):
    """
    从缓存中获取回复
    
    返回：缓存的回复，如果不存在则返回 None
    """
    return st.session_state.cache.get(cache_key)

def save_to_cache(cache_key, reply):
    """
    将回复保存到缓存
    
    优化：限制缓存大小，防止内存占用过大
    """
    # 如果缓存太大，清除最旧的条目（简单策略：保留最近 50 条）
    if len(st.session_state.cache) > 50:
        # 删除最旧的一条（字典的键是无序的，这里简化处理）
        oldest_key = next(iter(st.session_state.cache))
        del st.session_state.cache[oldest_key]
    
    st.session_state.cache[cache_key] = reply

# ============================================================================
# 辅助函数：限制消息历史（优化3: 限制消息历史）
# ============================================================================

def limit_message_history():
    """
    限制消息历史长度，防止无限增长
    
    优化点：
    - 只保留最近的 N 条消息
    - 防止内存占用过大
    - 防止 API 调用时 token 过多
    """
    if len(st.session_state.messages) > MAX_MESSAGE_HISTORY:
        # 保留最近的 MAX_MESSAGE_HISTORY 条消息
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGE_HISTORY:]

# ============================================================================
# 辅助函数：速率限制（防止 API 滥用）
# ============================================================================

def check_rate_limit():
    """
    检查速率限制
    
    返回：True 表示可以继续，False 表示需要等待
    """
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_request_time
    
    if time_since_last < RATE_LIMIT_SECONDS:
        return False, RATE_LIMIT_SECONDS - time_since_last
    
    st.session_state.last_request_time = current_time
    return True, 0

# ============================================================================
# 辅助函数：备用 API 调用方法（使用 requests 直接调用）
# ============================================================================

def call_api_direct(token, model_name, messages_for_api=None, prompt_text=None):
    """
    直接使用 requests 调用 Hugging Face API（备用方法）
    
    这个方法可以避免 InferenceClient 的响应解析问题
    支持文本生成和对话模型
    """
    # 判断是对话模型还是文本生成模型
    chat_models = [
        "moonshotai/Kimi-K2-Thinking",
        "deepseek-ai/DeepSeek-V3.2",
        "meta-llama/Llama-3.1-8B-Instruct"
    ]
    is_chat_model = model_name in chat_models or "chat" in model_name.lower() or "llama" in model_name.lower() or "kimi" in model_name.lower() or "deepseek" in model_name.lower() or "instruct" in model_name.lower()
    
    # 使用新的 API 端点（router.huggingface.co）
    # 旧的 api-inference.huggingface.co 已不再支持
    
    if is_chat_model and messages_for_api:
        # 对话模型：尝试使用新的 router API
        # 新 API 可能需要不同的格式
        url = f"https://router.huggingface.co/models/{model_name}"
        headers = {
            "Authorization": f"Bearer {token}" if token else None,
            "Content-Type": "application/json"
        }
        
        # 尝试 OpenAI 兼容格式
        payload = {
            "model": model_name,
            "messages": messages_for_api,
            "max_tokens": 150,
            "temperature": 0.7
        }
    else:
        # 文本生成模型：使用 text generation API
        if not prompt_text:
            # 如果没有提供 prompt_text，从 messages 构建
            if messages_for_api:
                prompt_text = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in messages_for_api[-5:]
                ])
                prompt_text += "\nAssistant:"
            else:
                raise ValueError("需要提供 prompt_text 或 messages_for_api")
        
        # 使用新的 router API
        url = f"https://router.huggingface.co/models/{model_name}"
        headers = {
            "Authorization": f"Bearer {token}" if token else None,
            "Content-Type": "application/json"
        }
        
        # 尝试新的 API 格式
        payload = {
            "inputs": prompt_text,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
    
    # 移除 None 值
    headers = {k: v for k, v in headers.items() if v is not None}
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            # 处理不同的响应格式
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    generated = result[0]["generated_text"]
                    # 如果是对话模型，可能需要提取回复部分
                    if is_chat_model and "Assistant:" in generated:
                        return generated.split("Assistant:")[-1].strip()
                    return generated
                elif "text" in result[0]:
                    return result[0]["text"]
                elif "message" in result[0]:
                    return result[0]["message"].get("content", str(result[0]))
            elif isinstance(result, dict):
                if "generated_text" in result:
                    generated = result["generated_text"]
                    if is_chat_model and "Assistant:" in generated:
                        return generated.split("Assistant:")[-1].strip()
                    return generated
                elif "text" in result:
                    return result["text"]
                elif "choices" in result and len(result["choices"]) > 0:
                    # OpenAI 格式的响应
                    if "message" in result["choices"][0]:
                        return result["choices"][0]["message"].get("content", "")
                    return str(result["choices"][0])
            
            # 如果都不匹配，返回字符串表示
            return str(result)
        elif response.status_code == 410:
            # API 端点已废弃
            error_msg = "API 端点已更新，正在尝试使用 InferenceClient..."
            raise Exception(f"API 端点已废弃: {error_msg}")
        elif response.status_code == 503:
            error_msg = response.json().get("error", "模型正在加载")
            raise Exception(f"模型正在加载，请稍后重试: {error_msg}")
        else:
            error_text = response.text[:500] if response.text else "未知错误"
            raise Exception(f"API 调用失败 ({response.status_code}): {error_text}")
    except requests.exceptions.Timeout:
        raise Exception("请求超时，请稍后重试")
    except requests.exceptions.ConnectionError:
        raise Exception("网络连接失败，请检查网络")
    except Exception as e:
        raise e

# ============================================================================
# 辅助函数：同步 API 调用（用于异步包装）
# ============================================================================

def call_api_sync(client, model_name, messages_for_api, prompt_text):
    """
    同步调用 API（在后台线程中运行）
    
    这个函数会在 ThreadPoolExecutor 中执行，不会阻塞主线程
    """
    try:
        # 对话模型列表
        chat_models_list = [
            "moonshotai/Kimi-K2-Thinking",
            "deepseek-ai/DeepSeek-V3.2",
            "meta-llama/Llama-3.1-8B-Instruct"
        ]
        
        if model_name in chat_models_list:
            # 对话模型
            response = client.chat_completion(
                model=model_name,
                messages=messages_for_api,
                max_tokens=150
            )
            # 安全地访问响应内容
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message'):
                    return response.choices[0].message.content
                else:
                    return str(response.choices[0])
            else:
                return str(response)
        else:
            # 文本生成模型
            response = client.text_generation(
                prompt_text,
                model=model_name,
                max_new_tokens=150,
                temperature=0.7,
                return_full_text=False  # 只返回新生成的文本
            )
            # text_generation 可能返回字符串或生成器
            if isinstance(response, str):
                return response
            elif hasattr(response, '__iter__'):
                # 如果是生成器或迭代器，转换为字符串
                try:
                    return ''.join(response) if not isinstance(response, str) else response
                except StopIteration:
                    # 处理 StopIteration（迭代器耗尽）
                    return str(response) if response else "生成失败：响应为空"
            else:
                return str(response)
    except StopIteration as e:
        # 专门处理 StopIteration 错误
        raise Exception(f"API 响应解析错误（StopIteration）: 可能是响应格式不符合预期。原始错误: {str(e)}")
    except Exception as e:
        raise e

# ============================================================================
# 辅助函数：异步 API 调用（优化4: 异步调用）
# ============================================================================

def call_api_async(client, model_name, messages_for_api, prompt_text):
    """
    异步调用 API
    
    优化点：
    - 使用 ThreadPoolExecutor 在后台线程执行同步 API 调用
    - 不阻塞 Streamlit 主线程
    - 用户界面保持响应
    
    注意：Streamlit 本身不支持真正的异步，这里使用线程池模拟异步
    """
    # 创建线程池执行器
    executor = ThreadPoolExecutor(max_workers=1)
    
    # 提交任务到线程池
    future = executor.submit(
        call_api_sync,
        client,
        model_name,
        messages_for_api,
        prompt_text
    )
    
    # 等待结果（这里仍然会阻塞，但在后台线程中执行）
    # 在实际应用中，可以使用 st.empty() 和轮询来显示进度
    try:
        result = future.result(timeout=60)  # 60 秒超时
        return result
    except Exception as e:
        raise e
    finally:
        executor.shutdown(wait=False)

# ============================================================================
# 主界面：侧边栏配置
# ============================================================================

with st.sidebar:
    st.title("⚙️ 配置")
    
    # Token 输入
    hf_token = st.text_input(
        "Hugging Face Token",
        type="password",
        value=os.environ.get("HF_TOKEN", ""),
        help="输入你的 Hugging Face Token（以 hf_ 开头）"
    )
    
    # Token 状态指示
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        if hf_token.startswith("hf_"):
            st.success("✅ Token 格式正确")
        else:
            st.error("❌ Token 格式错误（应以 hf_ 开头）")
    else:
        st.warning("⚠️ 未设置 Token")
        with st.expander("📖 如何获取 Token"):
            st.write("1. 访问 https://huggingface.co/settings/tokens")
            st.write("2. 登录你的 Hugging Face 账号")
            st.write("3. 点击 'New token' 创建新 Token")
            st.write("4. 选择 'Read' 权限（免费模型只需要读权限）")
            st.write("5. 复制生成的 Token（以 `hf_` 开头）")
            st.write("6. 粘贴到上面的输入框")
    
    # 模型选择
    model_name = st.selectbox(
        "选择模型",
        [
            "moonshotai/Kimi-K2-Thinking",  # Moonshot AI 的 Kimi 模型（推荐，中文友好）
            "deepseek-ai/DeepSeek-V3.2",  # DeepSeek V3.2 模型（推荐，性能强）
            "meta-llama/Llama-3.1-8B-Instruct",  # Llama 3.1（8B，热门）
        ],
        index=0,
        help="选择要使用的 AI 模型。推荐使用 Kimi 或 DeepSeek（中文友好）\n注意：切换模型会自动清除对话历史"
    )
    
    # 检测模型切换，自动清除对话历史
    if st.session_state.previous_model is not None and st.session_state.previous_model != model_name:
        # 模型已切换，清除对话历史
        st.session_state.messages = []
        st.session_state.cache = {}  # 同时清除缓存，因为缓存可能包含旧模型的回复
        st.info(f"🔄 已切换到 {model_name.split('/')[-1]}，对话历史已清除")
    
    # 更新上一次的模型
    st.session_state.previous_model = model_name
    
    # 高级设置
    with st.expander("⚙️ 高级设置"):
        use_cache = st.checkbox("启用缓存", value=True, help="缓存相同问题的回答，提高响应速度")
        # 默认使用 InferenceClient（已更新支持新端点），备用方法作为备选
        use_direct_api = st.checkbox("使用备用 API 方法", value=False, 
                                    help="如果遇到 API 错误，可以尝试启用此选项（默认使用 InferenceClient）")
        use_async = st.checkbox("异步调用", value=False, help="使用异步调用（实验性功能）")
        max_history = st.slider("最大消息历史", min_value=5, max_value=50, value=MAX_MESSAGE_HISTORY, 
                                help="限制保存的消息数量")
        
        # 更新全局变量
        MAX_MESSAGE_HISTORY = max_history
    
    # 统计信息
    st.divider()
    st.caption("📊 统计信息")
    st.caption(f"总请求数: {st.session_state.request_count}")
    st.caption(f"缓存条目: {len(st.session_state.cache)}")
    st.caption(f"当前消息数: {len(st.session_state.messages)}")
    
    # 清除按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清除对话历史"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("🗑️ 清除缓存"):
            st.session_state.cache = {}
            st.success("缓存已清除")

# ============================================================================
# 主界面：显示历史消息
# ============================================================================

st.title("🤖 Chatbot (优化版)")

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================================
# 主界面：处理用户输入
# ============================================================================

if prompt := st.chat_input("输入你的消息..."):
    # 输入验证
    if len(prompt) > 1000:
        st.error("❌ 输入过长，请限制在 1000 字符以内")
        st.stop()
    
    # 速率限制检查
    can_proceed, wait_time = check_rate_limit()
    if not can_proceed:
        st.warning(f"⏳ 请求过于频繁，请等待 {wait_time:.1f} 秒后再试")
        st.stop()
    
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 限制消息历史长度
    limit_message_history()
    
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 生成回复
    with st.chat_message("assistant"):
        # 先检查 Token
        if not hf_token:
            st.error("❌ **未设置 Token**")
            st.warning("请在侧边栏输入你的 Hugging Face Token")
            st.info("**如何获取 Token：**")
            st.write("1. 访问 https://huggingface.co/settings/tokens")
            st.write("2. 点击 'New token' 创建新 Token")
            st.write("3. 复制 Token（以 `hf_` 开头）")
            st.write("4. 粘贴到侧边栏的 Token 输入框")
            st.stop()
        
        # 检查 Token 格式
        if not hf_token.startswith("hf_"):
            st.error("❌ **Token 格式错误**")
            st.warning(f"Token 应该以 `hf_` 开头")
            st.info(f"当前输入：`{hf_token[:20]}...`（已隐藏）")
            st.info("请检查 Token 是否正确，或前往 https://huggingface.co/settings/tokens 重新生成")
            st.stop()
        
        with st.spinner("思考中..."):
            try:
                # 获取客户端（优化1: 客户端复用）
                client = get_client(hf_token)
                
                # 构建消息历史
                messages_for_api = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ]
                
                # 获取最近的消息（用于缓存键）
                recent_messages = messages_for_api[-5:] if len(messages_for_api) > 5 else messages_for_api
                
                # 检查缓存（优化2: 添加缓存）
                cache_key = get_cache_key(prompt, model_name, recent_messages) if use_cache else None
                cached_reply = get_cached_reply(cache_key) if cache_key else None
                
                if cached_reply:
                    # 使用缓存的回复
                    reply = cached_reply
                    st.info("💡 使用缓存结果")
                else:
                    # 调用 API
                    # 构建提示词（用于文本生成模型）
                    # 对话模型列表
                    chat_models_list = [
                        "moonshotai/Kimi-K2-Thinking",
                        "deepseek-ai/DeepSeek-V3.2",
                        "zai-org/GLM-4.7-Flash",
                        "zai-org/GLM-4.7",
                        "meta-llama/Llama-3.1-8B-Instruct",
                        "openai/gpt-oss-20b",
                        "openai/gpt-oss-120b",
                        "MiniMaxAI/MiniMax-M2.1"
                    ]
                    
                    if model_name not in chat_models_list:
                        prompt_text = "\n".join([
                            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                            for msg in messages_for_api[-5:]
                        ])
                        prompt_text += "\nAssistant:"
                    else:
                        prompt_text = None
                    
                    # 选择调用方式
                    # 默认使用 InferenceClient（已更新支持新端点），备用方法作为备选
                    if use_direct_api:
                        # 使用备用方法（直接 requests 调用）
                        reply = call_api_direct(hf_token, model_name, messages_for_api, prompt_text)
                    elif use_async:
                        # 异步调用
                        reply = call_api_async(client, model_name, messages_for_api, prompt_text)
                    else:
                        # 同步调用（原有方式）
                        # 对话模型列表已在上面定义，这里复用
                        if model_name in chat_models_list:
                            # 对话模型
                            response = client.chat_completion(
                                model=model_name,
                                messages=messages_for_api,
                                max_tokens=150
                            )
                            # 安全地访问响应内容
                            if hasattr(response, 'choices') and len(response.choices) > 0:
                                if hasattr(response.choices[0], 'message'):
                                    reply = response.choices[0].message.content
                                else:
                                    reply = str(response.choices[0])
                            else:
                                reply = str(response)
                        else:
                            # 文本生成模型
                            response = client.text_generation(
                                prompt_text,
                                model=model_name,
                                max_new_tokens=150,
                                temperature=0.7,
                                return_full_text=False  # 只返回新生成的文本，不包括输入
                            )
                            # text_generation 可能返回字符串或生成器
                            if isinstance(response, str):
                                reply = response
                            elif hasattr(response, '__iter__'):
                                # 如果是生成器或迭代器，转换为字符串
                                try:
                                    reply = ''.join(response) if not isinstance(response, str) else response
                                except StopIteration:
                                    # 处理 StopIteration（迭代器耗尽）
                                    reply = str(response) if response else "生成失败：响应为空"
                            else:
                                reply = str(response)
                    
                    # 保存到缓存（优化2: 添加缓存）
                    if cache_key and use_cache:
                        save_to_cache(cache_key, reply)
                    
                    # 更新请求计数
                    st.session_state.request_count += 1
                
                # 显示回复
                st.markdown(reply)
                
                # 添加助手回复到历史
                st.session_state.messages.append({"role": "assistant", "content": reply})
                
                # 再次限制消息历史（添加新消息后）
                limit_message_history()
                
            except StopIteration as e:
                # 专门处理 StopIteration 错误，自动尝试备用方法
                st.warning("⚠️ **检测到响应解析错误，正在尝试备用方法...**")
                
                try:
                    # 自动切换到备用 API 方法重试
                    # 对话模型列表
                    chat_models_list_retry = [
                        "moonshotai/Kimi-K2-Thinking",
                        "deepseek-ai/DeepSeek-V3.2",
                        "meta-llama/Llama-3.1-8B-Instruct"
                    ]
                    if model_name not in chat_models_list_retry:
                        prompt_text_retry = "\n".join([
                            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                            for msg in messages_for_api[-5:]
                        ])
                        prompt_text_retry += "\nAssistant:"
                    else:
                        prompt_text_retry = None
                    
                    reply = call_api_direct(hf_token, model_name, messages_for_api, prompt_text_retry)
                    st.success("✅ **已使用备用方法成功获取回复**")
                    
                    # 显示回复
                    st.markdown(reply)
                    
                    # 添加助手回复到历史
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    
                    # 限制消息历史
                    limit_message_history()
                    
                    # 更新请求计数
                    st.session_state.request_count += 1
                    
                except Exception as retry_error:
                    # 备用方法也失败了
                    st.error("🔄 **响应解析错误 (StopIteration)**")
                    st.warning("**问题：** API 返回的响应格式不符合预期")
                    st.info("**可能的原因：**")
                    st.write("1. Hugging Face API 响应格式变化")
                    st.write("2. 模型返回了特殊格式的响应")
                    st.write("3. API 版本不兼容")
                    st.write("4. 模型可能需要特殊权限或配置")
                    st.info("**解决方案：**")
                    st.write("1. ✅ 已自动尝试备用方法（失败）")
                    st.write("2. 尝试切换到其他模型（如 `moonshotai/Kimi-K2-Thinking` 或 `deepseek-ai/DeepSeek-V3.2`）")
                    st.write("3. 检查 Hugging Face API 文档是否有更新")
                    st.write("4. 更新 `huggingface_hub` 库：`pip install --upgrade huggingface_hub`")
                    st.write("5. 某些模型（如 Llama）可能需要申请访问权限")
                    
                    with st.expander("🔍 查看详细错误信息"):
                        st.code(f"原始错误类型: StopIteration\n原始错误信息: {str(e)}\n重试错误类型: {type(retry_error).__name__}\n重试错误信息: {str(retry_error)}\n模型: {model_name}\nToken 已设置: {'是' if hf_token else '否'}")
                    
                    # 记录错误日志
                    if "error_log" not in st.session_state:
                        st.session_state.error_log = []
                    st.session_state.error_log.append({
                        "time": time.time(),
                        "error_type": "StopIteration",
                        "error": str(e),
                        "retry_error": str(retry_error),
                        "model": model_name
                    })
                
            except Exception as e:
                # 详细的错误处理和诊断
                error_type = type(e).__name__
                error_str = str(e)
                
                # 检查是否是 HTTP 错误
                if "HTTPError" in error_type or "401" in error_str or "Unauthorized" in error_str:
                    st.error("🔐 **认证错误**")
                    st.warning("**可能的原因：**")
                    st.write("1. ❌ Token 未输入或为空")
                    st.write("2. ❌ Token 格式错误（应该以 `hf_` 开头）")
                    st.write("3. ❌ Token 已过期或无效")
                    st.write("4. ❌ Token 没有访问该模型的权限")
                    
                    # 检查 Token 状态
                    if not hf_token:
                        st.error("⚠️ **当前状态：未检测到 Token**")
                        st.info("请在侧边栏输入你的 Hugging Face Token")
                    else:
                        # 检查 Token 格式
                        if not hf_token.startswith("hf_"):
                            st.error(f"⚠️ **Token 格式错误**")
                            st.info(f"Token 应该以 `hf_` 开头，当前：`{hf_token[:10]}...`")
                        else:
                            st.error(f"⚠️ **Token 可能无效**")
                            st.info("请检查 Token 是否正确，或前往 https://huggingface.co/settings/tokens 重新生成")
                
                elif "404" in error_str or "Not Found" in error_str:
                    st.error("🔍 **模型未找到**")
                    st.warning(f"**问题：** 模型 `{model_name}` 不存在或无法访问")
                    st.info("**解决方案：**")
                    st.write("1. 检查模型名称是否正确")
                    st.write("2. 尝试切换到其他模型（如 `moonshotai/Kimi-K2-Thinking`）")
                    st.write("3. 某些模型可能需要特定的 Token 权限")
                
                elif "503" in error_str or "loading" in error_str.lower():
                    st.warning("⏳ **模型正在加载**")
                    st.info("Hugging Face 服务器正在加载模型，请稍等片刻后重试")
                    st.info("💡 提示：免费模型首次调用需要加载时间，通常需要 10-30 秒")
                
                elif "timeout" in error_str.lower() or "Timeout" in error_type:
                    st.error("⏱️ **请求超时**")
                    st.warning("API 调用超时，可能是网络问题或服务器繁忙")
                    st.info("**解决方案：**")
                    st.write("1. 检查网络连接")
                    st.write("2. 稍后重试")
                    st.write("3. 尝试使用异步调用（侧边栏 → 高级设置）")
                
                elif "Connection" in error_type or "连接" in error_str:
                    st.error("🌐 **网络连接错误**")
                    st.warning("无法连接到 Hugging Face API")
                    st.info("**解决方案：**")
                    st.write("1. 检查网络连接")
                    st.write("2. 检查防火墙设置")
                    st.write("3. 如果在中国大陆，可能需要使用代理")
                
                elif "410" in error_str or "no longer supported" in error_str.lower() or "router.huggingface.co" in error_str.lower():
                    st.error("🔄 **API 端点已更新**")
                    st.warning("**问题：** Hugging Face API 端点已更改")
                    st.info("**解决方案：**")
                    st.write("1. ✅ **已更新代码使用新端点**")
                    st.write("2. 确保 `huggingface_hub` 库是最新版本：")
                    st.code("pip install --upgrade huggingface_hub", language="bash")
                    st.write("3. 刷新页面，代码已默认使用 InferenceClient（支持新端点）")
                    st.write("4. 如果问题仍然存在，尝试取消勾选 '使用备用 API 方法'")
                    
                    with st.expander("🔍 查看详细错误信息"):
                        st.code(f"错误类型: {error_type}\n错误信息: {error_str}\n模型: {model_name}\nToken 已设置: {'是' if hf_token else '否'}")
                        st.info("💡 InferenceClient 应该已经支持新端点，请刷新页面重试")
                
                else:
                    # 其他未知错误
                    st.error(f"❌ **错误类型：** {error_type}")
                    st.error(f"**错误信息：** {error_str}")
                    st.info("💡 提示：请检查你的 Token 是否正确，或者尝试更换模型")
                
                # 显示详细错误信息（可展开）
                with st.expander("🔍 查看详细错误信息"):
                    st.code(f"错误类型: {error_type}\n错误信息: {error_str}\n模型: {model_name}\nToken 已设置: {'是' if hf_token else '否'}")
                
                # 记录错误日志
                if "error_log" not in st.session_state:
                    st.session_state.error_log = []
                st.session_state.error_log.append({
                    "time": time.time(),
                    "error_type": error_type,
                    "error": error_str,
                    "model": model_name
                })

