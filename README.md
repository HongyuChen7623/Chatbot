# 🤖 Chatbot - 基于 Hugging Face 的智能对话机器人

一个使用 Streamlit 和 Hugging Face API 构建的智能对话机器人，支持多种 AI 模型，具有缓存、客户端复用等优化功能。

## ✨ 功能特性

* 🎯 **多模型支持**：支持 Kimi、DeepSeek、Llama 等多种模型

* 💾 **智能缓存**：缓存相同问题的回答，提高响应速度

* 🔄 **客户端复用**：优化 API 调用，减少资源消耗

* 📊 **统计信息**：实时显示请求数、缓存条目等统计

* ⚡ **性能优化**：消息历史限制、速率限制、输入验证

* 🎨 **友好界面**：简洁美观的 Streamlit 界面

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install streamlit huggingface_hub requests
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 Hugging Face Token：

```
HF_TOKEN=your_huggingface_token_here
```

**如何获取 Token：**

1. 访问 https://huggingface.co/settings/tokens

2. 点击 "New token" 创建新 Token

3. 复制 Token（以 `hf_` 开头）

4. 粘贴到 `.env` 文件中

### 3. 运行应用

```bash
streamlit run chatbot_优化版.py
```

浏览器会自动打开，开始使用！

## 📖 使用方法

1. **输入 Token**：在侧边栏输入你的 Hugging Face Token（如果 `.env` 中已配置，会自动读取）

2. **选择模型**：在侧边栏选择要使用的 AI 模型

3. **开始对话**：在输入框输入消息，按回车发送

4. **查看统计**：侧边栏底部显示请求数、缓存条目等统计信息

## ⚙️ 高级设置

在侧边栏的"高级设置"中可以：

* **启用缓存**：缓存相同问题的回答（默认开启）

* **使用备用 API 方法**：如果遇到 API 错误可以尝试（默认关闭）

* **异步调用**：使用异步调用（实验性功能）

* **最大消息历史**：限制保存的消息数量（默认 20 条）

## 🛠️ 技术栈

* **前端框架**：Streamlit

* **AI 服务**：Hugging Face Inference API

* **Python 版本**：3.8+

## 📁 项目结构

```
.
├── chatbot_优化版.py          # 优化版主程序（推荐使用）
├── chatbot.py                 # 基础版本
├── chatbot_详细注释版.py      # 带详细注释的版本（学习用）
├── .env.example              # 环境变量模板
├── .gitignore                # Git 忽略文件
├── README.md                 # 本文件
└── 其他文档/
    ├── 技术解析-面试问答.md   # 技术详解
    ├── 优化说明.md            # 优化说明
    └── 自建Bot优势总结.md     # 优势对比
```

## 🔧 常见问题

### Q: Token 在哪里获取？

A: 访问 https://huggingface.co/settings/tokens，创建新 Token。

### Q: 遇到 "StopIteration" 错误怎么办？

A: 在侧边栏"高级设置"中启用"使用备用 API 方法"。

### Q: 如何清除对话历史？

A: 点击侧边栏的"清除对话历史"按钮。

### Q: 如何清除缓存？

A: 点击侧边栏的"清除缓存"按钮。

## 📝 注意事项

* ⚠️ **Token 是敏感信息**，不要分享给他人

* 💡 某些模型（如 Llama）可能需要申请访问权限

* 💡 免费模型首次调用可能需要加载时间（10-30秒）

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

* [Streamlit](https://streamlit.io/) - 快速构建 Web 应用

* [Hugging Face](https://huggingface.co/) - 提供 AI 模型和 API

**开始使用：** `streamlit run chatbot_优化版.py` 🚀
