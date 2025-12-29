# 📚 ASK MY DOCS

```text
    ___         __      __  ___              ____                
   /   |  _____/ /__   /  |/  /_  __        / __ \____  __________
  / /| | / ___/ //_/  / /|_/ / / / /       / / / / __ \/ ___/ ___/
 / ___ |(__  ) ,<    / /  / / /_/ /       / /_/ / /_/ / /__(__  ) 
/_/  |_/____/_/|_|  /_/  /_/\__, /        \____/\____/\___/____/  
                           /____/                                 
```

<div align="center">

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://python.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

**"Command your documents, summon the wisdom of AI."**
主宰你的文档，召唤 AI 的智慧。

[Installation](#installation) • [Usage](#usage) • [Features](#features) • [Tech Stack](#tech-stack)

</div>

---

## ⚡ What is AskMyDocs?

**AskMyDocs** 是一个基于 **LangChain** 构建的本地知识库问答系统。它不仅是一个简单的聊天机器人，更是你的专属私有文档管家。通过加载本地 PDF、TXT 等文档，它能让 LLM（如 Ollama 或 Gemini）在你的知识背景下提供精准、无幻觉的回答。

**让 AI 真正读懂你的本地资料。**

## 🚀 Features

- **📚 Local Knowledge Oracle**: 构建基于本地文档的向量数据库，实现 RAG (Retrieval-Augmented Generation)。
- **🤖 Hybrid Model Support**: 完美适配 `Ollama` 本地运行或 `Google Gemini` 高性能 API。
- **🛠️ Integrated Tooling**: 内置计算器等扩展工具，让 AI 能够处理复杂的逻辑运算。
- **💻 Dual-Mode Interface**: 同时提供极简 CLI 交互与直观的 Streamlit Web 界面。

## 📦 Installation

### 1. 克隆项目
```bash
git clone https://github.com/SaintFore/AskMyDocs.git
cd AskMyDocs
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量
创建 `.env` 文件并填入你的 API Key（如果使用 Gemini）：
```env
GOOGLE_API_KEY="your_google_api_key"
```

## 💻 Usage

### Web 模式 (推荐)
```bash
streamlit run app.py
```
访问 `http://localhost:8501`，上传文档并开始提问。

### CLI 模式
```bash
python cli.py --chunk-size 1000 --k 3
```

## 🛠️ Tech Stack

- **Framework**: LangChain
- **UI**: Streamlit
- **LLM Connectors**: Google Generative AI, Ollama
- **Vector Store**: FAISS (Local)

---

<div align="center">
Created with 🤖 by <a href="https://github.com/SaintFore">SaintFore</a>
</div>
