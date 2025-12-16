# 🤖 AskMyDocs

AskMyDocs 是一个基于 `LangChain` 的本地知识库问答系统。它能够加载本地文档，并使用大型语言模型来回答你的问题。

## ✨ 主要功能

- **本地知识库**: 使用你自己的文档作为知识库，让 AI 回答与你的文档相关的问题。
- **多种接口**: 同时提供了 Web 界面和命令行界面，满足不同场景下的使用需求。
- **高度可配置**: 支持自定义 `chunk size`, `chunk overlap`, `K` 等参数，以优化问答效果。
- **多模型支持**: 支持 `Ollama` 和 `Google Generative AI` 等多种大型语言模型。
- **工具扩展**: 集成了计算器等工具，让 AI 在需要时可以进行计算。

## 🛠️ 如何使用

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置 API Keys

在项目根目录下创建一个 `.env` 文件，并添加你的 API keys：

```
GOOGLE_API_KEY="your_google_api_key"
```

### 3. 准备知识库

将你的知识库文档（例如，一个 `.txt` 文件）放在 `books` 目录下。默认情况下，程序会加载 `books/Learning.txt` 文件。

### 4. 运行 Web 界面

```bash
streamlit run app.py
```

然后，在浏览器中打开 `http://localhost:8501`。

### 5. 运行命令行界面

```bash
python cli.py
```

## ⚙️ 配置

你可以在 Web 界面的侧边栏中配置 `chunk size`, `chunk overlap` 和 `K` 等参数。

在命令行界面中，你可以通过命令行参数来配置这些参数：

```bash
python cli.py --chunk-size 1000 --chunk-overlap 100 --k 3
```

## 🤝 贡献

欢迎任何形式的贡献！如果你有任何建议或问题，请随时提出 Issue。
