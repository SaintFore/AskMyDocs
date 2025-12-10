from dataclasses import dataclass
import os
import re
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.prompts import (
    MessagesPlaceholder,
    ChatPromptTemplate,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import tool, create_tool_calling_agent, AgentExecutor
from langchain_community.document_loaders import TextLoader

# google
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# OllamaLLM
# from langchain_ollama import OllamaLLM
# from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings


@dataclass
class AppConfig:
    chunk_size: int
    chunk_overlap: int
    k: int = 5
    db_path: str = "./chroma_db/"
    base_url: str = "http://192.168.31.60:11434"


def init_vectorstore(config: AppConfig) -> Chroma:

    # embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    embeddings_ollama = OllamaEmbeddings(
        base_url=config.base_url, model="embeddinggemma:300m"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", " "],
    )
    os.makedirs(config.db_path, exist_ok=True)
    vectorstore = Chroma(
        persist_directory=config.db_path, embedding_function=embeddings_ollama
    )

    if vectorstore._collection.count() == 0:
        print("数据库为空")
        loader = TextLoader("./books/Learning.txt", encoding="utf8")
        docs = loader.load()

        split = text_splitter.split_documents(documents=docs)
        split_clean = [
            Document(
                page_content=re.sub(r"\s+", " ", chunk.page_content),
                metadata=chunk.metadata,
            )
            for chunk in split
        ]
        split_clean = split_clean
        for single in split_clean[:50]:
            print(single.page_content)
            print("*" * 50)
            print(single.metadata)
            print("-" * 50)
        print(f"正在存入{len(split_clean)}个片段")
        vectorstore.add_documents(split_clean)
    else:
        print(f"已经有数据库: {vectorstore._collection}")
    return vectorstore


def init_agent(tools, config) -> AgentExecutor:
    # llm = ChatOllama(
    #     base_url=config.base_url, model="gemma3:4b"
    # )  # 绑定工具不能用ollamaLLM 但是可以使用ChatOllama
    llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")

    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个不仅能查书籍，遇到计算题还能使用计算器的人工智能。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            (
                "placeholder",
                "{agent_scratchpad}",
            ),  # 关键：给 AI 留出思考和调用工具的空间
        ]
    )
    # llm_bind_tools = llm.bind_tools(tools=tools)  # 后面的create_tool_calling_agent会自动绑定工具的
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor


def create_tools(retriever) -> list:

    @tool
    def search_book(query: str) -> str:
        """只有在需要的时候才查阅书籍，输入是查询的问题"""
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])

    @tool
    def calculate_multiply(a: int, b: int) -> int:
        """计算两个数字的乘积"""
        return a * b

    tools = [search_book, calculate_multiply]
    return tools
