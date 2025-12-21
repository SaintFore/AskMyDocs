from dataclasses import dataclass
import os
import re
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.prompts import (
    MessagesPlaceholder,
    ChatPromptTemplate,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader

# google
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI

# OllamaLLM
# from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


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


def init_agent(tools, config):
    llm = ChatOllama(
        base_url=config.base_url, model="ministral-3:3b"
    )  # 绑定工具不能用ollamaLLM 但是可以使用ChatOllama
    # llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")

    llm_bind_tools = llm.bind_tools(
        tools=tools
    )  # 后面的create_tool_calling_agent会自动绑定工具的，不过现在已经没有这个方法了。
    tool_node = ToolNode(tools)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个能查阅书籍，遇到计算题还能使用计算器的人工智能。",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
        ]
    )

    def call_llm(state: AgentState):
        chain = prompt | llm_bind_tools
        response = chain.invoke({"chat_history": state["messages"]})
        # 使用了add_messages，会自动append
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        # if hasattr(last, "tool_calls") and last.tool_calls:
        #     return "tools"
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)

    graph.add_node("llm", call_llm)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue)
    graph.add_edge("tools", "llm")

    return graph.compile()


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
