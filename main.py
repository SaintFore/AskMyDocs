from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain_core.prompts import (
    MessagesPlaceholder,
    ChatPromptTemplate,
)
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import AIMessage, HumanMessage

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from langchain.agents import tool, create_tool_calling_agent, AgentExecutor


import re

load_dotenv()
DB_PATH = "./chroma_db"
os.makedirs(DB_PATH, exist_ok=True)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " "],
)
base_url = "http://192.168.31.60:11434"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
embeddings_ollama = OllamaEmbeddings(base_url=base_url, model="embeddinggemma:300m")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_ollama = OllamaLLM(base_url=base_url, model="gemma3:4b")

vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings_ollama)
print(vectorstore._collection_metadata)

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


print(vectorstore._collection.count())

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


@tool
def search_book(query: str) -> str:
    """只有在需要的时候才查阅书籍，输入是查询的问题"""
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])


@tool
def calculate_multiply(a: int, b: int) -> int:
    """计算两个数字的乘积"""
    return a * b


chat_history = []
tools = [search_book, calculate_multiply]
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个不仅能查书籍，遇到计算题还能使用计算器的人工智能。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),  # 关键：给 AI 留出思考和调用工具的空间
    ]
)
# llm_bind_tools = llm.bind_tools(tools=tools)  # 后面的create_tool_calling_agent会自动绑定工具的

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


while True:
    user_input = input("\nHuman: ")
    if user_input.lower() in ["q", "quit", "exit"]:
        print("下次再见")
        break
    if not user_input.strip():
        continue

    print("AI正在思考...", end="", flush=True)

    response = agent_executor.invoke(
        {"input": user_input, "chat_history": chat_history}
    )
    print(f"\rAI: {response['output']}")

    # 历史
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response["output"]))
