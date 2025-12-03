from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain_core.prompts import (
    MessagesPlaceholder,
    PromptTemplate,
    ChatPromptTemplate,
)
from langchain_community.document_loaders import TextLoader, chatgpt
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage

# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from langchain.agents import tool, create_tool_calling_agent, AgentExecutor


import re

load_dotenv()

# print(docs)
DB_PATH = "./chroma_db"
os.makedirs(DB_PATH, exist_ok=True)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " "],
)
# for i in split_clean[:100]:
#     print(repr(i.page_content))
#     print("*" * 50)
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


# texts = ["hello world!", "你好，世界！", "cat", "dog"]
#
# vectors = embeddings.embed_documents(texts=texts)

# print(len(vectors[0]))
# print(vectors[0][:10])

# documents = [
#     "这里的晚餐真好吃",
#     "今天天气不错",
#     "猫喜欢吃鱼",
#     "我是一名程序员",
#     "The dog is barking",
# ]


print(vectorstore._collection.count())
# results = vectorstore.as_retriever().get_relevant_documents("测试")
# print(results)

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

# print("🕵️ Agent 开始执行...")
# agent_executor.invoke({"input": "刻意练习需要多长时间，把这个时间乘以10是多少"})
#
# contextualize_q_system_prompt = """
#     给定一段聊天历史和用户最新的问题，
#     如果该问题引用了历史中的上下文，请将其重新表述为一个独立的问题，使其不需要历史上下文也能被理解。
#     不要回答问题，只需返回改写后的问题；如果没有必要改写，则原样返回。
#     """
# contextualize_q_prompt = PromptTemplate.from_template(
#     contextualize_q_system_prompt
#     + "\n\n聊天历史:\n{chat_history}\n\n最新问题:\n{input}"
# )
#
# history_retriever = create_history_aware_retriever(
#     llm=llm, retriever=retriever, prompt=contextualize_q_prompt
# )
#
#
# qa_system_prompt = """
#     上下文 (Context):
#     {context}
#     """
#
# qa_prompt = PromptTemplate.from_template(qa_system_prompt + "\n问题: {input}")
# question_answer_chain = create_stuff_documents_chain(llm=llm_ollama, prompt=qa_prompt)
#
# rag_chain = create_retrieval_chain(history_retriever, question_answer_chain)

# # rag_chain = (
# #     {"context": retriever, "question": RunnablePassthrough()}
# #     | template
# #     | llm_ollama
# #     | StrOutputParser()
# # )
#
# # docs = retriever.get_relevant_documents("nothing to do")
# # print(docs)
# # print(len(docs))
#
# # question = "今天会下雨么"
# # print(f"问: {question}")
# # answer = rag_chain.invoke(question)
# # print(f"答: {answer}")
# # query = "coding"
# # # results = db.similarity_search(query=query, k=2)
# # print(results)
# #

while True:
    user_input = input("\nHuman: ")
    if user_input.lower() in ["q", "quit", "exit"]:
        print("下次再见")
        break
    if not user_input.strip():
        continue

    print("AI正在思考...", end="", flush=True)
    # response = rag_chain.invoke({"input": user_input, "chat_history": chat_history})

    response = agent_executor.invoke(
        {"input": user_input, "chat_history": chat_history}
    )
    # print(response)
    print(f"\rAI: {response['output']}")

    # 历史
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response["output"]))

    # source_docs = retriever.invoke(
    #     user_input
    # )  # 此retriever并非history_retriever，这里有bug
    # for i in chat_history:
    #     print(i)
    # print(len(response["context"]))
