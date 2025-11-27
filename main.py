from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage

# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

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
llm = GoogleGenerativeAI(model="gemini-2.5-flash")
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

contextualize_q_system_prompt = """
    给定一段聊天历史和用户最新的问题，
    如果该问题引用了历史中的上下文，请将其重新表述为一个独立的问题，使其不需要历史上下文也能被理解。
    不要回答问题，只需返回改写后的问题；如果没有必要改写，则原样返回。
    """
contextualize_q_prompt = PromptTemplate.from_template(
    contextualize_q_system_prompt
    + "\n\n聊天历史:\n{chat_history}\n\n最新问题:\n{input}"
)

history_retriever = create_history_aware_retriever(
    llm=llm_ollama, retriever=retriever, prompt=contextualize_q_prompt
)


qa_system_prompt = """
    你是一个基于本地知识库的 AI 助手。请根据以下上下文回答问题。如果不清楚就说不知道。
    上下文 (Context):
    {context}
    """

qa_prompt = PromptTemplate.from_template(qa_system_prompt + "\n问题: {input}")
question_answer_chain = create_stuff_documents_chain(llm=llm_ollama, prompt=qa_prompt)

rag_chain = create_retrieval_chain(history_retriever, question_answer_chain)

# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | template
#     | llm_ollama
#     | StrOutputParser()
# )

# docs = retriever.get_relevant_documents("nothing to do")
# print(docs)
# print(len(docs))

# question = "今天会下雨么"
# print(f"问: {question}")
# answer = rag_chain.invoke(question)
# print(f"答: {answer}")
# query = "coding"
#
# results = db.similarity_search(query=query, k=2)
# print(results)
#
chat_history = []

while True:
    user_input = input("\nHuman: ")
    if user_input.lower() in ["q", "quit", "exit"]:
        print("下次再见")
        break
    if not user_input.strip():
        continue

    print("AI正在思考...", end="", flush=True)
    response = rag_chain.invoke({"input": user_input, "chat_history": chat_history})

    print(f"\rAI: {response['answer']}")

    # 历史
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response["answer"]))

    # source_docs = retriever.invoke(
    #     user_input
    # )  # 此retriever并非history_retriever，这里有bug
    # for i in chat_history:
    #     print(i)
    # print(len(response["context"]))
