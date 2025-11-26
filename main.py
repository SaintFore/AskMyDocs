from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader

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

if vectorstore._collection.count() == 0:
    print("数据库为空")
    loader = TextLoader("./novel.txt", encoding="utf8")
    docs = loader.load()

    split = text_splitter.split_documents(documents=docs)
    split_clean = [
        Document(
            page_content=re.sub(r"\s+", "", chunk.page_content), metadata=chunk.metadata
        )
        for chunk in split
    ]
    split_clean = split_clean
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

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

template = PromptTemplate.from_template(
    """
    你是一个问答助手。请仅根据下面的上下文回答问题。如果不清楚，就说不知道。

    上下文 (Context):
    {context}

    问题 (Question):
    {question}
    """
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | template
    | llm_ollama
    | StrOutputParser()
)

# docs = retriever.get_relevant_documents("nothing to do")
# print(docs)
# print(len(docs))

question = "今天适合钓鱼么"
print(f"问: {question}")
answer = rag_chain.invoke(question)
print(f"答: {answer}")
# query = "coding"
#
# results = db.similarity_search(query=query, k=2)
# print(results)
