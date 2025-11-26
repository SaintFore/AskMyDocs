from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import re

load_dotenv()

loader = TextLoader("./novel.txt", encoding="utf8")
docs = loader.load()
# print(docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10,
    separators=["\n\n", "\n", " "],
)

split = text_splitter.split_documents(documents=docs)
split_clean = [
    Document(
        page_content=re.sub(r"\s+", "", chunk.page_content), metadata=chunk.metadata
    )
    for chunk in split
]
split_clean = split_clean[:50]
print(f"原文被切成了{len(split_clean)}个片段")
# for i in split_clean[:100]:
#     print(repr(i.page_content))
#     print("*" * 50)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = GoogleGenerativeAI(model="gemini-2.5-flash")

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

print("建立索引中")

db = Chroma.from_documents(documents=split_clean, embedding=embeddings)

retriever = db.as_retriever(search_kwargs={"k": 5})

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
    | llm
    | StrOutputParser()
)

question = "各种人物的外貌描写"
print(f"问: {question}")
answer = rag_chain.invoke(question)
print(f"答: {answer}")
# query = "coding"
#
# results = db.similarity_search(query=query, k=2)
# print(results)
