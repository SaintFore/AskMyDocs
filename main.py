from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = GoogleGenerativeAI(model="gemini-2.5-flash")

# texts = ["hello world!", "你好，世界！", "cat", "dog"]
#
# vectors = embeddings.embed_documents(texts=texts)

# print(len(vectors[0]))
# print(vectors[0][:10])

documents = [
    "这里的晚餐真好吃",
    "今天天气不错",
    "猫喜欢吃鱼",
    "我是一名程序员",
    "The dog is barking",
]

print("建立索引中")

db = Chroma.from_texts(documents, embedding=embeddings)

retriever = db.as_retriever(search_kwargs={"k": 1})

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

question = "今天应该去钓鱼么？"
print(f"问: {question}")
answer = rag_chain.invoke(question)
print(f"答: {answer}")
# query = "coding"
#
# results = db.similarity_search(query=query, k=2)
# print(results)
