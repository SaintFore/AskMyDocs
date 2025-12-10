from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

# import streamlit as st
import argparse
import backend as be


# st.set_page_config(page_title="My AI Agent", page_icon="🤖")
# st.title("🤖 本地全能知识库助手")

load_dotenv()


def parse_args() -> be.AppConfig:
    parser = argparse.ArgumentParser(description="知识工具")
    parser.add_argument("--chunk-size", type=int, default=500, help="文档切分大小")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="重叠大小")
    parser.add_argument("-k", type=int, default=5, help="搜索最近的n块数据")
    parser.add_argument(
        "--db-path", type=str, default="./chroma_db/", help="数据库位置"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://192.168.31.60:11434",
        help="ollama的地址",
    )

    args = parser.parse_args()
    return be.AppConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        k=args.k,
        db_path=args.db_path,
        base_url=args.base_url,
    )


def main():

    config = parse_args()

    retriever = be.init_vectorstore(config).as_retriever(search_kwargs={"k": config.k})
    tools = be.create_tools(retriever=retriever)
    agent_executor = be.init_agent(tools=tools, config=config)

    chat_history = []
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


if __name__ == "__main__":
    main()
