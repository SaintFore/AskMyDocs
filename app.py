from dotenv import load_dotenv
import streamlit as st
import backend as be
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(page_title="My AI Agent", page_icon="🤖")
st.title("🤖 本地全能知识库助手")

load_dotenv()

if "config" not in st.session_state:
    st.session_state["config"] = be.AppConfig(chunk_size=500, chunk_overlap=50, k=5)

config = st.session_state["config"]

with st.sidebar:
    st.header("⚙️ 配置面板")
    chunk_size = st.number_input("Chunk Size", value=500, step=50)
    chunk_overlap = st.number_input("Chunk Overlap", value=50)
    k_val = st.slider("检索数量 (K)", 1, 10, 5)
    if st.button("设置"):
        st.session_state["config"] = be.AppConfig(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k_val
        )
        st.success("配置已经更新")


@st.cache_resource
def create_agent(config: be.AppConfig):
    retriever = be.init_vectorstore(config).as_retriever(search_kwargs={"k": config.k})
    tools = be.create_tools(retriever=retriever)
    agent_executor = be.init_agent(tools=tools, config=config)
    return agent_executor


try:
    agent_executor = create_agent(config)
except Exception as e:
    st.error(f"Error {e}")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []


for msg in st.session_state.ui_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("输入问题..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.ui_messages.append({"role": "user", "content": prompt})

    with st.chat_message("ai"):
        placeholder = st.empty()
        placeholder.markdown("AI is thinking...")

        try:
            response = create_agent(config).invoke(
                {"input": prompt, "chat_history": st.session_state.chat_history}
            )

            ai_output = response["output"]
            placeholder.markdown(ai_output)
            # 更新状态
            st.session_state.ui_messages.append(
                {"role": "assistant", "content": ai_output}
            )
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=ai_output))
        except Exception as e:
            placeholder.error(e)
