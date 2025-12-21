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

    # 防止context爆炸
    recent_history = st.session_state.chat_history[-10:]
    current_history = recent_history + [HumanMessage(content=prompt)]
    with st.chat_message("ai"):
        # expanded=True 表示默认展开，让用户看到 AI 在干活
        status_container = st.status("🤖 AI 正在思考...", expanded=True)

        try:
            generated_msgs = []
            full_response = ""

            events = agent_executor.stream({"messages": current_history})

            for event in events:
                # event 字典的 key 是节点名 (如 'llm', 'tools')
                # value 是该节点的输出 (如 {'messages': [...]})
                for node_name, values in event.items():
                    if "messages" in values:
                        new_messages = values["messages"]
                        last_msg = new_messages[-1]

                        generated_msgs.append(last_msg)

                        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                            tool_name = last_msg.tool_calls[0]["name"]
                            tool_args = last_msg.tool_calls[0]["args"]
                            status_container.write(f"🛠️ **计划调用工具**: `{tool_name}`")
                            status_container.json(tool_args)

                        elif node_name == "tools":
                            # 为了不让大量文本刷屏，截取前200字
                            content_preview = last_msg.content[:200]
                            if len(last_msg.content) > 200:
                                content_preview += "..."
                            status_container.write(
                                f"📚 **工具返回结果**: {content_preview}"
                            )

                        elif (
                            isinstance(last_msg, AIMessage) and not last_msg.tool_calls
                        ):
                            full_response = last_msg.content
                            # 这里暂时不显示，等循环结束在外面统一显示，或者你想在 status 里也显示

            status_container.update(
                label="✅ 回答完成", state="complete", expanded=False
            )

            if full_response:
                st.markdown(full_response)
            else:
                # 兜底：万一循环里没抓到 content，尝试从 generated_msgs 找最后一条
                if generated_msgs and isinstance(generated_msgs[-1], AIMessage):
                    full_response = generated_msgs[-1].content
                    st.markdown(full_response)

            st.session_state.ui_messages.append(
                {"role": "assistant", "content": full_response}
            )

            # 更新 LangChain 记忆
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.extend(generated_msgs)

        except Exception as e:
            status_container.update(label="❌ 发生错误", state="error")
            st.error(f"处理过程中出错: {e}")
