import os
import requests
import streamlit as st

# Server URLs
BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CHAT_URL = f"{BASE_URL}/api/chat"
UPLOAD_URL = f"{BASE_URL}/api/upload"

st.set_page_config(
    page_title="GraphRAG Document Assistant",
    page_icon="🕸️",
    layout="wide",
)

st.markdown(
    """
<style>
    .stChatMessage { border-radius: 10px; padding: 10px; }
    .guardrail-alert { padding: 10px; border-radius: 8px; background-color: #451a03; color: #fde68a; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# SIDEBAR: DYNAMIC FILE UPLOAD
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("🕸️ GraphRAG AI")
    st.caption("FAISS Vector Search + NetworkX Knowledge Graph")

    st.divider()

    st.subheader("📁 Upload Document")
    uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("🚀 Process & Index PDF", use_container_width=True):
            with st.spinner("Processing PDF, vectorizing, and building graph..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    }
                    res = requests.post(UPLOAD_URL, files=files, timeout=120)

                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Indexed: {uploaded_file.name}")
                        st.info(
                            f"📄 Chunks: {data['chunks_count']} | 🕸️ Nodes: {data['graph_nodes']} | Edges: {data['graph_edges']}"
                        )
                        st.session_state.indexed_file = uploaded_file.name
                        st.session_state.messages = []  # Reset chat on new doc
                    else:
                        st.error(f"Error {res.status_code}: {res.json().get('detail')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------
# MAIN CHAT INTERFACE
# ---------------------------------------------------------------------
st.title("📄 Hybrid GraphRAG Assistant")

current_doc = st.session_state.get("indexed_file", None)
if current_doc:
    st.caption(f"Currently chatting with: **{current_doc}**")
else:
    st.info("👈 Please upload and process a PDF file from the sidebar to begin.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("guardrail_triggered"):
            st.markdown(
                f'<div class="guardrail-alert">{message["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(message["content"])

# Handle Query Prompt
if prompt := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving vector & graph context..."):
            try:
                response = requests.post(
                    CHAT_URL, json={"prompt": prompt}, timeout=40
                )
                if response.status_code == 200:
                    data = response.json()
                    bot_answer = data.get("answer", "")
                    is_guardrail = data.get("guardrail_triggered", False)

                    if is_guardrail:
                        st.markdown(
                            f'<div class="guardrail-alert">{bot_answer}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(bot_answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": bot_answer,
                            "guardrail_triggered": is_guardrail,
                        }
                    )
                else:
                    st.error(f"Server Error {response.status_code}")
            except requests.exceptions.RequestException:
                st.error("Could not reach backend server on port 8000.")