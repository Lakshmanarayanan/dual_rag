import streamlit as st
import requests

# ---------------------------------------------------------------------
# CONFIGURATION & STYLING
# ---------------------------------------------------------------------
BACKEND_URL = "http://localhost:8000/api/chat"

st.set_page_config(
    page_title="GraphRAG Document Assistant",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI polished look
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .guardrail-alert {
        padding: 10px;
        border-radius: 8px;
        background-color: #451a03;
        border: 1px solid #78350f;
        color: #fde68a;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# SIDEBAR: System Status & Metrics
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("🕸️ GraphRAG AI")
    st.caption("FAISS Vector Index + NetworkX Knowledge Graph")
    
    st.divider()
    
    st.subheader("🛡️ System Architecture")
    
    st.success("✅ FAISS Vector Index: Active")
    st.info("✅ NetworkX Graph: Ready")
    st.warning("🛡️ Input/Output Guardrails: Enabled")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------
# MAIN CHAT INTERFACE
# ---------------------------------------------------------------------
st.title("📄 PDF Document Assistant")
st.caption("Ask questions about your document context. Answers are retrieved via hybrid vector & graph search.")

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Existing Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("guardrail_triggered"):
            st.markdown(
                f'<div class="guardrail-alert">⚠️ <b>Guardrail Triggered:</b> {message["content"]}</div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(message["content"])

# User Prompt Input
if prompt := st.chat_input("Ask a question about your document..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context from FAISS & NetworkX Knowledge Graph..."):
            try:
                response = requests.post(
                    BACKEND_URL,
                    json={"prompt": prompt},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    bot_answer = data.get("answer", "No response content.")
                    is_guardrail = data.get("guardrail_triggered", False)
                    
                    if is_guardrail:
                        st.markdown(
                            f'<div class="guardrail-alert">⚠️ <b>Guardrail Triggered:</b> {bot_answer}</div>', 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(bot_answer)
                    
                    # Store response in session state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": bot_answer,
                        "guardrail_triggered": is_guardrail
                    })
                else:
                    error_msg = f"Error {response.status_code}: Unable to reach backend."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except requests.exceptions.RequestException as e:
                error_msg = "❌ Could not connect to backend server. Make sure FastAPI is running on port 8000."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})