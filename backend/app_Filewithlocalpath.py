import os
from contextlib import asynccontextmanager
import openai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from ingest import load_and_split_pdf, create_vector_db
from graph_builder import build_networkx_graph
from retriever import (
    retrieve_hybrid_context,
    validate_user_input,
    validate_llm_output,
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_PATH = os.getenv("PDF_PATH")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_pdf_index")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing in .env file!")

# Global dictionary to hold state across requests
state = {}


# ---------------------------------------------------------------------
# 1. LIFESPAN CONTEXT MANAGER (Replaces deprecated @app.on_event)
# ---------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs setup logic before startup and cleanup after shutdown."""
    print("🚀 Initializing backend databases and OpenAI client...")
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    chunks = load_and_split_pdf(PDF_PATH)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"⚡ Loading FAISS index from '{FAISS_INDEX_PATH}'...")
        vector_db = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
    else:
        vector_db = create_vector_db(chunks, FAISS_INDEX_PATH)

    kg = build_networkx_graph(chunks, client, max_chunks=5)

    # Save loaded resources to global state
    state["vector_db"] = vector_db
    state["kg"] = kg
    state["client"] = client
    print("✅ System initialized successfully!")

    yield  # Application runs while yielded

    # Cleanup logic (if needed on shutdown)
    state.clear()
    print("🛑 Backend resources cleaned up.")


# ---------------------------------------------------------------------
# 2. INITIALIZE FASTAPI APP
# ---------------------------------------------------------------------
app = FastAPI(title="GraphRAG Backend", lifespan=lifespan)

# Enable CORS for React / Streamlit frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request body schema
class ChatRequest(BaseModel):
    prompt: str


# ---------------------------------------------------------------------
# 3. CHAT ENDPOINT
# ---------------------------------------------------------------------
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    raw_input = request.prompt

    # 1. Input Guardrail Check
    is_valid, validated_input_or_err = validate_user_input(
        raw_input, max_length=500
    )

    if not is_valid:
        return {
            "answer": f"🚫 [Input Guardrail Triggered]: {validated_input_or_err}",
            "guardrail_triggered": True,
        }

    user_query = validated_input_or_err
    client = state["client"]
    vector_db = state["vector_db"]
    kg = state["kg"]

    # 2. Retrieve Context (FAISS + Knowledge Graph)
    vector_context, graph_context = retrieve_hybrid_context(
        user_query, vector_db, kg
    )

    system_prompt = """You are a strict, helpful assistant answering questions based ONLY on the provided context.
If the answer cannot be determined from the context, state "I do not have enough information in the provided document."

Context rules:
1. FAISS Vector Context: Unstructured raw document passages.
2. Knowledge Graph Context: Structural relationships between entities."""

    prompt = f"""Question: {user_query}

--- FAISS VECTOR CONTEXT ---
{vector_context}

--- KNOWLEDGE GRAPH CONTEXT ---
{graph_context}

Answer:"""

    # 3. Query OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    raw_answer = response.choices[0].message.content.strip()

    # 4. Output Guardrail Check
    safe_answer = validate_llm_output(raw_answer)

    return {"answer": safe_answer, "guardrail_triggered": False}