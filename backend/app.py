import os
import shutil
from contextlib import asynccontextmanager
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
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
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_pdf_index")
UPLOAD_DIR = "uploaded_pdfs"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing in .env file!")

# Shared state across requests
state = {"vector_db": None, "kg": None, "client": None, "current_pdf": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    state["client"] = openai.OpenAI(api_key=OPENAI_API_KEY)
    print("🚀 Backend ready for dynamic PDF uploads!")
    yield
    state.clear()


app = FastAPI(title="Dynamic GraphRAG Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# 1. PDF UPLOAD & INDEXING ENDPOINT
# ---------------------------------------------------------------------
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Receives PDF from Streamlit frontend, splits text, builds FAISS & KG."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files are supported."
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save uploaded PDF locally
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Load and Chunk Document
        chunks = load_and_split_pdf(file_path)

        # Build Vector Index
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_db = FAISS.from_documents(chunks, embeddings)

        # Build Knowledge Graph
        client = state["client"]
        kg = build_networkx_graph(chunks, client, max_chunks=5)

        # Update dynamic runtime state
        state["vector_db"] = vector_db
        state["kg"] = kg
        state["current_pdf"] = file.filename

        return {
            "message": f"Successfully indexed '{file.filename}'",
            "chunks_count": len(chunks),
            "graph_nodes": len(kg.nodes()),
            "graph_edges": len(kg.edges()),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process PDF: {str(e)}"
        )


# ---------------------------------------------------------------------
# 2. CHAT ENDPOINT
# ---------------------------------------------------------------------
class ChatRequest(BaseModel):
    prompt: str


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not state["vector_db"] or not state["kg"]:
        return {
            "answer": "⚠️ No PDF document has been uploaded yet. Please upload a PDF using the sidebar first.",
            "guardrail_triggered": True,
        }

    raw_input = request.prompt

    # Input Guardrail
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

    # Retrieve Context
    vector_context, graph_context = retrieve_hybrid_context(
        user_query, vector_db, kg
    )

    system_prompt = """You are a strict, helpful assistant answering questions based ONLY on the provided context.
If the answer cannot be determined from the context, state "I do not have enough information in the provided document." """

    prompt = f"""Question: {user_query}

--- FAISS VECTOR CONTEXT ---
{vector_context}

--- KNOWLEDGE GRAPH CONTEXT ---
{graph_context}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    raw_answer = response.choices[0].message.content.strip()
    safe_answer = validate_llm_output(raw_answer)

    return {"answer": safe_answer, "guardrail_triggered": False}