import os
import openai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from ingest import load_and_split_pdf, create_vector_db
from graph_builder import build_networkx_graph
from retriever import retrieve_hybrid_context

# Load configuration
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_PATH = os.getenv("PDF_PATH")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_pdf_index")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing in .env file!")


def initialize_backend():
    """Initializes vector store and knowledge graph."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    chunks = load_and_split_pdf(PDF_PATH)

    # Load or build FAISS
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"⚡ Loading FAISS index from '{FAISS_INDEX_PATH}'...")
        vector_db = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
    else:
        vector_db = create_vector_db(chunks, FAISS_INDEX_PATH)

    # Build Knowledge Graph
    kg = build_networkx_graph(chunks, client, max_chunks=5)

    return vector_db, kg, client


def main():
    vector_db, kg, client = initialize_backend()

    print("\n==================================================")
    print("🤖 Hybrid Graph-Vector Chatbot Initialized!")
    print("Type your questions below (or 'exit' to quit).")
    print("==================================================\n")

    system_prompt = """You are an assistant answering questions based on document context.
You receive:
1. FAISS Vector Context: Unstructured raw document passages.
2. Knowledge Graph Context: Structural relationships between entities.

Synthesize both sources to provide a concise and precise answer."""

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting chatbot...")
                break

            # Retrieve context across modules
            vector_context, graph_context = retrieve_hybrid_context(
                user_input, vector_db, kg
            )

            # Construct LLM input
            prompt = f"""Question: {user_input}

--- FAISS VECTOR CONTEXT ---
{vector_context}

--- KNOWLEDGE GRAPH CONTEXT ---
{graph_context}

Answer:"""

            # Call LLM
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            answer = response.choices[0].message.content.strip()
            print(f"\nBot: {answer}\n")
            print("-" * 50)

        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()