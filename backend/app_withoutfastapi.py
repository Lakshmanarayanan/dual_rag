import os
import openai
from dotenv import load_dotenv
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


def initialize_backend():
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
    return vector_db, kg, client


def main():
    vector_db, kg, client = initialize_backend()

    print("\n==================================================")
    print("🤖 Guardrailed Graph-Vector Chatbot Ready!")
    print("Type your questions below (or 'exit' to quit).")
    print("==================================================\n")

    system_prompt = """You are a strict, helpful assistant answering questions based ONLY on the provided context.
If the answer cannot be determined from the context, state "I do not have enough information in the provided document."

Context rules:
1. FAISS Vector Context: Unstructured raw document passages.
2. Knowledge Graph Context: Structural relationships between entities."""

    while True:
        try:
            raw_input = input("You: ")

            # Exit condition
            if raw_input.strip().lower() in ["exit", "quit", "q"]:
                print("Exiting chatbot...")
                break

            # ----------------------------------------------------
            # 🛡️ 1. INPUT GUARDRAIL CHECK
            # ----------------------------------------------------
            is_valid, validated_input_or_err = validate_user_input(
                raw_input, max_length=500
            )

            if not is_valid:
                print(f"\n🚫 [Input Guardrail Triggered]: {validated_input_or_err}\n")
                print("-" * 50)
                continue

            user_query = validated_input_or_err

            # 2. Retrieve context across modules
            vector_context, graph_context = retrieve_hybrid_context(
                user_query, vector_db, kg
            )

            prompt = f"""Question: {user_query}

--- FAISS VECTOR CONTEXT ---
{vector_context}

--- KNOWLEDGE GRAPH CONTEXT ---
{graph_context}

Answer:"""

            # 3. Call LLM
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )

            raw_answer = response.choices[0].message.content.strip()

            # ----------------------------------------------------
            # 🛡️ 2. OUTPUT GUARDRAIL CHECK
            # ----------------------------------------------------
            safe_answer = validate_llm_output(raw_answer)

            print(f"\nBot: {safe_answer}\n")
            print("-" * 50)

        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()