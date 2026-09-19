import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def load_and_split_pdf(pdf_path: str):
    """Loads a PDF file and splits it into text chunks."""
    print(f"📄 Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Document split into {len(chunks)} chunks.")
    return chunks


def create_vector_db(chunks, save_path: str):
    """Embeds text chunks into a FAISS index and persists it to disk."""
    print("⚡ Generating embeddings and building FAISS index...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(save_path)
    print(f"✅ FAISS index successfully saved to '{save_path}'.")
    return vector_db


if __name__ == "__main__":
    pdf_path = os.getenv("PDF_PATH")
    index_path = os.getenv("FAISS_INDEX_PATH", "faiss_pdf_index")

    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF_PATH invalid or missing: {pdf_path}")

    chunks = load_and_split_pdf(pdf_path)
    create_vector_db(chunks, index_path)