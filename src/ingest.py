import os
import glob

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_index")


def load_documents():
    """Load all .txt files from src/data as LangChain Documents."""
    pattern = os.path.join(DATA_DIR, "*.txt")
    file_paths = glob.glob(pattern)

    if not file_paths:
        raise FileNotFoundError(f"No .txt files found in {DATA_DIR}. Add some documents first.")

    docs = []
    for path in file_paths:
        loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())
    return docs


def split_documents(documents):
    """Split documents into chunks of ~1000 characters with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)


def build_faiss_index(chunks):
    """Create FAISS vector store from chunks and save it to disk."""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Ensure directory exists
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    print(f"Saved FAISS index to {INDEX_DIR}")


def main():
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents.")

    print("Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Building FAISS index...")
    build_faiss_index(chunks)
    print("Done!")


if __name__ == "__main__":
    main()
