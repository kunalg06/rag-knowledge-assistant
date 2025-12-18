import os
from typing import List, Dict

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.messages import HumanMessage

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")


def load_vectorstore():
    """Load FAISS index from disk."""
    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_DIR}. Run ingest.py first."
        )
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def retrieve_docs(query: str, k: int = 4):
    """Retrieve top-k similar chunks for the query."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)  # new-style retriever call
    return docs


def build_context_with_citations(docs) -> str:
    """Build a context string with simple [DOC_i] tags."""
    context_parts = []
    for i, doc in enumerate(docs, start=1):
        tag = f"[DOC_{i}]"
        source = doc.metadata.get("source", "unknown")
        context_parts.append(
            f"{tag} (source: {source})\n{doc.page_content}\n"
        )
    return "\n\n".join(context_parts)


def ask_llm(query: str, docs) -> Dict:
    """Call ChatOpenAI with context + query and return answer + sources."""
    context = build_context_with_citations(docs)

    system_prompt = (
        "You are a helpful assistant. Use the context to answer the question. "
        "Always mention the citation tags like [DOC_1], [DOC_2] in your answer "
        "so the user knows which document you used."
    )

    user_prompt = (
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer clearly and concisely. If you don't know, say you don't know."
    )

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,        # current langchain_openai arg name
        model="gpt-4o-mini",
        temperature=0.0,
    )

    messages = [
        HumanMessage(content=f"{system_prompt}\n\n{user_prompt}")
    ]

    # NEW: use invoke, not direct call
    response = llm.invoke(messages)

    # Collect sources file names
    sources: List[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        sources.append(f"DOC_{i}: {source}")

    return {
        "answer": response.content,
        "sources": sources,
    }


def answer_question(query: str) -> Dict:
    """High-level helper: retrieve docs and ask LLM."""
    docs = retrieve_docs(query, k=4)
    result = ask_llm(query, docs)
    return result
