# RAG Knowledge Assistant
Python + LangChain + OpenAI + FAISS + FastAPI

## Overview
A simple Retrieval-Augmented Generation (RAG) assistant that answers questions over your own documents. It loads local text files, indexes them in a vector database (FAISS), and uses an LLM to generate answers with citations like `[DOC_1]`.

## Problem
Traditional keyword search is weak at answering natural language questions over internal documents.  
This project shows how to build a small end‑to‑end RAG pipeline that can:
- Ingest your documents.
- Retrieve the most relevant chunks.
- Generate grounded answers that point back to the source text.

## Tech Stack
- Python
- LangChain
- OpenAI API (embeddings + chat)
- FAISS vector store
- FastAPI
- CLI helper script

## Architecture
1. **Ingestion**  
   `src/ingest.py` loads `.txt` files from `src/data/`, splits them into chunks, creates embeddings, and saves a FAISS index on disk.

2. **Retrieval + Generation**  
   `src/rag_chain.py` loads the FAISS index, retrieves the top‑k chunks for a user query, and calls an LLM with the retrieved context to generate an answer with simple `[DOC_i]` citations.

3. **Interfaces**  
   - **CLI**: `src/cli.py` → `python src/cli.py "your question"`  
   - **API**: `src/api.py` → FastAPI endpoint `POST /ask` that returns `{"answer": ..., "sources": [...]}`.

## Setup
### Prerequisites
- Python 3.10+
- OpenAI API key
- Git

### Installation
git clone https://github.com/kunalg06/rag-knowledge-assistant.git
cd rag-knowledge-assistant

python -m venv .venv

Windows:
.venv\Scripts\Activate.ps1
macOS / Linux:
source .venv/bin/activate
pip install -r requirements.txt

### Configuration
Create a `.env` file in the project root:
OPENAI_API_KEY=your_real_key_here
> Do not commit `.env` to Git.

## Usage
### 1. Add documents
Place your `.txt` files inside `src/data/`, for example:

- `src/data/doc1.txt`
- `src/data/doc2.txt`
- ...

### 2. Build the FAISS index
python src/ingest.py
This loads the files, splits them into chunks, creates embeddings, and saves the index under `faiss_index/`.

### 3. Ask questions via CLI
python src/cli.py "What are these documents about?"
You will see:
- The generated answer.
- A list of `DOC_i: filename` sources that were used.

### 4. Run the FastAPI server
uvicorn src.api:app --reload
Then send a request:
curl -X POST "http://127.0.0.1:8000/ask"
-H "Content-Type: application/json"
-d "{"query": "What is inside these documents?"}"

The response contains `answer` and `sources` in JSON.

## Evaluation
Basic evaluation is implemented in `src/eval.py`:

- `data/eval.jsonl` holds small evaluation examples:  
  `{"question": "...", "expected_keywords": ["..."]}`  
- The script calls the RAG pipeline for each question and computes very simple keyword‑based precision/recall.

Run:

python src/eval.py
This prints answers and aggregated metrics.

## Future Work
- Support PDFs / DOCX using LangChain document loaders.
- Add a simple web UI (Streamlit or frontend for FastAPI).
- Better evaluation (RAGAS‑style metrics, more test questions).
- Add other embedding providers or vector databases (e.g. Pinecone).