# rag-knowledge-assistant
Python + LangChain + OpenAI + Vector DB

# RAG Knowledge Assistant

## Overview
A Retrieval-Augmented Generation (RAG) assistant that answers questions over a custom document set using a vector database and an LLM.

## Problem
Traditional keyword search is weak at answering natural language questions over internal documents.  
Goal: build a system that can retrieve relevant chunks and generate grounded answers with citations.

## Tech Stack
- Python
- LangChain (or LlamaIndex)
- OpenAI API (or other LLM)
- FAISS (or Pinecone) vector store
- Streamlit / FastAPI (choose one)

## Architecture
1. Ingestion: Load PDFs/Markdown/HTML.
2. Chunking: Split documents into overlapping chunks.
3. Embeddings: Generate embeddings for each chunk.
4. Vector Store: Index chunks in FAISS/Pinecone.
5. Retrieval: Fetch top-k relevant chunks for a query.
6. Generation: Call LLM with retrieved context to answer, with citations.

## Setup
### Prerequisites
- Python 3.10+
- OpenAI API key (or other provider)
- `requirements.txt` installed

### Installation
- git clone https://github.com/kunalg06/rag-knowledge-assistant.git
- cd rag-knowledge-assistant
- pip install -r requirements.txt


### Configuration
- Create a `.env` file with:
  - `OPENAI_API_KEY=...`
- (Optional) Vector DB credentials if using Pinecone, etc.

### Usage
1. Run the ingestion script to index documents.
2. Launch the app:

3. Ask questions in the UI and view cited answers.

## Evaluation
- Manual relevance checks on retrieved chunks.
- Compare answers with and without retrieval.
- (Optional) Simple metrics: hit rate, answer length, etc.

## Future Work
- Add feedback collection and re-ranking.
- Multi-document summarisation.
- Access control/user auth.
