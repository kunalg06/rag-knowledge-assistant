from fastapi import FastAPI
from pydantic import BaseModel

from src.rag_chain import answer_question

app = FastAPI(title="RAG Knowledge Assistant")


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Accept a natural language query and return answer + sources."""
    result = answer_question(request.query)
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
    )
