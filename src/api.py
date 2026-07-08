import os
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

from src.retrieval.bm25_engine import BM25Retriever
from src.generation.llm_client import LLMClient
from src.main import INDEX_SAVE_DIR


app = FastAPI(
    title="RAG Against The Machine API",
    description="API for BM25 Retrieval and LLM Answer Generation",
    version="1.0.0"
)

# Global instances
retriever: Optional[BM25Retriever] = None
llm: Optional[LLMClient] = None


@app.on_event("startup")
def load_models() -> None:
    """Load the BM25 index and LLM model on startup."""
    global retriever, llm
    if not os.path.exists(INDEX_SAVE_DIR):
        raise RuntimeError(
            f"Index not found at {INDEX_SAVE_DIR}. "
            f"Please run 'python -m src index' first."
        )

    print("\033[96m\033[1m[API]\033[0m Loading BM25 Index...")
    retriever = BM25Retriever()
    retriever.load(INDEX_SAVE_DIR)

    print("\033[96m\033[1m[API]\033[0m Loading LLM...")
    llm = LLMClient()
    print("\033[92m\033[1m[API]\033[0m Ready!")


class SearchResponse(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str


class AnswerRequest(BaseModel):
    query: str
    k: int = 5


class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResponse]


@app.get("/health")
def health_check() -> dict:
    """Check if the API is running and models are loaded."""
    if retriever and llm:
        return {"status": "ok", "models_loaded": True}
    return {"status": "starting", "models_loaded": False}


@app.get("/search", response_model=List[SearchResponse])
def search(
    query: str = Query(..., description="The search query"),
    k: int = Query(3, description="Number of results to return")
) -> List[dict]:
    """Search for relevant chunks using BM25."""
    if not retriever:
        raise HTTPException(
            status_code=503, detail="Retriever not loaded yet"
        )

    chunks = retriever.search(query, k=k)
    return [c.model_dump() for c in chunks]


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    """Generate an answer using the retrieved context."""
    if not retriever or not llm:
        raise HTTPException(
            status_code=503, detail="Models not loaded yet"
        )

    # 1. Retrieve context
    chunks = retriever.search(req.query, k=req.k)

    # 2. Generate answer (uses the cache automatically)
    answer_text = llm.generate_answer(req.query, chunks)

    return AnswerResponse(
        query=req.query,
        answer=answer_text,
        sources=[SearchResponse(**c.model_dump()) for c in chunks]
    )


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
