"""
RAG Pipeline API
-----------------------------------
Central entry point for the Retrieval Augmented Generation service.

Features:
- Resource-based routing, proper status codes.
- Lifespan Management (Load models once, keep in memory).
- Threading (Handles CPU-bound inference in threadpool).
- Self-Documenting (Swagger/OpenAPI schemas).
"""

from contextlib import asynccontextmanager
import time
from typing import List

from fastapi import FastAPI, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from ragpipeline.generation import RAGGenerator
from ragpipeline.retrieval import Retriever

# --- GLOBAL STATE ---
# using a dict to allow mutable access inside the lifespan context.
pipeline_state = {
    "retriever": None,
    "generator": None,
}


# --- LIFESPAN MANAGER ( The "Startup/shutdown")
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the application.
    1. Initializes the heavy ML models.
    2. Yields control to the API to handle requests.
    3. Cleans up resources gracefully.
    """
    logger.info("API Startup: Initializing RAG Models...")

    try:
        pipeline_state["retriever"] = Retriever()
        pipeline_state["generator"] = RAGGenerator()
        logger.success("RAG Pipeline initialized and ready to serve.")
        yield
    except Exception as e:
        logger.critical(f"Fatal Error during startup: {e}")
        raise e
    finally:
        logger.info("API Shutdown: Cleaning up resources...")
        pipeline_state.clear()


app = FastAPI(
    title="RAG Inference Engine",
    description="API for Semantic Search and Retrieval Augmented Generation.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# --- The Contracts ----
class SourceDocument(BaseModel):
    """Represents a single retrieved chunk of context."""

    source: str
    context_preview: str
    score: float


class QueryRequest(BaseModel):
    """Input payload for a RAG query."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Question to ask.",
        example="What is Scaled Dot-Product Attention?",
    )
    k: int = Field(default=3, ge=1, le=10, description="Number of context chunks to retrieve.")
    use_hybrid_search: bool = Field(
        default=True, description="Whether to use keyword boosting logic."
    )


class QueryResponse(BaseModel):
    """Standardized response payload."""

    answer: str = Field(..., description="The generated answer from the LLM.")
    processing_time_ms: float = Field(..., description="Total processing time in ms.")
    sources: List[SourceDocument] = Field(
        default=[], description="List of supporting documents found."
    )


# --- ENDPOINTS ---


@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
def health_check():
    """
    K8s Health/Liveness Probe.
    Returns 200 OK if models are loaded. Returns 503 if initializing.
    """
    if not pipeline_state["retriever"] or not pipeline_state["generator"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="System is initializing."
        )
    return {"status": "healthy", "service": "rag-pipeline"}


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
def run_query(request: QueryRequest):
    """
    **Execute RAG Pipeline**

    1. **Retrieve**: Searches the vector database for relevant chunks
                    (Hybrid Search).
    2. **Generate**: Uses the LLM to synthesize an answer based
                    *only* on the context.

    Note: This endpoint is CPU-bound. It runs in a threadpool
            to avoid blocking the server.
    """
    t_start = time.time()

    retriever: Retriever = pipeline_state["retriever"]
    generator: RAGGenerator = pipeline_state["generator"]

    if not retriever or not generator:
        logger.error("Request received but models are not ready.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="System is initializing."
        )

    logger.info(f"Processing query: '{request.query}'")

    try:
        docs_with_scores = retriever.search(query=request.query, k=request.k)

        if not docs_with_scores:
            logger.warning("No relevant documents found.")
            return QueryResponse(
                answer="I could not find any relevant information in the knowledge base to answer your question.",
                processing_time_ms=(time.time() - t_start) * 1000,
                sources=[],
            )

        raw_docs = [doc for doc, _ in docs_with_scores]

        full_answer = ""
        for token in generator.generate_stream(request.query, raw_docs):
            full_answer += token

        response_sources = [
            SourceDocument(
                source=doc.metadata.get("source", "Unknown"),
                context_preview=doc.page_content[:100] + "...",
                score=score,
            )
            for doc, score in docs_with_scores
        ]

        t_end = time.time()

        return QueryResponse(
            answer=full_answer,
            processing_time_ms=(t_end - t_start) * 1000,
            sources=response_sources,
        )
    except Exception as e:
        logger.error(f"Pipeline processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Processing Error: {str(e)}",
        )
