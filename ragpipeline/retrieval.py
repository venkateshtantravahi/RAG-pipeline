"""
Retrieval Module (Reranking / Contextual Compression)
-----------------------------------------------------
Architecture:
1. Base Retriever: Fetches top-k * 4 candidates (High Recall).
2. Compressor: Cross-Encoder (Local) grades them (High Precision).
3. Result: Returns top-k highly relevant documents.
"""

from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from ragpipeline.config import CROSS_ENCODER_MODEL, EMBEDDING_MODEL_NAME, VECTOR_DB_DIR


class Retriever:
    def __init__(self):
        """
        Initializes the 2-stage pipeline.
        """
        if not VECTOR_DB_DIR.exists():
            raise FileNotFoundError(f"Vector DB not found at {VECTOR_DB_DIR}.")

        # Fast Vector Search
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store = Chroma(
            persist_directory=str(VECTOR_DB_DIR),
            embedding_function=self.embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )

        # retrieve 4x documents using base retriever
        # this ensures reranking has enough candidates
        self.base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})

        # Re-ranking
        # we use a small, fast local model specifically designed for ranking.
        # it takes (query, documents) pairs and outputs a similarity score.
        logger.info("Loading Cross-Encoder Model (cross-encoder/ms-macro-MiniLM-L-6-v2)")
        model = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_MODEL)

        self.compressor = CrossEncoderReranker(model=model, top_n=3)

        # Pipeline
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.base_retriever,
        )
        logger.success("Reranking Pipeline Initialized")

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """
        Executes the Reranking Pipeline.
        Args:
            query: User's question.
            k: Number of final docs to return (Default: 3).
        """
        # Update the compressor's "top_n" dynamically based on request
        self.compressor.top_n = k

        logger.info(f"Reranking Query: '{query}'")

        # Fetch 20 -> Score 20 -> Sort -> return top_k
        results = self.compression_retriever.invoke(query)

        # Document return
        docs_with_scores = []
        for doc in results:
            score = doc.metadata.get("relevance_score", 0.0)
            docs_with_scores.append((doc, score))

        return docs_with_scores


if __name__ == "__main__":
    # Test Case for Individual API call
    try:
        r = Retriever()
        query = "How does multi-head attention differ from standard attention?"

        docs = r.search(query, k=3)

        print(f"Top 3 Reranked Results for: '{query}'")
        for i, (doc, score) in enumerate(docs):
            print(f"\n[{i + 1}] Score: {score:.4f} | Source: {doc.metadata.get('source')}")
            print(f"    {doc.page_content[:200]}...")
    except Exception as ex:
        logger.error(f"Retrieval failed: {ex}")
