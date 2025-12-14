"""
Retrieval Module (Production Grade)
-----------------------------------
Handles semantic search against the ChromaDB vector store.
Optimized for low-latency HNSW lookups.
"""

import time
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from ragpipeline.config import (
    EMBEDDING_MODEL_NAME,
    VECTOR_DB_DIR,
)


class Retriever:
    def __init__(self):
        """
        Initializes the retrieval engine.

        OPTIMIZATION:
        - Loads the Embedding Model into RAM Once (Singleton Pattern).
        - Connects to the existing HNSW index in ChromaDB.
        """
        t0 = time.time()
        logger.info(f"Initializing Retriever with model: {EMBEDDING_MODEL_NAME}....")

        # Load the model
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        # Connect to Database
        if not VECTOR_DB_DIR.exists():
            logger.critical(f"Vector DB not found at {VECTOR_DB_DIR}")
            raise FileNotFoundError("ChromeDB index not found")

        self.vector_db = Chroma(
            persist_directory=str(VECTOR_DB_DIR), embedding_function=self.embedding_model
        )

        logger.success(f"Retriver initialized in {time.time() - t0:.2f} seconds")

    def _boost_score_for_keywords(self, doc: Document, score: float, query: str) -> float:
        """
        Custom Logic: If the exact query phrase appears in the text,
        artificially improve (lower) the distance score.
        """
        # Normalize text for matching
        doc_text = doc.page_content.lower()
        query_text = query.lower()

        # 1. Exact Phrase Match
        if query_text in doc_text:
            logger.debug(f"Exact Match found in {doc.metadata.get('source')} Boosting Score.")
            return score * 0.5

        # Partial keyword match
        query_words = set(query_text.split())
        doc_words = set(doc_text.split())
        common_words = query_words.intersection(doc_words)

        if len(common_words) / len(query_words) > 0.7:
            return score * 0.8

        return score

    def search(
        self, query: str, k: int = 5, score_threshold: float = 0.8
    ) -> List[Tuple[Document, float]]:
        """
        Performs semantic search with distance filtering.

        Args:
            query (str): The user's natural language question.
            k (int): Max number of chunks to retrieve.
            score_threshold (float): Cutoff for relevance (Lower distance = Better match).
                                     0.0 = Identical, 1.0 = Unrelated.

        Returns:
            List of (Document, score) tuples.
        """
        t_start = time.time()

        # HNSW Search
        # similarity_search_with_score to get Cosine Distance
        broad_k = k * 3
        raw_results = self.vector_db.similarity_search_with_score(query, k=broad_k)

        reranked_results = []
        for doc, original_score in raw_results:
            # Chroma returns L2 distance or Cosine Distance
            # For this model, lower is better, We filter out weak matches.
            new_score = self._boost_score_for_keywords(doc, original_score, query)
            reranked_results.append((doc, new_score))

        reranked_results.sort(key=lambda x: x[1])

        results = []
        for doc, score in reranked_results:
            if score <= score_threshold:
                results.append((doc, score))

        results = results[:k]

        latency_ms = (time.time() - t_start) * 1000
        logger.info(f"Retrieval: Found {len(results)} docs for '{query}' in  {latency_ms:.2f} ms")

        return results


if __name__ == "__main__":
    # ---- Test Block -----
    # Run this to benchmark your latency: python -m ragpipeline.retrieval
    try:
        retriever = Retriever()

        test_query = "What is attention mechanism?"
        results = retriever.search(test_query, k=3)

        print(f"\n---- Search Results for: '{test_query}' ----")
        for doc, score in results:
            print(f"\n[Distance: {score:.4f}] Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content: {doc.page_content.replace('\n', ' ')[:150]}...")

    except Exception as ex:
        logger.error(f"Retrieval failed: {ex}")
