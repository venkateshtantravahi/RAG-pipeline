from unittest.mock import patch

from langchain_core.documents import Document
import pytest
from ragpipeline.retrieval import Retriever


@patch("ragpipeline.retrieval.VECTOR_DB_DIR")
def test_init_db_not_found(mock_path):
    """Test if Retriever raises error when DB directory is missing."""
    mock_path.exists.return_value = False
    with pytest.raises(FileNotFoundError):
        Retriever()


@patch("ragpipeline.retrieval.VECTOR_DB_DIR")
@patch("ragpipeline.retrieval.Chroma")
@patch("ragpipeline.retrieval.HuggingFaceEmbeddings")
@patch("ragpipeline.retrieval.HuggingFaceCrossEncoder")
@patch("ragpipeline.retrieval.CrossEncoderReranker")
# CRITICAL: Patch the class to avoid Pydantic validation errors on Mock objects
@patch("ragpipeline.retrieval.ContextualCompressionRetriever")
def test_retriever_initialization(
    mock_compression_retriever,
    mock_reranker,
    mock_cross_encoder,
    mock_embeddings,
    mock_chroma,
    mock_path,
):
    """Test if the retriever initializes the 2-stage pipeline (Base + Reranker)."""
    mock_path.exists.return_value = True

    retriever = Retriever()

    # Verify components are initialized
    mock_chroma.assert_called()
    mock_cross_encoder.assert_called()
    mock_reranker.assert_called()

    # Ensure the pipeline was assembled
    mock_compression_retriever.assert_called_once()
    assert retriever.compression_retriever is not None


@patch("ragpipeline.retrieval.VECTOR_DB_DIR")
@patch("ragpipeline.retrieval.Chroma")
@patch("ragpipeline.retrieval.HuggingFaceEmbeddings")
@patch("ragpipeline.retrieval.HuggingFaceCrossEncoder")
@patch("ragpipeline.retrieval.CrossEncoderReranker")
@patch("ragpipeline.retrieval.ContextualCompressionRetriever")
def test_search_pipeline_execution(
    mock_compression_retriever,
    mock_reranker,
    mock_cross_encoder,
    mock_embeddings,
    mock_chroma,
    mock_path,
):
    """Test if the search method invokes the reranking pipeline and parses results."""
    mock_path.exists.return_value = True

    # Setup mock return from the pipeline
    # The pipeline returns Documents, usually with 'relevance_score' in metadata
    mock_doc1 = Document(
        page_content="High relevance doc", metadata={"relevance_score": 0.95, "source": "a.pdf"}
    )
    mock_doc2 = Document(
        page_content="Med relevance doc", metadata={"relevance_score": 0.75, "source": "b.pdf"}
    )

    # Mock the pipeline instance created in __init__
    mock_pipeline_instance = mock_compression_retriever.return_value
    mock_pipeline_instance.invoke.return_value = [mock_doc1, mock_doc2]

    retriever = Retriever()

    # Execute Search
    results = retriever.search("test query", k=2)

    # 1. Verify we updated the 'top_n' on the compressor
    mock_reranker_instance = mock_reranker.return_value
    assert mock_reranker_instance.top_n == 2

    # 2. Verify the pipeline was invoked
    mock_pipeline_instance.invoke.assert_called_with("test query")

    # 3. Verify output format (List of tuples: Doc, Score)
    assert len(results) == 2
    assert results[0][0] == mock_doc1
    assert results[0][1] == 0.95
    assert results[1][1] == 0.75
