from unittest.mock import patch

from langchain_core.documents import Document
import pytest
from ragpipeline.retrieval import Retriever


# -- Mock Data ---
@pytest.fixture
def mock_docs():
    """Returns a list of (Document, score) tuples as Chroma would return them."""
    return [
        (
            Document(
                page_content="This is a perfect match with keyword.", metadata={"source": "a.pdf"}
            ),
            0.2,
        ),
        (
            Document(
                page_content="This is okay but lacks the specific word.",
                metadata={"source": "b.pdf"},
            ),
            0.5,
        ),
        (Document(page_content="This is total garbage.", metadata={"source": "c.pdf"}), 0.9),
    ]


# --- Tests ----


@patch("ragpipeline.retrieval.VECTOR_DB_DIR")
@patch("ragpipeline.retrieval.Chroma")
@patch("ragpipeline.retrieval.HuggingFaceEmbeddings")
def test_init_db_not_found(mock_embeddings, mock_chroma, mock_path):
    """Test if Retriever raises error when DB directory is missing."""
    mock_path.exists.return_value = False

    with pytest.raises(FileNotFoundError):
        Retriever()


@patch("ragpipeline.retrieval.VECTOR_DB_DIR")
@patch("ragpipeline.retrieval.Chroma")
@patch("ragpipeline.retrieval.HuggingFaceEmbeddings")
def test_search_thresholding(mock_embeddings, mock_chroma, mock_path, mock_docs):
    """Test if the retriever filters out results above the threshold."""

    mock_path.exists.return_value = True
    mock_db_instance = mock_chroma.return_value
    mock_db_instance.similarity_search_with_score.return_value = mock_docs

    retriever = Retriever()

    results = retriever.search("test query", k=5, score_threshold=0.6)

    assert len(results) == 2
    assert results[0][1] == 0.2
    assert results[1][1] == 0.5


@patch("ragpipeline.retrieval.VECTOR_DB_DIR")
@patch("ragpipeline.retrieval.Chroma")
@patch("ragpipeline.retrieval.HuggingFaceEmbeddings")
def test_keyword_boosting(mock_embeddings, mock_chroma, mock_path):
    """Test if the Hybrid Search logic boosts exact matches."""
    mock_path.exists.return_value = True

    retriever = Retriever()

    doc = Document(page_content="The scaled dot-product attention is key.")
    original_score = 0.5
    query = "Scaled Dot-Product Attention"

    new_score = retriever._boost_score_for_keywords(doc, original_score, query)

    assert new_score < original_score
    assert new_score == 0.25
