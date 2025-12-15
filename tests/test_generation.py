from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
import pytest
from ragpipeline.generation import RAGGenerator


@pytest.fixture
def mock_retrieved_docs():
    return [
        Document(page_content="Transformers are fast.", metadata={"source": "ai_paper.pdf"}),
        Document(page_content="LSTMs are slow.", metadata={"source": "old_paper.pdf"}),
    ]


@patch("ragpipeline.generation.ChatOllama")
def test_generator_initialization(mock_ollama):
    """Test if Generator initializes with the correct model config."""
    generator = RAGGenerator()

    mock_ollama.assert_called_once()
    assert generator.chain is not None


def test_format_docs(mock_retrieved_docs):
    """Test if the Augmentation step formats context correctly."""
    with patch("ragpipeline.generation.ChatOllama"):
        generator = RAGGenerator()

    formatted_text = generator.format_docs(mock_retrieved_docs)

    assert "--- SOURCE: ai_paper.pdf ---" in formatted_text
    assert "Transformers are fast." in formatted_text
    assert "--- SOURCE: old_paper.pdf ---" in formatted_text
    assert "\n\n" in formatted_text


@patch("ragpipeline.generation.ChatOllama")
def test_generate_stream_no_docs(mock_ollama):
    """Test if generator handles empty context gracefully."""
    generator = RAGGenerator()

    stream = generator.generate_stream("Query", [])

    response = list(stream)
    assert len(response) == 1
    assert "could not find any relevant documents" in response[0]


@patch("ragpipeline.generation.ChatOllama")
def test_generate_stream_success(mock_ollama, mock_retrieved_docs):
    """Test if generator calls the chain and yields tokens."""
    generator = RAGGenerator()

    mock_chain = MagicMock()
    mock_chain.stream.return_value = ["Token1", "Token2", "Token3"]
    generator.chain = mock_chain

    stream = generator.generate_stream("What are Transformers?", mock_retrieved_docs)
    response = list(stream)

    assert response == ["Token1", "Token2", "Token3"]

    mock_chain.stream.assert_called_once()
    call_args = mock_chain.stream.call_args[0][0]
    assert "context" in call_args
    assert "question" in call_args
    assert "Transformers are fast" in call_args["context"]
