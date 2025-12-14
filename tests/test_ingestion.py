import json
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
import pytest
from ragpipeline import dataset, ingestion


@pytest.fixture
def sample_papers_config(tmp_path):
    """Creates a temporary papers.json file."""
    config_path = tmp_path / "papers.json"
    data = {"TestPaper.pdf": "http://example.com/test.pdf"}
    with open(config_path, "w") as f:
        json.dump(data, f)
    return config_path


@pytest.fixture
def mock_markdown_text():
    return """
    # Introduction
    This is the intro.
    
    ## Methods
    We used a transformer.
    
    # Conclusion
    It worked well.
    """


# ---- TESTS FOR DATASET.PY -------


def test_load_papers_config(sample_papers_config):
    """Test if config is loaded correctly from JSON."""
    # We patch the global variable PAPERS_CONFIG_PATH in the dataset module
    with patch("ragpipeline.dataset.PAPERS_CONFIG_PATH", sample_papers_config):
        config = dataset.load_papers_config()
        assert "TestPaper.pdf" in config
        assert config["TestPaper.pdf"] == "http://example.com/test.pdf"


@patch("ragpipeline.dataset.requests.get")
def test_download_file_sucess(mock_get, tmp_path):
    """Test Sucessful file download."""
    # Mock the network response
    mock_response = MagicMock()
    mock_response.headers.get.return_value = "100"
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    dest_path = tmp_path / "downloaded.pdf"

    dataset.download_file("http://fake-url.com", dest_path)

    # verify file was created and contains data
    assert dest_path.exists()
    assert dest_path.read_bytes() == b"chunk1chunk2"


# ------ TESTS FOR INGESTION.PY -------


def test_chunk_markdown_semantically(mock_markdown_text):
    """Test if headers split Markdown correctly."""
    source_name = "test_doc.pdf"

    docs = ingestion.chunk_markdown_semantically(mock_markdown_text, source_name)
    # We expect 3 chunks (Intro, Methods, Conclusion)
    assert len(docs) >= 3
    # Check Metadata
    assert docs[0].metadata["source"] == source_name
    # Check Content
    assert "Introduction" in docs[0].metadata["Header 1"]
    assert "This is the intro" in docs[0].page_content


@patch("ragpipeline.ingestion.text_from_rendered")
@patch("ragpipeline.ingestion.PdfConverter")
@patch("ragpipeline.ingestion.create_model_dict")
def test_convert_pdf_to_markdown(
    mock_create_model, mock_converter_class, mock_text_from_rendered, tmp_path
):
    """
    Test the conversion logic without loading real AI models.
    """
    # Setup Mocks
    # 1. Mock text extraction to return specific string
    mock_text_from_rendered.return_value = ("Mock Markdown Content", {}, {})

    # 2. Mock the converter instance behavior
    mock_instance = mock_converter_class.return_value
    mock_instance.return_value = MagicMock()  # The 'rendered' object

    # Execution
    input_pdf = tmp_path / "test.pdf"
    input_pdf.touch()
    output_dir = tmp_path / "processed"

    result = ingestion.convert_pdf_to_markdown(input_pdf, output_dir)

    # Assertions
    assert result == "Mock Markdown Content"
    mock_create_model.assert_called_once()
    mock_converter_class.assert_called_once()


@patch("ragpipeline.ingestion.Chroma")
@patch("ragpipeline.ingestion.HuggingFaceEmbeddings")
def test_embed_and_store(mock_embeddings, mock_chroma):
    """Test if embeddings and storage are triggered correctly."""
    docs = [Document(page_content="test", metadata={"source": "test"})]

    ingestion.embed_and_store(docs)

    mock_embeddings.assert_called_once()

    mock_chroma.from_documents.assert_called_once()

    call_args = mock_chroma.from_documents.call_args
    assert call_args[1]["documents"] == docs
