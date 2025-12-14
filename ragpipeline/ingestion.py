"""
Ingestion Module (Verified Marker API)
--------------------------------------
This script handles the ETL pipeline using 'marker-pdf' to convert PDFs into
Markdown (preserving Tables & Math) and storing them in ChromaDB.

Usage:
    python -m ragpipeline.ingestion --limit 5
"""

from pathlib import Path
import shutil
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# --- LangChain Imports ---
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

# --- Marker Imports  ---
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from tqdm import tqdm
import typer

# --- Config Imports ---
from ragpipeline.config import (
    EMBEDDING_MODEL_NAME,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    VECTOR_DB_DIR,
)

app = typer.Typer()


def convert_pdf_to_markdown(pdf_path: Path, output_base_dir: Path) -> Optional[str]:
    """
    Converts a single PDF to Markdown using the correct Marker API.
    """
    try:
        logger.info(f"Converting: {pdf_path.name}")

        # 1. Initialize Models (Lazy loading)
        model_dict = create_model_dict()

        # 2. Configure Converter
        converter = PdfConverter(
            artifact_dict=model_dict,
        )

        # 3. Run Conversion
        # Render returns a comprehensive object containing text, images, and metadata
        rendered = converter(str(pdf_path))

        # 4. Extract Text and Images
        full_text, _, images = text_from_rendered(rendered)

        # 5. Save Images
        doc_name = pdf_path.stem
        doc_images_dir = output_base_dir / "images" / doc_name
        doc_images_dir.mkdir(parents=True, exist_ok=True)

        for filename, image in images.items():
            save_path = doc_images_dir / filename
            image.save(save_path)

        return full_text

    except Exception as e:
        logger.error(f"Marker failed on {pdf_path.name}: {e}")
        return None


def chunk_markdown_semantically(markdown_text: str, source_name: str) -> List[Document]:
    """
    Splits Markdown by Headers (#, ##) to preserve context.
    """
    if not markdown_text:
        return []

    # 1. Split by Logic (Headers)
    # This ensures "Section 2.1" stays together
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_splits = markdown_splitter.split_text(markdown_text)

    # 2. Split by Size (Recursive fallback)
    # We use 1000 chars because Markdown tables and LaTeX math need width
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )

    final_docs = text_splitter.split_documents(header_splits)

    # 3. Add Source Metadata
    for doc in final_docs:
        doc.metadata["source"] = source_name

    return final_docs


def embed_and_store(documents: List[Document], batch_size: int = 50):
    """
    Embeds documents and upserts them to ChromaDB.
    """
    if not documents:
        return

    logger.info(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    logger.info(f"Upserting {len(documents)} chunks to Vector DB...")

    # Process in batches to manage RAM
    total_docs = len(documents)
    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        Chroma.from_documents(
            documents=batch, embedding=embeddings, persist_directory=str(VECTOR_DB_DIR)
        )
        logger.debug(f"Processed batch {i + 1}-{min(i + batch_size, total_docs)}")

    logger.success("Vector Store Successfully Updated!")


@app.command()
def main(
    limit: int = typer.Option(None, help="Limit number of PDFs to process (for testing)."),
    reset_db: bool = typer.Option(False, help="Delete existing Vector DB before starting."),
):
    """
    Main Ingestion Pipeline: PDF -> Marker (MD) -> Chunk -> Vector DB
    """
    # 1. Clean Slate (Optional)
    if reset_db and VECTOR_DB_DIR.exists():
        shutil.rmtree(VECTOR_DB_DIR)
        logger.warning("Deleted existing Vector DB.")

    # 2. Find Files
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDFs found in data/raw. Did you run 'make data'?")
        raise typer.Exit(code=0)

    if limit:
        pdf_files = pdf_files[:limit]
        logger.info(f"Testing Mode: Processing top {limit} files.")

    all_staged_docs = []

    # 4. Processing Loop
    for pdf_file in tqdm(pdf_files, desc="Processing Documents"):
        # A. Convert
        md_text = convert_pdf_to_markdown(pdf_file, PROCESSED_DATA_DIR)

        # B. Chunk
        if md_text:
            docs = chunk_markdown_semantically(md_text, pdf_file.name)
            all_staged_docs.extend(docs)
            logger.info(f"Generated {len(docs)} chunks from {pdf_file.name}")

    # 5. Store
    if all_staged_docs:
        embed_and_store(all_staged_docs)
    else:
        logger.warning("No valid documents were processed.")


if __name__ == "__main__":
    app()
