"""
Dataset Module
--------------
This script handles the data acquisition step.
It reads a JSON registry of papers and downloads them to the raw data directory.

Usage:
    python -m ragpipeline.dataset
    or
    make data
"""

import json
from pathlib import Path
from typing import Dict

from loguru import logger
import requests
from tqdm import tqdm
import typer

from ragpipeline.config import PROJ_ROOT, RAW_DATA_DIR

app = typer.Typer()

# Path to the registry file
PAPERS_CONFIG_PATH = PROJ_ROOT / "data" / "papers.json"


def load_papers_config() -> Dict[str, str]:
    """Loads the list of papers from the JSON configuration file."""
    if not PAPERS_CONFIG_PATH.exists():
        logger.error(f"Configuration file {PAPERS_CONFIG_PATH} does not exist.")
        raise typer.Exit(code=1)
    with open(PAPERS_CONFIG_PATH, "r") as f:
        return json.load(f)


def download_file(url: str, dest_path: Path) -> None:
    """
    Downloads a single file from a URL to a destination path.
    Skips download if the file already exists.
    """
    if dest_path.exists():
        # Debug level logging avoids cluttering the terminal for skipped files
        logger.debug(f"File {dest_path} already exists.")
        return

    try:
        # User-Agent is required to avoid 403 Forbidden errors from ArXiv
        headers = {"User-Agent": "Mozilla/5.0"}

        # Stream the download to handle large files efficiently
        response = requests.get(url, stream=True, timeout=30, headers=headers)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(dest_path, "wb") as f,
            tqdm(
                desc=dest_path.name,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)

        logger.success(f"Downloaded: {dest_path.name}")

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")


@app.command()
def main(
    output_dir: Path = RAW_DATA_DIR,
    force: bool = typer.Option(False, help="Force re-download all files."),
):
    """
    Main entry point to download the dataset.
    """
    logger.info(f"Start downloading dataset: {output_dir.name}")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load registry
    papers = load_papers_config()
    logger.info(f"Found {len(papers)} papers in registry.")

    # Processing loop
    for filename, url in papers.items():
        dest_path = output_dir / filename

        if force and dest_path.exists():
            dest_path.unlink()

        download_file(url, dest_path)

    logger.success("Dataset download complete.")


if __name__ == "__main__":
    app()
