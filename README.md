# RAG-Pipeline-Tutorial

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" alt="" />
</a>

An open-source Retrieval Augmented Generation (RAG) pipeline designed to answer technical questions about Large Language Model (LLM) architectures. 

This project uses **ArXiv** papers as its knowledge base and runs entirely locally using **Ollama** and **ChromaDB**.

---

## Quick Start

You can set up the entire project with just three commands.

### 1. Prerequisite
Ensure you have `Make`, `Conda` and [Ollama](https://ollama.com/) installed.

### 2. Install
Run these commands in your terminal:

```bash
# 1. Create the Conda environment
make create_environment

# 2. Activate the environment
conda activate rag-pipeline

# 3. Install dependencies
make requirements
```
### 3. Get the Data
Download the seminal ML papers (Attention is All You Need, Llama 2, etc.) automatically:
```bash
make data
```
This downloads the PDFs into `data/raw/`.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         RAGPipeline and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── RAGPipeline   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes RAGPipeline a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

### 4. Architecture: Data Ingestion (ETL)

We do not use standard text extraction. In technical domains (AI Research, Finance), losing the structure of **tables** and **equations** means losing the answer.

### The Strategy: "Markdown First"
Instead of flattening PDFs into plain text strings, we convert them into **Markdown**.

| Feature       | Plain Text (Standard)                                    | Markdown (Our Approach)                        |
|:--------------|:---------------------------------------------------------|:-----------------------------------------------|
| **Tables**    | Flattens into messy lines. <br> *(e.g. `Model 90% 80%`)* | Preserves structure using pipes. <br> *(e.g. ` | Model | Acc |`)* |
| **Math**      | Garbles equations. <br> *(e.g. `E m c 2`)*               | Preserves LaTeX. <br> *(e.g. `$E=mc^2$`)*      |
| **Structure** | Loses section boundaries.                                | Keeps Headers (`#`, `##`).                     |

### The Pipeline
1.  **Conversion:** We use **[Marker](https://github.com/datalab-to/marker)** to convert PDFs to Markdown + Images.
2.  **Chunking:** We use **Semantic Chunking**. instead of blindly cutting every 1000 characters, we split by **Headers** (`# Introduction`, `## Methods`) first. This keeps related text together.
3.  **Embedding:** Text is converted to 384-dimensional vectors using `sentence-transformers/all-MiniLM-L6-v2`.
4.  **Storage:** Vectors are stored in **ChromaDB**, using an **HNSW (Hierarchical Navigable Small World)** index for millisecond-latency search.

### Usage
To run the full ingestion pipeline (PDF → Vector DB):

```bash
# Process all PDFs in data/raw
make ingest

# Test with just 1 PDF to verify
python -m ragpipeline.ingestion --limit 1
```

### 5. Architecture: Retrieval Engine

We use a **Hybrid Search** strategy to balance semantic understanding with keyword precision.

### The Problem
Standard Vector Search (Dense Retrieval) often fails on specific technical terms (e.g., *"Scaled Dot-Product Attention"*) because the embedding model dilutes the specific phrase into a generic "math/AI" vector.

### Our Solution
We implemented a custom **Reranking Layer** in `retrieval.py`:

1.  **Broad Phase:** Retrieve top `3 * k` candidates using standard Vector Search (HNSW Index).
2.  **Rerank Phase:** Apply a "Keyword Boost" algorithm:
    * If the exact query phrase appears in the chunk, the distance score is **halved** (improving rank).
    * If >70% of query keywords appear, the score is reduced by **20%**.
3.  **Filter Phase:** Strict cutoff for any result with a distance score > `0.8` to prevent hallucinations.

Usage:
```bash
# Run standalone semantic search test
python -m ragpipeline.retrieval
```

