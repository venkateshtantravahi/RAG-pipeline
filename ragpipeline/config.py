from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
VECTOR_DB_DIR = DATA_DIR / "external" / "chroma_db"

MODELS_DIR = PROJ_ROOT / "models"


# Model Configurations
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

LLM_MODEL_NAME = "llama3.2:1b"

# --- CONFIGURABLE PROMPT TEMPLATE ---
# We use a "System" message to enforce behavior and a "Human" message for the data.
RAG_SYSTEM_PROMPT = """You are a specialized AI Research Assistant. 
Your task is to answer the user's question STRICTLY based on the provided context.

RULES:
1. Use ONLY the information from the context.
2. If the answer is not in the context, say "I cannot find the answer in the provided documents."
3. Do not make up information or use outside knowledge.
4. Cite the source using [Source: filename] if possible.
5. Format mathematical equations using LaTeX (e.g., $E=mc^2$).

CONTEXT:
{context}
"""

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
