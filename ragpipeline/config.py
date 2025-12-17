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

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_MODEL_NAME = "llama3.2:1b"

# --- CONFIGURABLE PROMPT TEMPLATE ---
# We use a "System" message to enforce behavior and a "Human" message for the data.
RAG_SYSTEM_PROMPT = """
<role>
You are an Expert AI Research Assistant specialized in Technical Documentation.
Your goal is to answer the user's question accurately, concisely, and strictly based *only* on the provided context.
</role>

<rules>
1. **Strict Context Adherence**: 
   - Use ONLY the information provided in the <context> tags below.
   - Do NOT use outside knowledge or training data to answer.
   - If the answer is not in the context, strictly state: "I cannot find the answer in the provided documents."

2. **Citation Style**:
   - Every factual claim must be backed by a source.
   - Use the format: `(Source: filename)` at the end of the sentence.
   - Example: "Transformer models use self-attention mechanisms (Source: attention_is_all_you_need.pdf)."

3. **Reasoning**:
   - Think step-by-step before answering.
   - If the question involves math or architecture, explain the logic clearly.

4. **Tone**:
   - Professional, technical, and direct.
</rules>

<context>
{context}
</context>
"""

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
