"""Configuration settings for Multi-Query RAG system."""
import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if (PROJECT_ROOT / "scifact_vector_db" / "scifact_chroma_db").exists():
    DATA_DIR = PROJECT_ROOT / "scifact_vector_db"
else:
    DATA_DIR = PROJECT_ROOT / "data" / "scifact"

CHROMA_DIR = DATA_DIR / "scifact_chroma_db"
EVAL_DIR = DATA_DIR / "scifact_eval"

# Ollama LLM Configuration
# Uses local MedGemma 1.5 4B IT model with 16k context window (configurable to 32k)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))  # 16k context window to bound VRAM

# Generation Temperatures
QUERY_GEN_TEMPERATURE = 0.4      # Slight diversity for orthogonal query generation
GROUNDED_GEN_TEMPERATURE = 0.0   # Deterministic greedy decoding for zero hallucination
JUDGE_TEMPERATURE = 0.0          # Impartial deterministic scoring

# Embedding Configuration
MODELS_DIR = PROJECT_ROOT / "models" / "bge-base-en-v1.5"
EMBEDDING_MODEL_NAME = str(MODELS_DIR) if (MODELS_DIR.exists() and any(MODELS_DIR.iterdir())) else "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")  # "auto", "cuda", or "cpu"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Retrieval & Fusion Parameters
DEFAULT_NUM_QUERIES = 4
DEFAULT_TOP_K_PER_QUERY = 5
DEFAULT_FINAL_TOP_K = 5
DEFAULT_RRF_K = 60  # Standard smoothing parameter for Reciprocal Rank Fusion

# ChromaDB collection name
CHROMA_COLLECTION_NAME = "scifact_collection"
