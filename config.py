"""
Central configuration for the RAG-Sarvam application.
All tunables live here so nothing is scattered across modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Use absolute path so .env is found even when Chainlit changes cwd
_PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_PROJECT_DIR / ".env")

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = _PROJECT_DIR
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
PAGE_IMAGES_DIR = DATA_DIR / "page_images"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
DOCUMENTS_JSON = DATA_DIR / "documents.json"

for _d in (DOCS_DIR, PAGE_IMAGES_DIR, FAISS_INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Sarvam API ───────────────────────────────────────────────────────────
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_BASE_URL = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai")

# Document Intelligence (OCR / Vision)
SARVAM_DOC_INTEL_URL = f"{SARVAM_BASE_URL}/doc-digitization/job/v1"
SARVAM_OCR_LANGUAGE = os.getenv("SARVAM_OCR_LANGUAGE", "en-IN")
SARVAM_OCR_OUTPUT_FORMAT = "md"  # markdown gives best structure
SARVAM_OCR_MAX_PAGES_PER_PDF = int(os.getenv("SARVAM_OCR_MAX_PAGES_PER_PDF", "10"))

# Chat Completions
SARVAM_CHAT_URL = f"{SARVAM_BASE_URL}/v1/chat/completions"
SARVAM_CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-m")
SARVAM_CHAT_TEMPERATURE = float(os.getenv("SARVAM_CHAT_TEMPERATURE", "0.2"))
SARVAM_CHAT_MAX_TOKENS = int(os.getenv("SARVAM_CHAT_MAX_TOKENS", "2048"))

# ── Embeddings ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384  # for all-MiniLM-L6-v2

# ── Chunking ─────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
CHUNK_MIN_SIZE = 80  # discard tiny fragments

# ── Retrieval ────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "8"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
FOOTNOTE_DOWNWEIGHT = float(os.getenv("FOOTNOTE_DOWNWEIGHT", "0.4"))  # multiplier for footnote/amendment chunks

# ── PDF Rendering ────────────────────────────────────────────────────────
PAGE_IMAGE_DPI = int(os.getenv("PAGE_IMAGE_DPI", "200"))

# ── Misc ─────────────────────────────────────────────────────────────────
OCR_POLL_INTERVAL = int(os.getenv("OCR_POLL_INTERVAL", "5"))  # seconds
OCR_MAX_WAIT = int(os.getenv("OCR_MAX_WAIT", "600"))  # seconds
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))  # seconds
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# ── Logging ──────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
