import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── LLM backend: "ollama" (fully free, local) or "groq" (free tier, cloud) ──
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")   # set in .env

# ── Ollama (local, free, no limits) ──────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")          # or mistral, gemma2

# ── Groq (free tier: 30 req/min, fast) ───────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama3-70b-8192")   # free on Groq

# ── SEC EDGAR ─────────────────────────────────────────────────────────────
EDGAR_BASE_URL  = "https://data.sec.gov"
USER_AGENT      = os.getenv("USER_AGENT", "SECAnalyzer your@email.com")

# ── Storage ───────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data" / "raw"
CHROMA_DIR      = BASE_DIR / "data" / "chroma_db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ── Embeddings (free, local) ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # ~80MB, downloads once automatically

# ── Chunking ──────────────────────────────────────────────────────────────
CHUNK_SIZE      = 800    # words
CHUNK_OVERLAP   = 100
MIN_CHUNK_WORDS = 30

# ── Retrieval ─────────────────────────────────────────────────────────────
TOP_K           = 6

# ── Drift ─────────────────────────────────────────────────────────────────
DRIFT_SECTIONS = [
    "risk factors",
    "competition",
    "management discussion",
    "business overview",
    "forward looking statements",
    "AI and technology strategy",
    "macroeconomic conditions",
    "regulatory and legal risks",
    "revenue and growth",
]