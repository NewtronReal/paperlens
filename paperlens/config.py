"""Central configuration. Everything is overridable through environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("PAPERLENS_DATA", ROOT / "data"))
CKPT_DIR = Path(os.getenv("PAPERLENS_CKPT", ROOT / "checkpoints"))
RESULTS_DIR = Path(os.getenv("PAPERLENS_RESULTS", ROOT / "results"))
FRONTEND_DIR = Path(os.getenv("PAPERLENS_FRONTEND", ROOT / "frontend" / "dist"))  # Vite build output
CACHE_DB = DATA_DIR / "cache.sqlite"

# External APIs (all optional, but an S2 key makes everything much faster and more reliable)
S2_API_KEY = os.getenv("S2_API_KEY") or None
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or None

# Optional LLM for the structured one-glance summary. Any OpenAI-compatible endpoint works,
# e.g. Ollama: LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
LLM_MODEL = os.getenv("LLM_MODEL") or None

# Demo mode uses a deterministic synthetic citation world: no network needed.
DEMO = os.getenv("PAPERLENS_DEMO", "0") == "1"

# Label order used everywhere (SciCite label set)
INTENT_LABELS = ["background", "method", "result"]

# Analysis limits (keep the API budget sane)
MAX_REFERENCES = int(os.getenv("PAPERLENS_MAX_REFS", 500))
MAX_CITATIONS = int(os.getenv("PAPERLENS_MAX_CITS", 400))
LINEAGE_DEPTH = int(os.getenv("PAPERLENS_LINEAGE_DEPTH", 2))
LINEAGE_BRANCH = int(os.getenv("PAPERLENS_LINEAGE_BRANCH", 3))
