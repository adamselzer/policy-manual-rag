"""Central configuration: models, paths, toggles, and key loading.

Embeddings and re-ranking run locally and free. Only answer generation and the
Ragas LLM-judge metrics need an API key, which is read from a gitignored .env so
it never lands in the repo.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")

# Local models (no API key).
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Generation + judge model (needs ANTHROPIC_API_KEY).
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "claude-sonnet-4-6")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")

# Retrieval defaults.
RETRIEVE_TOP_K = int(os.environ.get("RETRIEVE_TOP_K", "8"))
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "4"))

DATA_DIR = REPO / "data"
SECTIONS_DIR = DATA_DIR / "manual_sections"


def anthropic_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return key or None


def has_llm() -> bool:
    return anthropic_key() is not None
