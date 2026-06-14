"""The re-ranking stage, kept separate so it can be toggled on and off.

A cross-encoder re-ranker scores each (query, passage) pair jointly, which is more
accurate than the bi-encoder similarity used for first-pass retrieval but too slow
to run over the whole corpus. So it runs as a second stage over the top-k
candidates. Toggling it is exactly the ablation that shows its lift.

Runs locally and free.
"""

from __future__ import annotations

from llama_index.core.postprocessor import SentenceTransformerRerank

from .config import RERANK_MODEL, RERANK_TOP_N


def build_reranker(top_n: int = RERANK_TOP_N, model: str = RERANK_MODEL) -> SentenceTransformerRerank:
    return SentenceTransformerRerank(model=model, top_n=top_n)
