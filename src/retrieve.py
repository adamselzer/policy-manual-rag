"""Retrieval engine: vector-only or hybrid (BM25 + vector via RRF), with optional
metadata filtering and an optional cross-encoder re-rank stage.

Everything here is LLM-free. In particular the hybrid fusion uses num_queries=1 so
no LLM is invoked to generate query variants; the only models are the local
embedder and (optionally) the local re-ranker.
"""

from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import QueryBundle, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

from .config import EMBED_MODEL, RERANK_TOP_N, RETRIEVE_TOP_K
from .rerank import build_reranker
from .schema import RetrievedPassage

_EMBED_CACHE: dict[str, HuggingFaceEmbedding] = {}


def _embedder(model: str) -> HuggingFaceEmbedding:
    # One embedder per model name, reused across engines so we don't reload weights.
    if model not in _EMBED_CACHE:
        _EMBED_CACHE[model] = HuggingFaceEmbedding(model_name=model)
    return _EMBED_CACHE[model]


class RetrievalEngine:
    """Builds the indexes from a node set and retrieves passages for a query.

    mode:        "vector" | "hybrid"
    use_rerank:  apply the cross-encoder re-rank stage
    top_k:       candidates pulled in the first pass
    final_n:     passages returned (the re-rank top-n, or the first-pass top-n if
                 re-rank is off) so every configuration returns the same count.
    """

    def __init__(
        self,
        nodes: list[TextNode],
        *,
        mode: str = "hybrid",
        use_rerank: bool = True,
        embed_model: str = EMBED_MODEL,
        top_k: int = RETRIEVE_TOP_K,
        final_n: int = RERANK_TOP_N,
    ):
        if mode not in {"vector", "hybrid"}:
            raise ValueError(f"mode must be 'vector' or 'hybrid', got {mode!r}")
        self.mode = mode
        self.use_rerank = use_rerank
        self.top_k = top_k
        self.final_n = final_n

        embed = _embedder(embed_model)
        self._index = VectorStoreIndex(nodes, embed_model=embed, show_progress=False)
        vector_retriever = self._index.as_retriever(similarity_top_k=top_k)

        if mode == "hybrid":
            bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
            self._retriever = QueryFusionRetriever(
                [vector_retriever, bm25],
                similarity_top_k=top_k,
                num_queries=1,  # no LLM-based query expansion
                mode="reciprocal_rerank",
                use_async=False,
                llm=MockLLM(),  # never called at num_queries=1; avoids the OpenAI default
            )
        else:
            self._retriever = vector_retriever

        self._reranker = build_reranker(top_n=final_n) if use_rerank else None

    def retrieve(self, query: str, section_filter: set[str] | None = None) -> list[RetrievedPassage]:
        results = self._retriever.retrieve(query)
        if section_filter:
            results = [r for r in results if r.node.metadata.get("section_id") in section_filter]
        if self._reranker:
            results = self._reranker.postprocess_nodes(results, query_bundle=QueryBundle(query))
        else:
            results = results[: self.final_n]
        return [_to_passage(r) for r in results]


def _to_passage(nws) -> RetrievedPassage:
    md = nws.node.metadata
    return RetrievedPassage(
        section_id=md.get("section_id", "UNKNOWN"),
        title=md.get("title", ""),
        url=md.get("url", ""),
        heading=md.get("heading", ""),
        text=nws.node.get_content(),
        score=float(nws.score) if nws.score is not None else 0.0,
        strategy=md.get("strategy", ""),
    )
