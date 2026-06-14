"""Tests for retrieval metrics and the retrieval engine.

The metric tests are pure. The engine test builds a tiny in-memory index over a
handful of nodes (models are cached after first download) so it stays fast.
"""

from __future__ import annotations

from llama_index.core.schema import TextNode

from eval.retrieval_eval import _metrics_for_question
from src.retrieve import RetrievalEngine


def test_metrics_perfect_retrieval():
    m = _metrics_for_question(["BEM 502", "BEM 502"], {"BEM 502"})
    assert m["hit"] == 1.0
    assert m["context_precision"] == 1.0
    assert m["context_recall"] == 1.0
    assert m["mrr"] == 1.0
    assert m["citable_rate"] == 1.0


def test_metrics_partial_and_rank():
    # gold at rank 2 -> mrr 0.5; one of two relevant -> precision 0.5
    m = _metrics_for_question(["BEM 400", "BEM 213"], {"BEM 213"})
    assert m["hit"] == 1.0
    assert m["context_precision"] == 0.5
    assert m["mrr"] == 0.5


def test_metrics_miss_and_mixed():
    m = _metrics_for_question(["MIXED", "BEM 400"], {"BEM 213"})
    assert m["hit"] == 0.0
    assert m["mrr"] == 0.0
    assert m["citable_rate"] == 0.5  # one MIXED, unattributable


def _tiny_nodes() -> list[TextNode]:
    data = [
        ("BEM 502", "Self-employment income is gross receipts minus allowable business expenses of producing the income."),
        ("BEM 213", "Categorically eligible food assistance groups automatically meet the asset test."),
        ("BEM 554", "Allowable shelter expenses include rent or mortgage; the excess shelter deduction is capped except for senior or disabled members."),
        ("BAM 130", "A collateral contact is a confirmation of information from a source outside the household."),
    ]
    return [
        TextNode(text=t, id_=f"{sid}::0", metadata={"section_id": sid, "title": sid, "url": "http://x", "strategy": "section_aware"})
        for sid, t in data
    ]


def test_engine_retrieves_relevant_section():
    eng = RetrievalEngine(_tiny_nodes(), mode="hybrid", use_rerank=True, top_k=4, final_n=2)
    passages = eng.retrieve("How is self-employment income calculated?")
    assert passages
    assert passages[0].section_id == "BEM 502"
    assert passages[0].url.startswith("http")


def test_engine_rejects_bad_mode():
    import pytest

    with pytest.raises(ValueError):
        RetrievalEngine(_tiny_nodes(), mode="semantic")
