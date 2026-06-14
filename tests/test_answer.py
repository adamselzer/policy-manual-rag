"""Tests for grounded answer assembly: citation parsing and refusal.

These use a fake LLM (duck-typed .complete) so they run without an API key.
"""

from __future__ import annotations

from src.answer import REFUSAL_SENTINEL, _citations_from_text, answer_question
from src.judge import _parse_json
from src.schema import RetrievedPassage


class FakeLLM:
    def __init__(self, text: str):
        self._text = text

    def complete(self, prompt: str):  # mimics llama-index CompletionResponse via str()
        return self._text


def _passages() -> list[RetrievedPassage]:
    return [
        RetrievedPassage(section_id="BEM 502", title="Income From Self-Employment", url="http://x/502", text="...", score=5.0),
        RetrievedPassage(section_id="BEM 554", title="FAP Allowable Expenses", url="http://x/554", text="...", score=2.0),
    ]


def test_no_passages_refuses_without_llm():
    ans = answer_question("anything", [], llm=FakeLLM("should not be called"))
    assert ans.refused is True
    assert ans.citations == []


def test_refusal_sentinel_detected():
    ans = answer_question("q", _passages(), llm=FakeLLM(REFUSAL_SENTINEL))
    assert ans.refused is True
    assert ans.citations == []


def test_citations_parsed_and_mapped():
    text = "Self-employment income is gross receipts minus expenses [BEM 502]."
    ans = answer_question("q", _passages(), llm=FakeLLM(text))
    assert ans.refused is False
    assert [c.section_id for c in ans.citations] == ["BEM 502"]
    assert ans.citations[0].url == "http://x/502"


def test_citation_token_variants():
    cites = _citations_from_text("see [BAM 130] and [BEM 554]", _passages())
    ids = {c.section_id for c in cites}
    assert "BAM 130" in ids and "BEM 554" in ids


def test_judge_json_parsing():
    assert _parse_json('noise {"claims": 3, "supported": 2} trailing') == {"claims": 3, "supported": 2}
    assert _parse_json("no json here") == {}
