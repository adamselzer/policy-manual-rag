"""Tests for ingestion output and the two chunkers (no models needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.chunk import build_nodes, load_sections

SECTIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "manual_sections"
CORPUS_IDS = {"BEM 212", "BEM 213", "BEM 400", "BEM 500", "BEM 501", "BEM 502",
              "BEM 503", "BEM 550", "BEM 554", "BEM 556", "BAM 130"}


def test_all_sections_parsed():
    sections = load_sections()
    ids = {s["section_id"] for s in sections}
    assert CORPUS_IDS.issubset(ids)


def test_sections_have_clean_titles_and_text():
    for s in load_sections():
        assert s["title"] and s["title"] != s["section_id"], f"{s['section_id']} title not parsed"
        assert len(s["text"]) > 1000
        # boilerplate should be stripped
        assert "BRIDGES ELIGIBILITY MANUAL STATE OF MICHIGAN" not in s["text"]


def test_bam_130_title_correct():
    bam = json.loads((SECTIONS_DIR / "BAM_130.json").read_text())
    assert "VERIFICATION" in bam["title"].upper()


def test_cross_references_detected():
    # BEM 502 references other sections (e.g., BEM 504, BEM 501)
    bem502 = json.loads((SECTIONS_DIR / "BEM_502.json").read_text())
    assert any(ref.startswith("BEM") for ref in bem502["cross_references"])


def test_section_aware_chunks_are_attributable():
    nodes = build_nodes("section_aware")
    assert len(nodes) > 100
    for n in nodes:
        assert n.metadata["section_id"] in CORPUS_IDS
        assert n.metadata["section_id"] != "MIXED"
        assert n.metadata["url"].startswith("http")


def test_naive_chunks_include_boundary_crossers():
    nodes = build_nodes("naive")
    assert len(nodes) > 100
    assert any(n.metadata.get("crosses_boundary") for n in nodes), "expected some naive chunks to straddle sections"


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        build_nodes("semantic")
