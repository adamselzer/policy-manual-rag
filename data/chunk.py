"""Two chunkers: section-aware (the real one) and naive (the ablation baseline).

The whole faithfulness argument turns on chunking. A legal manual is a tree of
numbered sections; a citation is only as good as a chunk's knowledge of which
section it came from.

- section_aware: never crosses a BEM/BAM section boundary. Every chunk carries its
  exact section_id, title, and source URL, so any passage retrieved can be cited
  precisely. Long sections are split on paragraph boundaries up to a size cap, and
  the nearest preceding ALL-CAPS heading is carried as metadata.

- naive: concatenates the whole corpus and slices it into fixed-size windows,
  ignoring section boundaries. A chunk can straddle two sections, so it cannot be
  attributed to one. This is the baseline the ablation measures against.

Both return llama_index TextNodes ready to index.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from llama_index.core.schema import TextNode

SECTIONS_DIR = Path(__file__).resolve().parent / "manual_sections"

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP = 150

# A heading line: short, mostly uppercase, not a sentence.
_HEADING = re.compile(r"^[A-Z0-9][A-Z0-9 ,&/\-\.\?']{2,58}$")


def load_sections() -> list[dict]:
    files = sorted(SECTIONS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError("No parsed sections. Run data/ingest.py first.")
    return [json.loads(f.read_text()) for f in files]


def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not _HEADING.match(s):
        return False
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    return upper_ratio > 0.85 and len(s.split()) <= 8


def _pack_paragraphs(paragraphs: list[str], max_chars: int) -> list[str]:
    """Greedily pack paragraphs into chunks up to max_chars, never splitting a
    paragraph unless it alone exceeds the cap."""
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > max_chars:
            # flush buffer, then hard-split the oversized paragraph
            if buf:
                chunks.append(buf.strip())
                buf = ""
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars].strip())
            continue
        if len(buf) + len(para) + 2 <= max_chars:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf.strip())
            buf = para
    if buf:
        chunks.append(buf.strip())
    return [c for c in chunks if c]


def section_aware_nodes(max_chars: int = DEFAULT_MAX_CHARS) -> list[TextNode]:
    nodes: list[TextNode] = []
    for sec in load_sections():
        # Track the most recent heading as we walk the section's lines, and split
        # the body into paragraphs for packing.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", sec["text"]) if p.strip()]
        # Map each paragraph to its nearest preceding heading.
        chunks = _pack_paragraphs(paragraphs, max_chars)
        current_heading = ""
        for idx, chunk in enumerate(chunks):
            for line in chunk.splitlines():
                if _looks_like_heading(line):
                    current_heading = line.strip()
                    break
            nodes.append(
                TextNode(
                    text=chunk,
                    id_=f"{sec['manual']}_{sec['number']}::{idx}",
                    metadata={
                        "manual": sec["manual"],
                        "number": sec["number"],
                        "section_id": sec["section_id"],
                        "title": sec["title"],
                        "url": sec["url"],
                        "heading": current_heading,
                        "strategy": "section_aware",
                    },
                    excluded_embed_metadata_keys=["url", "strategy", "manual", "number"],
                    excluded_llm_metadata_keys=["url", "strategy", "manual", "number"],
                )
            )
    return nodes


def naive_nodes(max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP) -> list[TextNode]:
    """Fixed-size windows over the concatenated corpus, ignoring section boundaries.

    Each window records which sections it overlaps so the eval can show that a
    naive chunk often cannot be attributed to a single section.
    """
    sections = load_sections()
    spans: list[tuple[int, int, str]] = []  # (start, end, section_id) in the joined text
    parts: list[str] = []
    cursor = 0
    for sec in sections:
        body = sec["text"]
        parts.append(body)
        spans.append((cursor, cursor + len(body), sec["section_id"]))
        cursor += len(body) + 2  # the "\n\n" join
    joined = "\n\n".join(parts)

    def sections_in(a: int, b: int) -> list[str]:
        hit = [sid for (s, e, sid) in spans if s < b and e > a]
        return hit

    nodes: list[TextNode] = []
    step = max_chars - overlap
    idx = 0
    for start in range(0, len(joined), step):
        window = joined[start : start + max_chars].strip()
        if not window:
            continue
        overlapped = sections_in(start, start + max_chars)
        nodes.append(
            TextNode(
                text=window,
                id_=f"naive::{idx}",
                metadata={
                    "section_id": overlapped[0] if len(overlapped) == 1 else "MIXED",
                    "overlapping_sections": ", ".join(overlapped),
                    "crosses_boundary": len(overlapped) > 1,
                    "strategy": "naive",
                },
                excluded_embed_metadata_keys=["overlapping_sections", "crosses_boundary", "strategy"],
                excluded_llm_metadata_keys=["overlapping_sections", "crosses_boundary", "strategy"],
            )
        )
        idx += 1
    return nodes


def build_nodes(strategy: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[TextNode]:
    if strategy == "section_aware":
        return section_aware_nodes(max_chars)
    if strategy == "naive":
        return naive_nodes(max_chars)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")


if __name__ == "__main__":
    for strat in ("section_aware", "naive"):
        ns = build_nodes(strat)
        crossing = sum(1 for n in ns if n.metadata.get("crosses_boundary"))
        print(f"{strat:14}: {len(ns):3} chunks" + (f", {crossing} cross a section boundary" if strat == "naive" else ""))
