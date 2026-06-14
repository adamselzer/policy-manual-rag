"""Answer generation: grounded in retrieved passages, with enforced citations and
grounded refusal.

The safety rule of this domain drives the design: every policy assertion must be
traceable to a retrieved passage, and if the passages do not contain the answer,
the system refuses rather than inventing. A wrong policy answer becomes a wrong
determination becomes a wrongful denial, so refusal is a feature.

Generation uses Claude (via llama-index-llms-anthropic) when ANTHROPIC_API_KEY is
set. The model is instructed to answer only from the numbered passages, cite each
claim with its [SECTION], and emit a refusal sentinel when the context is
insufficient.
"""

from __future__ import annotations

import re

from llama_index.llms.anthropic import Anthropic

from .config import ANSWER_MODEL, anthropic_key
from .schema import Answer, Citation, RetrievedPassage

REFUSAL_SENTINEL = "INSUFFICIENT_CONTEXT"
SECTION_TOKEN = re.compile(r"\b(B[AE]M)\s*(\d{3})\b")

SYSTEM = (
    "You are a careful assistant for Michigan food assistance (SNAP/FAP) caseworkers. "
    "You answer ONLY from the numbered policy passages provided. Every sentence that "
    "states policy must end with a citation to the passage it came from, written as "
    "[BEM ###] or [BAM ###]. Do not use outside knowledge. If the passages do not "
    f"contain enough information to answer, reply with exactly '{REFUSAL_SENTINEL}' and "
    "nothing else. Be concise."
)

PROMPT = """\
Question: {question}

Numbered policy passages:
{context}

Answer the question using only these passages. Cite each policy statement with its
[SECTION]. If the passages do not answer the question, reply with exactly
'{sentinel}'.
"""


def build_llm(model: str = ANSWER_MODEL) -> Anthropic:
    key = anthropic_key()
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env to enable answer generation."
        )
    return Anthropic(
        model=model, api_key=key, max_tokens=700, temperature=0.0, system_prompt=SYSTEM
    )


def _format_context(passages: list[RetrievedPassage]) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        label = p.section_id + (f" ({p.heading})" if p.heading else "")
        blocks.append(f"[{i}] {label}:\n{p.text.strip()}")
    return "\n\n".join(blocks)


def _citations_from_text(text: str, passages: list[RetrievedPassage]) -> list[Citation]:
    """Map [BEM ###] tokens in the answer back to the retrieved passages."""
    cited = {f"{m} {n}" for m, n in SECTION_TOKEN.findall(text)}
    by_section: dict[str, RetrievedPassage] = {}
    for p in passages:
        by_section.setdefault(p.section_id, p)
    out = []
    for sid in cited:
        p = by_section.get(sid)
        out.append(Citation(section_id=sid, title=p.title if p else "", url=p.url if p else ""))
    return sorted(out, key=lambda c: c.section_id)


def answer_question(
    question: str,
    passages: list[RetrievedPassage],
    llm: Anthropic | None = None,
) -> Answer:
    if not passages:
        return Answer(question=question, answer=REFUSAL_SENTINEL, refused=True, passages=[])
    llm = llm or build_llm()
    prompt = PROMPT.format(
        question=question, context=_format_context(passages), sentinel=REFUSAL_SENTINEL
    )
    resp = llm.complete(prompt)
    text = str(resp).strip()
    refused = REFUSAL_SENTINEL in text
    citations = [] if refused else _citations_from_text(text, passages)
    return Answer(
        question=question,
        answer=text,
        citations=citations,
        passages=passages,
        refused=refused,
    )
