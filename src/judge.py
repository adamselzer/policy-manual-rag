"""LLM-judged RAG metrics, computed directly against Claude.

These implement the same metric definitions Ragas uses, without the Ragas
dependency (which was incompatible with the installed langchain 1.x at build time;
see pyproject.toml). Computing them directly is also more transparent: the prompts
below are exactly what each metric means.

  - faithfulness:      of the claims in the answer, the fraction entailed by the
                       retrieved passages. The safety metric: it catches an answer
                       that asserts something the policy text does not support.
  - answer_relevancy:  generate candidate questions from the answer, embed them and
                       the original question (local embeddings), report mean cosine
                       similarity. Penalizes evasive or off-topic answers.
  - context_precision: average precision of the retrieved passages, judging each
                       passage's usefulness for answering the question (rank-aware).
"""

from __future__ import annotations

import json
import re

import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .config import EMBED_MODEL
from .schema import RetrievedPassage

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict:
    m = _JSON.search(str(text))
    return json.loads(m.group(0)) if m else {}


def faithfulness(answer: str, passages: list[RetrievedPassage], llm) -> float:
    context = "\n\n".join(p.text for p in passages)
    prompt = (
        "Given a CONTEXT and an ANSWER, break the answer into discrete factual "
        "claims. Count how many claims are directly supported by the context. "
        'Reply ONLY as JSON: {"claims": <int>, "supported": <int>}.\n\n'
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    )
    data = _parse_json(llm.complete(prompt))
    claims = data.get("claims", 0)
    supported = data.get("supported", 0)
    if not claims:
        return 0.0
    return round(min(supported, claims) / claims, 4)


def answer_relevancy(question: str, answer: str, llm, embedder: HuggingFaceEmbedding | None = None) -> float:
    prompt = (
        "Read the ANSWER and generate three distinct questions that this answer "
        'would be a good answer to. Reply ONLY as JSON: {"questions": ["...","...","..."]}.\n\n'
        f"ANSWER:\n{answer}"
    )
    data = _parse_json(llm.complete(prompt))
    gen_qs = data.get("questions", [])
    if not gen_qs:
        return 0.0
    embedder = embedder or HuggingFaceEmbedding(model_name=EMBED_MODEL)
    q_vec = np.array(embedder.get_text_embedding(question))
    sims = []
    for gq in gen_qs:
        g_vec = np.array(embedder.get_text_embedding(gq))
        denom = np.linalg.norm(q_vec) * np.linalg.norm(g_vec)
        sims.append(float(np.dot(q_vec, g_vec) / denom) if denom else 0.0)
    return round(float(np.mean(sims)), 4)


def context_precision(question: str, passages: list[RetrievedPassage], llm) -> float:
    """Average precision over the ranked passages, judging each as useful or not."""
    useful: list[int] = []
    for p in passages:
        prompt = (
            "Is the following policy passage useful for answering the question? "
            'Reply ONLY as JSON: {"useful": true|false}.\n\n'
            f"QUESTION: {question}\n\nPASSAGE ({p.section_id}):\n{p.text}"
        )
        data = _parse_json(llm.complete(prompt))
        useful.append(1 if data.get("useful") else 0)
    if not any(useful):
        return 0.0
    # average precision: mean of precision@k at the ranks where a useful passage appears
    precisions = []
    hits = 0
    for k, u in enumerate(useful, start=1):
        if u:
            hits += 1
            precisions.append(hits / k)
    return round(sum(precisions) / len(precisions), 4)
