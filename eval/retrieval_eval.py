"""Offline retrieval metrics — no LLM required.

These measure whether the right policy sections reach the window, which is the
precondition for a faithful, citeable answer. All are computed against the gold
section labels in data/eval_questions.json:

  - hit@n            : did any returned passage come from a gold section?
  - context_precision: fraction of returned passages that are from a gold section
  - context_recall   : fraction of gold sections covered by the returned passages
  - mrr              : 1 / rank of the first gold-section passage
  - citable_rate     : fraction of returned passages attributable to a single
                       section (naive chunks that straddle a boundary are "MIXED"
                       and cannot be cited)

Because these need no LLM, the ablation that proves the design (section-aware
chunking + hybrid + re-rank) is fully reproducible offline.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from data.chunk import build_nodes
from src.retrieve import RetrievalEngine

QUESTIONS = json.loads((Path(__file__).resolve().parent.parent / "data" / "eval_questions.json").read_text())


@dataclass(frozen=True)
class Config:
    label: str
    chunking: str  # "section_aware" | "naive"
    mode: str  # "vector" | "hybrid"
    rerank: bool


def in_corpus_questions() -> list[dict]:
    return [q for q in QUESTIONS["questions"] if q["in_corpus"]]


def _metrics_for_question(retrieved_sections: list[str], gold: set[str]) -> dict[str, float]:
    n = len(retrieved_sections) or 1
    hits = [s in gold for s in retrieved_sections]
    hit = 1.0 if any(hits) else 0.0
    precision = sum(hits) / n
    recall = len(gold & set(retrieved_sections)) / (len(gold) or 1)
    mrr = 0.0
    for rank, s in enumerate(retrieved_sections, start=1):
        if s in gold:
            mrr = 1.0 / rank
            break
    citable = sum(1 for s in retrieved_sections if s != "MIXED") / n
    return {"hit": hit, "context_precision": precision, "context_recall": recall, "mrr": mrr, "citable_rate": citable}


def evaluate_config(cfg: Config, final_n: int = 4, top_k: int = 8) -> dict:
    nodes = build_nodes(cfg.chunking)
    engine = RetrievalEngine(nodes, mode=cfg.mode, use_rerank=cfg.rerank, final_n=final_n, top_k=top_k)
    rows = []
    for q in in_corpus_questions():
        passages = engine.retrieve(q["question"])
        sections = [p.section_id for p in passages]
        m = _metrics_for_question(sections, set(q["gold_sections"]))
        m["id"] = q["id"]
        m["retrieved"] = sections
        rows.append(m)
    keys = ["hit", "context_precision", "context_recall", "mrr", "citable_rate"]
    agg = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}
    return {"config": cfg.label, "n": len(rows), "aggregate": agg, "rows": rows}


if __name__ == "__main__":
    cfg = Config("full (section-aware + hybrid + rerank)", "section_aware", "hybrid", True)
    res = evaluate_config(cfg)
    print(json.dumps(res["aggregate"], indent=2))
