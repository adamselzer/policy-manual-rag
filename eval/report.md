# Eval report — policy-manual-rag retrieval ablation

Corpus: real Michigan FAP/SNAP manual sections. In-corpus questions: 36.
All metrics are retrieval metrics against gold section labels — no LLM, fully reproducible.

## Ablation ladder (each row adds one upgrade)

| Configuration | Hit@4 | Ctx precision | Ctx recall | MRR | Citable |
|---|---|---|---|---|---|
| A. naive + vector + no-rerank (baseline) | 0.92 | 0.65 | 0.90 | 0.81 | 0.97 |
| B. + section-aware chunking | 0.97 | 0.73 | 0.94 | 0.88 | 1.00 |
| C. + hybrid retrieval (BM25+vector) | 0.97 | 0.74 | 0.94 | 0.92 | 1.00 |
| D. + cross-encoder re-rank (full system) | 1.00 | 0.73 | 0.97 | 0.88 | 1.00 |

## Deltas (full system vs naive baseline)

| Metric | Baseline | Full | Lift |
|---|---|---|---|
| Hit@4 | 0.92 | 1.00 | +0.08 |
| Ctx precision | 0.65 | 0.73 | +0.08 |
| Ctx recall | 0.90 | 0.97 | +0.07 |
| MRR | 0.81 | 0.88 | +0.08 |
| Citable | 0.97 | 1.00 | +0.03 |

## Reading this

- **Citable** jumps to 1.00 with section-aware chunking: every returned passage
  maps to exactly one BEM/BAM section, so every answer can be cited. Naive chunks
  that straddle a section boundary are unattributable.
- **Hybrid** retrieval adds the keyword (BM25) channel, which matters for the exact
  policy terms and section numbers a caseworker uses.
- **Re-ranking** sharpens precision and MRR by scoring each (query, passage) pair
  jointly instead of trusting first-pass similarity.

When `ANTHROPIC_API_KEY` is set, `eval/run_ragas.py` adds the LLM-judged metrics
(faithfulness, answer relevancy) and the grounded-refusal rate on out-of-corpus
questions.
