# policy-manual-rag

Measured, citation-grounded retrieval-augmented Q&A over the Michigan SNAP
eligibility manual. Every policy assertion in an answer cites the exact manual
section it came from, and when the manual does not cover a question the system
declines instead of inventing an answer.

This is the RAG project in a four-part portfolio on AI in the public benefits
safety net. It is the cited-policy-lookup layer that the `rules-as-code-mcp`
server exposes as a tool and the `benefits-intake-agent` consumes.

> A hallucinated policy answer is a wrongful determination is a wrongful denial of
> food assistance. So faithfulness, the degree to which an answer is grounded in
> the retrieved policy text, is the safety case here. A RAG app
> that returns plausible text is table stakes. The point of this one is that it is
> measured.

## What it does

Ask a food assistance policy question. The system retrieves the relevant sections
of the real Michigan Bridges Eligibility Manual, generates an answer grounded only
in those passages with an inline `[BEM ###]` citation behind each statement, and
shows the exact source passages. If retrieval cannot ground an answer, it refuses.

## The corpus

A real, scoped subset of Michigan's Bridges Eligibility Manual (BEM) and Bridges
Administrative Manual (BAM): the eleven sections a caseworker reaches for when
determining food assistance (FAP, Michigan's name for SNAP).

| Section | Title |
|---|---|
| BEM 212 | Food Assistance Program group composition |
| BEM 213 | Categorical eligibility |
| BEM 400 | Assets |
| BEM 500 | Income overview |
| BEM 501 | Income from employment |
| BEM 502 | Income from self-employment |
| BEM 503 | Income, unearned |
| BEM 550 | FAP income budgeting |
| BEM 554 | FAP allowable expenses and expense budgeting |
| BEM 556 | Computing the food assistance budget |
| BAM 130 | Verification and collateral contacts |

Using the real manual matters because its structure is the thing the system
exploits: every page is stamped with its section number, so a citation points to
an exact section. The parsed sections are committed under `data/manual_sections/`;
`data/ingest.py` re-downloads and re-parses them from the State of Michigan site.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"             # retrieval stack (local models, no API key)

python data/ingest.py               # download + parse the manual (already committed)
python eval/ablations.py            # the retrieval ablation, no API key needed
pytest                              # 17 tests

# Optional, for generation + LLM-judged metrics:
pip install -e ".[llm,app]"
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
python eval/run_llm_eval.py --sample 15
streamlit run app/app.py
```

## Architecture

```
policy-manual-rag/
├── data/
│   ├── ingest.py            # download BEM/BAM PDFs -> structured sections (boilerplate stripped)
│   ├── chunk.py             # section-aware chunker (real) + naive chunker (ablation baseline)
│   ├── manual_sections/     # parsed sections (committed)
│   └── eval_questions.json  # 44 labeled questions (36 in-corpus w/ gold sections + 8 out-of-corpus)
├── src/
│   ├── config.py            # models, paths, .env key loading
│   ├── retrieve.py          # vector / hybrid (BM25+vector RRF) + metadata filter
│   ├── rerank.py            # toggleable cross-encoder re-rank stage
│   ├── answer.py            # generation with enforced citations + grounded refusal
│   ├── judge.py             # faithfulness / answer relevancy / context precision (LLM-judged)
│   └── schema.py
├── app/app.py               # Streamlit chat UI + source-passage panel
├── eval/
│   ├── retrieval_eval.py    # offline retrieval metrics (no LLM)
│   ├── ablations.py         # the ablation ladder -> report.md
│   └── run_llm_eval.py      # generation + LLM-judged metrics -> llm_report.md
└── tests/
```

**Stack** (chosen and verified against current docs): **LlamaIndex** for retrieval
over our own documents (clean node / retriever / postprocessor abstractions and
first-class hybrid + re-rank), a local **`bge-small`** embedder, an in-memory
vector index, a **BM25** sparse index, **RRF** fusion via `QueryFusionRetriever`,
and a local **cross-encoder** re-ranker. Retrieval is entirely LLM-free (the fusion
runs at `num_queries=1` so no LLM expands the query). Only answer generation and
the judge use Claude.

## Evaluation

Evaluation is the centerpiece, in two halves.

### 1. Retrieval ablation (offline, no API key, fully reproducible)

The differentiator. Each row adds one upgrade, against the gold section labels for
the 36 in-corpus questions.

| Configuration | Hit@4 | Ctx precision | Ctx recall | MRR | Citable |
|---|---|---|---|---|---|
| A. naive + vector + no-rerank (baseline) | 0.92 | 0.65 | 0.90 | 0.81 | 0.97 |
| B. + section-aware chunking | 0.97 | 0.73 | 0.94 | 0.88 | 1.00 |
| C. + hybrid retrieval (BM25+vector) | 0.97 | 0.74 | 0.94 | 0.92 | 1.00 |
| D. + cross-encoder re-rank (full system) | 1.00 | 0.73 | 0.97 | 0.88 | 1.00 |

Section-aware chunking takes **Citable to 1.00**: every returned passage maps to
exactly one BEM/BAM section, so every answer can be cited. Naive chunks that
straddle a section boundary are unattributable. Hybrid adds the keyword channel for
the exact policy terms and section numbers caseworkers use; re-ranking lifts Hit@4
to 1.00. Re-rank trades a little ranked precision for recall here, which is the
honest result and is left untuned. Regenerate with `python eval/ablations.py`.

### 2. Generation quality (LLM-judged, full pipeline, requires a key)

On the full system, 15 in-corpus questions and 8 out-of-corpus questions, judged by
Claude. Metric definitions follow Ragas; see the note below.

| Metric | Result |
|---|---|
| **Faithfulness** (answer grounded in retrieved policy) | **0.98** |
| Answer relevancy | 0.84 |
| Context precision | 0.85 |
| Citation accuracy (cited a gold section) | 0.79 |
| Grounded-refusal rate (out-of-corpus) | 0.88 |
| In-corpus over-refusal (lower is better) | 0.07 |

Faithfulness at 0.98 is the headline: answers almost never assert something the
retrieved policy does not support. Grounded refusal at 0.88 is its complement: when
asked about programs outside this corpus (unemployment, Medicaid spend-down, the
hearings process), the system declines rather than inventing. The 0.07 in-corpus
over-refusal is one question (the homestead-exclusion case) where retrieval did not
surface the supporting passage and the system correctly chose to decline rather
than guess. Regenerate with `python eval/run_llm_eval.py`.

### A note on Ragas

The build brief names Ragas for the LLM-judged metrics. At build time Ragas (both
0.2 and 0.4) was incompatible with the installed langchain 1.x line: it imports
modules removed in langchain 1.0, and pinning langchain down cascaded into
conflicts with langgraph and langchain-openai. Rather than ship a broken
dependency, the metrics are computed directly against Claude in `src/judge.py`,
using the same definitions Ragas uses (faithfulness as the fraction of answer
claims entailed by the context; answer relevancy as the similarity of
back-generated questions to the original; context precision as rank-aware
average precision). Computing them directly is also more transparent: the prompts
are exactly what each metric means.

## Grounded by construction

- Retrieval carries section IDs as metadata, so every passage knows its source.
- Generation is instructed to answer only from the numbered passages and to cite
  each statement; citations in the answer are mapped back to the retrieved sections.
- When the passages do not contain the answer, generation emits a refusal sentinel
  and the system declines. No passages, no answer.

## How it composes with the rest of the portfolio

`rules-as-code-mcp` exposes a `lookup_policy` tool that delegates here, so one MCP
server offers both deterministic determination and cited policy lookup. Each repo
also runs standalone.

## What I'd do differently at production scale

- **Cross-reference following.** Sections cite other sections ("see BEM 400"). The
  parser detects cross-references but retrieval does not yet follow them; doing so
  would raise recall on multi-section questions.
- **Wider corpus, section-level gold from the source.** The gold labels are
  hand-assigned; at scale I would derive them from the manual's own structure and
  cover the full BEM/BAM rather than the FAP subset.
- **Re-rank tuning and answer-time citation enforcement.** Citation accuracy (0.79)
  trails faithfulness because the model sometimes grounds a correct answer in a
  related section without emitting the gold section's token; enforcing citation at
  generation time would close that gap.
- **A hosted embedder/re-ranker** for throughput, with the same interfaces.

## Sources

- [LlamaIndex](https://developers.llamaindex.ai/), [BM25 retriever](https://developers.llamaindex.ai/python/framework/integrations/retrievers/bm25_retriever/).
- Ragas metric definitions: [faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/), [context precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/).
- Corpus: Michigan DHHS [Bridges Eligibility Manual](https://dhhs.michigan.gov/OLMWEB/EX/BP/Public/BEM/213.pdf) and Bridges Administrative Manual.
