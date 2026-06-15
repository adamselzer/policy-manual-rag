# Portfolio notes: policy-manual-rag

A plain-language account of what this project is and the judgment behind it.

## Pattern

**Retrieval-Augmented Generation (RAG).** Grounded question answering over a real
policy corpus, with citations and measured retrieval quality.

## Concept demonstrated

Context engineering and *measured* retrieval quality: getting the right policy text
into the model's window at the right moment, and proving it with numbers rather
than vibes. A RAG demo that returns plausible prose is easy. This project shows the
part that signals seniority: a hand-built labeled question set, an ablation that
isolates the lift from each design choice, and an LLM-judged faithfulness score that
treats a wrong answer as the safety failure it is in this domain.

## Why it matters in this domain

A caseworker who asks a policy question and gets a confident, wrong answer makes a
wrong determination, and a wrong determination can deny a family food assistance.
There is no "mostly right" that is safe here. So the system is built to be
grounded by construction (cite the section or do not assert it) and to refuse when
the manual does not cover the question. Faithfulness and grounded refusal are the
safety case; everything else serves them.

## Key design decisions and tradeoffs

1. **The real eligibility manual.** The corpus is eleven actual Michigan
   BEM/BAM sections. *Rejected:* a synthetic manual of the same shape, which would
   have been easier to parse and label. The real manual is harder (column layout,
   repeated page boilerplate, inconsistent headings) but it is the domain, it
   composes with the rules-as-code project that cites the same sections, and
   parsing it is itself part of the work. Building the real corpus also surfaced a
   wrong citation in the sibling project (BEM 213, not 556, is categorical
   eligibility), which I fixed there.

2. **Section-aware chunking over fixed-size chunking.** Chunks never cross a
   section boundary and carry their exact section ID. *Rejected:* naive fixed-size
   windows. The ablation shows why: naive chunking leaves some chunks straddling two
   sections and therefore uncitable, while section-aware chunking takes the citable
   rate to 1.00. In a domain where the citation is the product, that is decisive.

3. **Hybrid retrieval with a re-rank stage, each toggleable.** Vector similarity
   alone misses the exact policy terms and section numbers caseworkers use, so a
   BM25 channel is fused in, and a cross-encoder re-ranks the candidates.
   *Rejected:* vector-only retrieval. Making each stage toggleable is what turns the
   design into a measurable ablation rather than an assertion.

4. **Local, LLM-free retrieval.** Embeddings and re-ranking run on local models, and
   the hybrid fusion is configured so no LLM expands the query. *Rejected:* hosted
   embedding and re-rank APIs. The local path means the entire retrieval ablation,
   the differentiator, reproduces offline with no API key, which is the right
   default for a portfolio reviewer.

5. **Computing the LLM-judged metrics directly instead of via Ragas.** The brief
   names Ragas; at build time Ragas was incompatible with the installed langchain
   1.x and could not be made to import. *Rejected:* pinning the dependency tree
   until Ragas worked, which cascaded into further conflicts, or shipping a repo
   whose eval would not run. Implementing the same metric definitions directly
   against Claude keeps the eval runnable and makes each metric's meaning explicit
   in code.

## How it's evaluated

Two complementary halves.

- **Retrieval ablation (offline, reproducible):** a four-rung ladder adding
  section-aware chunking, then hybrid retrieval, then re-ranking, scored on Hit@4,
  context precision and recall, MRR, and the citable rate against gold section
  labels. Headline: Hit@4 0.92 -> 1.00 and citable 0.97 -> 1.00 from baseline to
  full system. No API key needed, so a reviewer can rerun it.
- **Generation quality (LLM-judged):** faithfulness, answer relevancy, context
  precision, citation accuracy, and grounded-refusal rate on the full pipeline.
  Headline: **faithfulness 0.98** and **grounded-refusal 0.88**, with a 0.07
  in-corpus over-refusal that is the system correctly declining when retrieval came
  up short.

These metrics were chosen because they map to the stake. Faithfulness is the safety
metric; grounded refusal is its complement; the retrieval metrics explain *why* the
generation is faithful (the right text reached the window). The labeled question
set is hand-built, which is what makes any of these numbers meaningful rather than
self-graded.

## What I'd do differently at production scale

- Follow cross-references during retrieval (sections cite other sections).
- Derive section-level gold labels from the manual's own structure and cover the
  full BEM/BAM rather than the FAP subset.
- Enforce citation at generation time to close the gap between faithfulness (0.98)
  and citation accuracy (0.79).
- Swap local models for hosted embedding/re-rank for throughput behind the same
  interfaces, and wire `lookup_policy` in the rules-as-code MCP server to this
  index.
