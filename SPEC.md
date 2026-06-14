# Project 2 — `policy-manual-rag`

**Pattern:** Retrieval-Augmented Generation
**One line:** Grounded Q&A over a state eligibility policy manual, where every answer cites the exact
policy section it came from — because in this domain a hallucinated answer is a wrongful determination.

A RAG app that returns plausible text is table stakes. A RAG app you have *measured* is what signals
seniority. This is an applied slice of context engineering: getting the right policy text into the
window at the right moment, with the receipts to prove it.

---

## Scope

**In scope**
- A corpus of real, public eligibility policy. Michigan's **Bridges Eligibility Manual (BEM)** and
  **Bridges Administrative Manual (BAM)** are ideal — public PDFs, genuinely complex, and in your
  domain. (If you'd rather not tie it to one state, generate a synthetic manual with the same shape:
  numbered sections, cross-references, income tables.)
- Section-aware ingestion and chunking (these manuals are structured — exploit it).
- **Hybrid retrieval**: keyword (BM25) + vector, with metadata filtering by manual / section /
  program.
- **Re-ranking** of retrieved chunks.
- Answers with **inline citations** back to the specific manual section, and an explicit "this isn't
  covered in the manual" path when retrieval is weak.
- A thin chat front-end with a source-passage panel so a reviewer can interrogate it.

**Explicitly out of scope**
- No eligibility *decisions* — this answers policy questions for a caseworker; it doesn't determine
  cases (that's Projects 1 and 4).

---

## The hard parts (where seniority shows)

1. **Section-aware chunking.** Naive fixed-size chunking shreds legal manuals mid-rule. Chunk along
   the manual's own section structure and carry section IDs as metadata so citations are exact.
2. **Citation fidelity.** Every sentence in an answer that asserts policy must point to a retrieved
   passage. If it can't, it doesn't get asserted.
3. **Grounded refusal.** When the question isn't answerable from the corpus, the system says so
   instead of inventing. In this domain that restraint is a feature, not a failure.
4. **Cross-references.** Policy sections reference other sections ("see BEM 500"). Bonus credit for
   following them during retrieval.

---

## Architecture / repo structure

```
policy-manual-rag/
├── README.md
├── data/
│   ├── ingest.py                # download/parse manual PDFs → structured sections
│   ├── chunk.py                 # section-aware chunking + metadata
│   ├── manual_sections/         # parsed sections
│   └── eval_questions.json      # YOUR labeled question set (the differentiator)
├── src/
│   ├── index.py                 # builds BM25 + vector indexes
│   ├── retrieve.py              # hybrid retrieval + metadata filter
│   ├── rerank.py                # re-ranking stage (toggleable for ablations)
│   ├── answer.py                # generation with enforced citations
│   └── schema.py
├── app/                         # chat UI + source-passage panel
├── eval/
│   ├── run_ragas.py             # faithfulness / answer relevancy / context precision
│   ├── ablations.py             # naive vs hybrid; with/without re-rank
│   └── report.md
└── pyproject.toml
```

## Tech stack

- **LlamaIndex** for ingestion/indexing/retrieval over your own data, or LangChain's RAG stack —
  either is defensible; pick one and justify it.
- A vector store (local FAISS/Chroma is fine) + a BM25 index for the keyword half.
- A re-ranker (cross-encoder or a hosted re-rank API).
- **Ragas** for evaluation — it gives faithfulness, answer relevancy, and context precision.
- Thin chat front-end (Streamlit or React) with a panel that shows the exact source passages behind
  each answer.

---

## The labeled question set (build this yourself — it's the whole point)

Write ~50 questions a real caseworker would ask, with reference answers and the manual section(s) that
support them. Examples in the SNAP/Medicaid vein: *"Does a stepparent's income count toward the
household?" "How is self-employment income calculated?" "What's the gross income limit for a
household of four?" "When does the 60-day verification clock start?"* Include a few deliberately
out-of-corpus questions to test grounded refusal. This hand-built set is what lets you walk an
interviewer through real numbers — it's what separates you from the candidate demoing a chat window
and hoping nobody asks how they know it's good.

---

## Evaluation (make it loud)

`eval/run_ragas.py` reports, on your labeled set:

- **Faithfulness** — are answers grounded in the retrieved passages? (This is the safety metric here.)
- **Answer relevancy.**
- **Context precision** — did retrieval pull the *right* sections?
- **Citation accuracy** — do the cited sections actually contain the claim? (A custom check on top
  of RAGAS.)
- **Grounded-refusal rate** on the out-of-corpus questions.

Then the headline: an **ablation table** showing the lift from each upgrade — naive chunking → 
section-aware, vector-only → hybrid, no re-rank → re-rank. Report the deltas. *"Adding re-ranking
moved faithfulness from X to Y"* is exactly the sentence that puts you in a different category.

---

## README framing

Open with the stake: a wrong policy answer becomes a wrong determination becomes a wrongful denial.
So faithfulness is the safety case, not a vanity metric. Then show the architecture, then lead the
results section with the ablation table.

## Interview one-liner

> "A hallucinated policy answer is a wrongful determination, so faithfulness is the whole game. I
> built my own labeled caseworker question set, ran RAGAS, and here's the faithfulness lift from
> section-aware chunking and re-ranking — measured, not vibes."

## Build order for Claude Code

1. `ingest.py` + `chunk.py` over the real manual; inspect the parsed sections.
2. `eval_questions.json` — write the labeled set early.
3. Indexes + hybrid retrieval + re-rank (re-rank toggleable for ablations).
4. `answer.py` with enforced citations and grounded refusal.
5. `run_ragas.py` + `ablations.py` + `report.md`.
6. Chat UI with source-passage panel.
