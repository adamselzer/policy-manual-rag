"""LLM-judged evaluation: faithfulness, answer relevancy, context precision,
citation accuracy, and grounded-refusal rate. Requires ANTHROPIC_API_KEY.

This is the generation-side complement to the offline retrieval ablation. It runs
the full pipeline (section-aware + hybrid + re-rank) over the labeled questions,
generates grounded answers, and scores them with the Claude-based judge in
src/judge.py.

Cost-conscious: one LLM and one embedder are reused. Use --sample N to limit the
in-corpus questions scored. Writes eval/llm_report.md.

Run:  python eval/run_llm_eval.py                 # full in-corpus set + all OOC
      python eval/run_llm_eval.py --sample 12      # cheaper
      python eval/run_llm_eval.py --check          # exit nonzero if quality gates fail
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.chunk import build_nodes
from eval.retrieval_eval import QUESTIONS
from src.answer import answer_question, build_llm
from src.config import EMBED_MODEL, has_llm
from src.judge import answer_relevancy, context_precision, faithfulness
from src.retrieve import RetrievalEngine

REPORT = Path(__file__).with_name("llm_report.md")


def main(argv: list[str]) -> int:
    if not has_llm():
        print("ANTHROPIC_API_KEY not set. Add it to .env to run the LLM eval.", file=sys.stderr)
        return 2

    sample = None
    if "--sample" in argv:
        sample = int(argv[argv.index("--sample") + 1])

    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    llm = build_llm()
    embedder = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    engine = RetrievalEngine(build_nodes("section_aware"), mode="hybrid", use_rerank=True, final_n=4)

    in_corpus = [q for q in QUESTIONS["questions"] if q["in_corpus"]]
    ooc = [q for q in QUESTIONS["questions"] if not q["in_corpus"]]
    if sample:
        in_corpus = in_corpus[:sample]

    faith, relev, ctxp, cite_hits, over_refusals, rows = [], [], [], 0, 0, []
    for q in in_corpus:
        passages = engine.retrieve(q["question"])
        ans = answer_question(q["question"], passages, llm=llm)
        if ans.refused:
            over_refusals += 1
            rows.append((q["id"], "REFUSED (in-corpus)", None, None, None, False))
            continue
        f = faithfulness(ans.answer, passages, llm)
        r = answer_relevancy(q["question"], ans.answer, llm, embedder)
        c = context_precision(q["question"], passages, llm)
        gold = set(q["gold_sections"])
        cited = {cit.section_id for cit in ans.citations}
        cite_hit = bool(cited & gold)
        faith.append(f); relev.append(r); ctxp.append(c); cite_hits += int(cite_hit)
        rows.append((q["id"], "answered", f, r, c, cite_hit))

    refused_ooc = sum(1 for q in ooc for _ in [0]
                      if answer_question(q["question"], engine.retrieve(q["question"]), llm=llm).refused)

    n = len(in_corpus)
    answered = n - over_refusals
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0
    summary = {
        "faithfulness": mean(faith),
        "answer_relevancy": mean(relev),
        "context_precision": mean(ctxp),
        "citation_accuracy": round(cite_hits / answered, 4) if answered else 0.0,
        "in_corpus_over_refusal": round(over_refusals / n, 4) if n else 0.0,
        "grounded_refusal_rate_ooc": round(refused_ooc / len(ooc), 4) if ooc else 0.0,
    }
    report = render(summary, rows, n, len(ooc))
    print(report)
    REPORT.write_text(report)

    if "--check" in argv:
        ok = summary["faithfulness"] >= 0.80 and summary["grounded_refusal_rate_ooc"] >= 0.75
        if not ok:
            print("\nFAIL: faithfulness or grounded-refusal gate not met.", file=sys.stderr)
            return 1
    return 0


def render(summary: dict, rows: list, n_in: int, n_ooc: int) -> str:
    lines = [
        "# Eval report — policy-manual-rag generation (LLM-judged)",
        "",
        f"Full pipeline (section-aware + hybrid + re-rank). In-corpus questions: {n_in}; "
        f"out-of-corpus: {n_ooc}. Judge and generation: Claude.",
        "",
        "## Headline metrics",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Faithfulness (answer grounded in retrieved policy) | {summary['faithfulness']:.2f} |",
        f"| Answer relevancy | {summary['answer_relevancy']:.2f} |",
        f"| Context precision | {summary['context_precision']:.2f} |",
        f"| Citation accuracy (cited a gold section) | {summary['citation_accuracy']:.2f} |",
        f"| Grounded-refusal rate (out-of-corpus) | {summary['grounded_refusal_rate_ooc']:.2f} |",
        f"| In-corpus over-refusal (lower is better) | {summary['in_corpus_over_refusal']:.2f} |",
        "",
        "Faithfulness is the safety metric: a wrong policy answer becomes a wrong",
        "determination. Grounded-refusal is its complement: the system declines when the",
        "corpus does not contain the answer instead of inventing one.",
        "",
        "## Per-question (in-corpus)",
        "",
        "| Question | Outcome | Faith | Relev | CtxP | Cited gold |",
        "|---|---|---|---|---|---|",
    ]
    for qid, outcome, f, r, c, cite in rows:
        fmt = lambda x: "-" if x is None else f"{x:.2f}"
        lines.append(f"| `{qid}` | {outcome} | {fmt(f)} | {fmt(r)} | {fmt(c)} | {'yes' if cite else '-'} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
