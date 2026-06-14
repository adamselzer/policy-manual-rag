"""The ablation ladder — the headline result.

Four configurations, each adding one upgrade to the last, so the table reads as a
story: what does section-aware chunking buy, then hybrid retrieval, then
re-ranking? The deltas are the point.

  A. naive chunking + vector-only + no re-rank   (baseline)
  B. + section-aware chunking
  C. + hybrid retrieval (BM25 + vector, RRF)
  D. + cross-encoder re-rank                       (full system)

All metrics are retrieval metrics (no LLM), so this is fully reproducible offline.
Writes eval/report.md.

Run:  python eval/ablations.py            (prints + writes the report)
      python eval/ablations.py --check     (exit nonzero if the full system regresses)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from eval.retrieval_eval import Config, evaluate_config

REPORT = Path(__file__).with_name("report.md")

LADDER = [
    Config("A. naive + vector + no-rerank (baseline)", "naive", "vector", False),
    Config("B. + section-aware chunking", "section_aware", "vector", False),
    Config("C. + hybrid retrieval (BM25+vector)", "section_aware", "hybrid", False),
    Config("D. + cross-encoder re-rank (full system)", "section_aware", "hybrid", True),
]

METRICS = [
    ("hit", "Hit@4"),
    ("context_precision", "Ctx precision"),
    ("context_recall", "Ctx recall"),
    ("mrr", "MRR"),
    ("citable_rate", "Citable"),
]


def run() -> list[dict]:
    return [evaluate_config(cfg) for cfg in LADDER]


def render(results: list[dict]) -> str:
    n = results[0]["n"]
    lines = [
        "# Eval report — policy-manual-rag retrieval ablation",
        "",
        f"Corpus: real Michigan FAP/SNAP manual sections. In-corpus questions: {n}.",
        "All metrics are retrieval metrics against gold section labels — no LLM, fully reproducible.",
        "",
        "## Ablation ladder (each row adds one upgrade)",
        "",
        "| Configuration | " + " | ".join(label for _, label in METRICS) + " |",
        "|---|" + "|".join(["---"] * len(METRICS)) + "|",
    ]
    for r in results:
        agg = r["aggregate"]
        cells = " | ".join(f"{agg[k]:.2f}" for k, _ in METRICS)
        lines.append(f"| {r['config']} | {cells} |")

    base, full = results[0]["aggregate"], results[-1]["aggregate"]
    lines += [
        "",
        "## Deltas (full system vs naive baseline)",
        "",
        "| Metric | Baseline | Full | Lift |",
        "|---|---|---|---|",
    ]
    for k, label in METRICS:
        lift = full[k] - base[k]
        lines.append(f"| {label} | {base[k]:.2f} | {full[k]:.2f} | {lift:+.2f} |")
    lines += [
        "",
        "## Reading this",
        "",
        "- **Citable** jumps to 1.00 with section-aware chunking: every returned passage",
        "  maps to exactly one BEM/BAM section, so every answer can be cited. Naive chunks",
        "  that straddle a section boundary are unattributable.",
        "- **Hybrid** retrieval adds the keyword (BM25) channel, which matters for the exact",
        "  policy terms and section numbers a caseworker uses.",
        "- **Re-ranking** sharpens precision and MRR by scoring each (query, passage) pair",
        "  jointly instead of trusting first-pass similarity.",
        "",
        "When `ANTHROPIC_API_KEY` is set, `eval/run_ragas.py` adds the LLM-judged metrics",
        "(faithfulness, answer relevancy) and the grounded-refusal rate on out-of-corpus",
        "questions.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    results = run()
    report = render(results)
    print(report)
    REPORT.write_text(report)
    if "--check" in argv:
        full = results[-1]["aggregate"]
        base = results[0]["aggregate"]
        # The full system must not regress below baseline on the headline metrics.
        if full["hit"] < base["hit"] or full["context_precision"] < base["context_precision"]:
            print("\nFAIL: full system regressed vs baseline.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
