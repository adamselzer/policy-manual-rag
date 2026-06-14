# Eval report — policy-manual-rag generation (LLM-judged)

Full pipeline (section-aware + hybrid + re-rank). In-corpus questions: 15; out-of-corpus: 8. Judge and generation: Claude.

## Headline metrics

| Metric | Result |
|---|---|
| Faithfulness (answer grounded in retrieved policy) | 0.98 |
| Answer relevancy | 0.84 |
| Context precision | 0.85 |
| Citation accuracy (cited a gold section) | 0.79 |
| Grounded-refusal rate (out-of-corpus) | 0.88 |
| In-corpus over-refusal (lower is better) | 0.07 |

Faithfulness is the safety metric: a wrong policy answer becomes a wrong
determination. Grounded-refusal is its complement: the system declines when the
corpus does not contain the answer instead of inventing one.

## Per-question (in-corpus)

| Question | Outcome | Faith | Relev | CtxP | Cited gold |
|---|---|---|---|---|---|
| `grp-separate-food` | answered | 1.00 | 0.80 | 1.00 | yes |
| `grp-spouse` | answered | 1.00 | 0.90 | 1.00 | yes |
| `grp-children-under-22` | answered | 1.00 | 0.73 | 0.83 | yes |
| `cat-elig-asset` | answered | 1.00 | 0.92 | 0.81 | yes |
| `cat-elig-what` | answered | 1.00 | 0.87 | 0.75 | yes |
| `cat-elig-gambling` | answered | 1.00 | 0.80 | 1.00 | yes |
| `asset-countable` | answered | 1.00 | 0.86 | 0.25 | yes |
| `asset-vehicle` | answered | 1.00 | 0.77 | 0.81 | - |
| `asset-home` | REFUSED (in-corpus) | - | - | - | - |
| `inc-earned-vs-unearned` | answered | 0.75 | 0.90 | 0.83 | yes |
| `inc-excluded` | answered | 1.00 | 0.87 | 1.00 | - |
| `emp-income-count` | answered | 1.00 | 0.85 | 1.00 | yes |
| `emp-tips` | answered | 1.00 | 0.81 | 1.00 | yes |
| `se-income-calc` | answered | 1.00 | 0.87 | 0.92 | yes |
| `se-expenses` | answered | 1.00 | 0.86 | 0.64 | - |
