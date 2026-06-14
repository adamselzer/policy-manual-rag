# Website blurb — policy-manual-rag

Drop-in copy for a portfolio page. A short paragraph plus highlight bullets.

---

## policy-manual-rag

A retrieval system that answers food assistance policy questions from the real
Michigan eligibility manual, with a citation to the exact section behind every
statement. In public benefits a confident wrong answer is a wrongful
determination, so the system is built to be grounded by construction: it cites the
policy section or it does not make the claim, and when the manual does not cover a
question it declines rather than inventing one. The work is in the measurement. I
hand-built a labeled caseworker question set, ran an ablation that isolates the lift
from section-aware chunking, hybrid retrieval, and re-ranking, and scored answer
faithfulness with an LLM judge. It is the cited-policy-lookup layer of a four-part
portfolio on AI in the safety net.

**Highlights**

- Grounded Q&A over eleven real sections of the Michigan Bridges Eligibility
  Manual, with an inline citation to the exact BEM/BAM section behind each
  statement.
- Faithfulness of 0.98 and a grounded-refusal rate of 0.88 on out-of-corpus
  questions, judged by an LLM against a hand-built labeled set.
- A retrieval ablation that reproduces offline with no API key: section-aware
  chunking, hybrid (keyword + vector) retrieval, and re-ranking move Hit@4 from
  0.92 to 1.00 and the citable rate from 0.97 to 1.00.
- Refuses when the corpus cannot ground an answer, which is the safe failure in
  this domain.
- Composes with the rules-as-code MCP server, which exposes this index as a cited
  policy-lookup tool.

---

*Voice note: drafted in the plain, finding-first house style from your other
writing (no em dashes, evidence over decoration). Worth a pass against the live
aselzer.com voice before publishing.*
