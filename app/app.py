"""Streamlit chat UI over the Michigan SNAP manual, with a source-passage panel.

A reviewer can ask a caseworker question, read the grounded answer with inline
[SECTION] citations, and interrogate the exact passages behind it in the side
panel, with their retrieval scores. The sidebar exposes the retrieval knobs
(hybrid vs vector, re-rank on/off) so the ablation is visible live.

Without ANTHROPIC_API_KEY the app still runs in retrieval-only mode: it shows the
passages it would ground an answer in. With a key it generates the cited answer.

Run:  streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app._ui import apply_theme, case_study, footer, header
from data.chunk import build_nodes
from src.answer import answer_question, build_llm
from src.config import has_llm
from src.retrieve import RetrievalEngine

st.set_page_config(page_title="Michigan SNAP policy RAG", layout="wide")
apply_theme(st)


@st.cache_resource(show_spinner="Building indexes over the manual...")
def get_engine(mode: str, rerank: bool) -> RetrievalEngine:
    return RetrievalEngine(build_nodes("section_aware"), mode=mode, use_rerank=rerank, final_n=4)


@st.cache_resource(show_spinner=False)
def get_llm():
    return build_llm() if has_llm() else None


CASE_STUDY = """
<p class="cs-lead">Grounded answers to a caseworker's policy questions, every claim cited to the manual section it came from, with a refusal when the manual does not cover it.</p>

<h2>Who it is for, and the need</h2>
<p>A benefits caseworker needs to look up a policy point quickly and trust the answer. A general chatbot is dangerous here: a confident, wrong policy answer becomes a wrong eligibility determination. The need is grounded question-answering over the real eligibility manual, where every statement is traceable to a section and the system declines when the manual cannot support an answer.</p>

<h2>What it does</h2>
<p>It ingests real Michigan BEM/BAM sections, retrieves the relevant passages for a question (keyword and semantic search, then a re-rank), and generates an answer that cites each policy statement. When retrieval cannot ground an answer, it refuses instead of inventing one.</p>

<h2>Technical decisions</h2>
<div class="dec"><b>The real eligibility manual</b><p>The corpus is eleven actual BEM/BAM sections. It is harder to parse but it is the domain, and it composes with the rules engine that cites the same sections.</p><p class="alt">Instead of: a tidy synthetic manual that would not prove anything about real policy text.</p></div>
<div class="dec"><b>Section-aware chunking</b><p>Chunks never cross a section boundary and carry their section id, so every passage is citable. Naive fixed-size chunks that straddle a boundary cannot be attributed.</p><p class="alt">Instead of: fixed-size windows that shred a rule mid-sentence.</p></div>
<div class="dec"><b>Hybrid retrieval with a re-rank, each toggleable</b><p>Keyword search catches exact section numbers and policy terms; vector search catches meaning; a cross-encoder re-ranks the top candidates. Each stage toggles, which turns the design into a measured ablation.</p><p class="alt">Instead of: vector-only retrieval asserted to be good.</p></div>
<div class="dec"><b>Measured with numbers</b><p>An offline retrieval ablation (no API key) plus an LLM-judged faithfulness score on a hand-built question set. Faithfulness is the safety metric.</p><p class="alt">Instead of: shipping Ragas, which conflicted with the installed langchain; the metrics are computed directly against Claude with the same definitions.</p></div>

<h2>Design decisions</h2>
<div class="dec"><b>The answer sits beside its sources</b><p>A source-passage panel shows the exact text behind the answer, with retrieval scores, so a reviewer can interrogate the grounding.</p><p class="alt">Instead of: an answer with no way to check where it came from.</p></div>
<div class="dec"><b>Citations as artifacts</b><p>Section ids are set in monospace and link to the source, treated like the legal references they are.</p><p class="alt">Instead of: burying the citation in prose.</p></div>
<div class="dec"><b>The retrieval knobs are visible</b><p>Hybrid/vector and re-rank toggles live in the sidebar, so the ablation the eval measures can be felt live.</p><p class="alt">Instead of: hiding the pipeline behind a single opaque answer.</p></div>

<h2>Honest limitations</h2>
<ul>
<li>Gold section labels are hand-assigned; at scale they would come from the manual's own structure.</li>
<li>Cross-references between sections are detected but not yet followed during retrieval.</li>
<li>The LLM-judged faithfulness run is sampled, not the full set.</li>
</ul>
"""

header(
    st,
    "Policy lookup · food assistance",
    "Michigan SNAP eligibility: policy Q&A",
    "Grounded answers over a real subset of the Michigan Bridges Eligibility Manual. "
    "Every policy statement cites the BEM/BAM section it came from. When the manual "
    "does not cover a question, the system declines rather than inventing.",
)

with st.sidebar:
    st.header("Retrieval settings")
    mode = st.radio("Retrieval", ["hybrid", "vector"], index=0)
    rerank = st.checkbox("Cross-encoder re-rank", value=True)
    st.divider()
    if has_llm():
        st.success("Answer generation: ON (Claude)")
    else:
        st.warning("No ANTHROPIC_API_KEY: retrieval-only mode. Add a key to .env for answers.")

tab_demo, tab_about = st.tabs(["Live demo", "How it works"])

with tab_demo:
    question = st.text_input(
        "Ask a food assistance policy question",
        placeholder="How is self-employment income counted for food assistance?",
    )
    if question:
        engine = get_engine(mode, rerank)
        passages = engine.retrieve(question)
        answer_col, source_col = st.columns([3, 2])

        with answer_col:
            st.subheader("Answer")
            llm = get_llm()
            if llm is not None:
                with st.spinner("Generating grounded answer..."):
                    ans = answer_question(question, passages, llm=llm)
                if ans.refused:
                    st.info("Not covered by the manual corpus. The system declined to answer rather than invent one.")
                else:
                    st.write(ans.answer)
                    if ans.citations:
                        st.markdown("**Citations**")
                        for c in ans.citations:
                            st.markdown(f"- [{c.section_id} · {c.title}]({c.url})")
            else:
                st.info("Retrieval-only mode. The passages the answer would be grounded in are shown on the right.")

        with source_col:
            st.subheader("Source passages")
            for i, p in enumerate(passages, start=1):
                label = f"{i}. {p.section_id}" + (f" · {p.heading}" if p.heading else "")
                with st.expander(f"{label}  (score {p.score:.2f})"):
                    st.markdown(f"[{p.title}]({p.url})")
                    st.write(p.text)

with tab_about:
    case_study(st, CASE_STUDY)

footer(st, "Real Michigan BEM/BAM policy, retrieved and cited")
