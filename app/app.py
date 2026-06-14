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

from data.chunk import build_nodes
from src.answer import answer_question, build_llm
from src.config import has_llm
from src.retrieve import RetrievalEngine

st.set_page_config(page_title="Michigan SNAP policy RAG", layout="wide")


@st.cache_resource(show_spinner="Building indexes over the manual...")
def get_engine(mode: str, rerank: bool) -> RetrievalEngine:
    return RetrievalEngine(build_nodes("section_aware"), mode=mode, use_rerank=rerank, final_n=4)


@st.cache_resource(show_spinner=False)
def get_llm():
    return build_llm() if has_llm() else None


st.title("Michigan SNAP eligibility — policy Q&A")
st.caption(
    "Grounded answers over a real subset of the Michigan Bridges Eligibility Manual. "
    "Every policy statement cites the BEM/BAM section it came from. If the manual "
    "does not cover the question, the system declines rather than inventing."
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
                        st.markdown(f"- [{c.section_id} — {c.title}]({c.url})")
        else:
            st.info("Retrieval-only mode. The passages the answer would be grounded in are shown on the right.")

    with source_col:
        st.subheader("Source passages")
        for i, p in enumerate(passages, start=1):
            label = f"{i}. {p.section_id}" + (f" — {p.heading}" if p.heading else "")
            with st.expander(f"{label}  (score {p.score:.2f})"):
                st.markdown(f"[{p.title}]({p.url})")
                st.write(p.text)
