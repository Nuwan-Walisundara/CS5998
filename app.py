"""Stage 7/8 - Validation dashboard.   Run:  streamlit run app.py"""
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src.pipeline import analyse_files
from src.questions import llm_questions, ollama_available, template_questions, to_email_text

st.set_page_config(page_title="WSO2 Requirements Validator", layout="wide")
st.title("WSO2 Integration Requirements Validator")
st.caption("Flags potential gaps and issues for analyst review. It never changes the requirements.")

with st.sidebar:
    st.header("Settings")
    sim_method = st.selectbox("Similarity representation", ["embedding", "tfidf"])
    threshold = st.slider("Similarity threshold", 0.50, 0.99, config.SIMILARITY_THRESHOLD, 0.01)
    scope = st.radio("Duplicate scope", ["all", "within"], format_func=lambda s: {"all": "Within + across documents", "within": "Within document"}[s])
    llm_ok = ollama_available()
    st.markdown(f"**Ollama:** {'🟢 available' if llm_ok else '⚪ not running (template questions only)'}")
    st.caption(f"Model: `{config.OLLAMA_MODEL}`")

uploads = st.file_uploader("Upload requirement documents (.md / .docx)", type=["md", "docx"], accept_multiple_files=True)
samples = sorted(config.SYNTHETIC_DIR.glob("*.md"))
use_sample = st.checkbox(f"Use synthetic sample documents ({len(samples)})", value=not uploads)

paths = []
if uploads:
    tmp = Path(tempfile.mkdtemp())
    for u in uploads:
        p = tmp / u.name; p.write_bytes(u.getbuffer()); paths.append(p)
elif use_sample:
    paths = samples

if not paths:
    st.info("Upload a document or select the sample documents."); st.stop()


@st.cache_data(show_spinner="Analysing requirements…")
def run(paths_key, sim_method, threshold, scope):
    r = analyse_files([Path(p) for p in paths_key], sim_method=sim_method, sim_threshold=threshold,
                      sim_scope=scope, questions=False)
    return r.units, r.mappings, r.findings, r.similar_pairs


units, mappings, findings, pairs = run(tuple(map(str, paths)), sim_method, threshold, scope)

c = st.columns(5)
c[0].metric("Documents", units["doc"].nunique())
c[1].metric("Requirement statements", int(units["is_nlp_unit"].sum()))
c[2].metric("Findings", len(findings))
c[3].metric("High severity", int((findings["severity"] == "High").sum()))
c[4].metric("Possible duplicates", len(pairs))

tab_f, tab_q, tab_r, tab_s, tab_m = st.tabs(["⚠️ Findings", "❓ Clarification questions", "📋 Requirements",
                                             "🔁 Similar requirements", "🗺️ Mappings"])

with tab_f:
    f1, f2, f3 = st.columns(3)
    sev = f1.multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    src = f2.multiselect("Source", sorted(findings["source"].dropna().unique()), default=sorted(findings["source"].dropna().unique()))
    docs = f3.multiselect("Document", sorted(findings["doc"].unique()), default=sorted(findings["doc"].unique()))
    view = findings[findings.severity.isin(sev) & findings.source.isin(src) & findings.doc.isin(docs)]
    st.dataframe(view[["finding_id", "doc", "req_id", "field", "category", "issue_type", "severity", "source", "detail", "text"]]
                 if "category" in view else view, width="stretch", hide_index=True)
    st.bar_chart(findings.groupby("issue_type").size().sort_values(ascending=False))

with tab_q:
    st.write("Questions are generated **only from the findings above**. Review, edit or delete them before sending.")
    sel = st.multiselect("Findings to include", findings["finding_id"].tolist(), default=findings["finding_id"].tolist())
    sub = findings[findings.finding_id.isin(sel)]
    b1, b2 = st.columns(2)
    if b1.button("Generate template questions"):
        st.session_state.q, st.session_state.qinfo = template_questions(sub), {"model": None}
    if b2.button("Generate with Ollama (LLM)", disabled=not llm_ok):
        with st.spinner(f"Asking {config.OLLAMA_MODEL}…"):
            st.session_state.q, st.session_state.qinfo = llm_questions(sub)
    if "q" in st.session_state:
        info = st.session_state.qinfo
        if info.get("model"):
            st.caption(f"LLM latency: {info.get('latency_s')} s | fallback used: {info.get('fallback')} | "
                       f"unknown issue IDs dropped: {info.get('dropped_unknown_ids')} | issues covered by template: {info.get('uncovered')}")
        edited = st.data_editor(st.session_state.q.assign(approve=True), num_rows="dynamic",
                                width="stretch", hide_index=True,
                                column_config={"question": st.column_config.TextColumn(width="large")})
        final = edited[edited["approve"]].drop(columns="approve")
        d1, d2 = st.columns(2)
        d1.download_button("Download CSV", final.to_csv(index=False), "clarification_questions.csv")
        doc_name = ", ".join(final["doc"].dropna().unique()) or "requirements"
        d2.download_button("Download email text", to_email_text(final, doc_name), "clarification_email.txt")

with tab_r:
    cols = [c for c in ["req_id", "doc", "section", "field", "text_clean", "category", "category_conf",
                        "category_source", "quality_label", "is_empty"] if c in units]
    only_nlp = st.toggle("Requirement statements only", value=True)
    st.dataframe(units.loc[units.is_nlp_unit, cols] if only_nlp else units[cols], width="stretch", hide_index=True)
    if "category" in units:
        st.bar_chart(units.loc[units.is_nlp_unit, "category"].value_counts())

with tab_s:
    st.dataframe(pairs, width="stretch", hide_index=True)

with tab_m:
    st.dataframe(mappings, width="stretch", hide_index=True)
