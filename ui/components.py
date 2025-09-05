from __future__ import annotations
import io
import streamlit as st
from typing import Dict, Any


def show_header():
    st.title("📄 Resume Analyzer AI")
    st.caption("Agentic, RAG-powered insights for stronger resumes. Deployed on Render.")


def show_footer():
    st.divider()
    st.caption("Built with Streamlit • LangGraph • Google Generative AI • In-memory RAG")


def show_alert(level: str, msg: str):
    if level == "error":
        st.error(msg)
    elif level == "warning":
        st.warning(msg)
    else:
        st.info(msg)


def _metric(label: str, value: int, help_text: str = ""):
    st.metric(label, value, help=help_text)


def show_scores(scores: Dict[str, Any] | None):
    if not scores:
        st.info("No scores available yet.")
        return
    st.subheader("📊 Scores")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _metric("Overall", scores.get("overall_score", 0))
    with c2:
        _metric("Content", scores.get("content_score", 0))
    with c3:
        _metric("Format", scores.get("format_score", 0))
    with c4:
        _metric("Keywords", scores.get("keyword_score", 0))
    with c5:
        _metric("Impact", scores.get("impact_score", 0))
    st.caption(scores.get("feedback", ""))


def show_analysis_markdown(text: str):
    st.subheader("💡 Detailed Analysis & Suggestions")
    if not text:
        st.info("Run an analysis to see suggestions here.")
        return
    st.markdown(text, unsafe_allow_html=False)


def side_about_panel():
    with st.sidebar:
        st.header("⚙️ Settings")
        st.text_input("GENAI_MODEL", value="gemini-2.0-flash-lite", disabled=True, help="Configured via env var in Render.")
        st.text_input("EMBEDDINGS_MODEL", value="models/embedding-001", disabled=True)
        st.divider()
        st.header("ℹ️ About")
        st.write("This app analyzes resumes using an agentic graph + RAG. Provide PDF/TXT or paste text.")
        st.write("No files are persisted; temporary files are removed by the OS on container restart.")


def build_report_bytes(result: Dict[str, Any]) -> bytes:
    lines = ["# Resume Analysis Report\n"]
    scores = result.get("scores") or {}
    if scores:
        lines.append("## Scores\n")
        for k in ["overall_score", "content_score", "format_score", "keyword_score", "impact_score"]:
            if k in scores:
                lines.append(f"- **{k.replace('_', ' ').title()}**: {scores[k]}")
        fb = scores.get("feedback")
        if fb:
            lines.append(f"\n> {fb}\n")
    analysis = result.get("analysis")
    if analysis:
        lines.append("\n## Detailed Analysis\n")
        lines.append(analysis)
    buf = "\n".join(lines)
    return buf.encode("utf-8")