import os
import traceback
import streamlit as st
from ui.components import (
    show_header, show_footer, show_scores, show_alert,
    show_analysis_markdown, side_about_panel, build_report_bytes
)
from core.pipeline import analyze_resume, analyze_resume_from_file

st.set_page_config(page_title="Resume Analyzer AI", page_icon="📄", layout="wide")

# --- Sidebar (About & Config) ---
side_about_panel()

# --- Header ---
show_header()

TAB_UPLOAD, TAB_PASTE, TAB_ADVICE = st.tabs([
    "Upload PDF/TXT", "Paste Text", "General Advice"
])

with TAB_UPLOAD:
    st.subheader("Upload your resume")
    uploaded = st.file_uploader("Choose a .pdf or .txt file", type=["pdf", "txt"])
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        run_btn = st.button("Analyze", type="primary", use_container_width=True, disabled=uploaded is None)
    with col_btn2:
        st.caption("Max size ~10MB. We never store your files on the server beyond processing.")

    if run_btn and uploaded:
        try:
            temp_path = f"/tmp/{uploaded.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded.read())
            result = analyze_resume_from_file(temp_path, verbose=False)

            if "error" in result:
                show_alert("error", result["error"]) 
            else:
                show_scores(result.get("scores"))
                show_analysis_markdown(result.get("analysis", ""))

                # Allow user to download a report
                report = build_report_bytes(result)
                st.download_button(
                    label="Download Report (Markdown)",
                    data=report,
                    file_name="resume_analysis_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
        except Exception as e:
            show_alert("error", f"Processing failed: {e}")
            st.exception(e)

with TAB_PASTE:
    st.subheader("Paste your resume text")
    text = st.text_area("Paste complete resume text", height=280, placeholder="Paste here…")
    if st.button("Analyze Text", type="primary"):
        if not text or len(text) < 200:
            show_alert("warning", "Please paste the complete resume (at least ~200 characters) for best results.")
        else:
            try:
                result = analyze_resume(text, verbose=False)
                if "error" in result:
                    show_alert("error", result["error"]) 
                else:
                    show_scores(result.get("scores"))
                    show_analysis_markdown(result.get("analysis", ""))
                    report = build_report_bytes(result)
                    st.download_button(
                        label="Download Report (Markdown)",
                        data=report,
                        file_name="resume_analysis_report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
            except Exception:
                show_alert("error", "Unexpected error while analyzing pasted text.")
                st.code(traceback.format_exc())

with TAB_ADVICE:
    st.subheader("Ask for general resume advice")
    q = st.text_area("Ask a question (e.g., 'How do I improve bullet points?')", height=160)
    if st.button("Get Advice", type="primary"):
        if not q.strip():
            show_alert("warning", "Please enter a question.")
        else:
            try:
                result = analyze_resume(q.strip(), verbose=False)
                if "error" in result:
                    show_alert("error", result["error"]) 
                else:
                    show_analysis_markdown(result.get("analysis", ""))
            except Exception:
                show_alert("error", "Unexpected error while generating advice.")
                st.code(traceback.format_exc())

show_footer()