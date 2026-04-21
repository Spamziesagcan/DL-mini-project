from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from pipeline import run_resume_screening_pipeline


def _save_uploaded_pdf(uploaded_file) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def main() -> None:
    st.set_page_config(page_title="Resume Screening", layout="centered")

    st.title("Resume Screening")
    st.caption("Upload a resume PDF and paste a job description to compare LSTM and CNN match scores.")

    resume_pdf = st.file_uploader("Resume PDF", type=["pdf"])
    job_description_text = st.text_area(
        "Job Description",
        height=220,
        placeholder="Paste the job description here.",
    )

    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

    if not analyze_clicked:
        return

    if resume_pdf is None:
        st.error("Please upload a resume PDF.")
        return

    if not job_description_text.strip():
        st.error("Please enter a job description.")
        return

    temp_pdf_path = _save_uploaded_pdf(resume_pdf)
    analysis_result = None

    try:
        analysis_result = run_resume_screening_pipeline(str(temp_pdf_path), job_description_text)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
    finally:
        temp_pdf_path.unlink(missing_ok=True)

    if analysis_result is None:
        return

    final_match_score = float(analysis_result["final_match_score"])
    baseline_tfidf_score = float(analysis_result["baseline_tfidf_score"])
    lstm_score = float(analysis_result["lstm_score"])
    cnn_score = float(analysis_result["cnn_score"])
    matched_skills = list(analysis_result["matched_skills"])
    missing_skills = list(analysis_result["missing_skills"])

    st.metric("Match Score", f"{final_match_score:.1f}%")

    score_columns = st.columns(3)
    score_columns[0].metric("LSTM Score", f"{lstm_score:.1f}%")
    score_columns[1].metric("CNN Score", f"{cnn_score:.1f}%")
    score_columns[2].metric("TF-IDF Baseline", f"{baseline_tfidf_score:.1f}%")

    st.subheader("Matched Skills")
    st.write(", ".join(matched_skills) if matched_skills else "None")

    st.subheader("Missing Skills")
    st.write(", ".join(missing_skills) if missing_skills else "None")


if __name__ == "__main__":
    main()