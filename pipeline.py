from __future__ import annotations

from baseline import compute_tfidf_similarity
from preprocessing import clean_text, extract_text_from_pdf
from models.similarity_runtime import predict_similarity_scores
from utils import extract_skills


def _extract_skill_match(resume_text: str, jd_text: str) -> tuple[list[str], list[str]]:
    resume_skills, _ = extract_skills(resume_text)
    jd_skills, _ = extract_skills(jd_text)

    matched_skills = [skill for skill in jd_skills if skill in resume_skills]
    missing_skills = [skill for skill in jd_skills if skill not in resume_skills]
    return matched_skills, missing_skills


def run_resume_screening_pipeline(
    resume_pdf_path: str,
    job_description_text: str,
) -> dict[str, object]:
    """Run the resume screening pipeline and return the final match summary."""
    resume_raw_text = extract_text_from_pdf(resume_pdf_path)
    resume_clean_text = clean_text(resume_raw_text)
    jd_clean_text = clean_text(job_description_text)

    tfidf_score = compute_tfidf_similarity(resume_clean_text, jd_clean_text)
    dl_scores = predict_similarity_scores(resume_raw_text, job_description_text)
    matched_skills, missing_skills = _extract_skill_match(resume_raw_text, job_description_text)

    lstm_score = float(dl_scores["lstm_score"])
    cnn_score = float(dl_scores["cnn_score"])
    final_match_score = (lstm_score + cnn_score) / 2.0

    return {
        "final_match_score": round(final_match_score * 100, 2),
        "baseline_tfidf_score": round(tfidf_score * 100, 2),
        "lstm_score": round(lstm_score * 100, 2),
        "cnn_score": round(cnn_score * 100, 2),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
    }