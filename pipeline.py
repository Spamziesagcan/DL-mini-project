from __future__ import annotations

from baseline import compute_tfidf_similarity
from preprocessing import clean_text, extract_text_from_pdf
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
    matched_skills, missing_skills = _extract_skill_match(resume_raw_text, job_description_text)

    return {
        "final_match_score": round(tfidf_score * 100, 2),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
    }