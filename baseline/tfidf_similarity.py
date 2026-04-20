from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    resume_text = (resume_text or "").strip()
    jd_text = (jd_text or "").strip()

    if not resume_text and not jd_text:
        return 0.0

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return max(0.0, min(1.0, float(score)))