from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .skill_extraction import extract_skills


@dataclass(frozen=True)
class MatchExplanation:
    matched_skills: list[str]
    missing_skills: list[str]
    resume_only_skills: list[str]
    jd_only_skills: list[str]
    resume_skill_count: int
    jd_skill_count: int
    overlap_ratio: float
    recommendation: str


@dataclass(frozen=True)
class MatchResult:
    resume_path: str
    job_description_preview: str
    lstm_score: float
    cnn_score: float
    ensemble_score: float
    explanation: MatchExplanation
    score_chart_path: str | None = None
    skill_chart_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["resume_path"] = str(Path(self.resume_path))
        return payload


def _preview_text(text: str, limit: int = 90) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _build_recommendation(
    ensemble_score: float,
    matched_count: int,
    missing_count: int,
    overlap_ratio: float,
) -> str:
    if ensemble_score >= 80 and missing_count <= 2:
        return "Strong fit. Prioritize this candidate for the next screening round."
    if ensemble_score >= 65 and matched_count >= max(2, missing_count):
        return "Promising fit. Validate the remaining skill gaps during interview screening."
    if overlap_ratio >= 0.7 and matched_count >= max(3, missing_count):
        return "Skill-aligned profile. Model confidence is conservative, so use the skill overlap to guide manual review."
    if ensemble_score >= 50:
        return "Partial fit. Candidate has relevant overlap but needs targeted upskilling or role narrowing."
    return "Weak fit for this job description. The current resume does not align with the required stack."


def build_match_explanation(
    resume_text: str,
    job_description_text: str,
    lstm_score: float,
    cnn_score: float,
) -> MatchResult:
    resume_skills, _ = extract_skills(resume_text)
    jd_skills, _ = extract_skills(job_description_text)

    resume_skill_set = set(resume_skills)
    jd_skill_set = set(jd_skills)
    matched_skills = sorted(resume_skill_set & jd_skill_set)
    missing_skills = sorted(jd_skill_set - resume_skill_set)
    resume_only_skills = sorted(resume_skill_set - jd_skill_set)
    denominator = max(1, len(jd_skill_set))
    overlap_ratio = round(len(matched_skills) / denominator, 4)
    ensemble_score = round((lstm_score + cnn_score) / 2, 2)

    explanation = MatchExplanation(
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        resume_only_skills=resume_only_skills,
        jd_only_skills=missing_skills,
        resume_skill_count=len(resume_skill_set),
        jd_skill_count=len(jd_skill_set),
        overlap_ratio=overlap_ratio,
        recommendation=_build_recommendation(
            ensemble_score,
            len(matched_skills),
            len(missing_skills),
            overlap_ratio,
        ),
    )

    return MatchResult(
        resume_path="",
        job_description_preview=_preview_text(job_description_text),
        lstm_score=round(lstm_score, 2),
        cnn_score=round(cnn_score, 2),
        ensemble_score=ensemble_score,
        explanation=explanation,
    )
