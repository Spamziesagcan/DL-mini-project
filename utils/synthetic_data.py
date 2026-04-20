from __future__ import annotations

from random import Random
from typing import Sequence


def generate_synthetic_training_data(
    resume_texts: Sequence[str],
    job_description_texts: Sequence[str],
    negative_ratio: float = 1.0,
    random_state: int | None = None,
) -> tuple[list[tuple[str, str]], list[int]]:
    """Build simple positive and negative resume-job pairs for training."""
    rng = Random(random_state)

    resumes = [text.strip() for text in resume_texts if text and text.strip()]
    job_descriptions = [text.strip() for text in job_description_texts if text and text.strip()]
    pair_count = min(len(resumes), len(job_descriptions))

    if pair_count == 0:
        return [], []

    x: list[tuple[str, str]] = []
    y: list[int] = []

    for index in range(pair_count):
        x.append((resumes[index], job_descriptions[index]))
        y.append(1)

    negative_count = int(pair_count * negative_ratio)
    if negative_count <= 0:
        return x, y

    if len(job_descriptions) > 1:
        shift = rng.randrange(1, len(job_descriptions))
        for index in range(negative_count):
            resume_index = index % pair_count
            jd_index = (resume_index + shift) % len(job_descriptions)
            x.append((resumes[resume_index], job_descriptions[jd_index]))
            y.append(0)
    elif len(resumes) > 1:
        shift = rng.randrange(1, len(resumes))
        for index in range(negative_count):
            jd_index = index % pair_count
            resume_index = (jd_index + shift) % len(resumes)
            x.append((resumes[resume_index], job_descriptions[jd_index]))
            y.append(0)

    return x, y