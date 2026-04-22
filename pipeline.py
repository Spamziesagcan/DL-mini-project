from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from preprocessing import extract_text_from_pdf
from models.similarity_runtime import predict_similarity_scores, warmup_similarity_models
from utils.reporting import MatchResult, build_match_explanation
from utils.visualization import create_comparison_graph, create_match_graphs


def run_resume_screening_pipeline(
    resume_pdf_path: str,
    job_description_text: str,
    output_dir: str | None = None,
    generate_graphs: bool = False,
) -> MatchResult:
    resume_raw_text = extract_text_from_pdf(resume_pdf_path)
    dl_scores = predict_similarity_scores(resume_raw_text, job_description_text)

    result = build_match_explanation(
        resume_raw_text,
        job_description_text,
        lstm_score=float(dl_scores["lstm_score"]) * 100,
        cnn_score=float(dl_scores["cnn_score"]) * 100,
    )
    result = replace(result, resume_path=str(Path(resume_pdf_path).resolve()))

    if generate_graphs and output_dir:
        score_chart_path, skill_chart_path = create_match_graphs(result, output_dir)
        result = replace(
            result,
            score_chart_path=score_chart_path,
            skill_chart_path=skill_chart_path,
        )

    return result


def compare_resumes_against_job(
    resume_pdf_paths: Sequence[str],
    job_description_text: str,
    output_dir: str | None = None,
    generate_graphs: bool = False,
) -> tuple[list[MatchResult], str | None]:
    results = [
        run_resume_screening_pipeline(
            resume_pdf_path=resume_path,
            job_description_text=job_description_text,
            output_dir=output_dir,
            generate_graphs=generate_graphs,
        )
        for resume_path in resume_pdf_paths
    ]
    results = sorted(results, key=lambda item: item.ensemble_score, reverse=True)

    comparison_chart_path = None
    if generate_graphs and output_dir and results:
        comparison_chart_path = create_comparison_graph(results, output_dir)

    return results, comparison_chart_path


def export_results_to_json(results: Sequence[MatchResult], path: str) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.to_dict() for result in results]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(output_path)


def warmup_pipeline() -> None:
    warmup_similarity_models()
