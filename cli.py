from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline import compare_resumes_against_job, export_results_to_json, run_resume_screening_pipeline, warmup_pipeline
from utils.reporting import MatchResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resume-match",
        description="Resume screening CLI with CNN and LSTM scoring, explanations, and graphs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    match_parser = subparsers.add_parser("match", help="Score one resume against one job description.")
    match_parser.add_argument("--resume", required=True, help="Path to the resume PDF.")
    _add_job_description_args(match_parser)
    match_parser.add_argument("--output-dir", default="outputs", help="Directory for charts and exported files.")
    match_parser.add_argument("--json-out", help="Optional path to save the structured result as JSON.")
    match_parser.add_argument("--no-graphs", action="store_true", help="Skip graph generation.")

    compare_parser = subparsers.add_parser("compare", help="Compare multiple resumes against one job description.")
    compare_parser.add_argument("--resumes", nargs="+", required=True, help="Paths to resume PDFs.")
    _add_job_description_args(compare_parser)
    compare_parser.add_argument("--output-dir", default="outputs", help="Directory for charts and exported files.")
    compare_parser.add_argument("--json-out", help="Optional path to save structured results as JSON.")
    compare_parser.add_argument("--no-graphs", action="store_true", help="Skip graph generation.")

    warmup_parser = subparsers.add_parser("warmup", help="Prepare reusable model artifacts for fast later runs.")
    warmup_parser.add_argument("--output-dir", default="artifacts", help="Artifact directory hint shown in output.")
    return parser


def _add_job_description_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--jd-text", help="Inline job description text.")
    group.add_argument("--jd-file", help="Path to a text file containing the job description.")


def _load_job_description(args: argparse.Namespace) -> str:
    if args.jd_text:
        return args.jd_text.strip()
    return Path(args.jd_file).read_text(encoding="utf-8").strip()


def _print_match_result(result: MatchResult) -> None:
    explanation = result.explanation
    print(f"Resume:      {result.resume_path}")
    print(f"Job:         {result.job_description_preview}")
    print(f"LSTM Score:  {result.lstm_score:.2f}%")
    print(f"CNN Score:   {result.cnn_score:.2f}%")
    print(f"Ensemble:    {result.ensemble_score:.2f}%")
    print(f"Overlap:     {explanation.overlap_ratio * 100:.1f}% of JD skills covered")
    print(f"Matched:     {', '.join(explanation.matched_skills) if explanation.matched_skills else 'None'}")
    print(f"Missing:     {', '.join(explanation.missing_skills) if explanation.missing_skills else 'None'}")
    print(f"Resume Only: {', '.join(explanation.resume_only_skills) if explanation.resume_only_skills else 'None'}")
    print(f"Decision:    {explanation.recommendation}")
    if result.score_chart_path:
        print(f"Score Chart: {result.score_chart_path}")
    if result.skill_chart_path:
        print(f"Skill Chart: {result.skill_chart_path}")


def _run_match(args: argparse.Namespace) -> int:
    jd_text = _load_job_description(args)
    result = run_resume_screening_pipeline(
        resume_pdf_path=args.resume,
        job_description_text=jd_text,
        output_dir=args.output_dir,
        generate_graphs=not args.no_graphs,
    )
    _print_match_result(result)
    if args.json_out:
        json_path = export_results_to_json([result], args.json_out)
        print(f"JSON:        {json_path}")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    jd_text = _load_job_description(args)
    results, comparison_chart_path = compare_resumes_against_job(
        resume_pdf_paths=args.resumes,
        job_description_text=jd_text,
        output_dir=args.output_dir,
        generate_graphs=not args.no_graphs,
    )

    for index, result in enumerate(results, start=1):
        print(f"\nRank {index}")
        print("-" * 72)
        _print_match_result(result)

    if comparison_chart_path:
        print(f"\nComparison Chart: {comparison_chart_path}")
    if args.json_out:
        json_path = export_results_to_json(results, args.json_out)
        print(f"JSON:             {json_path}")
    return 0


def _run_warmup(_: argparse.Namespace) -> int:
    warmup_pipeline()
    print("Artifacts are ready for later CLI runs.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "match":
        return _run_match(args)
    if args.command == "compare":
        return _run_compare(args)
    if args.command == "warmup":
        return _run_warmup(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
