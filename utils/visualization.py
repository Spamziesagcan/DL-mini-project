from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .reporting import MatchResult


def create_match_graphs(
    result: MatchResult,
    output_dir: str | Path,
) -> tuple[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_stem = Path(result.resume_path).stem.replace(" ", "_")
    score_chart = output_path / f"{safe_stem}_score_summary.png"
    skill_chart = output_path / f"{safe_stem}_skill_alignment.png"

    _plot_score_chart(result, score_chart)
    _plot_skill_chart(result, skill_chart)
    return str(score_chart), str(skill_chart)


def create_comparison_graph(
    results: Sequence[MatchResult],
    output_dir: str | Path,
    file_name: str = "comparison_scores.png",
) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart_path = output_path / file_name

    labels = [Path(result.resume_path).stem for result in results]
    ensemble_scores = [result.ensemble_score for result in results]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, ensemble_scores, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    plt.ylim(0, 100)
    plt.ylabel("Ensemble Score (%)")
    plt.title("Candidate Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, score in zip(bars, ensemble_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 1, f"{score:.1f}%", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=160)
    plt.close()
    return str(chart_path)


def _plot_score_chart(result: MatchResult, path: Path) -> None:
    labels = ["LSTM", "CNN", "Ensemble"]
    scores = [result.lstm_score, result.cnn_score, result.ensemble_score]
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, scores, color=colors)
    plt.ylim(0, 100)
    plt.ylabel("Score (%)")
    plt.title("Model Match Scores")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 1, f"{score:.1f}%", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_skill_chart(result: MatchResult, path: Path) -> None:
    explanation = result.explanation
    labels = ["Matched", "Missing", "Resume Only"]
    counts = [
        len(explanation.matched_skills),
        len(explanation.missing_skills),
        len(explanation.resume_only_skills),
    ]
    colors = ["#59a14f", "#e15759", "#9c755f"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, counts, color=colors)
    plt.ylabel("Skill Count")
    plt.title("Skill Alignment Breakdown")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, count + 0.1, str(count), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
