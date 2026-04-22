from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import shutil
from unittest.mock import patch

import cli
from utils.reporting import build_match_explanation


def _result(resume_path: str, ensemble_score: float = 76.0):
    base = build_match_explanation("python react sql", "python sql docker", 74.0, 78.0)
    return base.__class__(
        resume_path=resume_path,
        job_description_preview=base.job_description_preview,
        lstm_score=base.lstm_score,
        cnn_score=base.cnn_score,
        ensemble_score=ensemble_score,
        explanation=base.explanation,
        score_chart_path="score.png",
        skill_chart_path="skills.png",
    )


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests/.tmp_cli")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("cli.run_resume_screening_pipeline", return_value=_result("resume.pdf"))
    def test_match_command_prints_scores(self, _: object) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main(
                [
                    "match",
                    "--resume",
                    "resume.pdf",
                    "--jd-text",
                    "python sql docker",
                    "--no-graphs",
                ]
            )

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("LSTM Score:", output)
        self.assertIn("CNN Score:", output)
        self.assertIn("Decision:", output)

    @patch("cli.compare_resumes_against_job", return_value=([_result("a.pdf", 88.0), _result("b.pdf", 70.0)], "comparison.png"))
    def test_compare_command_prints_ranked_results(self, _: object) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main(
                [
                    "compare",
                    "--resumes",
                    "a.pdf",
                    "b.pdf",
                    "--jd-text",
                    "python sql docker",
                    "--no-graphs",
                ]
            )

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("Rank 1", output)
        self.assertIn("Comparison Chart: comparison.png", output)

    def test_match_command_reads_job_description_file(self) -> None:
        jd_path = self.temp_dir / "job.txt"
        jd_path.write_text("python sql docker", encoding="utf-8")
        stdout = io.StringIO()
        with patch("cli.run_resume_screening_pipeline", return_value=_result("resume.pdf")):
            with redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "match",
                        "--resume",
                        "resume.pdf",
                        "--jd-file",
                        str(jd_path),
                        "--no-graphs",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("resume.pdf", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
