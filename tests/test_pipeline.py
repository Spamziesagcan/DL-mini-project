from __future__ import annotations

import json
import unittest
from pathlib import Path
import shutil
from unittest.mock import patch

from pipeline import compare_resumes_against_job, export_results_to_json, run_resume_screening_pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests/.tmp_pipeline")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("pipeline.predict_similarity_scores", return_value={"lstm_score": 0.61, "cnn_score": 0.73})
    @patch("pipeline.extract_text_from_pdf", return_value="Python React SQL REST APIs Git")
    def test_run_resume_screening_pipeline_returns_explanation_and_graphs(self, _: object, __: object) -> None:
        result = run_resume_screening_pipeline(
            resume_pdf_path="resume.pdf",
            job_description_text="Need Python SQL Docker",
            output_dir=str(self.temp_dir),
            generate_graphs=True,
        )

        self.assertEqual(result.lstm_score, 61.0)
        self.assertEqual(result.cnn_score, 73.0)
        self.assertEqual(result.ensemble_score, 67.0)
        self.assertTrue(Path(result.score_chart_path).is_file())
        self.assertTrue(Path(result.skill_chart_path).is_file())

    @patch("pipeline.run_resume_screening_pipeline")
    def test_compare_resumes_sorts_results_descending(self, mocked_run: object) -> None:
        mocked_run.side_effect = [
            self._fake_result("a.pdf", 51.0),
            self._fake_result("b.pdf", 83.0),
        ]

        results, _ = compare_resumes_against_job(
            resume_pdf_paths=["a.pdf", "b.pdf"],
            job_description_text="python",
            output_dir=None,
            generate_graphs=False,
        )

        self.assertEqual([Path(item.resume_path).name for item in results], ["b.pdf", "a.pdf"])

    def test_export_results_to_json_writes_serialized_payload(self) -> None:
        path = self.temp_dir / "results.json"
        export_results_to_json([self._fake_result("resume.pdf", 75.0)], str(path))
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["ensemble_score"], 75.0)

    def _fake_result(self, resume_path: str, ensemble_score: float):
        from utils.reporting import build_match_explanation

        result = build_match_explanation("python react", "python docker", 70.0, 80.0)
        return result.__class__(
            resume_path=resume_path,
            job_description_preview=result.job_description_preview,
            lstm_score=result.lstm_score,
            cnn_score=result.cnn_score,
            ensemble_score=ensemble_score,
            explanation=result.explanation,
            score_chart_path=None,
            skill_chart_path=None,
        )


if __name__ == "__main__":
    unittest.main()
