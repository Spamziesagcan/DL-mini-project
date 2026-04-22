from __future__ import annotations

import unittest

from utils.reporting import build_match_explanation


class ReportingTests(unittest.TestCase):
    def test_build_match_explanation_identifies_overlap_and_gaps(self) -> None:
        resume_text = "Python React SQL Docker Git REST APIs"
        jd_text = "Backend developer with Python SQL Docker and Linux"

        result = build_match_explanation(resume_text, jd_text, lstm_score=72.5, cnn_score=77.5)

        self.assertEqual(result.ensemble_score, 75.0)
        self.assertIn("python", result.explanation.matched_skills)
        self.assertIn("sql", result.explanation.matched_skills)
        self.assertIn("docker", result.explanation.matched_skills)
        self.assertIn("linux", result.explanation.missing_skills)
        self.assertIn("react", result.explanation.resume_only_skills)
        self.assertGreater(result.explanation.overlap_ratio, 0)


if __name__ == "__main__":
    unittest.main()
