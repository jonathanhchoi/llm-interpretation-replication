"""Offline integrity checks for the public replication package."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SURVEY_ANALYSIS_DIR = REPO_ROOT / "survey_analysis"
sys.path.insert(0, str(SURVEY_ANALYSIS_DIR))

from survey_analysis_consolidated import load_cleaned_question_responses  # noqa: E402


SURVEY_PATHS = [
    REPO_ROOT / "data" / "word_meaning_survey_results.csv",
    REPO_ROOT / "data" / "word_meaning_survey_results_part_2.csv",
]


def canonicalize(value: object) -> str:
    translation = str.maketrans(
        {
            "\u201c": '"',
            "\u201d": '"',
            "\u2018": "'",
            "\u2019": "'",
            "\u00a0": " ",
        }
    )
    return " ".join(str(value).translate(translation).split()).casefold()


class ReplicationInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.responses, cls.exclusions = load_cleaned_question_responses(
            [str(path) for path in SURVEY_PATHS],
            require_consent=False,
        )
        cls.human_questions = {canonicalize(question) for question in cls.responses}

    def test_preregistered_sample_and_question_counts(self) -> None:
        self.assertEqual(self.exclusions["raw_count"], 1008)
        self.assertEqual(self.exclusions["duration_excluded"], 0)
        self.assertEqual(self.exclusions["identical_excluded"], 9)
        self.assertEqual(self.exclusions["attention_failed"], 115)
        self.assertEqual(self.exclusions["total_excluded"], 124)
        self.assertEqual(self.exclusions["final_count"], 884)
        self.assertEqual(self.exclusions["n_substantive_questions"], 100)
        self.assertEqual(len(self.responses), 100)
        self.assertEqual(sum(map(len, self.responses.values())), 8807)

    def test_attention_check_is_not_a_substantive_question(self) -> None:
        joined = "\n".join(self.human_questions)
        self.assertNotIn("2 + 2", joined)
        self.assertNotIn("2+2", joined)
        self.assertIn(
            canonicalize(
                'Is an "algorithmic recommendation" an "editorial decision"?'
            ),
            self.human_questions,
        )

    def test_closed_source_results_cover_the_exact_battery(self) -> None:
        path = (
            REPO_ROOT
            / "results"
            / "closed_source_evaluation"
            / "closed_source_evaluation_results.csv"
        )
        frame = pd.read_csv(path)
        self.assertEqual(len(frame), 100)
        self.assertFalse(frame["question"].duplicated().any())
        self.assertEqual(
            {canonicalize(question) for question in frame["question"]},
            self.human_questions,
        )
        for column in (
            "gpt_weighted_confidence",
            "gemini_confidence",
            "claude_confidence",
        ):
            self.assertEqual(frame[column].notna().sum(), 100)
        expected_valid_counts = {
            "gpt_response": 100,
            "gemini_response": 99,
            "claude_response": 100,
        }
        for column, expected_count in expected_valid_counts.items():
            valid = frame[column].astype(str).str.match(
                r"^\s*(yes|no)\b",
                case=False,
            )
            self.assertEqual(int(valid.sum()), expected_count)
            if column == "gemini_response":
                self.assertEqual(
                    frame.loc[~valid, "question"].tolist(),
                    ['Is an "NFT" a "security"?'],
                )

    def test_closed_source_metadata_and_correlations_are_complete(self) -> None:
        output_dir = REPO_ROOT / "results" / "closed_source_evaluation"
        metadata = json.loads((output_dir / "evaluation_metadata.json").read_text())
        self.assertEqual(
            metadata["model_snapshots"],
            {
                "gpt": "gpt-4.1-2025-04-14",
                "gemini": "gemini-2.5-pro",
                "claude": "claude-opus-4-1-20250805",
            },
        )
        self.assertEqual(metadata["n_substantive_questions"], 100)
        self.assertEqual(metadata["human_sample"]["final_count"], 884)
        self.assertEqual(metadata["bootstrap_resamples"], 10_000)
        self.assertEqual(
            metadata["valid_binary_responses"],
            {"gpt": 100, "gemini": 99, "claude": 100},
        )

        correlations = json.loads((output_dir / "correlations.json").read_text())
        self.assertEqual(len(correlations), 3)
        for result in correlations.values():
            self.assertEqual(result["n"], 100)
            self.assertTrue(math.isfinite(result["correlation"]))
            self.assertTrue(-1.0 <= result["correlation"] <= 1.0)

        comparisons = json.loads(
            (output_dir / "human_comparisons.json").read_text()
        )
        expected_maes = {
            "gpt": 0.22966479196837444,
            "claude": 0.2289349809558382,
            "gemini": 0.3432004522637239,
        }
        for model, expected in expected_maes.items():
            self.assertAlmostEqual(
                comparisons["models"][model]["mae"],
                expected,
                places=12,
            )
        self.assertAlmostEqual(
            comparisons["baselines"]["always_50"]["mae"],
            0.17315890387616442,
            places=12,
        )
        self.assertAlmostEqual(
            comparisons["baselines"]["normal_human"]["mae"],
            0.19311542207731097,
            places=12,
        )

    def test_human_distribution_output_is_complete(self) -> None:
        frame = pd.read_csv(
            REPO_ROOT / "survey_analysis" / "human_response_distribution.csv"
        )
        self.assertEqual(len(frame), 100)
        self.assertEqual(
            {canonicalize(question) for question in frame["question"]},
            self.human_questions,
        )
        self.assertEqual(int((frame["sd"] >= 30.0).sum()), 41)
        self.assertAlmostEqual(float(frame["sd"].median()), 29.0158291137, places=6)

    def test_base_tuned_output_uses_six_exact_checkpoints(self) -> None:
        payload = json.loads(
            (REPO_ROOT / "results" / "base_vs_tuned_49q.json").read_text()
        )
        metadata = payload["metadata"]
        self.assertEqual(metadata["human_cleaned_n"], 884)
        self.assertEqual(metadata["matched_questions_per_checkpoint"], 49)
        self.assertEqual(metadata["bootstrap_resamples"], 10_000)

        expected_ids = {
            "tiiuae/falcon-7b",
            "tiiuae/falcon-7b-instruct",
            "stabilityai/stablelm-base-alpha-7b",
            "stabilityai/stablelm-tuned-alpha-7b",
            "togethercomputer/RedPajama-INCITE-7B-Base",
            "togethercomputer/RedPajama-INCITE-7B-Instruct",
        }
        found_ids = {
            family[role]["model_id"]
            for family in payload["families"].values()
            for role in ("base", "tuned")
        }
        self.assertEqual(found_ids, expected_ids)
        for family in payload["families"].values():
            self.assertEqual(family["base"]["n_questions"], 49)
            self.assertEqual(family["tuned"]["n_questions"], 49)
        expected_differences = {
            "Falcon": 0.07327356634356916,
            "StableLM": -0.0349819747508274,
            "RedPajama": -0.00190610201693589,
        }
        for family, expected in expected_differences.items():
            self.assertAlmostEqual(
                payload["families"][family]["tuned_minus_base"][
                    "observed_difference"
                ],
                expected,
                places=12,
            )

    def test_topk_audit_matches_reported_sample(self) -> None:
        frame = pd.read_csv(
            REPO_ROOT / "results" / "gpt41_confidence_topk_audit.csv"
        )
        bounds = pd.to_numeric(frame["error_bound"], errors="raise")
        self.assertEqual(len(bounds), 500)
        self.assertLess(float(bounds.median()), 0.001)
        self.assertAlmostEqual(float(bounds.quantile(0.95)), 0.121404, places=5)
        self.assertTrue(bounds.between(0.0, 100.0).all())


if __name__ == "__main__":
    unittest.main()
