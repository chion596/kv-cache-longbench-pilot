import copy
import unittest

from src.analyze_group_size import analyze_details


class GroupSizeDetailTests(unittest.TestCase):
    def fixture(self):
        reference = {}
        rows = {"passage_retrieval_en:0": {"pred": "Paragraph 1", "answers": ["Paragraph 1"],
                    "output_ids": [1, 2], "context_sha256": "a", "input_tokens": 100, "stop_reason": "eos"},
                "passage_retrieval_en:1": {"pred": "Paragraph 1", "answers": ["Paragraph 1"],
                    "output_ids": [1, 2], "context_sha256": "b", "input_tokens": 200, "stop_reason": "eos"}}
        for cache in ("dynamic", "int4_g16", "int4_g64"):
            reference[("qwen3-4b", "passage_retrieval_en", "context_then_question", cache)] = copy.deepcopy(rows)
        matrix = {("qwen3-4b", "passage_retrieval_en", "context_then_question", "int4_g32"): copy.deepcopy(rows)}
        return matrix, reference

    def test_changed_first_token_does_not_imply_wrong_answer(self):
        matrix, reference = self.fixture()
        rows = next(iter(matrix.values()))
        rows["passage_retrieval_en:0"]["output_ids"] = [9, 2]
        rows["passage_retrieval_en:1"]["pred"] = "Paragraph 2"
        original = copy.deepcopy((matrix, reference))
        summary, contrasts, trajectories = analyze_details(matrix, reference,
            {"passage_retrieval_en": lambda p, a, **_: float(p == a)})
        row = summary[0]
        self.assertEqual(row["changed_first_token_full_score"], 1)
        self.assertEqual(row["changed_first_token_score_same"], 1)
        self.assertEqual(row["changed_first_token_score_lower"], 0)
        self.assertEqual(row["same_first_token_score_lower"], 1)
        self.assertEqual(row["baseline_correct_first_number_lost"], 1)
        self.assertEqual(len(contrasts), 3)
        self.assertEqual(contrasts[0]["delta_pp"], -50)
        self.assertEqual(len(trajectories), 2)
        self.assertEqual((matrix, reference), original)

    def test_official_score_and_first_number_are_distinct(self):
        matrix, reference = self.fixture()
        rows = next(iter(matrix.values()))
        rows["passage_retrieval_en:0"]["pred"] = "Paragraph 1. Or Paragraph 2?"
        summary, contrasts, _ = analyze_details(matrix, reference,
            {"passage_retrieval_en": lambda p, a, **_: float(p == a)})
        self.assertEqual(summary[0]["g32_score_pct"], 50)
        self.assertEqual(summary[0]["first_number_correct"], 2)
        self.assertTrue(all(r["delta_pp"] == 0 for r in contrasts))

    def test_qa_uses_official_metric_without_number_heuristic(self):
        matrix, reference = self.fixture()
        matrix = {(k[0], "multifieldqa_en", *k[2:]): rows for k, rows in matrix.items()}
        reference = {(k[0], "multifieldqa_en", *k[2:]): rows for k, rows in reference.items()}
        summary, contrasts, trajectories = analyze_details(matrix, reference,
            {"multifieldqa_en": lambda p, a, **_: float(p == a)})
        self.assertIsNone(summary[0]["first_number_correct"])
        self.assertFalse(contrasts)
        self.assertIsNone(trajectories[0]["int4_g32_first_number_correct"])


if __name__ == "__main__":
    unittest.main()
