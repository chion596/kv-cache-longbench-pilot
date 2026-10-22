"""Тесты контроля предельной длины ответа; GPU не требуется."""
import unittest

from src.compare_answer_lengths import compare_cache
from src.run_quality import effective_generation_limit


class ShortAnswerControlTests(unittest.TestCase):
    def _record(self, limit, prediction, cache="int4_g64"):
        return {
            "task_id": "passage_retrieval_en:10",
            "dataset": "passage_retrieval_en",
            "cache": cache,
            "input_ids_sha256": "same-input-hash",
            "middle_truncated": False,
            "model": "/models/Qwen3-1.7B",
            "max_new_tokens": limit,
            "pred": prediction,
            "answers": ["Paragraph 8"],
            "all_classes": None,
        }

    @staticmethod
    def fake_metric(prediction, truth, all_classes=None):
        import re
        target = re.findall(r"\d+", truth)[0]
        numbers = re.findall(r"\d+", prediction)
        return 0 if not numbers else sum(v == target for v in numbers) / len(numbers)

    def test_generation_limit_override(self):
        self.assertEqual(effective_generation_limit(32, None), 32)
        self.assertEqual(effective_generation_limit(32, 8), 8)
        with self.assertRaises(ValueError):
            effective_generation_limit(32, 0)

    def test_shortened_continuation_changes_metric_not_first_number(self):
        old = self._record(32, "8\nParagraph 19")
        new = self._record(8, "8")
        result = compare_cache({old["task_id"]: old}, {new["task_id"]: new},
                               self.fake_metric, expected_n=1)
        self.assertEqual(result["changed_first_number"], 0)
        self.assertAlmostEqual(result["old_score"], 50)
        self.assertAlmostEqual(result["new_score"], 100)

    def test_reject_mismatched_input_hash(self):
        old = self._record(32, "8")
        new = self._record(8, "8")
        new["input_ids_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "input_ids_sha256"):
            compare_cache({old["task_id"]: old}, {new["task_id"]: new},
                          self.fake_metric, expected_n=1)

    def test_reject_incorrect_output_limit(self):
        old = self._record(32, "8")
        new = self._record(32, "8")
        with self.assertRaisesRegex(ValueError, "пределы генерации"):
            compare_cache({old["task_id"]: old}, {new["task_id"]: new},
                          self.fake_metric, expected_n=1)


if __name__ == "__main__":
    unittest.main()
