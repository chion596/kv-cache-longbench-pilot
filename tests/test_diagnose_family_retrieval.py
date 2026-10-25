"""Проверка поиска первого номера и диагностики формата ответа."""
import unittest
from src.diagnose_family_retrieval import first_int, expected_numbers, analyze_model


class FamilyRetrievalDiagnosisTests(unittest.TestCase):
    def test_first_int_and_reference(self):
        self.assertEqual(first_int("Paragraph 24, next 5"), 24)
        self.assertEqual(first_int(""), None)
        self.assertEqual(expected_numbers({"answers": ["Paragraph 3"]}), {3})

    def test_different_continuation_does_not_change_first_number(self):
        def row(pred, peak):
            return {
                "pred": pred, "answers": ["Paragraph 3"], "all_classes": [],
                "output_tokens": 2, "max_new_tokens": 32,
                "peak_gpu_gib": peak,
            }
        matrix = {
            "dynamic": {"a": row("3.", 4)},
            "int4_g64": {"a": row("3 and 24", 3)},
        }
        def score(pred, truth, all_classes=None):
            return float(pred == "3.")
        summaries, changed = analyze_model(matrix, score)
        self.assertEqual(summaries[1]["first_number_correct"], 1)
        self.assertEqual(summaries[1]["multiple_numbers"], 1)
        self.assertEqual(summaries[1]["mean_paired_peak_memory_saving_pct"], 25)
        self.assertEqual(changed[0]["delta_pp"], -100)


if __name__ == "__main__":
    unittest.main()
