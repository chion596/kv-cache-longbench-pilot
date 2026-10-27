"""Тесты парного различия F1 от постпроцессинга."""
import unittest

from src.score_qa_continuations import analyze_continuations


def exact_metric(pred, truth, all_classes=None):
    return float(pred == truth)


def record(pred):
    return {"pred": pred, "answers": ["yes"], "all_classes": None}


class QAContinuationPairedTests(unittest.TestCase):
    def test_raw_vs_first_line_and_difference_of_deltas(self):
        matrix = {
            "dynamic": {"a": record("yes"), "b": record("yes")},
            "int4_g64": {"a": record("yes\nwrong continuation"), "b": record("yes")},
        }
        summary, contrasts = analyze_continuations(matrix, exact_metric)
        first = next(x for x in summary if x["cache"] == "int4_g64")
        self.assertEqual(first["raw_f1_pct"], 50.0)
        self.assertEqual(first["first_line_f1_pct"], 100.0)
        self.assertEqual(first["first_minus_raw_pp"], 50.0)
        self.assertEqual(first["first_line_better"], 1)
        results = {x["comparison"]: x for x in contrasts}
        self.assertEqual(results["raw_delta"]["delta_pp"], -50.0)
        self.assertEqual(results["first_line_delta"]["delta_pp"], 0)
        self.assertEqual(results["difference_of_deltas"]["delta_pp"], 50.0)

    def test_rejects_unpaired_task_ids(self):
        matrix = {
            "dynamic": {"a": record("yes")},
            "int4_g16": {"b": record("yes")},
        }
        with self.assertRaisesRegex(ValueError, "разных заданиях"):
            analyze_continuations(matrix, exact_metric)


if __name__ == "__main__":
    unittest.main()
