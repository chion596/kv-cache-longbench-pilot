"""Проверка диагностического удаления лишнего продолжения без изменения JSONL."""
import unittest
from src.diagnose_family_qa import first_line, diagnose


class QADiagnosisTests(unittest.TestCase):
    def test_first_nonempty_line(self):
        self.assertEqual(first_line("  Vice Admiral\n\n### reasoning"), "Vice Admiral")
        self.assertEqual(first_line("\n\n  result\nnext"), "result")
        self.assertEqual(first_line(" \n "), "")

    def test_changes_only_diagnostic_score(self):
        row = {
            "pred": "yes\n\nextra explanation",
            "answers": ["yes"], "all_classes": None,
            "max_new_tokens": 64, "output_tokens": 64,
        }
        matrix = {"dynamic": {"qa:1": row}}
        def fake_score(pred, truth, all_classes=None):
            return float(pred == truth)
        results, examples = diagnose(matrix, fake_score)
        self.assertEqual(results[0]["raw_f1_pct"], 0)
        self.assertEqual(results[0]["first_line_f1_pct"], 100)
        self.assertEqual(results[0]["improved"], 1)
        self.assertEqual(results[0]["hit_limit"], 1)
        self.assertEqual(examples[0]["task_id"], "qa:1")
        self.assertEqual(row["pred"], "yes\n\nextra explanation")


if __name__ == "__main__":
    unittest.main()
