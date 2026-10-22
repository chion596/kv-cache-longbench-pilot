"""Тесты диагностической метрики первого номера без GPU."""
import unittest

from src.analyze_retrieval_prefix import analyze, first_number, reference_number


class FirstNumberTests(unittest.TestCase):
    def test_first_number_is_exact_sequence(self):
        self.assertEqual(first_number("88\nParagraph 8"), "88")
        self.assertEqual(first_number("8\nParagraph 8"), "8")
        self.assertIsNone(first_number("Номер неизвестен"))

    def test_reference_number(self):
        self.assertEqual(reference_number(["Paragraph 24"]), "24")
        with self.assertRaisesRegex(ValueError, "Не найден номер"):
            reference_number(["Ответ неизвестен"])

    def test_first_number_differs_from_full_official(self):
        rows = {}
        for cache, pred in (
            ("dynamic", "8\nParagraph 8"),
            ("int4_g16", "8\nParagraph 8"),
            ("int4_g64", "8\nParagraph 19"),
        ):
            rows[cache] = {
                "retrieval:1": {
                    "task_id": "retrieval:1", "answers": ["Paragraph 8"],
                    "pred": pred, "output_tokens": 4, "max_new_tokens": 32
                }
            }
        def metric(pred, answer, all_classes=None):
            gold = answer.split()[-1]
            import re
            numbers = re.findall(r"\d+", pred)
            return sum(n == gold for n in numbers) / len(numbers)
        summaries = analyze(rows, metric)
        quant = [r for r in summaries if r["cache"] == "int4_g64"][0]
        self.assertEqual(quant["первое_число_верно"], 1)
        self.assertEqual(quant["официальный_балл_проц"], 50.0)


if __name__ == "__main__":
    unittest.main()
