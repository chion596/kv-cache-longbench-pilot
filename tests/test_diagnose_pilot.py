"""Локальные тесты парной диагностики: для них не нужен GPU или LongBench."""
import tempfile
import unittest
from pathlib import Path

from src.common import write_jsonl
from src.diagnose_pilot import build_diagnostics, load_paired_data


class DiagnosePilotTests(unittest.TestCase):
    def _make_rows(self, root):
        dataset = "passage_retrieval_en"
        preds = {
            "dynamic": ("yes", "yes"),
            "int4_g16": ("yes", "yes"),
            "int4_g64": ("no", "yes"),
        }
        for cache, answers in preds.items():
            rows = []
            for i, pred in enumerate(answers):
                rows.append({
                    "dataset": dataset, "cache": cache,
                    "task_id": f"{dataset}:{i}",
                    "input_ids_sha256": f"hash-{i}",
                    "max_new_tokens": 16,
                    "middle_truncated": False,
                    "answers": ["yes"], "all_classes": None,
                    "pred": pred, "output_tokens": 1,
                })
            write_jsonl(root / f"{dataset}_{cache}.jsonl", rows)

    def test_direct_paired_advantage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_rows(root)
            by_cache = load_paired_data(root, "passage_retrieval_en")
            metric = lambda pred, truth, all_classes=None: float(pred == truth)
            cases = build_diagnostics(by_cache, metric)
            self.assertEqual(len(cases), 2)
            self.assertEqual(sum(c["g16_minus_g64"] for c in cases), 1.0)
            self.assertEqual(sum(c["dynamic_minus_g64"] for c in cases), 1.0)

    def test_reject_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_rows(root)
            path = root / "passage_retrieval_en_int4_g64.jsonl"
            from src.common import read_jsonl
            rows = list(read_jsonl(path))
            rows[0]["input_ids_sha256"] = "wrong"
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, "input_ids_sha256"):
                load_paired_data(root, "passage_retrieval_en")


if __name__ == "__main__":
    unittest.main()
