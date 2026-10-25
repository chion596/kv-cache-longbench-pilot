"""QA LongBench: проверка полных пар и одинаковых входов."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.common import read_jsonl, write_jsonl
from src.score_family_qa import read_qa_matrix
from src.score_family_retrieval import analyze_results


class FamilyQATests(unittest.TestCase):
    dataset = "multifieldqa_en"
    caches = ("dynamic", "int4_g16", "int4_g64")

    def fixture(self, folder):
        for cache in self.caches:
            records = []
            for i in range(3):
                records.append({
                    "dataset": self.dataset,
                    "task_id": f"{self.dataset}:{i}",
                    "source_index": i,
                    "cache": cache,
                    "answers": ["yes"],
                    "all_classes": [],
                    "pred": "no" if cache == "int4_g64" and i == 2 else "yes",
                    "input_ids_sha256": f"same-{i}",
                    "max_new_tokens": 64,
                    "middle_truncated": False,
                    "model": "SmolLM3-3B",
                    "dtype": "float16",
                    "residual_length": 128,
                    "prompt_mode": "raw_longbench_official",
                    "gpu": "V100",
                    "output_tokens": 2,
                })
            write_jsonl(folder / f"{self.dataset}_{cache}.jsonl", records)

    def test_f1_pipeline_and_paired_contrasts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            matrix = read_qa_matrix(root, self.dataset, self.caches, 3)
            def fake_metric(pred, truth, all_classes=None):
                return float(pred == truth)
            summaries, contrasts = analyze_results(matrix, fake_metric)
            self.assertEqual(len(summaries), 3)
            self.assertAlmostEqual(summaries[2]["quality_pct"], 66.667)
            self.assertEqual(len(contrasts), 3)

    def test_mismatch_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            path = root / f"{self.dataset}_int4_g64.jsonl"
            rows = list(read_jsonl(path))
            rows[2]["input_ids_sha256"] = "not-the-same"
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, "input_ids_sha256"):
                read_qa_matrix(root, self.dataset, self.caches, 3)

    def test_incomplete_matrix_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / f"{self.dataset}_int4_g16.jsonl").unlink()
            with self.assertRaises(FileNotFoundError):
                read_qa_matrix(root, self.dataset, self.caches, 3)


if __name__ == "__main__":
    unittest.main()
