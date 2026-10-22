"""Тесты парного анализа SmolLM3: одинаковые задания и отдельная метрика."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.common import read_jsonl, write_jsonl
from src.score_family_retrieval import analyze_results, load_results


def fake_score(pred, answer, all_classes=None):
    return float(pred == answer)


class SmolLM3RetrievalTests(unittest.TestCase):
    caches = ("dynamic", "int4_g16", "int4_g64")

    def write_fixture(self, folder):
        for cache in self.caches:
            records = []
            for i in range(2):
                records.append({
                    "task_id": f"passage_retrieval_en:{i}",
                    "source_index": i,
                    "dataset": "passage_retrieval_en",
                    "cache": cache,
                    "model": "/scratch/models/SmolLM3-3B",
                    "gpu": "Tesla V100-SXM2-32GB",
                    "dtype": "float16",
                    "residual_length": 128,
                    "prompt_mode": "raw_longbench_official",
                    "input_ids_sha256": f"smollm-{i}",
                    "middle_truncated": False,
                    "max_new_tokens": 32,
                    "output_tokens": 2,
                    "answers": ["correct"],
                    "all_classes": None,
                    "pred": "wrong" if cache == "int4_g64" and i == 1
                            else "correct",
                })
            write_jsonl(folder / f"passage_retrieval_en_{cache}.jsonl", records)

    def test_paired_scores_and_bootstrap(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write_fixture(folder)
            matrix = load_results(folder, self.caches, expected_n=2)
            summaries, contrasts = analyze_results(matrix, fake_score)
            self.assertEqual(len(summaries), 3)
            self.assertEqual(len(contrasts), 3)
            delta = next(
                row for row in contrasts
                if row["reference"] == "int4_g16"
                and row["candidate"] == "int4_g64"
            )
            self.assertEqual(delta["delta_pp"], -50.0)
            self.assertEqual(
                next(x for x in summaries if x["cache"] == "int4_g64")["zero_score_count"],
                1,
            )

    def test_mismatched_hash_is_rejected(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write_fixture(folder)
            path = folder / "passage_retrieval_en_int4_g64.jsonl"
            rows = list(read_jsonl(path))
            rows[0]["input_ids_sha256"] = "wrong"
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, "input_ids_sha256"):
                load_results(folder, self.caches, expected_n=2)

    def test_incomplete_matrix_is_rejected(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write_fixture(folder)
            (folder / "passage_retrieval_en_int4_g16.jsonl").unlink()
            with self.assertRaises(FileNotFoundError):
                load_results(folder, self.caches, expected_n=2)

    def test_mismatched_tasks_are_rejected(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write_fixture(folder)
            path = folder / "passage_retrieval_en_int4_g64.jsonl"
            rows = list(read_jsonl(path))
            rows[0]["source_index"] = 999
            rows[0]["task_id"] = "passage_retrieval_en:999"
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, "отличаются задания"):
                load_results(folder, self.caches, expected_n=2)


if __name__ == "__main__":
    unittest.main()
