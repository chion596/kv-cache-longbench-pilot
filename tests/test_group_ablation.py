"""Тесты расширенного сравнения q_group_size без GPU."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from src.common import DATASETS, write_jsonl
from src.score_group_ablation import (
    GROUP_CACHES,
    analyze_matrix,
    read_matrix,
)
from src.run_quality import GENERATION_CACHES, make_cache


class GroupSizeAblationTests(unittest.TestCase):
    def make_fixture(self, folder: Path):
        for dataset in DATASETS:
            for cache in GROUP_CACHES:
                rows = []
                for index in range(2):
                    task_id = f"{dataset}:{index}"
                    rows.append({
                        "dataset": dataset,
                        "cache": cache,
                        "task_id": task_id,
                        "source_index": index,
                        "input_ids_sha256": f"{dataset}-same-{index}",
                        "max_new_tokens": 64,
                        "middle_truncated": False,
                        "model": "Qwen3-1.7B",
                        "dtype": "float16",
                        "residual_length": 128,
                        "prompt_mode": "raw_longbench_official",
                        "answers": ["yes"],
                        "all_classes": None,
                        "input_tokens": 500,
                        "output_tokens": 64,
                        "pred": "no" if cache == "int4_g128" else "yes",
                    })
                write_jsonl(folder / f"{dataset}_{cache}.jsonl", rows)

    def test_complete_matrix_and_pairwise_comparisons(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_fixture(root)
            matrix = read_matrix(root, expected_n=2)
            metrics = {dataset: lambda prediction, reference, **_:
                       float(prediction == reference) for dataset in DATASETS}
            datasets, comparisons, examples = analyze_matrix(matrix, metrics)
            self.assertEqual(len(datasets), 15)
            self.assertEqual(len(comparisons), 27)
            self.assertEqual(len(examples), 30)
            direct = [row for row in comparisons
                      if row["reference"] == "int4_g16"
                      and row["candidate"] == "int4_g128"]
            self.assertEqual(len(direct), 3)
            self.assertTrue(all(row["delta_pp"] == -100.0 for row in direct))

    def test_refuses_incomplete_matrix(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_fixture(root)
            (root / "passage_retrieval_en_int4_g32.jsonl").unlink()
            with self.assertRaises(FileNotFoundError):
                read_matrix(root, expected_n=2)

    def test_refuses_different_input_tokens(self):
        from src.common import read_jsonl
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_fixture(root)
            path = root / "multifieldqa_en_int4_g128.jsonl"
            rows = list(read_jsonl(path))
            rows[0]["input_ids_sha256"] = "other"
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, "input_ids_sha256"):
                read_matrix(root, expected_n=2)

    def test_refuses_different_model_or_output_limit(self):
        from src.common import read_jsonl
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_fixture(root)
            path = root / "multifieldqa_zh_int4_g32.jsonl"
            rows = list(read_jsonl(path))
            rows[1]["max_new_tokens"] = 8
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, "max_new_tokens"):
                read_matrix(root, expected_n=2)

    def test_cache_setup_recognizes_all_four_group_sizes(self):
        class DummyCache:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_transformers = SimpleNamespace(
            DynamicCache=DummyCache,
            QuantizedCache=DummyCache,
        )
        model = SimpleNamespace(config=object())
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            for size in (16, 32, 64, 128):
                name = f"int4_g{size}"
                self.assertIn(name, GENERATION_CACHES)
                obj = make_cache(model, name, 128)
                self.assertEqual(obj.kwargs["q_group_size"], size)
                self.assertEqual(obj.kwargs["nbits"], 4)
                self.assertEqual(obj.kwargs["backend"], "hqq")


if __name__ == "__main__":
    unittest.main()
