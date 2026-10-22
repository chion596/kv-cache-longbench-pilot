"""Проверки анализа качества и GPU-памяти без CUDA и LongBench."""
import unittest

from src.common import DATASETS
from src.score_group_ablation import GROUP_CACHES
from src.score_memory_tradeoff import summarize_memory


class MemoryTradeoffTests(unittest.TestCase):
    def setUp(self):
        self.matrix = {}
        for dataset in DATASETS:
            for cache in GROUP_CACHES:
                self.matrix[(dataset, cache)] = {}
                for idx in range(2):
                    peak = 10.0 if cache == "dynamic" else 8.0
                    self.matrix[(dataset, cache)][f"{dataset}:{idx}"] = {
                        "task_id": f"{dataset}:{idx}",
                        "cache": cache,
                        "gpu": "Tesla V100-SXM2-32GB",
                        "peak_gpu_gib": peak,
                        "answers": ["верно"],
                        "pred": "верно",
                        "all_classes": None,
                    }
        self.metrics = {
            dataset: (lambda pred, truth, **kwargs: float(pred == truth))
            for dataset in DATASETS
        }

    def test_mean_saving_uses_identical_tasks(self):
        table = summarize_memory(self.matrix, self.metrics)
        self.assertEqual(len(table), 15)
        quantized = next(
            row for row in table
            if row["dataset"] == "passage_retrieval_en"
            and row["cache"] == "int4_g64"
        )
        self.assertEqual(quantized["mean_paired_peak_saving_pct"], 20.0)
        self.assertEqual(quantized["quality_pct"], 100.0)
        self.assertEqual(quantized["median_peak_allocated_gib"], 8.0)

    def test_rejects_different_gpu(self):
        key = ("multifieldqa_zh", "int4_g32")
        self.matrix[key]["multifieldqa_zh:1"]["gpu"] = "Другой GPU"
        with self.assertRaisesRegex(ValueError, "Разные GPU"):
            summarize_memory(self.matrix, self.metrics)

    def test_rejects_nonpositive_peak(self):
        key = ("multifieldqa_en", "int4_g128")
        self.matrix[key]["multifieldqa_en:0"]["peak_gpu_gib"] = 0
        with self.assertRaisesRegex(ValueError, "Недопустимое"):
            summarize_memory(self.matrix, self.metrics)


if __name__ == "__main__":
    unittest.main()
