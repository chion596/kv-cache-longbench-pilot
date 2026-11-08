import copy
import unittest

from src.analyze_protocol_details import analyze_details, first_difference
from src.protocol import load_config


class ProtocolDetailsTests(unittest.TestCase):
    def fixture(self):
        config = load_config()
        config.update(models=["qwen3-4b", "smollm3-3b"], datasets=["passage_retrieval_en"], n=2)
        matrix = {}
        for model in config["models"]:
            for protocol in config["protocols"]:
                for cache in config["caches"]:
                    rows = {}
                    for index in range(2):
                        tid = f"passage_retrieval_en:{index}"
                        rows[tid] = {"pred": "Paragraph 1", "answers": ["Paragraph 1"],
                            "all_classes": None, "context_sha256": str(index),
                            "output_ids": [1, 2], "input_tokens": 200, "context_split": 60,
                            "stop_reason": "eos", "quantize_calls_by_layer": [0, 0] if cache == "dynamic" else [2, 2]}
                    matrix[(model, "passage_retrieval_en", protocol, cache)] = rows
        return matrix, config

    def test_first_difference_includes_prefix_length_and_exact_match(self):
        self.assertIsNone(first_difference([1, 2], [1, 2]))
        self.assertEqual(first_difference([1, 2], [1, 3]), 1)
        self.assertEqual(first_difference([1], [1, 2]), 1)

    def test_drift_and_refresh_counts_are_separate_from_quality(self):
        matrix, config = self.fixture()
        row = matrix[("qwen3-4b", "passage_retrieval_en", "context_then_question", "dynamic")]["passage_retrieval_en:0"]
        row["output_ids"] = [1, 3]  # Text and official score unchanged.
        quant = matrix[("qwen3-4b", "passage_retrieval_en", "context_then_question", "int4_g64")]["passage_retrieval_en:0"]
        quant.update(quantize_calls_by_layer=[4, 4], output_ids=[4, 5], pred="Paragraph 2")
        metric = {"passage_retrieval_en": lambda pred, truth, **_: float(pred == truth)}
        dynamic, caches, _, _ = analyze_details(matrix, config, metric)
        drift = next(r for r in dynamic if r["model"] == "qwen3-4b")
        self.assertEqual(drift["different_output_ids"], 1)
        self.assertEqual(drift["different_text"], 0)
        self.assertEqual(drift["different_score"], 0)
        q = next(r for r in caches if r["model"] == "qwen3-4b" and r["cache"] == "int4_g64" and r["protocol"] == "context_then_question")
        self.assertEqual(q["repeated_quantization_count"], 1)
        self.assertEqual(q["first_token_changed_vs_dynamic"], 1)
        self.assertEqual(q["first_number_correct"], 1)
        self.assertEqual(q["baseline_correct_first_number_lost"], 1)

    def test_direct_model_comparison_and_input_records_are_preserved(self):
        matrix, config = self.fixture()
        for protocol in config["protocols"]:
            for row in matrix[("qwen3-4b", "passage_retrieval_en", protocol, "int4_g64")].values():
                row["pred"] = "Paragraph 2"
        original = copy.deepcopy(matrix)
        _, _, comparisons, _ = analyze_details(matrix, config,
            {"passage_retrieval_en": lambda pred, truth, **_: float(pred == truth)})
        self.assertEqual(matrix, original)
        self.assertEqual(len(comparisons), 2)
        self.assertTrue(all(r["delta_pp"] == 100 and r["cluster_ci95_low_pp"] == 100 for r in comparisons))

    def test_missing_quantization_metadata_is_not_reported_as_initial_only(self):
        matrix, config = self.fixture()
        row = matrix[("qwen3-4b", "passage_retrieval_en", "single_prefill", "int4_g64")]["passage_retrieval_en:0"]
        del row["quantize_calls_by_layer"]
        _, caches, _, _ = analyze_details(matrix, config,
            {"passage_retrieval_en": lambda pred, truth, **_: float(pred == truth)})
        q = next(r for r in caches if r["model"] == "qwen3-4b" and r["cache"] == "int4_g64" and r["protocol"] == "single_prefill")
        self.assertEqual(q["quantization_metadata_missing_count"], 1)
        self.assertEqual(q["initial_quantization_only_count"], 1)


if __name__ == "__main__":
    unittest.main()
