import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from src.common import read_jsonl, write_jsonl
from src.group_size import reference_fingerprints
from src.protocol import digest_json, result_path
from src.residual import cells, load_config, read_matrix, references_for, selected_ids, validate_budget
from src.score_residual import analyze


class ResidualTests(unittest.TestCase):
    def fixture(self, root, phase="full"):
        config = load_config()
        config.update(n=2, smoke_n=1)
        reference = {}
        for model in config["models"]:
            for cache in ("dynamic", "int4_g16", "int4_g32", "int4_g64"):
                cell = {"model_name": model, "dataset": "passage_retrieval_en",
                        "protocol": "context_then_question", "cache": cache}
                p = {**cell, "manifest_sha256": model, "model_artifacts": {"files": model},
                    "longbench_files_sha256": {"metrics.py": "test"}, "versions": {"torch": "test"},
                    "gpu": "V100", "cuda": "test", "dtype": "float16", "attention": "sdpa",
                    "residual_length": 128, "source_hashes": {}}
                reference[(model, cell["dataset"], cell["protocol"], cache)] = {
                    f"passage_retrieval_en:{i}": {**cell, "task_id": f"passage_retrieval_en:{i}",
                    "source_index": i, "input_ids_sha256": str(i), "context_sha256": str(i),
                    "input_tokens": 200, "context_split": 55, "max_new_tokens": 32,
                    "answers": ["Paragraph 1"], "all_classes": None, "middle_truncated": False,
                    "output_ids": [1, 2], "output_tokens": 2, "stop_reason": "eos",
                    "pred": "Paragraph 2" if cache in ("int4_g32", "int4_g64") else "Paragraph 1",
                    "quantize_calls_by_layer": [4, 4], "provenance": p, "run_fingerprint": digest_json(p)}
                    for i in range(2)}
        for cell in cells(config):
            refs = references_for(reference, cell)
            p = {**next(iter(refs["dynamic"].values()))["provenance"], **cell, "series": config["series"],
                "phase": phase, "config_sha256": digest_json(config), "residual_length": 512,
                "reference_results_sha256": digest_json(refs), "reference_fingerprints": reference_fingerprints(refs),
                "control_source_hashes": {"residual.py": "same"}}
            rows = [{**refs[cell["cache"]][tid], "pred": "Paragraph 1", "residual_length": 512,
                "prefill_split": 55, "query_tokens": 145, "quantize_calls_by_layer": [2, 2],
                "provenance": p, "run_fingerprint": digest_json(p)} for tid in selected_ids(refs, config, phase)]
            write_jsonl(result_path(root, cell), rows)
        return config, reference

    def mutate(self, root, config, change):
        path = result_path(root, cells(config)[0])
        rows = list(read_jsonl(path))
        change(rows[0])
        rows[0]["run_fingerprint"] = digest_json(rows[0]["provenance"])
        write_jsonl(path, rows)

    def test_plan_has_320_full_and_12_smoke_answers(self):
        config = load_config()
        self.assertEqual(len(cells(config)), 4)
        self.assertEqual(len(cells(config)) * config["n"], 320)
        self.assertEqual(len(cells(config)) * config["smoke_n"], 12)

    def test_paired_analysis_preserves_original_data_and_uses_same_cache_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            config, reference = self.fixture(temp)
            matrix = read_matrix(temp, config, "full", reference)
            original = copy.deepcopy((matrix, reference))
            summary, contrasts, examples = analyze(matrix, reference, config,
                lambda pred, truth, **_: float(pred == truth))
            self.assertEqual((matrix, reference), original)
            self.assertEqual((len(summary), len(contrasts), len(examples)), (4, 20, 8))
            primary = [r for r in contrasts if r["comparison"] == "residual512-residual128" and r["metric"] == "retrieval_score"]
            self.assertEqual(len(primary), 4)
            self.assertTrue(all(r["delta_pp"] == 100 and r["cluster_ci95_low_pp"] == 100 for r in primary))
            self.assertTrue(all(r["delta_pp"] == 0 for r in contrasts if r["comparison"].startswith('residual_effect:')))

    def test_refresh_or_changed_first_token_invalidates_control(self):
        for change, message in ((lambda r: r.update(quantize_calls_by_layer=[4, 4]), "исходная квантизация"),
                                (lambda r: r.update(output_ids=[9, 2]), "Первый токен")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp:
                config, reference = self.fixture(temp)
                self.mutate(temp, config, change)
                with self.assertRaisesRegex(ValueError, message):
                    read_matrix(temp, config, "full", reference)

    def test_only_residual_is_allowed_to_change(self):
        for change, message in ((lambda r: r.update(answers=['Paragraph 9']), "Изменились входы"),
                                (lambda r: r["provenance"].update(gpu='other'), "Несопоставимые условия"),
                                (lambda r: r.update(residual_length=128), "Неверный residual_length"),
                                (lambda r: r.update(prefill_split=54), "граница prefill")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp:
                config, reference = self.fixture(temp)
                self.mutate(temp, config, change)
                with self.assertRaisesRegex(ValueError, message):
                    read_matrix(temp, config, "full", reference)

    def test_baseline_prediction_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            config, reference = self.fixture(temp)
            next(iter(next(iter(reference.values())).values()))["pred"] = "changed"
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                read_matrix(temp, config, "full", reference)

    def test_missing_and_incomplete_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            config, reference = self.fixture(temp)
            path = result_path(temp, cells(config)[0])
            write_jsonl(path, list(read_jsonl(path))[:1])
            with self.assertRaisesRegex(ValueError, "Неполная ячейка"):
                read_matrix(temp, config, "full", reference)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                read_matrix(temp, config, "full", reference)

    def test_query_and_answer_must_fit_below_residual(self):
        config = load_config()
        task = {"task_id": 'test', "input_tokens": 493, "context_split": 100, "max_new_tokens": 32}
        validate_budget(task, config)
        task["input_tokens"] = 580
        with self.assertRaisesRegex(ValueError, "query\\+answer"):
            validate_budget(task, config)

    def test_smoke_subset_uses_shared_source_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            config, reference = self.fixture(temp, "smoke")
            reference = {k: dict(reversed(list(v.items()))) for k, v in reference.items()}
            matrix = read_matrix(temp, config, "smoke", reference)
            self.assertTrue(all(set(v) == {'passage_retrieval_en:0'} for v in matrix.values()))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "Численная проверка cache требует torch/transformers")
class NumericalResidualTests(unittest.TestCase):
    def test_larger_residual_suppresses_refresh_without_changing_first_token(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from transformers.cache_utils import Cache, QuantizedLayer
        from src.run_quality import greedy_generate

        class ZeroLayer(QuantizedLayer):
            def _quantize(self, tensor, axis):
                return torch.zeros_like(tensor)
            def _dequantize(self, tensor):
                return tensor

        torch.set_num_threads(1)
        torch.manual_seed(2026)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=512, bos_token_id=None, eos_token_id=None, pad_token_id=0)
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).eval()
        tokenizer = SimpleNamespace(eos_token_id=None, decode=lambda ids, **_: str(ids))

        def cache_factory(_model, _cache, residual):
            return Cache(layers=[ZeroLayer(residual_length=residual) for _ in range(2)])

        with patch('src.run_quality.make_cache', side_effect=cache_factory):
            prompt = [i % 63 for i in range(170)]
            _, old = greedy_generate(model, tokenizer, prompt, 'int4_g32', 128, 6,
                                     prefill_split=5, track_quantization=True)
            _, new = greedy_generate(model, tokenizer, prompt, 'int4_g32', 512, 6,
                                     prefill_split=5, track_quantization=True)
        self.assertEqual(old['quantize_calls_by_layer'], [4, 4])
        self.assertEqual(new['quantize_calls_by_layer'], [2, 2])
        self.assertEqual(old['output_ids'][0], new['output_ids'][0])


if __name__ == "__main__":
    unittest.main()
