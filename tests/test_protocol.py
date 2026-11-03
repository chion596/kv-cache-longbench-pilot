import importlib.util
from pathlib import Path
import tempfile
import unittest

from src.common import read_jsonl, write_jsonl
from src.prepare_protocol import prepare_chat_task
from src.protocol import cells, digest_json, load_config, result_path
from src.run_protocol import output_lock, validate_resume
from src.score_protocol import analyze, cluster_ci, read_matrix


class ChatTokenizer:
    chat_template = "user content; enable_thinking; assistant"

    def apply_chat_template(self, messages, **kwargs):
        if kwargs != {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}:
            raise AssertionError(kwargs)
        return "<user>" + messages[0]["content"] + "</user><assistant><think></think>"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class ProtocolTests(unittest.TestCase):
    def source(self):
        return {"context": "A" * 100, "input": "Where?", "answers": ["here"]}

    def test_prepared_split_reconstructs_identical_full_chat_ids(self):
        tokenizer = ChatTokenizer()
        source = self.source()
        row = prepare_chat_task(source, 4, "multifieldqa_en", "{context}\nQ: {input}", 64, tokenizer, 1000)
        expected = tokenizer.encode(tokenizer.apply_chat_template(
            [{"role": "user", "content": source["context"] + "\nQ: Where?"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False))
        split = row["context_split"]
        self.assertEqual(row["input_ids"][:split] + row["input_ids"][split:], expected)
        self.assertEqual(chr(row["input_ids"][split]), "\n")
        self.assertEqual(row["input_ids"][split - 1], ord("A"))

    def test_middle_truncation_preserves_question_and_chat_suffix(self):
        row = prepare_chat_task(self.source(), 4, "multifieldqa_en", "{context}\nQ: {input}", 64, ChatTokenizer(), 90)
        suffix = "".join(chr(i) for i in row["input_ids"][row["context_split"]:])
        self.assertEqual(suffix, "\nQ: Where?</user><assistant><think></think>")
        self.assertTrue(row["middle_truncated"])

    def test_unsupported_thinking_switch_fails(self):
        tokenizer = ChatTokenizer()
        tokenizer.chat_template = "No switch"
        with self.assertRaisesRegex(ValueError, "enable_thinking"):
            prepare_chat_task(self.source(), 0, "multifieldqa_en", "{context} {input}", 64, tokenizer, 1000)

    def fixture(self, root, phase="smoke"):
        config = load_config()
        config.update(models=["qwen3-1.7b", "smollm3-3b"], datasets=["multifieldqa_en"], n=2, smoke_n=2)
        for cell in cells(config):
            provenance = {**cell, "phase": phase, "config_sha256": digest_json(config),
                "longbench_files_sha256": {"metrics.py": "same-metrics"},
                "manifest_sha256": "same-manifest", "versions": {"torch": "test"}, "gpu": "test", "cuda": "test",
                "dtype": "float16", "attention": "sdpa", "residual_length": 128, "source_hashes": {"runner": "same"}}
            rows = [{**cell, "task_id": "multifieldqa_en:" + str(i), "source_index": i,
                "input_ids_sha256": str(i), "context_sha256": "same-context", "context_split": 10,
                "max_new_tokens": 64, "middle_truncated": False, "answers": ["yes"], "all_classes": None,
                "pred": "yes", "output_ids": [11, 12], "output_tokens": 2, "stop_reason": "eos",
                "provenance": provenance, "run_fingerprint": digest_json(provenance),
                "reference_matches": True, "reference_output_ids": [11, 12]} for i in range(2)]
            write_jsonl(result_path(root, cell), rows)
        return config

    def test_complete_smoke_validates_and_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.fixture(temp)
            matrix = read_matrix(temp, config, "smoke")
            summaries, contrasts, examples = analyze(matrix, config,
                {"multifieldqa_en": lambda pred, truth, **_: float(pred == truth)})
            self.assertEqual(len(summaries), 12)
            self.assertEqual(len(examples), 24)
            self.assertTrue(all(r["documents"] == 1 for r in contrasts))
            self.assertTrue(all(r["delta_pp"] == 0 for r in contrasts))

    def test_smoke_rejects_dynamic_split_difference_even_with_same_text(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.fixture(temp)
            cell = cells(config)[3]  # first model, split, Dynamic
            path = result_path(temp, cell)
            rows = list(read_jsonl(path))
            rows[0]["output_ids"] = [11, 13]
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, "Dynamic изменился"):
                read_matrix(temp, config, "smoke")

    def test_missing_cache_cell_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.fixture(temp)
            result_path(temp, cells(config)[-1]).unlink()
            with self.assertRaises(FileNotFoundError):
                read_matrix(temp, config, "smoke")

    def test_smoke_rejects_first_token_difference_in_hqq(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.fixture(temp)
            path = result_path(temp, cells(config)[1])
            rows = list(read_jsonl(path))
            rows[0]["output_ids"] = [13, 12]
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, "Первый токен"):
                read_matrix(temp, config, "smoke")

    def test_harm_difference_is_computed_directly_between_models(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.fixture(temp, phase="full")
            matrix = read_matrix(temp, config, "full")
            for protocol in config["protocols"]:
                for row in matrix[("qwen3-1.7b", "multifieldqa_en", protocol, "int4_g64")].values():
                    row["pred"] = "wrong"
            _, contrasts, _ = analyze(matrix, config,
                {"multifieldqa_en": lambda pred, truth, **_: float(pred == truth)})
            direct = [r for r in contrasts if r["comparison"] == "g64_loss:smollm3-3b-qwen3-1.7b"]
            self.assertEqual(len(direct), 2)
            self.assertTrue(all(r["delta_pp"] == 100 for r in direct))

    def test_resume_rejects_changed_environment_and_duplicates(self):
        tasks = [{"task_id": "x", "input_ids_sha256": "a", "context_sha256": "c", "context_split": 2, "max_new_tokens": 3}]
        row = {**tasks[0], "run_fingerprint": "one", "output_ids": [1], "output_tokens": 1}
        self.assertEqual(validate_resume([row], tasks, "one"), {"x"})
        with self.assertRaisesRegex(ValueError, "изменились"):
            validate_resume([row], tasks, "two")
        with self.assertRaisesRegex(ValueError, "Повторная"):
            validate_resume([row, row], tasks, "one")

    def test_cluster_bootstrap_keeps_questions_together(self):
        differences = {"a": 1., "b": 1., "c": -1., "d": -1.}
        paired = {"a": "doc1", "b": "doc1", "c": "doc2", "d": "doc2"}
        self.assertEqual(cluster_ci(differences, paired, iters=1000), (-1., 1.))
        self.assertEqual(cluster_ci(differences, {k: "one" for k in differences}, iters=100), (0., 0.))

    def test_matrix_declares_36_cells_and_2880_generations(self):
        config = load_config()
        self.assertEqual(len(cells(config)), 36)
        self.assertEqual(len(cells(config)) * config["n"], 2880)

    def test_same_output_cannot_be_written_by_two_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "answers.jsonl"
            with output_lock(path):
                with self.assertRaisesRegex(RuntimeError, "другой процесс"):
                    with output_lock(path):
                        pass
            with output_lock(path):
                pass  # После выхода блокировка снята.


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "Численные проверки требуют torch и transformers; стандартные тесты работают без них")
class NumericalDecodeTests(unittest.TestCase):
    def models(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM, SmolLM3Config, SmolLM3ForCausalLM
        torch.set_num_threads(1)
        for config_class, model_class in ((Qwen3Config, Qwen3ForCausalLM), (SmolLM3Config, SmolLM3ForCausalLM)):
            torch.manual_seed(2026)
            config = config_class(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                max_position_embeddings=256, bos_token_id=None, eos_token_id=None, pad_token_id=0)
            config._attn_implementation = "sdpa"
            yield model_class(config).eval()

    def test_manual_and_split_decode_match_generate_on_two_architectures(self):
        from types import SimpleNamespace
        from src.run_quality import greedy_generate
        from src.run_protocol import reference_generate
        tokenizer = SimpleNamespace(eos_token_id=None, pad_token_id=0,
                                    decode=lambda ids, **_: ",".join(map(str, ids)))
        for model in self.models():
            with self.subTest(model=type(model).__name__):
                prompt = [1, 5, 9, 7, 3, 12, 2]
                _, full = greedy_generate(model, tokenizer, prompt, "dynamic", 128, 6)
                _, split = greedy_generate(model, tokenizer, prompt, "dynamic", 128, 6, prefill_split=4)
                self.assertEqual(full["output_ids"], split["output_ids"])
                self.assertEqual(full["output_ids"], reference_generate(model, tokenizer, prompt, 6))
                self.assertEqual(full["stop_reason"], "max_new_tokens")

    def test_quantized_prefill_returns_original_states_but_decode_reads_quantized(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        from transformers.cache_utils import Cache, DynamicCache, QuantizedLayer
        from src.run_quality import greedy_generate

        class ZeroLayer(QuantizedLayer):
            def _quantize(self, tensor, axis):
                return torch.zeros_like(tensor)
            def _dequantize(self, tensor):
                return tensor

        for model in self.models():
            with self.subTest(model=type(model).__name__), torch.inference_mode():
                normal = DynamicCache(config=model.config)
                approximate = Cache(layers=[ZeroLayer() for _ in range(model.config.num_hidden_layers)])
                prompt = torch.tensor([[1, 5, 9, 7, 3]])
                a = model(input_ids=prompt, past_key_values=normal, use_cache=True).logits
                b = model(input_ids=prompt, past_key_values=approximate, use_cache=True).logits
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                token = torch.tensor([[4]])
                a = model(input_ids=token, past_key_values=normal, use_cache=True).logits
                b = model(input_ids=token, past_key_values=approximate, use_cache=True).logits
                self.assertFalse(torch.allclose(a, b))
                tokenizer = SimpleNamespace(eos_token_id=None,
                    decode=lambda ids, **_: ",".join(map(str, ids)))
                def zero_cache(*_):
                    return Cache(layers=[ZeroLayer(residual_length=128)
                        for _ in range(model.config.num_hidden_layers)])
                with patch("src.run_quality.make_cache", side_effect=zero_cache):
                    prompt = [i % 63 for i in range(150)]
                    _, full = greedy_generate(model, tokenizer, prompt, "int4_g64", 128, 3,
                                              track_quantization=True)
                    _, split = greedy_generate(model, tokenizer, prompt, "int4_g64", 128, 3,
                        prefill_split=5, track_quantization=True)
                self.assertEqual(full["quantize_calls_by_layer"], [2, 2])
                self.assertEqual(split["quantize_calls_by_layer"], [4, 4])


if __name__ == "__main__":
    unittest.main()
