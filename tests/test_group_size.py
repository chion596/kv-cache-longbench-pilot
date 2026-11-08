import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.common import digest_ids, read_jsonl, write_jsonl
from src.download_data import sha256_file
from src.group_size import (cells, load_config, prepared_tasks, read_matrix, reference_fingerprints, references_for,
                            selected_task_ids, validate_environment)
from src.protocol import digest_json, load_config as protocol_config, model_signature, result_path
from src.score_group_size import analyze


class GroupSizeTests(unittest.TestCase):
    def fixture(self, root, phase="full"):
        config = load_config()
        config.update(n=2, smoke_n=1)
        reference = {}
        for cell in cells(config):
            for cache in ("dynamic", "int4_g16", "int4_g64"):
                old_cell = {**cell, "cache": cache}
                p = {**old_cell, "manifest_sha256": cell["model_name"], "model_artifacts": {"files": "test"},
                    "longbench_files_sha256": {"metrics.py": "test"}, "versions": {"torch": "test"},
                    "gpu": "test-gpu", "cuda": "test-cuda", "dtype": "float16", "attention": "sdpa",
                    "residual_length": 128, "source_hashes": {}}
                reference[tuple(old_cell.values())] = {f"{cell['dataset']}:{i}": {**old_cell,
                    "task_id": f"{cell['dataset']}:{i}", "source_index": i,
                    "input_ids_sha256": str(i), "context_sha256": str(i), "context_split": 10,
                    "input_tokens": 30, "answers": ["yes"], "all_classes": None, "max_new_tokens": 32,
                    "middle_truncated": False, "output_ids": [11, 12], "output_tokens": 2,
                    "stop_reason": "eos", "pred": "no" if cache == "int4_g64" else "yes",
                    "provenance": p, "run_fingerprint": digest_json(p)} for i in range(2)}
        for cell in cells(config):
            refs = references_for(reference, cell)
            p = {**next(iter(refs["dynamic"].values()))["provenance"], **cell,
                "series": config["series"], "phase": phase, "config_sha256": digest_json(config),
                "reference_fingerprints": reference_fingerprints(refs), "reference_results_sha256": digest_json(refs),
                "control_source_hashes": {"run_group_size.py": "test"}}
            rows = [{**refs["dynamic"][tid], **cell, "provenance": p,
                "run_fingerprint": digest_json(p), "quantize_calls_by_layer": [2, 2]}
                for tid in selected_task_ids(reference, cell, config, phase)]
            write_jsonl(result_path(root, cell), rows)
        return config, reference

    def mutate(self, root, config, change):
        path = result_path(root, cells(config)[0])
        rows = list(read_jsonl(path))
        change(rows[0])
        rows[0]["run_fingerprint"] = digest_json(rows[0]["provenance"])
        write_jsonl(path, rows)

    def test_plan_is_eight_cells_and_640_new_generations(self):
        config = load_config()
        self.assertEqual(len(cells(config)), 8)
        self.assertEqual(len(cells(config)) * config["n"], 640)
        self.assertEqual(len(cells(config)) * config["smoke_n"], 24)

    def test_complete_matrix_uses_paired_baselines_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            config, reference = self.fixture(temp)
            matrix = read_matrix(temp, config, "full", reference)
            original = copy.deepcopy((matrix, reference))
            metrics = {d: lambda pred, truth, **_: float(pred == truth) for d in config["datasets"]}
            summaries, contrasts, examples = analyze(matrix, reference, config, metrics)
            self.assertEqual((matrix, reference), original)
            self.assertEqual((len(summaries), len(contrasts), len(examples)), (8, 32, 16))
            improvements = [r for r in contrasts if r["comparison"] == "int4_g32-int4_g64"]
            self.assertEqual(len(improvements), 8)
            self.assertTrue(all(r["delta_pp"] == 100 and r["cluster_ci95_low_pp"] == 100 for r in improvements))
            self.assertTrue(all(r["delta_pp"] == 0 for r in contrasts if "split_loss" in r["comparison"]))

    def test_input_and_environment_changes_are_rejected(self):
        for change in (lambda row: row.update(answers=["changed"]),
                       lambda row: row["provenance"].update(gpu="another-gpu")):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                config, reference = self.fixture(temp)
                self.mutate(temp, config, change)
                with self.assertRaisesRegex(ValueError, "Изменились|Условия отличаются"):
                    read_matrix(temp, config, "full", reference)

    def test_modified_baseline_outputs_are_rejected_even_if_run_fingerprint_is_same(self):
        with tempfile.TemporaryDirectory() as temp:
            config, reference = self.fixture(temp)
            rows = next(iter(reference.values()))
            next(iter(rows.values()))["pred"] = "modified after smoke"
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                read_matrix(temp, config, "full", reference)

    def test_first_token_invariant_is_checked_on_full_results_too(self):
        with tempfile.TemporaryDirectory() as temp:
            config, reference = self.fixture(temp)
            self.mutate(temp, config, lambda row: row.update(output_ids=[13, 12]))
            with self.assertRaisesRegex(ValueError, "Первый токен"):
                read_matrix(temp, config, "full", reference)

    def test_smoke_subset_does_not_depend_on_baseline_file_order(self):
        with tempfile.TemporaryDirectory() as temp:
            config, reference = self.fixture(temp, "smoke")
            for k, rows in reference.items():
                reference[k] = dict(reversed(list(rows.items())))
            # Digest sorts object keys, so changing dict insertion order preserves
            # the sealed baseline and the selected shared source indices.
            matrix = read_matrix(temp, config, "smoke", reference)
            self.assertTrue(all(set(rows) == {f"{k[1]}:0"} for k, rows in matrix.items()))

    def test_missing_cell_and_truncated_cell_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            config, reference = self.fixture(temp)
            path = result_path(temp, cells(config)[0])
            rows = list(read_jsonl(path))
            write_jsonl(path, rows[:1])
            with self.assertRaisesRegex(ValueError, "Неполный результат"):
                read_matrix(temp, config, "full", reference)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                read_matrix(temp, config, "full", reference)

    def test_package_and_hardware_changes_fail_before_inference(self):
        p = {"versions": {"torch": "same"}, "source_hashes": {}, "gpu": "V100", "cuda": "12.6"}
        validate_environment(p, versions={"torch": "same"}, gpu="V100", cuda="12.6")
        for arguments in ({"versions": {"torch": "new"}},
                          {"versions": {"torch": "same"}, "gpu": "other"},
                          {"versions": {"torch": "same"}, "cuda": "new"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_environment(p, **arguments)

    def test_prepared_inputs_and_weight_files_are_bound_to_original_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, reference = self.fixture(root / "results")
            cell = cells(config)[0]
            refs = references_for(reference, cell)
            weights = root / "weights"
            weights.mkdir()
            (weights / "config.json").write_text('{}\n')
            (weights / "model.safetensors").write_bytes(b'fixture')
            folder = root / 'data/prepared/protocol_v1' / cell["model_name"]
            folder.mkdir(parents=True)
            filename = folder / (cell["dataset"] + '.jsonl')
            tasks = []
            for tid, row in refs["dynamic"].items():
                row.update(input_ids_sha256=digest_ids([1, 2, 3]), input_tokens=3, context_split=2)
                tasks.append({**{k: row[k] for k in ("task_id", "input_ids_sha256", "input_tokens", "context_split",
                    "context_sha256", "answers", "all_classes", "max_new_tokens", "middle_truncated")},
                    "input_ids": [1, 2, 3]})
            write_jsonl(filename, tasks)
            manifest = {"config_sha256": digest_json(protocol_config()), "model_name": cell["model_name"],
                "datasets": {cell["dataset"]: {"file_sha256": sha256_file(filename),
                    "task_hashes": {t["task_id"]: t["input_ids_sha256"] for t in tasks}}}}
            manifest_path = folder / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            for row in refs["dynamic"].values():
                row["provenance"].update(manifest_sha256=sha256_file(manifest_path), model_artifacts=model_signature(weights))
            with patch('src.group_size.Path', side_effect=lambda path: root / path):
                self.assertEqual(prepared_tasks(cell, weights, refs), tasks)
                (weights / "model.safetensors").write_bytes(b'changed fixture')
                with self.assertRaisesRegex(ValueError, "Модель/токенизатор"):
                    prepared_tasks(cell, weights, refs)


if __name__ == "__main__":
    unittest.main()
