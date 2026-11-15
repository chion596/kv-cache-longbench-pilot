import copy
from contextlib import chdir
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.common import digest_ids, read_jsonl, write_jsonl
from src.download_data import sha256_file
from src.expansion import TASK_FIELDS
from src.mixed_precision import cell, cells, load_config, read_matrix
from src.mixed_cache import policy_bits
from src.protocol import digest_json, result_path
from src.score_mixed_precision import analyze, untruncated


class MixedMatrixTests(unittest.TestCase):
    def test_prepare_and_input_preflight_from_sealed_parent_are_idempotent(self):
        from src.expansion import load_config as original_config
        from src.prepare_mixed_precision import main as prepare
        from src.protocol import model_signature
        from src.run_mixed_precision import current_inputs
        config, parent = load_config(), original_config()
        model = config['models'][0]
        with tempfile.TemporaryDirectory() as temp, chdir(temp):
            weights = Path(temp)/'model'; weights.mkdir()
            (weights/'config.json').write_text('{"num_hidden_layers":2}')
            root = Path('data/prepared/expansion_v1'); folder = root/model; folder.mkdir(parents=True)
            selection = {'series': parent['series'], 'config_sha256': digest_json(parent),
                'archive_sha256': parent['archive_sha256'], 'datasets': {d: {'source_count': 80,
                    'source_indices': list(range(80)), 'context_hashes': {str(i): f'{d}:{i}' for i in range(80)}}
                    for d in config['datasets']}}
            (root/'selection.json').write_text(json.dumps(selection))
            datasets, originals = {}, {}
            for dataset in config['datasets']:
                ids = [1, 2, 3, 4, 5]
                tasks = [{'task_id': f'{dataset}:{i}', 'source_index': i, 'dataset': dataset,
                    'input_ids': ids, 'input_ids_sha256': digest_ids(ids), 'context_sha256': f'{dataset}:{i}',
                    'context_split': 3, 'input_tokens': 5, 'max_new_tokens': 32, 'middle_truncated': False,
                    'answers': ['Paragraph 1' if dataset == 'passage_retrieval_en' else 'yes'], 'all_classes': None}
                    for i in range(80)]
                originals[dataset] = tasks
                write_jsonl(folder/f'{dataset}.jsonl', tasks)
                datasets[dataset] = {'file_sha256': sha256_file(folder/f'{dataset}.jsonl'),
                    'task_metadata': {t['task_id']: {k: t[k] for k in TASK_FIELDS} for t in tasks}}
            manifest = {'model_name': model, 'config_sha256': digest_json(parent), 'hidden_layers': 2,
                'selection_sha256': sha256_file(root/'selection.json'), 'model_artifacts': model_signature(weights),
                'versions': {k: 'test' for k in ('torch', 'transformers', 'hqq')},
                'prepare_source_hashes': {}, 'longbench_files_sha256': {'metrics.py': 'test'}, 'datasets': datasets}
            path = folder/'manifest.json'; path.write_text(json.dumps(manifest))
            for dataset, tasks in originals.items():
                for policy in ('dynamic', 'int4_g64'):
                    c = cell(model, dataset, policy, config)
                    p = {**c, 'series': parent['series'], 'phase': 'full', 'config_sha256': digest_json(parent),
                        'manifest_sha256': sha256_file(path), 'selection_sha256': manifest['selection_sha256'],
                        **{k: manifest[k] for k in ('model_artifacts', 'versions', 'prepare_source_hashes', 'longbench_files_sha256')},
                        **{k: parent[k] for k in ('dtype', 'attention', 'residual_length', 'prompt_mode')}}
                    rows = [{**{k: v for k, v in t.items() if k != 'input_ids'}, **c, 'provenance': p,
                        'run_fingerprint': digest_json(p), 'pred': t['answers'][0], 'output_ids': [7], 'output_tokens': 1,
                        'stop_reason': 'eos', 'prefill_split': 3, 'query_tokens': 2,
                        'quantize_calls_by_layer': [0, 0] if policy == 'dynamic' else [2, 2]} for t in tasks]
                    write_jsonl(result_path('results/expansion_v1/full', c), rows)
            args = ['prepare_mixed_precision', '--name', model, '--model', str(weights)]
            with patch('sys.argv', args), patch('importlib.metadata.version', return_value='test'):
                prepare()
                target = Path('data/prepared/mixed_precision_v1')/model/'manifest.json'
                first = target.read_bytes()
                prepare()
                self.assertEqual(target.read_bytes(), first)
                for dataset in config['datasets']:
                    tasks, _, _, refs = current_inputs(model, dataset, weights, config)
                    self.assertEqual(len(tasks), 80)
                    self.assertEqual(set(refs), {'dynamic', 'k4_v4'})
                with (folder/f"{config['datasets'][0]}.jsonl").open('a') as handle:
                    handle.write('\n')
                with self.assertRaisesRegex(ValueError, 'Изменён источник'):
                    current_inputs(model, config['datasets'][0], weights, config)
                self.assertEqual(target.read_bytes(), first)

    def fixture(self, root, phase='smoke'):
        config = {**load_config(), 'n': 2, 'smoke_n': 1}
        prepared, results = Path(root)/'prepared', Path(root)/'results'
        for model in config['models']:
            datasets, all_tasks = {}, {}
            for dataset in config['datasets']:
                ids = [1, 2, 3, 4, 5]
                tasks = [{'task_id': f'{dataset}:{i}', 'source_index': i, 'dataset': dataset,
                    'input_ids': ids, 'input_ids_sha256': digest_ids(ids), 'context_sha256': f'{dataset}:{i}',
                    'context_split': 3, 'input_tokens': 5, 'max_new_tokens': 32, 'middle_truncated': i == 1,
                    'answers': ['Paragraph 1' if dataset == 'passage_retrieval_en' else 'yes'], 'all_classes': None}
                    for i in range(2)]
                all_tasks[dataset] = tasks
                folder = prepared/model
                write_jsonl(folder/f'{dataset}.jsonl', tasks)
                datasets[dataset] = {'task_ids': [t['task_id'] for t in tasks],
                    'file_sha256': sha256_file(folder/f'{dataset}.jsonl'),
                    'task_hashes': {t['task_id']: digest_json(t) for t in tasks},
                    'task_metadata': {t['task_id']: {k: t[k] for k in TASK_FIELDS} for t in tasks}}
            manifest = {'model_name': model, 'config_sha256': digest_json(config), 'hidden_layers': 2,
                        'model_artifacts': {'path': model}, 'versions': {'test': '1'}, 'datasets': datasets}
            path = prepared/model/'manifest.json'
            path.write_text(json.dumps(manifest))
            for dataset in config['datasets']:
                for policy in config['policies']:
                    c = cell(model, dataset, policy, config)
                    p = {**c, 'series': config['series'], 'phase': phase, 'config_sha256': digest_json(config),
                        'manifest_sha256': sha256_file(path), 'versions': manifest['versions'],
                        'model_artifacts': manifest['model_artifacts'], 'source_hashes': {'test': '1'}, 'gpu': 'test', 'cuda': 'test'}
                    nbytes = {'dynamic': 1000, 'k4_v4': 300, 'k8_v4': 500, 'k4_v8': 500, 'k16_v4': 750}[policy]
                    bits = list(policy_bits(policy))
                    calls = [0, 0] if policy == 'dynamic' else [int(b < 16) for b in bits]
                    rows = []
                    for task in all_tasks[dataset][:1] if phase == 'smoke' else all_tasks[dataset]:
                        rows.append({**{k: v for k, v in task.items() if k != 'input_ids'}, **c,
                            'provenance': p, 'run_fingerprint': digest_json(p), 'task_sha256': digest_json(task),
                            'pred': task['answers'][0] if policy != 'k4_v8' else 'wrong',
                            'output_ids': [7, 8], 'output_tokens': 2, 'stop_reason': 'eos',
                            'stored_bits_by_layer': [bits]*2, 'hqq_calls_by_layer': [calls]*2,
                            'cache_device_bytes': {'cuda:0': nbytes}, 'cache_gpu_bytes_after_query': nbytes,
                            'cache_host_bytes_after_query': 0, 'cache_storage_count': 4, 'cache_tokens_after_query': 5,
                            'prefill_sec': 1., 'decode_sec': 2.,
                            'exact_logits_match': True, 'stock8_logits_match': True, 'stock4_logits_match': True})
                    write_jsonl(result_path(results/phase, c), rows)
        return config, prepared, results

    def test_full_matrix_paired_quality_and_common_subset(self):
        with tempfile.TemporaryDirectory() as root:
            config, prepared, results = self.fixture(root, 'full')
            matrix = read_matrix(config, 'full', prepared, results)
            metrics = {d: lambda p, a, **_: float(p == a) for d in config['datasets']}
            summaries, contrasts, budgets = analyze(matrix, config, metrics)
            self.assertEqual(len(summaries), 20)
            self.assertEqual(len(contrasts), 28)
            direct = [r for r in contrasts if r['comparison'] == 'k8_v4-k4_v8']
            self.assertTrue(all(r['delta_pp'] == 100 for r in direct))
            self.assertTrue(all(r['max_absolute_bytes_difference'] == 0 for r in budgets))
            counts, sensitivity = untruncated(matrix, config, metrics)
            self.assertTrue(all(r['common_untruncated_n'] == 1 for r in counts))
            self.assertTrue(all(r['n'] == 1 for r in sensitivity))
        config = load_config()
        self.assertEqual(len(cells(config))*config['n'], 1600)
        self.assertEqual(len(cells(config))*config['smoke_n'], 60)

    def test_smoke_rejects_wrong_bits_memory_budget_snapshot_and_failed_parity(self):
        for policy, mutation, message in [
            ('k8_v4', lambda r: r.update(stored_bits_by_layer=[[4, 4]]*2), 'битность'),
            ('k8_v4', lambda r: r.update(hqq_calls_by_layer=[[2, 1]]*2), 'refresh'),
            ('dynamic', lambda r: r.update(stock8_logits_match=False), 'сверка'),
            ('k4_v4', lambda r: r.update(cache_tokens_after_query=6), 'snapshot'),
            ('k8_v4', lambda r: r.update(cache_device_bytes={'cuda:0': 501}, cache_gpu_bytes_after_query=501), 'разную измеренную'),
        ]:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as root:
                config, prepared, results = self.fixture(root)
                c = cell(config['models'][0], config['datasets'][0], policy, config)
                path = result_path(results/'smoke', c)
                rows = list(read_jsonl(path)); mutation(rows[0]); write_jsonl(path, rows)
                with self.assertRaisesRegex(ValueError, message):
                    read_matrix(config, 'smoke', prepared, results)

    def test_incomplete_and_mixed_environment_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            config, prepared, results = self.fixture(root)
            path = result_path(results/'smoke', cell(config['models'][0], config['datasets'][0], 'dynamic', config))
            rows = list(read_jsonl(path))
            write_jsonl(path, [])
            with self.assertRaisesRegex(ValueError, 'Неполная'):
                read_matrix(config, 'smoke', prepared, results)
            rows[0]['provenance']['gpu'] = 'other'
            rows[0]['run_fingerprint'] = digest_json(rows[0]['provenance'])
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, 'среда'):
                read_matrix(config, 'smoke', prepared, results)


@unittest.skipUnless(all(importlib.util.find_spec(p) for p in ('torch', 'transformers', 'hqq')),
                     'Нужны torch, transformers и настоящий HQQ')
class NumericalMixedTests(unittest.TestCase):
    def model(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.set_num_threads(1); torch.manual_seed(42)
        c = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=32, eos_token_id=None)
        c._attn_implementation = 'sdpa'
        return Qwen3ForCausalLM(c).eval()

    def test_true_hqq_parity_bit_metadata_native_decoder_and_memory_budget(self):
        from src.mixed_cache import native_generate
        from src.run_mixed_precision import smoke_checks
        from src.run_quality import greedy_generate
        model = self.model()
        tok = SimpleNamespace(eos_token_id=None, decode=lambda x, **_: str(x))
        task = {'input_ids': list(range(1, 22)), 'context_split': 16, 'max_new_tokens': 6}
        before = copy.deepcopy(task)
        memories = {}
        for policy in load_config()['policies']:
            _, stats, trace = native_generate(model, tok, task, policy, trace=True)
            memories[policy] = stats['cache_host_bytes_after_query']
            bits = list(policy_bits(policy))
            bits = [32 if b == 16 else b for b in bits]  # exact CPU states are FP32; production is FP16.
            self.assertEqual(stats['stored_bits_by_layer'], [bits]*2)
            self.assertEqual(stats['cache_tokens_after_query'], 21)
            if policy in ('dynamic', 'k4_v4'):
                checks = smoke_checks(model, tok, task, policy, stats, trace, 512)
                self.assertTrue(all(checks.values()))
                _, old = greedy_generate(model, tok, task['input_ids'], 'dynamic' if policy == 'dynamic' else 'int4_g64',
                    512, 6, prefill_split=16)
                self.assertEqual(stats['output_ids'], old['output_ids'])
        self.assertEqual(memories['k8_v4'], memories['k4_v8'])
        self.assertLess(memories['k4_v4'], memories['k8_v4'])
        self.assertLess(memories['k16_v4'], memories['dynamic'])
        self.assertEqual(task, before)

    def test_storage_deduplication_includes_metadata_exact_state_and_empty_views(self):
        import torch
        from src.mixed_cache import ExactTensor, storage_bytes
        tensor = torch.zeros(100, dtype=torch.float32)
        other = torch.zeros(2, dtype=torch.float16)
        result = storage_bytes([tensor[:1], {'scale': tensor[2:], 'zero': other}, ExactTensor(tensor[:0])])
        self.assertEqual(result['cache_host_bytes_after_query'], 404)
        self.assertEqual(result['cache_storage_count'], 2)

    def test_eos_and_refresh_guards(self):
        import torch
        from src.mixed_cache import make_mixed_cache, native_generate
        model = self.model()
        tok = SimpleNamespace(eos_token_id=None, decode=lambda x, **_: str(x))
        task = {'input_ids': list(range(1, 22)), 'context_split': 16, 'max_new_tokens': 6}
        _, stats, _ = native_generate(model, tok, task, 'dynamic')
        tok.eos_token_id = stats['output_ids'][0]
        _, stopped, _ = native_generate(model, tok, task, 'dynamic')
        self.assertEqual(stopped['output_tokens'], 1)
        self.assertEqual(stopped['stop_reason'], 'eos')
        with self.assertRaisesRegex(ValueError, 'refresh'):
            native_generate(model, tok, task, 'k8_v4', residual=8)
        cache, _ = make_mixed_cache(model, 'k8_v4', residual=2)
        layer = cache.layers[0]
        k, v = torch.randn(1, 1, 16, 32), torch.randn(1, 1, 16, 32)
        layer.update(k, v); layer.update(k[:, :, :1], v[:, :, :1])
        with self.assertRaisesRegex(ValueError, 'refresh'):
            layer.update(k[:, :, :1], v[:, :, :1])


if __name__ == '__main__':
    unittest.main()
