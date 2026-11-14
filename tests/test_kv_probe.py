import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from src.common import digest_ids, read_jsonl, write_jsonl
from src.download_data import sha256_file
from src.kv_probe import cell, load_config, prepare_task, read_matrix
from src.protocol import digest_json, result_path
from src.score_kv_probe import analyze


class Tokenizer:
    def encode(self, text, **_):
        return list(map(ord, text))

    def decode(self, ids, **_):
        return ''.join(map(chr, ids))


def task(i=0):
    ids = [1, 2, 3, 4, 5]
    return {'task_id': f'passage_retrieval_en:{i}', 'source_index': i,
        'input_ids': ids, 'input_ids_sha256': digest_ids(ids), 'context_split': 3,
        'input_tokens': len(ids), 'context_sha256': str(i), 'answers': ['Paragraph 12']}


class ProbeTests(unittest.TestCase):
    def test_gold_target_and_prefix_are_common_and_original_task_is_unchanged(self):
        source = task()
        original = copy.deepcopy(source)
        result = prepare_task(source, Tokenizer(), load_config())
        self.assertEqual(Tokenizer().decode(result['target_ids']), '12')
        self.assertEqual(Tokenizer().decode(result['forced_prefix_ids']), 'The abstract is from Paragraph ')
        self.assertEqual(source, original)

    def test_ambiguous_gold_boundary_merge_and_residual_overflow_are_rejected(self):
        config = load_config()
        t = task()
        t['answers'] = ['Paragraph 1 or 2']
        with self.assertRaisesRegex(ValueError, 'один эталонный'):
            prepare_task(t, Tokenizer(), config)

        class MergingTokenizer(Tokenizer):
            def encode(self, text, **_):
                return [len(text)]

        with self.assertRaisesRegex(ValueError, 'Tokenization'):
            prepare_task(task(), MergingTokenizer(), config)
        with self.assertRaisesRegex(ValueError, 'residual'):
            prepare_task(task(), Tokenizer(), {**config, 'residual_length': 32})

    def fixture(self, root, phase='smoke'):
        config = {**load_config(), 'n': 2, 'smoke_n': 1}
        prepared, results = Path(root)/'prepared', Path(root)/'results'
        for model in config['models']:
            tasks = [prepare_task(task(i), Tokenizer(), config) for i in range(config['n'])]
            folder = prepared/model
            write_jsonl(folder/'tasks.jsonl', tasks)
            manifest = {'model_name': model, 'model_artifacts': {'path': model}, 'versions': {'test': '1'},
                'config_sha256': digest_json(config), 'tasks_sha256': sha256_file(folder/'tasks.jsonl'),
                'task_ids': [t['task_id'] for t in tasks], 'hidden_layers': 2,
                'task_hashes': {t['task_id']: digest_json(t) for t in tasks}}
            path = folder/'manifest.json'
            path.write_text(json.dumps(manifest))
            p = {'series': config['series'], 'model_name': model, 'phase': phase,
                'config_sha256': digest_json(config), 'manifest_sha256': sha256_file(path),
                'versions': manifest['versions'], 'model_artifacts': manifest['model_artifacts'],
                'source_hashes': {'test': '1'}, 'gpu': 'test', 'cuda': 'test'}
            for policy in config['policies']:
                policy_provenance = {**p, 'policy': policy}
                selected = [int(policy.startswith(('k_', 'kv_'))), int(policy.startswith(('v_', 'kv_')))]
                nll = 1 if policy.startswith('k_') else 2 if policy.startswith('v_') else 5 if policy.startswith('kv_') else 0
                rows = [{**{k: t[k] for k in ('task_id', 'context_sha256', 'input_ids_sha256', 'target_ids', 'gold_number')},
                    'policy': policy, 'task_sha256': digest_json(t), 'provenance': policy_provenance,
                    'run_fingerprint': digest_json(policy_provenance),
                    'gold_nll_per_token': [nll, nll], 'gold_nll_mean': nll, 'gold_nll_sum': 2*nll,
                    'first_number_token_margin': -nll, 'first_number_token_top1': t['target_ids'][0],
                    'query_first_token': 5, 'hqq_calls_by_layer': [selected]*2,
                    'probe_sec': 1.0,
                    'stock_logits_match': True, 'stock_query_token_match': True}
                    for t in (tasks[:1] if phase == 'smoke' else tasks)]
                write_jsonl(result_path(results/phase, cell(model, policy, config)), rows)
        return config, prepared, results

    def test_matrix_and_nonadditive_effect_use_paired_rows(self):
        with tempfile.TemporaryDirectory() as root:
            config, prepared, results = self.fixture(root)
            matrix = read_matrix(config, 'smoke', prepared, results)
            summary, contrasts = analyze(matrix, config)
            self.assertEqual(len(summary), 14)
            self.assertEqual(len(contrasts), 40)
            nonadd = [r for r in contrasts if r['metric'] == 'gold_nll_mean' and r['comparison'].startswith('nonadditivity')]
            self.assertTrue(all(r['delta'] == 2 and r['cluster_ci95_low'] == 2 for r in nonadd))
            self.assertEqual(config['n']*14, 28)
        full = load_config()
        self.assertEqual(len(full['models'])*len(full['policies'])*full['n'], 280)

    def test_missing_rows_changed_targets_wrong_sides_and_unverified_stock_rejected(self):
        for mutation, message in (
            (lambda r: r.update(target_ids=[99, 99]), 'Изменена задача'),
            (lambda r: r.update(hqq_calls_by_layer=[[1, 0], [1, 0]]), 'стороны'),
            (lambda r: r.update(stock_logits_match=False), 'smoke'),
            (lambda r: r.update(gold_nll_mean=99), 'NLL'),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as root:
                config, prepared, results = self.fixture(root)
                path = result_path(results/'smoke', cell(config['models'][0], 'kv_g32', config))
                rows = list(read_jsonl(path))
                mutation(rows[0])
                write_jsonl(path, rows)
                with self.assertRaisesRegex(ValueError, message):
                    read_matrix(config, 'smoke', prepared, results)
        with tempfile.TemporaryDirectory() as root:
            config, prepared, results = self.fixture(root)
            path = result_path(results/'smoke', cell(config['models'][0], 'dynamic', config))
            write_jsonl(path, [])
            with self.assertRaisesRegex(ValueError, 'Неполная'):
                read_matrix(config, 'smoke', prepared, results)


@unittest.skipUnless(all(importlib.util.find_spec(p) for p in ('torch', 'transformers', 'hqq')),
                     'Численная проверка требует torch, transformers и HQQ')
class NumericalProbeTests(unittest.TestCase):
    def model(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=32)
        config._attn_implementation = 'sdpa'
        return Qwen3ForCausalLM(config).eval()

    def test_actual_hqq_both_matches_stock_and_exact_matches_dynamic(self):
        import torch
        from src.kv_probe import evaluate, parity, selective_cache
        from src.run_quality import make_cache
        model = self.model()
        t = {'input_ids': list(range(1, 22)), 'context_split': 16,
             'forced_prefix_ids': [23, 24], 'target_ids': [25, 26]}
        for policy in ('dynamic', 'kv_g32', 'kv_g64'):
            cache, calls = selective_cache(model, policy, 512)
            stats, logits = evaluate(model, t, cache, calls)
            if policy == 'dynamic':
                reference, count = selective_cache(model, 'exact_g64', 512)
            else:
                reference, count = make_cache(model, 'int4_'+policy.split('_')[1], 512), None
            other, expected = evaluate(model, t, reference, count)
            self.assertTrue(parity(logits, expected))
            self.assertEqual(stats['query_first_token'], other['query_first_token'])
            expected_nll = torch.nn.functional.cross_entropy(logits, torch.tensor(t['target_ids'])).item()
            self.assertAlmostEqual(stats['gold_nll_mean'], expected_nll, places=6)
        # Gold alignment checked against an ordinary complete forward, without cache intervention.
        with torch.inference_mode():
            seq = t['input_ids']+t['forced_prefix_ids']+t['target_ids'][:-1]
            output = model(input_ids=torch.tensor([seq])).logits[0, -len(t['target_ids']):].float()
        cache, calls = selective_cache(model, 'dynamic', 512)
        _, logits = evaluate(model, t, cache, calls)
        torch.testing.assert_close(logits, output, rtol=1e-5, atol=1e-6)

    def test_only_selected_side_changes_and_refresh_is_rejected(self):
        import torch
        from src.kv_probe import selective_cache
        model = self.model()
        keys, values = torch.randn(1, 1, 16, 32), torch.randn(1, 1, 16, 32)
        for policy in ('k_g32', 'v_g32'):
            cache, calls = selective_cache(model, policy, 2)
            layer = cache.layers[0]
            layer.update(keys, values)
            k, v = layer._dequantize(layer._quantized_keys), layer._dequantize(layer._quantized_values)
            if policy.startswith('k_'):
                self.assertFalse(torch.equal(k, keys))
                self.assertTrue(torch.equal(v, values))
                self.assertEqual(calls[0], [1, 0])
            else:
                self.assertTrue(torch.equal(k, keys))
                self.assertFalse(torch.equal(v, values))
                self.assertEqual(calls[0], [0, 1])
            layer.update(keys[:, :, :1], values[:, :, :1])
            with self.assertRaisesRegex(ValueError, 'refresh'):
                layer.update(keys[:, :, :1], values[:, :, :1])


if __name__ == '__main__':
    unittest.main()
