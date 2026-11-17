import copy
from contextlib import chdir
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from src.audit_expansion import Catalog, audit
from src.check_zh_confirmation import tokenization_summary
from src.common import read_jsonl, write_jsonl
from src.download_data import sha256_file
from src.mixed_cache import policy_bits
from src.prepare_zh_confirmation import main as prepare, prepare_tasks, stress_order, validate_preview
from src.protocol import digest_json, model_signature, result_path
from src.score_zh_confirmation import analyze, load_metric
from src.zh_confirmation import (DATASET, PREPARED, ROOT, cell, cells, current_inputs,
    load_config, load_prepared, read_matrix)


class CharTokenizer:
    chat_template = 'enable_thinking'

    def apply_chat_template(self, messages, **kwargs):
        return '<user>'+messages[0]['content']+'</user><assistant>'

    def encode(self, text, **kwargs):
        return list(map(ord, text))


class ZhConfirmationTests(unittest.TestCase):
    def fixture(self):
        config = load_config()
        raw = [{'context': f'文档{i}。'*8, 'input': '这是摘要。', 'answers': ['段落1']} for i in range(200)]
        # Тот же seed выбирает membership независимо от длины input.
        from src.audit_expansion import eligible_preview
        records = [{'source_index': i, 'context_sha256': hashlib.sha256(r['context'].encode()).hexdigest()}
                   for i, r in enumerate(raw)]
        indices = eligible_preview(records, set(), set(), 120, 2043)['preview_source_indices']
        raw[indices[-1]]['input'] = '长'*815
        archive = Path('data/raw/longbench-data.zip'); archive.parent.mkdir(parents=True)
        with zipfile.ZipFile(archive, 'w') as z:
            for dataset, rows in [(DATASET, raw), ('passage_retrieval_en', [{'context': 'old English context'}])]:
                z.writestr(f'data/{dataset}.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        config['archive_sha256'] = sha256_file(archive)
        published = {'datasets': [], 'revision': 'test', 'archive_sha256_from_upstream_lfs': config['archive_sha256']}
        fake_root = Path('fake-source'); (fake_root/'docs').mkdir(parents=True)
        published_path = fake_root/'docs/data_independence_audit_2026-11-03.json'
        published_path.write_text(json.dumps(published))
        (fake_root/'src').mkdir()
        for name in ('check_zh_confirmation.py', 'audit_expansion.py'):
            shutil.copyfile(ROOT/'src'/name, fake_root/'src'/name)
        code = Path('external/LongBench/LongBench'); (code/'config').mkdir(parents=True)
        template = '{context}\n摘要：{input}\n答案：'
        (code/'config/dataset2prompt.json').write_text(json.dumps({DATASET: template}))
        (code/'config/dataset2maxlen.json').write_text(json.dumps({DATASET: 32}))
        (code/'metrics.py').write_text(
            'import re\ndef retrieval_zh_score(p, a, **kwargs):\n'
            '    target = re.search(r"段落(\\d+)", a)[1]\n'
            '    found = re.findall(r"\\d+", p)\n'
            '    return sum(x == target for x in found)/len(found) if found else 0.0\n')
        official = {f: sha256_file(code/f) for f in ('config/dataset2prompt.json', 'config/dataset2maxlen.json', 'metrics.py')}
        models, summaries = {}, []
        for name in config['models']:
            weights = Path('models')/name; weights.mkdir(parents=True)
            (weights/'config.json').write_text('{"num_hidden_layers":2}')
            models[name] = str(weights)
            parent = Path('data/prepared/mixed_precision_v1')/name/'manifest.json'
            parent.parent.mkdir(parents=True)
            parent.write_text(json.dumps({'versions': {k: 'test' for k in ('torch', 'transformers', 'hqq')},
                'model_artifacts': model_signature(weights), 'longbench_files_sha256': official}))
            summaries.append({'model': name, 'parent_manifest_sha256': sha256_file(parent),
                'chat_template_sha256': digest_json(CharTokenizer.chat_template),
                **tokenization_summary(raw, indices, template, CharTokenizer())})
        write_jsonl('results/old/english.jsonl', [{'dataset': 'passage_retrieval_en', 'source_index': 0}])
        with zipfile.ZipFile(archive) as z:
            history = audit(Catalog(z), published, ['data/prepared', 'results'], [DATASET], 120, 2043)
        preview = {'n': 120, 'seed': 2043, 'dataset': DATASET, 'archive_sha256': config['archive_sha256'],
            'source_sha256': sha256_file(ROOT/'src/check_zh_confirmation.py'),
            'audit_source_sha256': sha256_file(ROOT/'src/audit_expansion.py'),
            'published_audit_sha256': sha256_file(published_path), 'official_files_sha256': official,
            'local_history_ready': True, 'all_query_budgets_fit_residual512': False,
            'history': history, 'tokenizer_precheck': summaries,
            'preview_context_hashes': {str(i): records[i]['context_sha256'] for i in indices}}
        Path('results/zh_confirmation_preflight_2026-11-17.json').write_text(json.dumps(preview))
        return config, models, fake_root, preview, raw, template

    def run_prepare(self, config, models, fake_root):
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: CharTokenizer()),
            AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(max_position_embeddings=32768, num_hidden_layers=2)))
        args = ['prepare', '--qwen17', models['qwen3-1.7b'], '--qwen4', models['qwen3-4b']]
        with (patch('src.prepare_zh_confirmation.load_config', return_value=config),
              patch('src.prepare_zh_confirmation.ROOT', fake_root),
              patch('sys.argv', args), patch.dict('sys.modules', {'transformers': fake_transformers}),
              patch('importlib.metadata.version', return_value='test')):
            prepare()

    def write_smoke(self, config):
        for model in config['models']:
            tasks, manifest, path = load_prepared(model, config)
            for policy in config['policies']:
                c = cell(model, DATASET, policy, config)
                provenance = {**c, 'series': config['series'], 'phase': 'smoke', 'config_sha256': digest_json(config),
                    'manifest_sha256': sha256_file(path), 'versions': manifest['versions'],
                    'model_artifacts': manifest['model_artifacts'], 'source_hashes': {'fixture': 'same'}, 'gpu': 'test', 'cuda': 'test'}
                bits = list(policy_bits(policy)); calls = [0, 0] if policy == 'dynamic' else [1, 1]
                memory = {'dynamic': 1000, 'k4_v4': 300, 'k8_v4': 500, 'k4_v8': 500}[policy]
                rows = []
                for task in tasks[:config['smoke_n']]:
                    rows.append({**{k: v for k, v in task.items() if k != 'input_ids'}, **c,
                        'provenance': provenance, 'run_fingerprint': digest_json(provenance), 'task_sha256': digest_json(task),
                        'pred': '段落1' if policy in ('dynamic', 'k8_v4') else '段落2',
                        'output_ids': [7] if policy == 'dynamic' else [8], 'output_tokens': 1, 'stop_reason': 'eos',
                        'hqq_calls_by_layer': [calls]*2, 'stored_bits_by_layer': [bits]*2,
                        'cache_device_bytes': {'cuda:0': memory}, 'cache_gpu_bytes_after_query': memory,
                        'cache_host_bytes_after_query': 0, 'cache_storage_count': 4, 'cache_tokens_after_query': task['input_tokens'],
                        'prefill_sec': 1., 'decode_sec': 2., 'exact_logits_match': True,
                        'stock8_logits_match': True, 'stock4_logits_match': True})
                write_jsonl(result_path('results/zh_confirmation_v1/smoke', c), rows)

    def test_sealed_preview_prepare_stress_smoke_and_resume(self):
        with tempfile.TemporaryDirectory() as temp, chdir(temp):
            config, models, root, preview, raw, template = self.fixture()
            self.assertTrue(all(s['query_plus_answer_ge_512_count'] == 1 for s in preview['tokenizer_precheck']))
            self.run_prepare(config, models, root)
            first = {str(p): p.read_bytes() for p in PREPARED.rglob('*') if p.is_file()}
            self.run_prepare(config, models, root)
            self.assertEqual(first, {str(p): p.read_bytes() for p in PREPARED.rglob('*') if p.is_file()})
            for name in config['models']:
                with patch('importlib.metadata.version', return_value='test'):
                    tasks, _, _ = current_inputs(name, models[name], config)
                self.assertEqual(len(tasks), 120)
                self.assertGreater(tasks[0]['input_tokens']-tasks[0]['context_split'], 512)
                self.assertEqual(max(t['input_tokens']-t['context_split'] for t in tasks), tasks[0]['input_tokens']-tasks[0]['context_split'])
                self.assertEqual({t['source_index'] for t in tasks}, set(preview['history']['datasets'][0]['preview_source_indices']))
            self.write_smoke(config)
            matrix = read_matrix(config, 'smoke')
            self.assertEqual(sum(map(len, matrix.values())), 24)
            from src.run_zh_confirmation import smoke_environment
            with self.assertRaisesRegex(ValueError, 'Код изменился'):
                smoke_environment(config, config['models'][0], {'fixture': 'changed'})
            smoke_environment(config, config['models'][0], {'fixture': 'same'})
            metric = load_metric('external/LongBench/LongBench')
            summaries, contrasts, budgets, examples = analyze(matrix, {**config, 'bootstrap_iterations': 200}, metric)
            self.assertEqual((len(summaries), len(contrasts), len(budgets), len(examples)), (8, 10, 2, 24))
            primary = [c for c in contrasts if c['role'] == 'primary']
            self.assertTrue(all(c['delta_pp'] == 100 and c['ci_level'] == .975 for c in primary))
            self.assertEqual(len(primary), 2)
            k8 = [s for s in summaries if s['policy'] == 'k8_v4']
            self.assertTrue(all(s['score_equal_vs_dynamic'] == 3 and s['identical_output_ids_vs_dynamic'] == 0 for s in k8))
            self.assertTrue(all(b['equal_bytes_count'] == 3 for b in budgets))
            path = Path('data/prepared/mixed_precision_v1/qwen3-1.7b/manifest.json')
            path.write_text(path.read_text()+'\n')
            with patch('importlib.metadata.version', return_value='test'), self.assertRaisesRegex(ValueError, 'Изменены'):
                current_inputs('qwen3-1.7b', models['qwen3-1.7b'], config)
            self.assertEqual(first, {str(p): p.read_bytes() for p in PREPARED.rglob('*') if p.is_file()})

    def test_history_change_blocks_freeze_instead_of_resampling(self):
        with tempfile.TemporaryDirectory() as temp, chdir(temp):
            config, models, root, preview, *_ = self.fixture()
            used = preview['history']['datasets'][0]['preview_source_indices'][0]
            write_jsonl('results/another_run/zh.jsonl', [{'dataset': DATASET, 'source_index': used}])
            with self.assertRaisesRegex(ValueError, 'Доступные документы изменились'):
                self.run_prepare(config, models, root)
            self.assertFalse(PREPARED.exists())

    def test_budget_overflow_is_not_filtered_and_matrix_remains_frozen(self):
        config = load_config()
        self.assertEqual(len(cells(config))*config['n'], 960)
        self.assertEqual(config['residual_length'], 1024)
        raw = [{'context': '文档。'*50, 'input': '长'*1100, 'answers': ['段落1']}]
        before = copy.deepcopy(raw)
        hashes = {'0': hashlib.sha256(raw[0]['context'].encode()).hexdigest()}
        with self.assertRaisesRegex(ValueError, 'refresh'):
            prepare_tasks(raw, [0], CharTokenizer(), '{context}\n{input}', config, hashes)
        self.assertEqual(raw, before)


@unittest.skipUnless(all(importlib.util.find_spec(p) for p in ('torch', 'transformers', 'hqq')),
                     'Нужны torch, transformers и настоящий HQQ')
class ZhNumericalTests(unittest.TestCase):
    def test_815_token_query_fits_1024_and_real_hqq_budgets_and_parity(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from src.mixed_cache import native_generate
        from src.run_mixed_precision import smoke_checks
        torch.set_num_threads(1); torch.manual_seed(42)
        c = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=32, eos_token_id=None)
        c._attn_implementation = 'sdpa'
        model = Qwen3ForCausalLM(c).eval()
        tok = SimpleNamespace(eos_token_id=None, decode=lambda x, **_: str(x))
        task = {'input_ids': [i % 63+1 for i in range(64+815)], 'context_split': 64, 'max_new_tokens': 32}
        with self.assertRaisesRegex(ValueError, 'refresh'):
            native_generate(model, tok, task, 'k8_v4', residual=512)
        memory = {}
        for policy in load_config()['policies']:
            _, stats, trace = native_generate(model, tok, task, policy, residual=1024, trace=True)
            memory[policy] = stats['cache_host_bytes_after_query']
            self.assertEqual(stats['hqq_calls_by_layer'], [[0, 0] if policy == 'dynamic' else [1, 1]]*2)
            if policy in ('dynamic', 'k4_v4'):
                self.assertTrue(all(smoke_checks(model, tok, task, policy, stats, trace, 1024).values()))
        self.assertEqual(memory['k8_v4'], memory['k4_v8'])


if __name__ == '__main__':
    unittest.main()
