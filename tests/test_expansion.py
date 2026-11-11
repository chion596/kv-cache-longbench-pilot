import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from src.audit_expansion import Catalog, DEFAULT_DATASETS, audit
from src.common import digest_ids, read_jsonl, write_jsonl
from src.download_data import sha256_file
from src.expansion import (CODE_FILES, PREPARE_FILES, ROOT, TASK_FIELDS, freeze_selection, load_config,
                           read_matrix, validate_selection)
from src.prepare_expansion import prepare_tasks
from src.protocol import cells, digest_json, result_path
from src.score_expansion import analyze, common_untruncated, runtime_estimate


class ExpansionTests(unittest.TestCase):
    def fixture(self, root, phase='full'):
        root = Path(root)
        config = load_config()
        config.update(n=2, smoke_n=1)
        prepared = root / 'prepared'
        prepared.mkdir()
        selection = {'series': config['series'], 'config_sha256': digest_json(config),
            'archive_sha256': config['archive_sha256'], 'datasets': {d: {'source_count': 10,
                'source_indices': [0, 1], 'context_hashes': {str(i): hashlib.sha256(f'{d}:{i}'.encode()).hexdigest()
                for i in range(2)}} for d in config['datasets']}}
        (prepared / 'selection.json').write_text(json.dumps(selection))
        for model in config['models']:
            folder = prepared / model
            folder.mkdir()
            manifest = {'config_sha256': digest_json(config), 'model_name': model,
                'selection_sha256': sha256_file(prepared / 'selection.json'), 'versions': {'torch': 'test'},
                'model_artifacts': {'model': model}, 'longbench_files_sha256': {'metrics.py': 'test'},
                'prepare_source_hashes': {name: 'same' for name in PREPARE_FILES},
                'hidden_layers': 2, 'datasets': {}}
            for dataset in config['datasets']:
                metadata = {}
                for i in range(2):
                    metadata[f'{dataset}:{i}'] = dict(zip(TASK_FIELDS, (digest_ids([1, 2, 3]),
                        selection['datasets'][dataset]['context_hashes'][str(i)], 1, 3, 32, False,
                        ['Paragraph 1' if dataset == 'passage_retrieval_en' else 'yes'], None)))
                manifest['datasets'][dataset] = {'task_metadata': metadata}
            manifest_path = folder / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            for cell in (c for c in cells(config) if c['model_name'] == model):
                p = {'series': config['series'], **cell, 'phase': phase, 'config_sha256': digest_json(config),
                    'selection_sha256': manifest['selection_sha256'], 'manifest_sha256': sha256_file(manifest_path),
                    'versions': manifest['versions'], 'model_artifacts': manifest['model_artifacts'],
                    'prepare_source_hashes': manifest['prepare_source_hashes'],
                    'longbench_files_sha256': manifest['longbench_files_sha256'], 'gpu': 'V100', 'cuda': 'test',
                    'source_hashes': {name: 'same' for name in CODE_FILES},
                    **{k: config[k] for k in ('dtype', 'attention', 'residual_length', 'prompt_mode')}}
                rows = []
                for i in range(config['smoke_n'] if phase == 'smoke' else config['n']):
                    tid = f"{cell['dataset']}:{i}"
                    rows.append({**manifest['datasets'][cell['dataset']]['task_metadata'][tid], **cell,
                        'task_id': tid, 'source_index': i, 'output_ids': [11, 12], 'output_tokens': 2,
                        'stop_reason': 'eos', 'pred': 'Paragraph 1' if cell['dataset'] == 'passage_retrieval_en' else 'yes',
                        'prefill_split': 1 if cell['protocol'] == 'context_then_question' else None,
                        'query_tokens': 2 if cell['protocol'] == 'context_then_question' else None,
                        'quantize_calls_by_layer': [0, 0] if cell['cache'] == 'dynamic' else [2, 2],
                        'provenance': p, 'run_fingerprint': digest_json(p),
                        'reference_output_ids': [11, 12], 'reference_matches': True})
                write_jsonl(result_path(root / 'results', cell), rows)
        return config, prepared, selection

    def test_plan_has_3840_full_and_144_smoke_answers(self):
        config = load_config()
        self.assertEqual(len(cells(config)), 48)
        self.assertEqual(len(cells(config))*config['n'], 3840)
        self.assertEqual(len(cells(config))*config['smoke_n'], 144)

    def test_complete_smoke_and_all_pairwise_model_contrasts(self):
        with tempfile.TemporaryDirectory() as temp:
            config, prepared, _ = self.fixture(temp, 'smoke')
            matrix = read_matrix(Path(temp)/'results', config, 'smoke', prepared)
            for r in matrix[('qwen3-1.7b', 'hotpotqa', 'context_then_question', 'int4_g64')].values():
                r['pred'] = 'wrong'
            original = copy.deepcopy(matrix)
            metrics = {d: lambda pred, truth, **_: float(pred == truth) for d in config['datasets']}
            summary, contrasts, examples, drift = analyze(matrix, config, metrics)
            self.assertEqual(matrix, original)
            self.assertEqual((len(summary), len(contrasts), len(examples), len(drift)), (48, 78, 48, 6))
            direct = [r for r in contrasts if r['comparison'] == 'int4_g64_loss:qwen3-4b-qwen3-1.7b'
                      and r['dataset'] == 'hotpotqa' and r['protocol'] == 'context_then_question']
            self.assertEqual(direct[0]['delta_pp'], 100)

    def test_dynamic_split_drift_is_reported_without_invalidating_quantized_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            config, prepared, _ = self.fixture(temp, 'smoke')
            cell = cells(config)[4]  # first model, retrieval split Dynamic
            path = result_path(Path(temp)/'results', cell)
            rows = list(read_jsonl(path))
            rows[0]['output_ids'] = [13, 12]
            write_jsonl(path, rows)
            matrix = read_matrix(Path(temp)/'results', config, 'smoke', prepared)
            *_, drift = analyze(matrix, config, {d: lambda p, a, **_: float(p == a) for d in config['datasets']})
            self.assertEqual(drift[0]['different_output_ids'], 1)

    def test_refresh_input_change_and_first_token_change_are_rejected(self):
        for change in (lambda row: row.update(quantize_calls_by_layer=[4, 4]),
                       lambda row: row.update(answers=['wrong']),
                       lambda row: row.update(output_ids=[13, 12])):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                config, prepared, _ = self.fixture(temp)
                path = result_path(Path(temp)/'results', cells(config)[1])
                rows = list(read_jsonl(path))
                change(rows[0])
                write_jsonl(path, rows)
                with self.assertRaises(ValueError):
                    read_matrix(Path(temp)/'results', config, 'full', prepared)

    def test_smoke_generate_mismatch_and_missing_cells_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            config, prepared, _ = self.fixture(temp, 'smoke')
            path = result_path(Path(temp)/'results', cells(config)[0])
            rows = list(read_jsonl(path))
            rows[0]['reference_matches'] = False
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, 'model.generate'):
                read_matrix(Path(temp)/'results', config, 'smoke', prepared)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                read_matrix(Path(temp)/'results', config, 'smoke', prepared)

    def test_contexts_must_be_unique_across_datasets(self):
        with tempfile.TemporaryDirectory() as temp:
            config, _, selection = self.fixture(temp)
            selection['datasets']['hotpotqa']['context_hashes']['0'] = selection['datasets']['passage_retrieval_en']['context_hashes']['0']
            with self.assertRaisesRegex(ValueError, 'пересекаются'):
                validate_selection(selection, config)

    def test_sensitivity_subset_excludes_a_question_if_any_model_truncates_it(self):
        with tempfile.TemporaryDirectory() as temp:
            config, prepared, _ = self.fixture(temp)
            matrix = read_matrix(Path(temp)/'results', config, 'full', prepared)
            for k, rows in matrix.items():
                if k[0] == 'qwen3-1.7b' and k[1] == 'hotpotqa':
                    rows['hotpotqa:1']['middle_truncated'] = True
            counts, contrasts = common_untruncated(matrix, config,
                {d: lambda p, a, **_: float(p == a) for d in config['datasets']})
            self.assertEqual(counts[1]['common_untruncated_questions'], 1)
            self.assertEqual(len(contrasts), 39)
            self.assertTrue(all(r['n'] == 1 and r['dataset'] == 'hotpotqa' for r in contrasts))

    def test_runtime_extrapolation_uses_only_recorded_generation_time(self):
        matrix = {('m', 'd', 'p', 'c'): {'a': {'prefill_sec': 2, 'decode_sec': 3},
                                       'b': {'prefill_sec': 4, 'decode_sec': 1}}}
        rows = runtime_estimate(matrix, {'n': 80})
        self.assertEqual(rows[0]['timed_answers'], 2)
        self.assertAlmostEqual(rows[0]['estimated_full_generation_hours'], 400/3600)
        del matrix[('m', 'd', 'p', 'c')]['b']['decode_sec']
        self.assertEqual(runtime_estimate(matrix, {'n': 80})[0]['timed_answers'], 1)

    def audit_fixture(self, root):
        root = Path(root)
        archive_path = root/'source.zip'
        entries = {}
        with zipfile.ZipFile(archive_path, 'w') as archive:
            for dataset in DEFAULT_DATASETS:
                raw = '\n'.join(json.dumps({'context': f'{dataset}-{i}'}) for i in range(6))+'\n'
                entries[dataset] = hashlib.sha256(raw.encode()).hexdigest()
                archive.writestr(f'data/{dataset}.jsonl', raw)
        published = {'revision': 'fixture', 'archive_sha256_from_upstream_lfs': sha256_file(archive_path),
            'datasets': [{'dataset': d, 'member_sha256': entries[d], 'pilot_indices': [0], 'heldout_indices': [1]}
                         for d in DEFAULT_DATASETS[:3]]}
        published_path = root/'published.json'
        published_path.write_text(json.dumps(published))
        history = root/'history'
        history.mkdir()
        (history/'old.jsonl').write_text('{"dataset":"passage_retrieval_en","source_index":0}\n')
        config = load_config()
        config.update(n=2, smoke_n=1, archive_sha256=sha256_file(archive_path))
        with zipfile.ZipFile(archive_path) as archive:
            report = audit(Catalog(archive), published, [history], list(DEFAULT_DATASETS), 2, config['sampling_seed'])
        report.update(archive_sha256=sha256_file(archive_path), source_revision=published['revision'],
            published_audit_sha256=sha256_file(published_path), audit_source_sha256=sha256_file(ROOT/'src/audit_expansion.py'))
        audit_path = root/'audit.json'
        audit_path.write_text(json.dumps(report))
        return config, audit_path, archive_path, published_path, root/'new', history

    def test_freeze_does_not_resample_after_new_inputs_appear(self):
        with tempfile.TemporaryDirectory() as temp:
            config, audit_path, archive, published, root, history = self.audit_fixture(temp)
            selection = freeze_selection(config, audit_path, archive, published, root, [history])
            index = selection['datasets']['passage_retrieval_en']['source_indices'][0]
            (history/'new.jsonl').write_text(json.dumps({'dataset': 'passage_retrieval_en', 'source_index': index})+'\n')
            self.assertEqual(selection, freeze_selection(config, audit_path, archive, published, root, [history]))

    def test_stale_audit_is_rejected_before_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            config, audit_path, archive, published, root, history = self.audit_fixture(temp)
            (history/'new.jsonl').write_text('{"dataset":"passage_retrieval_en","source_index":2}\n')
            with self.assertRaisesRegex(ValueError, 'История изменилась'):
                freeze_selection(config, audit_path, archive, published, root, [history])

    def test_preparation_keeps_same_full_ids_for_both_prefill_protocols(self):
        from test_protocol import ChatTokenizer
        config = load_config()
        source = [{'context': 'Document', 'input': 'Where?', 'answers': ['yes']}]
        selection = {'datasets': {'hotpotqa': {'context_hashes': {'0': hashlib.sha256(b'Document').hexdigest()}}}}
        task = prepare_tasks(source, [0], 'hotpotqa', '{context}\nQ:{input}', ChatTokenizer(), config, selection)[0]
        self.assertEqual(task['input_ids_sha256'], digest_ids(task['input_ids']))
        split = task['context_split']
        self.assertTrue(0 < split < len(task['input_ids']))
        config['residual_length'] = 10
        with self.assertRaisesRegex(ValueError, 'Query\\+answer'):
            prepare_tasks(source, [0], 'hotpotqa', '{context}\nQ:{input}', ChatTokenizer(), config, selection)


if __name__ == '__main__':
    unittest.main()
