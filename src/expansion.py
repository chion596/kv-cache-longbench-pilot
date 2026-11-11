"""Зафиксированные новые документы, входы и проверки серии expansion_v1."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

from src.audit_expansion import Catalog, DEFAULT_DATASETS, audit
from src.common import read_jsonl
from src.download_data import sha256_file
from src.protocol import cells, digest_json, result_path
from src.run_protocol import output_lock
from src.score_protocol import key

ROOT = Path(__file__).resolve().parents[1]
PREPARED = Path('data/prepared/expansion_v1')
AUDIT = Path('results/expansion_audit_2026-11-11.json')
PUBLISHED = Path('docs/data_independence_audit_2026-11-03.json')
CODE_FILES = ('expansion.py', 'prepare_expansion.py', 'run_expansion.py',
              'prepare_protocol.py', 'run_protocol.py', 'run_quality.py', 'protocol.py', 'common.py')
PREPARE_FILES = ('expansion.py', 'prepare_expansion.py', 'prepare_protocol.py', 'common.py', 'audit_expansion.py')
TASK_FIELDS = ('input_ids_sha256', 'context_sha256', 'context_split', 'input_tokens',
               'max_new_tokens', 'middle_truncated', 'answers', 'all_classes')


def load_config():
    config = json.loads((ROOT / 'experiments/expansion_v1.json').read_text(encoding='utf-8'))
    if (config['series'] != 'expansion_v1' or config['datasets'] != ['passage_retrieval_en', 'hotpotqa']
            or config['models'] != ['qwen3-1.7b', 'qwen3-4b', 'smollm3-3b']
            or config['protocols'] != ['single_prefill', 'context_then_question']
            or config['caches'] != ['dynamic', 'int4_g16', 'int4_g32', 'int4_g64']
            or config['n'] != 80 or config['smoke_n'] != 3 or config['sampling_seed'] != 2031
            or config['max_input_tokens'] != 16384
            or config['residual_length'] != 512 or config['dtype'] != 'float16' or config['attention'] != 'sdpa'
            or config['prompt_mode'] != 'native_chat_no_thinking'
            or config['max_new_tokens'] != {'passage_retrieval_en': 32, 'hotpotqa': 32}):
        raise ValueError('Условия expansion_v1 изменились; для другого протокола нужна отдельная серия')
    for field in ('models', 'datasets', 'protocols', 'caches'):
        if not config[field] or len(config[field]) != len(set(config[field])):
            raise ValueError(f'Пустая/повторная ось: {field}')
    return config


def validate_selection(selection, config):
    if selection.get('series') != config['series'] or selection.get('config_sha256') != digest_json(config):
        raise ValueError('Выборка подготовлена по другой матрице')
    if selection.get('archive_sha256') != config['archive_sha256']:
        raise ValueError('Выборка относится к другому архиву')
    seen = set()
    for dataset in config['datasets']:
        data = selection['datasets'][dataset]
        indices = data['source_indices']
        hashes = data['context_hashes']
        if (len(indices) != config['n'] or len(set(indices)) != len(indices)
                or any(type(i) is not int or not 0 <= i < data['source_count'] for i in indices)
                or set(hashes) != {str(i) for i in indices} or len(set(hashes.values())) != len(indices)):
            raise ValueError(f'Нужны ровно {config["n"]} разных документов: {dataset}')
        if seen.intersection(hashes.values()):
            raise ValueError('Между выбранными наборами пересекаются source context')
        seen.update(hashes.values())


def freeze_selection(config, audit_path=AUDIT, archive_path='data/raw/longbench-data.zip',
                     published_path=PUBLISHED, root=PREPARED, history_roots=('data/prepared', 'results')):
    root = Path(root)
    path = root / 'selection.json'
    with output_lock(path):
        if path.exists():
            selection = json.loads(path.read_text(encoding='utf-8'))
            validate_selection(selection, config)
            if selection['audit_sha256'] != sha256_file(Path(audit_path)):
                raise ValueError('Аудит изменился после фиксации выборки')
            return selection
        provided = json.loads(Path(audit_path).read_text(encoding='utf-8'))
        if not provided.get('local_history_scan_passed') or provided.get('history_scan_errors'):
            raise ValueError('Сначала нужен успешный аудит локальной истории')
        archive_hash = sha256_file(Path(archive_path))
        if archive_hash != config['archive_sha256']:
            raise ValueError('Архив отличается от зафиксированного')
        published = json.loads(Path(published_path).read_text(encoding='utf-8'))
        with zipfile.ZipFile(archive_path) as archive:
            catalog = Catalog(archive)
            current = audit(catalog, published, history_roots, list(DEFAULT_DATASETS),
                            config['n'], config['sampling_seed'])
            current.update(archive_sha256=archive_hash, source_revision=published['revision'],
                published_audit_sha256=sha256_file(Path(published_path)),
                audit_source_sha256=sha256_file(ROOT / 'src/audit_expansion.py'))
            if current != provided:
                raise ValueError('История изменилась после аудита; нужен новый --output у src.audit_expansion')
            summaries = {row['dataset']: row for row in current['datasets']}
            datasets = {}
            for dataset in config['datasets']:
                item = summaries[dataset]
                if item['preview_n'] != config['n']:
                    raise ValueError(f'Недостаточно новых документов: {dataset}/{item["preview_n"]}')
                metadata, _ = catalog.get(dataset)
                indices = item['preview_source_indices']
                datasets[dataset] = {'source_count': len(metadata), 'member_sha256': item['member_sha256'],
                    'source_indices': indices, 'context_hashes': {str(i): metadata[i]['context_sha256'] for i in indices}}
        selection = {'series': config['series'], 'config_sha256': digest_json(config),
            'archive_sha256': archive_hash, 'audit_sha256': sha256_file(Path(audit_path)),
            'audit_source_sha256': current['audit_source_sha256'], 'selection_seed': config['sampling_seed'],
            'history_jsonl_files': current['history_jsonl_files'], 'recognized_history_rows': current['recognized_history_rows'],
            'datasets': datasets}
        validate_selection(selection, config)
        with path.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(selection, ensure_ascii=False, indent=2) + '\n')
        return selection


def load_manifest(model, config, root=PREPARED):
    root = Path(root)
    selection = json.loads((root / 'selection.json').read_text(encoding='utf-8'))
    validate_selection(selection, config)
    path = root / model / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (manifest['config_sha256'] != digest_json(config) or manifest['model_name'] != model
            or manifest['selection_sha256'] != sha256_file(root / 'selection.json')):
        raise ValueError('Model manifest не совпал с зафиксированной выборкой')
    return manifest, path, selection


def validate_row(row, cell, config, phase, manifest, manifest_path, selection, *, manifest_hash=None):
    tid = row['task_id']
    expected = manifest['datasets'][cell['dataset']]['task_metadata']
    if tid not in expected or tid != f"{cell['dataset']}:{row['source_index']}":
        raise ValueError(f'Чужая задача: {tid}')
    if any(row.get(field) != expected[tid][field] for field in TASK_FIELDS):
        raise ValueError(f'Изменились входы/эталоны: {tid}')
    if (row['max_new_tokens'] != config['max_new_tokens'][cell['dataset']]
            or row['input_tokens'] > config['max_input_tokens']
            or row['input_tokens'] - row['context_split'] + row['max_new_tokens'] >= config['residual_length']):
        raise ValueError(f'Нарушены лимиты входа/ответа/residual: {tid}')
    p = row['provenance']
    if (any(row.get(k) != v or p.get(k) != v for k, v in cell.items())
            or p.get('series') != config['series'] or p.get('phase') != phase
            or p.get('config_sha256') != digest_json(config) or row.get('run_fingerprint') != digest_json(p)
            or p.get('manifest_sha256') != (manifest_hash or sha256_file(manifest_path))
            or p.get('selection_sha256') != manifest['selection_sha256']
            or p.get('versions') != manifest['versions'] or p.get('model_artifacts') != manifest['model_artifacts']
            or p.get('prepare_source_hashes') != manifest['prepare_source_hashes']
            or p.get('longbench_files_sha256') != manifest['longbench_files_sha256']):
        raise ValueError(f'Некорректный fingerprint expansion: {tid}')
    for field in ('dtype', 'attention', 'residual_length', 'prompt_mode'):
        if p.get(field) != config[field]:
            raise ValueError(f'Неверный {field}: {tid}')
    if row['context_sha256'] != selection['datasets'][cell['dataset']]['context_hashes'][str(row['source_index'])]:
        raise ValueError(f'Context не совпал с выбранным документом: {tid}')
    if (not row.get('output_ids') or len(row['output_ids']) != row['output_tokens']
            or not 1 <= row['output_tokens'] <= row['max_new_tokens']
            or row['stop_reason'] not in ('eos', 'max_new_tokens')):
        raise ValueError(f'Неполная генерация: {tid}')
    split = row['context_split'] if cell['protocol'] == 'context_then_question' else None
    if row.get('prefill_split') != split or row.get('query_tokens') != (row['input_tokens']-split if split is not None else None):
        raise ValueError(f'Не совпала граница prefill: {tid}')
    calls = row.get('quantize_calls_by_layer')
    count = 0 if cell['cache'] == 'dynamic' else 2
    if not isinstance(calls, list) or len(calls) != manifest['hidden_layers'] or any(type(v) is not int or v != count for v in calls):
        raise ValueError(f'Неожиданная повторная квантизация/метаданные: {tid}/{calls}')


def read_matrix(root, config, phase, prepared_root=PREPARED):
    matrix = {}
    manifests = {model: load_manifest(model, config, prepared_root) for model in config['models']}
    manifest_hashes = {model: sha256_file(values[1]) for model, values in manifests.items()}
    common = None
    for cell in cells(config):
        path = result_path(root, cell)
        if not path.is_file():
            raise FileNotFoundError(f'Нет результата {path}; проверьте logs/lb-expansion-v1-*.err')
        manifest, manifest_path, selection = manifests[cell['model_name']]
        indices = selection['datasets'][cell['dataset']]['source_indices']
        if phase == 'smoke':
            indices = indices[:config['smoke_n']]
        expected = {f"{cell['dataset']}:{i}" for i in indices}
        rows = {}
        for row in read_jsonl(path):
            tid = row['task_id']
            if tid not in expected or tid in rows:
                raise ValueError(f'Повторная/чужая задача: {path}/{tid}')
            validate_row(row, cell, config, phase, manifest, manifest_path, selection,
                         manifest_hash=manifest_hashes[cell['model_name']])
            p = row['provenance']
            environment = {name: p[name] for name in ('versions', 'gpu', 'cuda', 'source_hashes', 'prepare_source_hashes', 'selection_sha256', 'longbench_files_sha256')}
            if common is None:
                common = environment
            if common != environment:
                raise ValueError('Между ячейками изменились среда, source hashes или выборка')
            rows[tid] = row
        if set(rows) != expected or len({r['run_fingerprint'] for r in rows.values()}) != 1:
            raise ValueError(f'Неполная/смешанная ячейка: {path}: {len(rows)}/{len(expected)}')
        matrix[key(cell)] = rows
    for model in config['models']:
        for dataset in config['datasets']:
            base = matrix[(model, dataset, 'single_prefill', 'dynamic')]
            if phase == 'smoke' and any(r.get('reference_matches') is not True or r.get('reference_output_ids') != r['output_ids'] for r in base.values()):
                raise ValueError('Не пройдена сверка Dynamic single с model.generate')
            for cache in config['caches'][1:]:
                other = matrix[(model, dataset, 'single_prefill', cache)]
                if any(base[t]['output_ids'][0] != other[t]['output_ids'][0] for t in base):
                    raise ValueError('Первый токен single изменился до чтения приближённого cache')
    for dataset in config['datasets']:
        base = matrix[(config['models'][0], dataset, 'single_prefill', 'dynamic')]
        for model in config['models'][1:]:
            other = matrix[(model, dataset, 'single_prefill', 'dynamic')]
            if set(base) != set(other) or any(any(base[t][f] != other[t][f] for f in ('answers', 'all_classes', 'context_sha256')) for t in base):
                raise ValueError('Между моделями различаются исходные документы/эталоны')
    return matrix
