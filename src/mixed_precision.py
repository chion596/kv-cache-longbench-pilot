"""Матрица mixed_precision_v1: новые native-ответы и измеренная память cache."""
from __future__ import annotations

import importlib.metadata
import inspect
import itertools
import json
import math
from pathlib import Path

from src.common import read_jsonl
from src.download_data import sha256_file
from src.expansion import ROOT, TASK_FIELDS
from src.mixed_cache import policy_bits
from src.protocol import digest_json, result_path, validate_task

PREPARED = Path('data/prepared/mixed_precision_v1')
RESULTS = Path('results/mixed_precision_v1')
FILES = ('mixed_precision.py', 'mixed_cache.py', 'prepare_mixed_precision.py', 'run_mixed_precision.py',
         'score_mixed_precision.py', 'run_quality.py', 'run_expansion.py', 'expansion.py', 'protocol.py', 'run_protocol.py', 'common.py')


def load_config():
    c = json.loads((ROOT/'experiments/mixed_precision_v1.json').read_text())
    expected = {'series': 'mixed_precision_v1', 'reference_series': 'expansion_v1',
        'models': ['qwen3-1.7b', 'qwen3-4b'], 'datasets': ['passage_retrieval_en', 'hotpotqa'],
        'policies': ['dynamic', 'k4_v4', 'k8_v4', 'k4_v8', 'k16_v4'],
        'protocol': 'context_then_question', 'n': 80, 'smoke_n': 3, 'group_size': 64,
        'residual_length': 512, 'dtype': 'float16', 'attention': 'sdpa', 'max_new_tokens': 32}
    if any(c.get(k) != v for k, v in expected.items()):
        raise ValueError('Изменена матрица mixed_precision_v1; нужна новая серия')
    return c


def cell(model, dataset, policy, config):
    return dict(model_name=model, dataset=dataset, protocol=config['protocol'], cache=policy)


def cells(config):
    return [cell(m, d, p, config) for m, d, p in itertools.product(config['models'], config['datasets'], config['policies'])]


def versions():
    return {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'hqq')}


def source_hashes():
    from transformers import cache_utils
    from hqq.core import quantize
    return {**{name: sha256_file(ROOT/'src'/name) for name in FILES},
        'transformers.cache_utils': sha256_file(Path(inspect.getfile(cache_utils))),
        'hqq.core.quantize': sha256_file(Path(inspect.getfile(quantize)))}


def load_prepared(model, dataset, config, root=PREPARED):
    folder = Path(root)/model
    path = folder/'manifest.json'
    manifest = json.loads(path.read_text())
    data = manifest['datasets'][dataset]
    tasks = list(read_jsonl(folder/f'{dataset}.jsonl'))
    if (manifest['config_sha256'] != digest_json(config) or manifest['model_name'] != model
            or sha256_file(folder/f'{dataset}.jsonl') != data['file_sha256']
            or len(tasks) != config['n'] or [t['task_id'] for t in tasks] != data['task_ids']
            or len({t['task_id'] for t in tasks}) != len(tasks)):
        raise ValueError('Неполные/изменённые входы mixed precision')
    for task in tasks:
        validate_task(task)
        if (digest_json(task) != data['task_hashes'][task['task_id']]
                or task['input_tokens'] != len(task['input_ids'])
                or task['max_new_tokens'] != config['max_new_tokens']
                or task['input_tokens']-task['context_split']+task['max_new_tokens'] >= config['residual_length']):
            raise ValueError('Изменена задача или нарушен запас residual')
    return tasks, manifest, path


def read_references(manifest, dataset, tasks):
    result = {}
    expected = {t['task_id']: t for t in tasks}
    data = manifest['datasets'][dataset]
    for policy, info in data['reference_results'].items():
        path = Path(info['path'])
        if sha256_file(path) != info['sha256']:
            raise ValueError(f'Изменён исходный результат: {path}')
        all_rows = list(read_jsonl(path))
        if (len(all_rows) != len(data['task_ids']) or {r['task_id'] for r in all_rows} != set(data['task_ids'])):
            raise ValueError('Неполный/повторный source baseline')
        rows = {r['task_id']: r for r in all_rows if r['task_id'] in expected}
        if set(rows) != set(expected) or any(any(rows[t][k] != expected[t][k]
                for k in ('input_ids_sha256', 'context_sha256')) for t in rows):
            raise ValueError('Source baseline относится к другим входам')
        result[policy] = rows
    return result


def validate_result(row, c, config, phase, manifest, manifest_hash):
    p = row['provenance']
    tid = row['task_id']
    data = manifest['datasets'][c['dataset']]
    if (any(row.get(k) != v or p.get(k) != v for k, v in c.items())
            or p.get('series') != config['series'] or p.get('phase') != phase
            or p.get('config_sha256') != digest_json(config) or p.get('manifest_sha256') != manifest_hash
            or p.get('versions') != manifest['versions'] or p.get('model_artifacts') != manifest['model_artifacts']
            or row.get('run_fingerprint') != digest_json(p)
            or row.get('task_sha256') != data['task_hashes'].get(tid)):
        raise ValueError('Неверные входы/условия mixed precision')
    expected = data['task_metadata'][tid]
    if any(row.get(k) != expected[k] for k in TASK_FIELDS):
        raise ValueError('Изменены metadata/эталоны задачи')
    ids = row.get('output_ids', [])
    if (not ids or len(ids) != row['output_tokens'] or not 1 <= len(ids) <= config['max_new_tokens']
            or row['stop_reason'] not in ('eos', 'max_new_tokens') or not isinstance(row.get('pred'), str)):
        raise ValueError('Неполная native-генерация')
    bits = policy_bits(c['cache'])
    calls = [0, 0] if c['cache'] == 'dynamic' else [int(b < 16) for b in bits]
    if (row.get('hqq_calls_by_layer') != [calls]*manifest['hidden_layers']
            or row.get('stored_bits_by_layer') != [list(bits)]*manifest['hidden_layers']):
        raise ValueError('Неверная битность или refresh')
    device_bytes = row.get('cache_device_bytes', {})
    gpu_bytes = sum(v for k, v in device_bytes.items() if k.startswith('cuda'))
    if (not device_bytes or any(type(v) is not int or v < 0 for v in device_bytes.values())
            or row.get('cache_gpu_bytes_after_query') != gpu_bytes or gpu_bytes <= 0
            or row.get('cache_host_bytes_after_query') != sum(v for k, v in device_bytes.items() if k.startswith('cpu'))
            or type(row.get('cache_storage_count')) is not int or row['cache_storage_count'] <= 0
            or row.get('cache_tokens_after_query') != row['input_tokens']):
        raise ValueError('Некорректное измерение cache bytes/точки snapshot')
    if any(not math.isfinite(row[k]) or row[k] < 0 for k in ('prefill_sec', 'decode_sec')):
        raise ValueError('Некорректное время')
    if phase == 'smoke':
        required = ('exact_logits_match', 'stock8_logits_match') if c['cache'] == 'dynamic' else (
            ('stock4_logits_match',) if c['cache'] == 'k4_v4' else ())
        if any(row.get(k) is not True for k in required):
            raise ValueError('Не пройдена обязательная stock/exact сверка')


def read_matrix(config, phase, prepared_root=PREPARED, results_root=RESULTS):
    matrix, common = {}, None
    for c in cells(config):
        tasks, manifest, path = load_prepared(c['model_name'], c['dataset'], config, prepared_root)
        if phase == 'smoke':
            tasks = tasks[:config['smoke_n']]
        expected = {t['task_id'] for t in tasks}
        filename = result_path(Path(results_root)/phase, c)
        if not filename.exists():
            raise FileNotFoundError(f'Матрица ещё не готова: {filename}; проверьте logs/lb-mixed-v1-*.err')
        rows = {}
        for row in read_jsonl(filename):
            tid = row['task_id']
            if tid in rows or tid not in expected:
                raise ValueError('Повторная/чужая задача')
            validate_result(row, c, config, phase, manifest, sha256_file(path))
            env = {k: row['provenance'][k] for k in ('versions', 'source_hashes', 'gpu', 'cuda', 'config_sha256')}
            if common is not None and env != common:
                raise ValueError('Между ячейками изменились среда или код')
            common = env
            rows[tid] = row
        if set(rows) != expected or len({r['run_fingerprint'] for r in rows.values()}) != 1:
            raise ValueError(f'Неполная/смешанная ячейка: {filename}')
        matrix[c['model_name'], c['dataset'], c['cache']] = rows
    for model in config['models']:
        for dataset in config['datasets']:
            a, b = matrix[model, dataset, 'k8_v4'], matrix[model, dataset, 'k4_v8']
            if any(a[t]['cache_gpu_bytes_after_query'] != b[t]['cache_gpu_bytes_after_query'] for t in a):
                raise ValueError('K8/V4 и K4/V8 имеют разную измеренную память; нужен разбор до full')
    for dataset in config['datasets']:
        reference = matrix[config['models'][0], dataset, 'dynamic']
        for model in config['models'][1:]:
            other = matrix[model, dataset, 'dynamic']
            if set(other) != set(reference) or any(any(reference[t][k] != other[t][k]
                    for k in ('context_sha256', 'answers', 'all_classes')) for t in reference):
                raise ValueError('Модели исследуются на разных исходных документах')
    return matrix
