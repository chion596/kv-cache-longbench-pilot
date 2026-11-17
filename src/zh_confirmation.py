"""Отдельная ZH-матрица: 120 новых документов и общий residual1024."""
from __future__ import annotations

import json
from pathlib import Path
import re

from src.download_data import sha256_file
from src.mixed_precision import (ROOT, cell, cells, load_prepared as mixed_prepared,
    read_matrix as mixed_matrix, source_hashes as mixed_sources, validate_result, versions)
from src.protocol import digest_json, model_signature, result_path

PREPARED = Path('data/prepared/zh_confirmation_v1')
RESULTS = Path('results/zh_confirmation_v1')
PREFLIGHT = Path('results/zh_confirmation_preflight_2026-11-17.json')
DATASET = 'passage_retrieval_zh'
PREPARE_FILES = ('zh_confirmation.py', 'prepare_zh_confirmation.py', 'prepare_protocol.py',
                 'prepare_data.py', 'audit_expansion.py', 'check_zh_confirmation.py', 'common.py', 'protocol.py')
FILES = ('zh_confirmation.py', 'prepare_zh_confirmation.py', 'run_zh_confirmation.py',
         'score_zh_confirmation.py', *PREPARE_FILES)


def load_config():
    config = json.loads((ROOT/'experiments/zh_confirmation_v1.json').read_text())
    expected = {'series': 'zh_confirmation_v1', 'parent_series': 'mixed_precision_v1',
        'models': ['qwen3-1.7b', 'qwen3-4b'], 'datasets': [DATASET],
        'policies': ['dynamic', 'k4_v4', 'k8_v4', 'k4_v8'], 'protocol': 'context_then_question',
        'prompt_mode': 'native_chat_no_thinking', 'n': 120, 'smoke_n': 3, 'sampling_seed': 2043,
        'max_input_tokens': 16384, 'max_new_tokens': 32, 'group_size': 64, 'residual_length': 1024,
        'dtype': 'float16', 'attention': 'sdpa', 'bootstrap_iterations': 10000, 'bootstrap_seed': 2043,
        'primary_ci_level': .975, 'secondary_ci_level': .95,
        'archive_sha256': 'cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64',
        'secondary_comparisons': ['k8_v4-dynamic', 'k4_v8-dynamic', 'k4_v4-dynamic', 'k8_v4-k4_v4']}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Условия zh_confirmation_v1 изменились; нужна новая серия')
    return config


def prepare_hashes():
    return {name: sha256_file(ROOT/'src'/name) for name in PREPARE_FILES}


def source_hashes():
    return {**mixed_sources(), **{name: sha256_file(ROOT/'src'/name) for name in FILES}}


def load_selection(config, root=PREPARED):
    path = Path(root)/'selection.json'
    selection = json.loads(path.read_text())
    indices, hashes = selection['source_indices'], selection['context_hashes']
    if (selection['series'] != config['series'] or selection['config_sha256'] != digest_json(config)
            or selection['archive_sha256'] != config['archive_sha256']
            or len(indices) != config['n'] or indices != sorted(set(indices))
            or any(type(i) is not int or not 0 <= i < selection['source_count'] for i in indices)
            or set(hashes) != {str(i) for i in indices} or len(set(hashes.values())) != config['n']
            or any(not re.fullmatch('[0-9a-f]{64}', h) for h in hashes.values())
            or len(selection['smoke_task_ids']) != config['smoke_n']
            or len(set(selection['smoke_task_ids'])) != config['smoke_n']
            or not set(selection['smoke_task_ids']) <= {f'{DATASET}:{i}' for i in indices}):
        raise ValueError('Повреждена выборка ZH или изменены 120 документов')
    return selection, path


def load_prepared(model, config, root=PREPARED):
    tasks, manifest, path = mixed_prepared(model, DATASET, config, root)
    selection, selection_path = load_selection(config, root)
    if (manifest['selection_sha256'] != sha256_file(selection_path)
            or manifest['archive_sha256'] != config['archive_sha256']
            or {t['source_index'] for t in tasks} != set(selection['source_indices'])
            or [t['task_id'] for t in tasks[:config['smoke_n']]] != selection['smoke_task_ids']):
        raise ValueError('ZH inputs не совпали с фиксированной выборкой/smoke')
    for task in tasks:
        if (task['dataset'] != DATASET or task['task_id'] != f"{DATASET}:{task['source_index']}"
                or task['context_sha256'] != selection['context_hashes'][str(task['source_index'])]
                or task['prompt_mode'] != config['prompt_mode']
                or not task['answers'] or any(not re.fullmatch(r'段落\d+', a) for a in task['answers'])):
            raise ValueError('Изменены context/prompt/эталоны ZH')
    return tasks, manifest, path


def current_inputs(model, model_path, config):
    tasks, manifest, path = load_prepared(model, config)
    if (manifest['versions'] != versions() or manifest['model_artifacts'] != model_signature(model_path)
            or manifest['prepare_source_hashes'] != prepare_hashes()
            or sha256_file(Path(manifest['preflight_path'])) != manifest['preflight_sha256']
            or sha256_file(Path(manifest['parent_manifest_path'])) != manifest['parent_manifest_sha256']):
        raise ValueError('Изменены модель/tokenizer/версии/подготовка/preflight/parent manifest')
    return tasks, manifest, path


def read_matrix(config, phase, prepared_root=PREPARED, results_root=RESULTS):
    for model in config['models']:
        load_prepared(model, config, prepared_root)
    missing = [result_path(Path(results_root)/phase, c) for c in cells(config)
               if not result_path(Path(results_root)/phase, c).is_file()]
    if missing:
        raise FileNotFoundError(f'ZH-матрица {phase} ещё не готова: нет {len(missing)}/8 файлов; '
                                f'например {missing[0]}; проверьте logs/lb-zh-v1-*.err')
    return mixed_matrix(config, phase, prepared_root, results_root)
