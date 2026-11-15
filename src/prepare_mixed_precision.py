"""Копируем проверенные expansion-входы в отдельный неизменный manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common import read_jsonl, write_jsonl
from src.download_data import sha256_file
from src.expansion import TASK_FIELDS, load_config as expansion_config, validate_row as validate_original
from src.mixed_precision import PREPARED, cell, load_config
from src.protocol import digest_json, result_path
from src.run_expansion import load_inputs
from src.run_protocol import output_lock


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', choices=config['models'], required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    datasets, all_tasks, original = {}, {}, None
    for dataset in config['datasets']:
        tasks, manifest, path, selection, _ = load_inputs(cell(args.name, dataset, 'dynamic', config), args.model, expansion_config())
        if original is not None and original != manifest:
            raise ValueError('Manifest источника изменился во время подготовки')
        original = manifest
        all_tasks[dataset] = tasks
        references = {}
        for policy, old_cache in (('dynamic', 'dynamic'), ('k4_v4', 'int4_g64')):
            c = cell(args.name, dataset, old_cache, config)
            source = result_path('results/expansion_v1/full', c)
            rows = list(read_jsonl(source))
            if len(rows) != len(tasks) or {r['task_id'] for r in rows} != {t['task_id'] for t in tasks}:
                raise ValueError('Нужны завершённые source baselines expansion_v1')
            for row in rows:
                validate_original(row, c, expansion_config(), 'full', manifest, path, selection)
            references[policy] = {'path': str(source), 'sha256': sha256_file(source)}
        datasets[dataset] = {'source_path': str(path.parent/f'{dataset}.jsonl'),
            'source_sha256': manifest['datasets'][dataset]['file_sha256'],
            'reference_results': references,
            'task_ids': [t['task_id'] for t in tasks],
            'task_hashes': {t['task_id']: digest_json(t) for t in tasks},
            'task_metadata': {t['task_id']: {k: t[k] for k in TASK_FIELDS} for t in tasks}}
    folder = PREPARED/args.name
    with output_lock(folder/'manifest.json'):
        metadata = {'series': config['series'], 'model_name': args.name, 'config_sha256': digest_json(config),
            'versions': original['versions'], 'model_artifacts': original['model_artifacts'],
            'hidden_layers': original['hidden_layers'], 'longbench_files_sha256': original['longbench_files_sha256'],
            'source_manifest_path': str(path), 'source_manifest_sha256': sha256_file(path),
            'selection_path': 'data/prepared/expansion_v1/selection.json',
            'selection_sha256': original['selection_sha256'],
            'prepare_source_sha256': sha256_file(Path(__file__)), 'datasets': datasets}
        target = folder/'manifest.json'
        if target.exists():
            old = json.loads(target.read_text())
            for dataset, tasks in all_tasks.items():
                prepared = folder/f'{dataset}.jsonl'
                datasets[dataset]['file_sha256'] = sha256_file(prepared)
                if list(read_jsonl(prepared)) != tasks:
                    raise ValueError('Подготовленные задачи изменились')
            if old != metadata:
                raise ValueError('Условия подготовки изменились; файлы не перезаписываем')
            print('Уже подготовлено:', target)
            return
        if any((folder/f'{d}.jsonl').exists() for d in config['datasets']):
            raise FileExistsError('Остались входы без manifest; проверьте каталог вручную')
        for dataset, tasks in all_tasks.items():
            prepared = folder/f'{dataset}.jsonl'
            write_jsonl(prepared, tasks)
            datasets[dataset]['file_sha256'] = sha256_file(prepared)
        with target.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print('Mixed precision: подготовлены неизменные inputs:', target, '80 документов × 2 набора')


if __name__ == '__main__':
    main()
