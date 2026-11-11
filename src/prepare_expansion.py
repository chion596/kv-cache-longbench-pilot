"""Подготовить модельные token IDs для новых документов expansion_v1 на CPU."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from src.common import load_official_configs, write_jsonl
from src.download_data import sha256_file
from src.expansion import AUDIT, PREPARED, PREPARE_FILES, ROOT, TASK_FIELDS, freeze_selection, load_config
from src.prepare_data import read_dataset_from_zip
from src.prepare_protocol import prepare_chat_task
from src.protocol import code_revision, digest_json, model_signature


def prepare_tasks(raw, indices, dataset, template, tokenizer, config, selection):
    tasks = [prepare_chat_task(raw[i], i, dataset, template, config['max_new_tokens'][dataset],
                              tokenizer, config['max_input_tokens']) for i in indices]
    for task in tasks:
        if task['context_sha256'] != selection['datasets'][dataset]['context_hashes'][str(task['source_index'])]:
            raise ValueError('Context не совпал с выборкой')
        if task['input_tokens'] - task['context_split'] + task['max_new_tokens'] >= config['residual_length']:
            raise ValueError(f"Query+answer не помещается до refresh: {task['task_id']}")
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--audit', default=str(AUDIT))
    parser.add_argument('--archive', default='data/raw/longbench-data.zip')
    parser.add_argument('--longbench-code', default='external/LongBench/LongBench')
    args = parser.parse_args()
    config = load_config()
    if args.name not in config['models']:
        parser.error('Модель отсутствует в матрице')
    destination = PREPARED / args.name
    if destination.exists():
        raise FileExistsError(f'Входы уже подготовлены: {destination}; перезапись запрещена')
    selection = freeze_selection(config, args.audit, args.archive)
    if sha256_file(Path(args.archive)) != selection['archive_sha256']:
        raise ValueError('Архив изменился после фиксации выборки')
    previous = json.loads((Path('data/prepared/protocol_v1') / args.name / 'manifest.json').read_text())
    artifacts = model_signature(args.model)
    if artifacts != previous['model_artifacts']:
        raise ValueError('Модель/токенизатор отличаются от предыдущей серии')
    code = Path(args.longbench_code)
    hashes = {name: sha256_file(code / name) for name in
              ('config/dataset2prompt.json', 'config/dataset2maxlen.json', 'metrics.py')}
    if hashes != previous['longbench_files_sha256']:
        raise ValueError('Изменились официальные шаблоны/метрики LongBench')
    templates, limits = load_official_configs(code)
    if any(limits[d] != config['max_new_tokens'][d] for d in config['datasets']):
        raise ValueError('Пределы ответа не совпали с официальным протоколом')
    from transformers import AutoConfig, AutoTokenizer
    import transformers
    if transformers.__version__ != previous['transformers']:
        raise ValueError('Transformers отличается от предыдущей подготовки')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if model_config.max_position_embeddings < config['max_input_tokens'] + max(config['max_new_tokens'].values()):
        raise ValueError('Контекст модели короче заданного протокола')
    prepared = {}
    with zipfile.ZipFile(args.archive) as archive:
        for dataset in config['datasets']:
            source = read_dataset_from_zip(archive, dataset)
            prepared[dataset] = prepare_tasks(source, selection['datasets'][dataset]['source_indices'],
                dataset, templates[dataset], tokenizer, config, selection)
    manifest = {'series': config['series'], 'config_sha256': digest_json(config), 'model_name': args.name,
        'selection_sha256': sha256_file(PREPARED / 'selection.json'), 'archive_sha256': selection['archive_sha256'],
        'model_artifacts': artifacts, 'hidden_layers': model_config.num_hidden_layers,
        'prepare_source_hashes': {name: sha256_file(ROOT / 'src' / name) for name in PREPARE_FILES},
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'hqq')},
        'longbench_files_sha256': hashes, 'longbench_git_commit': code_revision(code.parent, environ={}),
        'chat_template_sha256': digest_json(tokenizer.chat_template), 'datasets': {}}
    staging = Path(tempfile.mkdtemp(prefix='.expansion_', dir=PREPARED))
    try:
        for dataset, tasks in prepared.items():
            path = staging / f'{dataset}.jsonl'
            write_jsonl(path, tasks)
            manifest['datasets'][dataset] = {'file_sha256': sha256_file(path),
                'middle_truncated': sum(r['middle_truncated'] for r in tasks),
                'query_tokens_max': max(r['input_tokens'] - r['context_split'] for r in tasks),
                'task_metadata': {r['task_id']: {field: r[field] for field in TASK_FIELDS} for r in tasks}}
            print(args.name, dataset, 'n=', len(tasks), 'truncated=', manifest['datasets'][dataset]['middle_truncated'])
        (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print('Подготовлено:', destination)


if __name__ == '__main__':
    main()
