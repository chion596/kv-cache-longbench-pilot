"""Зафиксировать CPU-preview ZH без исключения длинного вопроса; без inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from src.audit_expansion import Catalog, audit
from src.check_zh_confirmation import audit_ready, tokenization_summary
from src.common import write_jsonl
from src.download_data import sha256_file
from src.expansion import TASK_FIELDS
from src.prepare_data import read_dataset_from_zip
from src.prepare_protocol import prepare_chat_task
from src.protocol import digest_json, model_signature
from src.run_protocol import output_lock
from src.zh_confirmation import (DATASET, PREFLIGHT, PREPARED, ROOT, current_inputs,
    load_config, prepare_hashes, versions)


def validate_preview(preview, config):
    if (preview.get('n') != config['n'] or preview.get('seed') != config['sampling_seed']
            or preview.get('dataset') != DATASET or preview.get('archive_sha256') != config['archive_sha256']
            or not preview.get('local_history_ready') or not audit_ready(preview['history'], config['n'])
            or preview['history']['datasets'][0]['dataset'] != DATASET
            or preview['history']['selection_seed'] != config['sampling_seed']):
        raise ValueError('Нужен полный независимый CPU-preview на 120 ZH context')
    summaries = preview['tokenizer_precheck']
    if ({s['model'] for s in summaries} != set(config['models']) or len(summaries) != 2
            or any(s['n'] != config['n'] or s['max_query_plus_answer_tokens'] >= config['residual_length'] for s in summaries)):
        raise ValueError('query+32 не помещается в общий residual1024; случаи не отбрасываем')


def prepare_tasks(raw, indices, tokenizer, template, config, context_hashes):
    tasks = [prepare_chat_task(raw[i], i, DATASET, template, config['max_new_tokens'],
                              tokenizer, config['max_input_tokens']) for i in indices]
    for task in tasks:
        if task['context_sha256'] != context_hashes[str(task['source_index'])]:
            raise ValueError('Context не совпал с CPU-preview')
        if task['input_tokens']-task['context_split']+task['max_new_tokens'] >= config['residual_length']:
            raise ValueError('Риск refresh: выборку не фильтруем, нужен новый residual/протокол')
    return tasks


def stress_order(tasks_by_model, smoke_n):
    """Общие cases по длинам обоих tokenizers; качество не читается."""
    maps = [{t['task_id']: t for t in tasks} for tasks in tasks_by_model.values()]
    ids = sorted(maps[0], key=lambda tid: maps[0][tid]['source_index'])
    if any(set(m) != set(ids) for m in maps):
        raise ValueError('Модели получили разные ZH-документы')
    query = lambda tid: max(m[tid]['input_tokens']-m[tid]['context_split'] for m in maps)
    length = lambda tid: max(m[tid]['input_tokens'] for m in maps)
    candidates = [max(ids, key=query), min(ids, key=length), max(ids, key=length), *ids]
    chosen = list(dict.fromkeys(candidates))[:smoke_n]
    return chosen+[tid for tid in ids if tid not in chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qwen17', required=True)
    parser.add_argument('--qwen4', required=True)
    parser.add_argument('--preflight', default=str(PREFLIGHT))
    parser.add_argument('--archive', default='data/raw/longbench-data.zip')
    parser.add_argument('--longbench-code', default='external/LongBench/LongBench')
    args = parser.parse_args()
    config = load_config()
    models = dict(zip(config['models'], (args.qwen17, args.qwen4)))
    with output_lock(PREPARED.parent/'zh_confirmation_v1.prepare'):
        if PREPARED.exists():
            for name, model in models.items():
                current_inputs(name, model, config)
            print('ZH уже подготовлена; входы и metadata проверены, перезаписи нет:', PREPARED)
            return
        preview = json.loads(Path(args.preflight).read_text())
        validate_preview(preview, config)
        if (sha256_file(Path(args.archive)) != config['archive_sha256']
                or preview['source_sha256'] != sha256_file(ROOT/'src/check_zh_confirmation.py')
                or preview['audit_source_sha256'] != sha256_file(ROOT/'src/audit_expansion.py')):
            raise ValueError('Архив или код CPU-preview изменился')
        published_path = ROOT/'docs/data_independence_audit_2026-11-03.json'
        if sha256_file(published_path) != preview['published_audit_sha256']:
            raise ValueError('Опубликованная история изменилась')
        published = json.loads(published_path.read_text())
        code = Path(args.longbench_code)
        official = {f: sha256_file(code/f) for f in ('config/dataset2prompt.json', 'config/dataset2maxlen.json', 'metrics.py')}
        if official != preview['official_files_sha256']:
            raise ValueError('Официальный код изменился после CPU-preview')
        templates = json.loads((code/'config/dataset2prompt.json').read_text())
        limits = json.loads((code/'config/dataset2maxlen.json').read_text())
        if limits[DATASET] != config['max_new_tokens']:
            raise ValueError('Изменён официальный предел ответа')
        with zipfile.ZipFile(args.archive) as archive:
            history = audit(Catalog(archive), published, ['data/prepared', 'results'], [DATASET],
                            config['n'], config['sampling_seed'])
            if not audit_ready(history, config['n']) or history['datasets'] != preview['history']['datasets']:
                raise ValueError('Доступные документы изменились после preview; нужен новый аудит, без переотбора по качеству')
            raw = read_dataset_from_zip(archive, DATASET)
        indices = history['datasets'][0]['preview_source_indices']
        contexts = preview['preview_context_hashes']
        from transformers import AutoConfig, AutoTokenizer
        tasks_by_model, manifests = {}, {}
        for name, model in models.items():
            parent_path = Path('data/prepared/mixed_precision_v1')/name/'manifest.json'
            parent = json.loads(parent_path.read_text())
            old = next(s for s in preview['tokenizer_precheck'] if s['model'] == name)
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            model_config = AutoConfig.from_pretrained(model, local_files_only=True)
            if (sha256_file(parent_path) != old['parent_manifest_sha256']
                    or parent['versions'] != versions() or parent['model_artifacts'] != model_signature(model)
                    or parent['longbench_files_sha256'] != official
                    or old['chat_template_sha256'] != digest_json(tokenizer.chat_template)
                    or model_config.max_position_embeddings < config['max_input_tokens']+config['max_new_tokens']):
                raise ValueError('Изменены model/tokenizer/версии/parent или недостаточен контекст')
            summary = tokenization_summary(raw, indices, templates[DATASET], tokenizer)
            if any(summary[k] != old[k] for k in summary):
                raise ValueError('Токенизация отличается от CPU-preview')
            tasks_by_model[name] = prepare_tasks(raw, indices, tokenizer, templates[DATASET], config, contexts)
            manifests[name] = {'series': config['series'], 'model_name': name, 'config_sha256': digest_json(config),
                'archive_sha256': config['archive_sha256'], 'versions': versions(), 'model_artifacts': parent['model_artifacts'],
                'hidden_layers': model_config.num_hidden_layers, 'longbench_files_sha256': official,
                'chat_template_sha256': old['chat_template_sha256'], 'parent_manifest_path': str(parent_path),
                'parent_manifest_sha256': sha256_file(parent_path), 'preflight_path': args.preflight,
                'preflight_sha256': sha256_file(Path(args.preflight)), 'prepare_source_hashes': prepare_hashes()}
            print(name, 'n=', len(tasks_by_model[name]), 'max query+32=', summary['max_query_plus_answer_tokens'],
                  'residual=', config['residual_length'], 'truncated=', summary['middle_truncated'])
        order = stress_order(tasks_by_model, config['smoke_n'])
        selection = {'series': config['series'], 'config_sha256': digest_json(config),
            'archive_sha256': config['archive_sha256'], 'preflight_sha256': sha256_file(Path(args.preflight)),
            'sampling_seed': config['sampling_seed'], 'source_count': len(raw), 'source_indices': indices,
            'context_hashes': contexts, 'smoke_task_ids': order[:config['smoke_n']], 'history_at_freeze': history}
        PREPARED.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.zh_confirmation_', dir=PREPARED.parent))
        try:
            (staging/'selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2)+'\n')
            for name, tasks in tasks_by_model.items():
                by_id = {t['task_id']: t for t in tasks}
                tasks = [by_id[tid] for tid in order]
                folder = staging/name
                write_jsonl(folder/f'{DATASET}.jsonl', tasks)
                manifest = {**manifests[name], 'selection_sha256': sha256_file(staging/'selection.json'),
                    'datasets': {DATASET: {'file_sha256': sha256_file(folder/f'{DATASET}.jsonl'),
                        'task_ids': order, 'task_hashes': {t['task_id']: digest_json(t) for t in tasks},
                        'task_metadata': {t['task_id']: {k: t[k] for k in TASK_FIELDS} for t in tasks}}}}
                (folder/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
            if PREPARED.exists():
                raise FileExistsError('ZH уже существует; перезапись запрещена')
            staging.rename(PREPARED)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        print('ZH inputs зафиксированы:', PREPARED, '; smoke:', selection['smoke_task_ids'])
        print('Ни один из 120 документов не исключён; GPU не запускался')


if __name__ == '__main__':
    main()
