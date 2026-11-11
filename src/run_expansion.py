"""Одна Slurm-ячейка expansion_v1: только новая независимая выборка."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path

from src.common import read_jsonl
from src.download_data import sha256_file
from src.expansion import (CODE_FILES, PREPARED, ROOT, TASK_FIELDS, load_config, load_manifest,
                           read_matrix, validate_row)
from src.protocol import cells, code_revision, digest_json, model_signature, result_path, validate_task
from src.run_protocol import output_lock, reference_generate, validate_resume
from src.run_quality import greedy_generate
from src.score_protocol import key


def load_inputs(cell, model_path, config):
    manifest, path, selection = load_manifest(cell['model_name'], config)
    if model_signature(model_path) != manifest['model_artifacts']:
        raise ValueError('Изменились модель или tokenizer после подготовки')
    data = manifest['datasets'][cell['dataset']]
    tasks_path = path.parent / f"{cell['dataset']}.jsonl"
    if sha256_file(tasks_path) != data['file_sha256']:
        raise ValueError('Подготовленные входы изменились')
    tasks = list(read_jsonl(tasks_path))
    expected = {f"{cell['dataset']}:{i}" for i in selection['datasets'][cell['dataset']]['source_indices']}
    if len(tasks) != config['n'] or {r['task_id'] for r in tasks} != expected:
        raise ValueError('Неполный/повторный набор входов')
    for task in tasks:
        validate_task(task)
        if any(task[field] != data['task_metadata'][task['task_id']][field] for field in TASK_FIELDS):
            raise ValueError('Входы не совпали с manifest')
        if (task['context_sha256'] != selection['datasets'][cell['dataset']]['context_hashes'][str(task['source_index'])]
                or task['max_new_tokens'] != config['max_new_tokens'][cell['dataset']]
                or task['input_tokens'] - task['context_split'] + task['max_new_tokens'] >= config['residual_length']):
            raise ValueError('Context или лимиты входа отличаются от выбранного протокола')
    if any(sha256_file(ROOT / 'src' / name) != digest for name, digest in manifest['prepare_source_hashes'].items()):
        raise ValueError('Код подготовки изменился после создания token IDs')
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'hqq')}
    if versions != manifest['versions']:
        raise ValueError('Версии пакетов изменились после подготовки')
    return tasks, manifest, path, selection, versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell', type=int, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    config = load_config()
    matrix = cells(config)
    if not 0 <= args.cell < len(matrix):
        parser.error('Индекс ячейки вне матрицы expansion')
    cell = matrix[args.cell]
    tasks, manifest, manifest_path, selection, versions = load_inputs(cell, args.model, config)
    if args.phase == 'smoke':
        tasks = tasks[:config['smoke_n']]
    if args.check_data:
        if args.phase == 'full':
            smoke = read_matrix('results/expansion_v1/smoke', config, 'smoke')
            old = next(iter(smoke[key(cell)].values()))['provenance']
            sources = {name: sha256_file(ROOT / 'src' / name) for name in CODE_FILES}
            if old['source_hashes'] != sources:
                raise ValueError('Код генерации изменился после smoke; full пока не отправляем')
        print('Expansion: входы/модель/версии проверены:', cell, 'n=', len(tasks))
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('GPU-запуск выполняется через Slurm')
    provenance = {'series': config['series'], **cell, 'phase': args.phase,
        'config_sha256': digest_json(config), 'selection_sha256': manifest['selection_sha256'],
        'manifest_sha256': sha256_file(manifest_path), 'model_artifacts': manifest['model_artifacts'],
        'prepare_source_hashes': manifest['prepare_source_hashes'],
        'longbench_files_sha256': manifest['longbench_files_sha256'], 'versions': versions,
        'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
        'source_hashes': {name: sha256_file(ROOT / 'src' / name) for name in CODE_FILES},
        **{field: config[field] for field in ('dtype', 'attention', 'residual_length', 'prompt_mode')}}
    fingerprint = digest_json(provenance)
    if args.phase == 'full':
        smoke = read_matrix('results/expansion_v1/smoke', config, 'smoke')
        old = next(iter(smoke[key(cell)].values()))['provenance']
        if {k: v for k, v in old.items() if k != 'phase'} != {k: v for k, v in provenance.items() if k != 'phase'}:
            raise ValueError('Условия изменились после smoke expansion')
    output = result_path(Path('results/expansion_v1') / args.phase, cell)
    commit = code_revision(ROOT)
    with output_lock(output):
        previous = list(read_jsonl(output)) if output.exists() else []
        completed = validate_resume(previous, tasks, fingerprint)
        for row in previous:
            validate_row(row, cell, config, args.phase, manifest, manifest_path, selection)
        if len(completed) == len(tasks):
            print('Уже завершено:', output)
            return
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
            attn_implementation=config['attention'], local_files_only=True).to('cuda').eval()
        with output.open('a', encoding='utf-8') as handle:
            for task in tasks:
                if task['task_id'] in completed:
                    continue
                gc.collect()
                torch.cuda.empty_cache()
                split = task['context_split'] if cell['protocol'] == 'context_then_question' else None
                pred, stats = greedy_generate(model, tokenizer, task['input_ids'], cell['cache'],
                    config['residual_length'], task['max_new_tokens'], prefill_split=split, track_quantization=True)
                row = {k: v for k, v in task.items() if k != 'input_ids'}
                row.update({**stats, **cell, 'pred': pred, 'provenance': provenance,
                    'run_fingerprint': fingerprint, 'code_commit': commit})
                if args.phase == 'smoke' and cell['cache'] == 'dynamic' and split is None:
                    reference = reference_generate(model, tokenizer, task['input_ids'], task['max_new_tokens'])
                    row.update(reference_output_ids=reference, reference_matches=reference == stats['output_ids'])
                validate_row(row, cell, config, args.phase, manifest, manifest_path, selection)
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                handle.flush()
                print(task['task_id'], cell, 'stop=', stats['stop_reason'], 'pred=', repr(pred[:120]), flush=True)
                if row.get('reference_matches') is False:
                    raise ValueError('Ручная генерация не совпала с model.generate; результат сохранён для разбора')
        print('Сохранено:', output)


if __name__ == '__main__':
    main()
