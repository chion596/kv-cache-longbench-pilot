"""Один Slurm job на модель; семь K/V-политик, только teacher forcing."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from src.common import read_jsonl
from src.download_data import sha256_file
from src.kv_probe import (RESULTS, ROOT, cell, evaluate, load_config, load_prepared,
    parity, read_matrix, selective_cache, source_hashes, validate_result, versions)
from src.protocol import code_revision, digest_json, model_signature, result_path
from src.run_protocol import output_lock
from src.run_quality import make_cache


def current_inputs(name, model_path, config):
    tasks, manifest, path = load_prepared(name, config)
    if versions() != manifest['versions'] or model_signature(model_path) != manifest['model_artifacts']:
        raise ValueError('Модель/tokenizer/версии изменились после подготовки')
    if sha256_file(ROOT/'src/prepare_kv_probe.py') != manifest['prepare_source_sha256']:
        raise ValueError('Код подготовки изменился после создания входов')
    for path_key, hash_key in (('reference_manifest_path', 'reference_manifest_sha256'),
                              ('reference_tasks_path', 'reference_tasks_sha256'),
                              ('reference_selection_path', 'reference_selection_sha256')):
        if sha256_file(Path(manifest[path_key])) != manifest[hash_key]:
            raise ValueError(f'Изменён исходный артефакт: {manifest[path_key]}')
    for filename, digest in manifest['reference_results_sha256'].items():
        if sha256_file(Path(filename)) != digest:
            raise ValueError(f'Изменены reference-результаты: {filename}')
    return tasks, manifest, path


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', choices=config['models'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    tasks, manifest, manifest_path = current_inputs(args.name, args.model, config)
    sources = source_hashes()
    smoke = read_matrix(config, 'smoke') if args.phase == 'full' else None
    if smoke is not None:
        old = next(iter(smoke[args.name, 'dynamic'].values()))['provenance']
        if old['source_hashes'] != sources:
            raise ValueError('Код изменился после smoke; full не отправляем')
    if args.phase == 'smoke':
        tasks = tasks[:config['smoke_n']]
    if args.check_data:
        print('KV probe: проверены входы, эталоны, исходные результаты, модель и версии:',
              args.name, 'n=', len(tasks), 'политик=', len(config['policies']))
        return
    import torch
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError('GPU-расчёт выполняется через Slurm')
    provenance = {'series': config['series'], 'model_name': args.name, 'phase': args.phase,
        'config_sha256': digest_json(config), 'manifest_sha256': sha256_file(manifest_path),
        'versions': versions(), 'model_artifacts': manifest['model_artifacts'], 'source_hashes': sources,
        'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda}
    if smoke is not None:
        if {k: v for k, v in old.items() if k not in ('phase', 'policy')} != {k: v for k, v in provenance.items() if k != 'phase'}:
            raise ValueError('Условия full отличаются от smoke')
    model = None
    commit = code_revision(ROOT)
    for policy in config['policies']:
        policy_provenance = {**provenance, 'policy': policy}
        fingerprint = digest_json(policy_provenance)
        output = result_path(RESULTS/args.phase, cell(args.name, policy, config))
        with output_lock(output):
            completed = set()
            for row in read_jsonl(output) if output.exists() else []:
                tid = row['task_id']
                validate_result(row, args.name, policy, config, args.phase, manifest, sha256_file(manifest_path))
                if row['run_fingerprint'] != fingerprint or tid in completed or tid not in {t['task_id'] for t in tasks}:
                    raise ValueError('Resume: смешаны условия или задачи')
                completed.add(tid)
            if len(completed) == len(tasks):
                print('Уже завершено:', output)
                continue
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
                    attn_implementation=config['attention'], local_files_only=True).to('cuda').eval()
            with output.open('a', encoding='utf-8') as handle:
                for task in tasks:
                    if task['task_id'] in completed:
                        continue
                    gc.collect()
                    torch.cuda.empty_cache()
                    cache, calls = selective_cache(model, policy, config['residual_length'])
                    stats, logits = evaluate(model, task, cache, calls)
                    del cache
                    checks = {}
                    if args.phase == 'smoke' and policy in ('dynamic', 'kv_g32', 'kv_g64'):
                        reference_policy = 'exact_g64' if policy == 'dynamic' else 'int4_g'+policy.split('_g')[1]
                        if policy == 'dynamic':
                            reference, count = selective_cache(model, reference_policy, config['residual_length'])
                        else:
                            reference = make_cache(model, reference_policy, config['residual_length'])
                            count = None
                        ref_stats, ref_logits = evaluate(model, task, reference, count)
                        checks['stock_logits_match'] = parity(logits, ref_logits)
                        checks['stock_query_token_match'] = stats['query_first_token'] == ref_stats['query_first_token']
                        if not checks['stock_query_token_match']:
                            raise ValueError('Не совпал первый токен вопроса со stock cache')
                        del reference, ref_logits
                    del logits
                    if policy in ('dynamic', 'kv_g32', 'kv_g64'):
                        ref_name = 'dynamic' if policy == 'dynamic' else 'int4_g'+policy.split('_g')[1]
                        checks['expansion_query_token_match'] = (stats['query_first_token'] ==
                            manifest['reference_first_tokens'][ref_name][task['task_id']])
                    row = {k: task[k] for k in ('task_id', 'context_sha256', 'input_ids_sha256', 'gold_number', 'target_ids')}
                    row.update(**stats, **checks, policy=policy, task_sha256=digest_json(task),
                        provenance=policy_provenance, run_fingerprint=fingerprint, code_commit=commit)
                    validate_result(row, args.name, policy, config, args.phase, manifest, sha256_file(manifest_path))
                    handle.write(json.dumps(row, ensure_ascii=False)+'\n')
                    handle.flush()
                    print(task['task_id'], policy, 'NLL=', round(stats['gold_nll_mean'], 5),
                          'margin=', round(stats['first_number_token_margin'], 5), flush=True)
        print('Сохранено:', output)


if __name__ == '__main__':
    main()
