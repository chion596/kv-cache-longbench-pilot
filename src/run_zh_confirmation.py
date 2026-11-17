"""Одна Qwen в Slurm: четыре политики, native ZH, residual1024."""
from __future__ import annotations

import argparse
import json

from src.common import read_jsonl
from src.download_data import sha256_file
from src.mixed_cache import native_generate
from src.protocol import code_revision, digest_json, result_path
from src.run_mixed_precision import cleanup, smoke_checks
from src.run_protocol import output_lock, validate_resume
from src.zh_confirmation import (DATASET, RESULTS, ROOT, cell, current_inputs, load_config,
    read_matrix, source_hashes, validate_result, versions)


def smoke_environment(config, model, sources):
    smoke = read_matrix(config, 'smoke')
    for rows in smoke.values():
        if next(iter(rows.values()))['provenance']['source_hashes'] != sources:
            raise ValueError('Код изменился после ZH smoke; full не отправляем')
    return {p: next(iter(smoke[model, DATASET, p].values()))['provenance'] for p in config['policies']}


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', choices=config['models'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    tasks, manifest, path = current_inputs(args.name, args.model, config)
    sources = source_hashes()
    smoke = smoke_environment(config, args.name, sources) if args.phase == 'full' else None
    if args.phase == 'smoke':
        tasks = tasks[:config['smoke_n']]
    if args.check_data:
        print('ZH inputs/модель/версии проверены:', args.name, 'n=', len(tasks),
              'policies=', len(config['policies']), 'residual=', config['residual_length'])
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('GPU-вычисления выполняются через Slurm')
    base = {'series': config['series'], 'phase': args.phase, 'config_sha256': digest_json(config),
        'manifest_sha256': sha256_file(path), 'versions': versions(), 'model_artifacts': manifest['model_artifacts'],
        'source_hashes': sources, 'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda}
    model = tokenizer = None
    for policy in config['policies']:
        c = cell(args.name, DATASET, policy, config)
        provenance = {**base, **c}
        if smoke is not None and ({k: v for k, v in smoke[policy].items() if k != 'phase'} !=
                                  {k: v for k, v in provenance.items() if k != 'phase'}):
            raise ValueError('Условия ZH full отличаются от smoke')
        fingerprint = digest_json(provenance)
        output = result_path(RESULTS/args.phase, c)
        with output_lock(output):
            previous = list(read_jsonl(output)) if output.exists() else []
            done = validate_resume(previous, tasks, fingerprint)
            for row in previous:
                validate_result(row, c, config, args.phase, manifest, sha256_file(path))
            if len(done) == len(tasks):
                print('Уже завершено:', output)
                continue
            if model is None:
                tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
                    attn_implementation=config['attention'], local_files_only=True).to('cuda').eval()
            with output.open('a', encoding='utf-8') as handle:
                for task in tasks:
                    if task['task_id'] in done:
                        continue
                    cleanup()
                    traced = args.phase == 'smoke' and policy in ('dynamic', 'k4_v4')
                    pred, stats, trace = native_generate(model, tokenizer, task, policy,
                        residual=config['residual_length'], trace=traced)
                    checks = smoke_checks(model, tokenizer, task, policy, stats, trace,
                                          config['residual_length']) if traced else {}
                    del trace
                    row = {k: v for k, v in task.items() if k != 'input_ids'}
                    row.update(**stats, **checks, **c, pred=pred, task_sha256=digest_json(task),
                        provenance=provenance, run_fingerprint=fingerprint, code_commit=code_revision(ROOT))
                    validate_result(row, c, config, args.phase, manifest, sha256_file(path))
                    handle.write(json.dumps(row, ensure_ascii=False)+'\n'); handle.flush()
                    print(task['task_id'], policy, 'MiB=', round(stats['cache_gpu_bytes_after_query']/2**20, 3),
                          'stop=', stats['stop_reason'], 'pred=', repr(pred[:100]), flush=True)
        print('Сохранено:', output)


if __name__ == '__main__':
    main()
