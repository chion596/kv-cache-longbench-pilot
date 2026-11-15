"""Slurm job на модель/набор: пять политик, свежие baselines и cache bytes."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from src.common import read_jsonl
from src.download_data import sha256_file
from src.mixed_cache import assert_parity, native_generate
from src.mixed_precision import (ROOT, RESULTS, cell, load_config, load_prepared, read_matrix,
    read_references, source_hashes, validate_result, versions)
from src.protocol import code_revision, digest_json, model_signature, result_path
from src.run_protocol import output_lock, validate_resume


def current_inputs(model, dataset, model_path, config):
    tasks, manifest, path = load_prepared(model, dataset, config)
    if manifest['versions'] != versions() or manifest['model_artifacts'] != model_signature(model_path):
        raise ValueError('Модель/tokenizer/версии изменились после подготовки')
    pairs = [(manifest['source_manifest_path'], manifest['source_manifest_sha256']),
             (manifest['selection_path'], manifest['selection_sha256']),
             (manifest['datasets'][dataset]['source_path'], manifest['datasets'][dataset]['source_sha256'])]
    if (any(sha256_file(Path(p)) != h for p, h in pairs)
            or sha256_file(ROOT/'src/prepare_mixed_precision.py') != manifest['prepare_source_sha256']):
        raise ValueError('Изменён источник или код подготовки')
    return tasks, manifest, path, read_references(manifest, dataset, tasks)


def cleanup():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stock_factory(bits, residual):
    from transformers import QuantizedCache
    def factory(model):
        return QuantizedCache(backend='hqq', config=model.config, nbits=bits, axis_key=1,
            axis_value=1, q_group_size=64, residual_length=residual), None
    return factory


def smoke_checks(model, tokenizer, task, policy, stats, trace, residual):
    checks = {}
    if policy == 'dynamic':
        cleanup()
        _, other, values = native_generate(model, tokenizer, task, 'k16_v16', residual=residual, trace=True)
        checks['exact_logits_match'] = assert_parity(stats, trace, other, values)
        del values
        cleanup()
        _, stock, reference = native_generate(model, tokenizer, task, 'k8_v8', residual=residual,
            trace=True, factory=stock_factory(8, residual))
        cleanup()
        _, mixed, values = native_generate(model, tokenizer, task, 'k8_v8', residual=residual, trace=True)
        checks['stock8_logits_match'] = assert_parity(stock, reference, mixed, values)
    elif policy == 'k4_v4':
        cleanup()
        _, other, values = native_generate(model, tokenizer, task, policy, residual=residual,
            trace=True, factory=stock_factory(4, residual))
        checks['stock4_logits_match'] = assert_parity(stats, trace, other, values)
    return checks


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', choices=config['models'], required=True)
    parser.add_argument('--dataset', choices=config['datasets'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    tasks, manifest, path, references = current_inputs(args.name, args.dataset, args.model, config)
    sources = source_hashes()
    smoke = read_matrix(config, 'smoke') if args.phase == 'full' else None
    if smoke is not None:
        old = next(iter(smoke[args.name, args.dataset, 'dynamic'].values()))['provenance']
        if old['source_hashes'] != sources:
            raise ValueError('Код изменился после smoke; full не отправляем')
    if args.phase == 'smoke':
        tasks = tasks[:config['smoke_n']]
    if args.check_data:
        print('Mixed precision: inputs/версии/модель/source baselines проверены:', args.name, args.dataset,
              'n=', len(tasks), 'policies=', len(config['policies']))
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
        c = cell(args.name, args.dataset, policy, config)
        provenance = {**base, **c}
        if smoke is not None:
            old = next(iter(smoke[args.name, args.dataset, policy].values()))['provenance']
            if {k: v for k, v in old.items() if k != 'phase'} != {k: v for k, v in provenance.items() if k != 'phase'}:
                raise ValueError('Условия full отличаются от smoke')
        fingerprint = digest_json(provenance)
        output = result_path(RESULTS/args.phase, c)
        with output_lock(output):
            previous = list(read_jsonl(output)) if output.exists() else []
            completed = validate_resume(previous, tasks, fingerprint)
            for row in previous:
                validate_result(row, c, config, args.phase, manifest, sha256_file(path))
            if len(completed) == len(tasks):
                print('Уже завершено:', output)
                continue
            if model is None:
                tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
                    attn_implementation=config['attention'], local_files_only=True).to('cuda').eval()
            with output.open('a', encoding='utf-8') as handle:
                for task in tasks:
                    if task['task_id'] in completed:
                        continue
                    cleanup()
                    traced = args.phase == 'smoke' and policy in ('dynamic', 'k4_v4')
                    pred, stats, trace = native_generate(model, tokenizer, task, policy,
                        residual=config['residual_length'], trace=traced)
                    checks = smoke_checks(model, tokenizer, task, policy, stats, trace, config['residual_length']) if traced else {}
                    del trace
                    if policy in references:
                        checks['expansion_output_ids_match'] = stats['output_ids'] == references[policy][task['task_id']]['output_ids']
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
