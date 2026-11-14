"""Общий префикс ответа и раздельный K/V-контроль stock HQQ; отдельная серия."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import re
import time

from src.common import read_jsonl
from src.download_data import sha256_file
from src.expansion import ROOT, load_config as expansion_config, validate_row as validate_reference
from src.protocol import digest_json, result_path, validate_task
from src.run_quality import make_cache

PREPARED = Path('data/prepared/kv_probe_v1')
RESULTS = Path('results/kv_probe_v1')
FILES = ('kv_probe.py', 'prepare_kv_probe.py', 'run_kv_probe.py',
         'run_quality.py', 'run_expansion.py', 'expansion.py', 'protocol.py',
         'run_protocol.py', 'common.py')


def load_config():
    config = json.loads((ROOT/'experiments/kv_probe_v1.json').read_text())
    if (config['series'] != 'kv_probe_v1' or config['reference_series'] != 'expansion_v1'
            or config['models'] != ['qwen3-1.7b', 'qwen3-4b']
            or config['dataset'] != 'passage_retrieval_en' or config['protocol'] != 'context_then_question'
            or config['policies'] != ['dynamic', 'k_g32', 'v_g32', 'kv_g32', 'k_g64', 'v_g64', 'kv_g64']
            or config['n'] != 20 or config['smoke_n'] != 3 or config['residual_length'] != 512
            or config['forced_prefix'] != 'The abstract is from Paragraph '
            or config['dtype'] != 'float16' or config['attention'] != 'sdpa'):
        raise ValueError('Изменён протокол kv_probe_v1; нужна отдельная серия')
    return config


def cell(model, policy, config):
    return dict(model_name=model, dataset=config['dataset'], protocol=config['protocol'], cache=policy)


def source_hashes():
    # Частные методы cache привязаны к содержимому реально установленной реализации.
    from transformers import cache_utils
    from hqq.core import quantize
    return {**{name: sha256_file(ROOT/'src'/name) for name in FILES},
            'transformers.cache_utils': sha256_file(Path(inspect.getfile(cache_utils))),
            'hqq.core.quantize': sha256_file(Path(inspect.getfile(quantize)))}


def versions():
    return {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'hqq')}


def prepare_task(task, tokenizer, config):
    validate_task(task)
    numbers = {int(n) for answer in task['answers'] for n in re.findall(r'\d+', answer)}
    if len(numbers) != 1:
        raise ValueError('Нужен один эталонный номер абзаца')
    gold = str(next(iter(numbers)))
    prefix = tokenizer.encode(config['forced_prefix'], add_special_tokens=False)
    full = tokenizer.encode(config['forced_prefix']+gold, add_special_tokens=False)
    if not prefix or full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError('Tokenization префикса зависит от номера; протокол надо пересмотреть')
    target = full[len(prefix):]
    if tokenizer.decode(target, skip_special_tokens=False) != gold:
        raise ValueError('Целевые токены должны кодировать только номер')
    query = len(task['input_ids'])-task['context_split']
    if query+len(prefix)+len(target) >= config['residual_length']:
        raise ValueError('Недостаточный residual: query + prefix + target >= 512')
    return {**task, 'forced_prefix_ids': prefix, 'target_ids': target, 'gold_number': int(gold)}


def load_reference(model, tasks, manifest, manifest_path, selection):
    """Три старых split-ячейки: только проверка и первый свободный токен."""
    config = expansion_config()
    refs, hashes = {}, {}
    expected = set(manifest['datasets']['passage_retrieval_en']['task_metadata'])
    for cache in ('dynamic', 'int4_g32', 'int4_g64'):
        c = dict(model_name=model, dataset='passage_retrieval_en', protocol='context_then_question', cache=cache)
        path = result_path('results/expansion_v1/full', c)
        rows = list(read_jsonl(path))
        if len(rows) != len(expected) or {r['task_id'] for r in rows} != expected:
            raise ValueError(f'Неполная/повторная reference-ячейка: {path}')
        for row in rows:
            validate_reference(row, c, config, 'full', manifest, manifest_path, selection)
        by_id = {r['task_id']: r for r in rows}
        refs[cache] = {t['task_id']: by_id[t['task_id']]['output_ids'][0] for t in tasks}
        hashes[str(path)] = sha256_file(path)
    return refs, hashes


def load_prepared(model, config, root=PREPARED):
    folder = Path(root)/model
    path = folder/'manifest.json'
    manifest = json.loads(path.read_text())
    tasks = list(read_jsonl(folder/'tasks.jsonl'))
    if (manifest['config_sha256'] != digest_json(config) or manifest['model_name'] != model
            or manifest['tasks_sha256'] != sha256_file(folder/'tasks.jsonl')
            or len(tasks) != config['n'] or len({t['task_id'] for t in tasks}) != len(tasks)
            or [t['task_id'] for t in tasks] != manifest['task_ids']):
        raise ValueError('Входы kv_probe не совпали с manifest')
    for task in tasks:
        validate_task(task)
        if digest_json(task) != manifest['task_hashes'][task['task_id']]:
            raise ValueError('Изменена задача kv_probe')
    return tasks, manifest, path


@dataclass
class ExactState:
    tensor: object


def selective_cache(model, policy, residual_length):
    """Stock update; для точной стороны обходим только quantize/dequantize.

    Порядок K,V проверяется счётчиком: допустимы только первые два вызова.
    Любой refresh останавливает пробу. Политика exact служит smoke-контролем.
    """
    if policy == 'dynamic':
        cache = make_cache(model, 'dynamic', residual_length)
        return cache, [[0, 0] for _ in cache.layers]
    match = re.fullmatch(r'(k|v|kv|exact)_g(32|64)', policy)
    if not match:
        raise ValueError(f'Неизвестная политика K/V: {policy}')
    side, group = match.groups()
    cache = make_cache(model, f'int4_g{group}', residual_length)
    calls = [[0, 0] for _ in cache.layers]
    for index, layer in enumerate(cache.layers):
        quantize, dequantize = layer._quantize, layer._dequantize
        state = {'calls': 0}

        def encode(tensor, axis, *, quantize=quantize, index=index, state=state):
            slot = state['calls']
            if slot >= 2:
                raise ValueError('Неожиданный refresh во время teacher forcing')
            state['calls'] += 1
            selected = side == 'kv' or side == ('k' if slot == 0 else 'v')
            if selected:
                calls[index][slot] += 1
                return quantize(tensor, axis=axis)
            return ExactState(tensor.clone())

        def decode(packed, *, dequantize=dequantize):
            return packed.tensor if isinstance(packed, ExactState) else dequantize(packed)

        layer._quantize, layer._dequantize = encode, decode
    return cache, calls


def evaluate(model, task, cache, calls):
    """Никакого свободного decode: query, общий prefix, затем gold[:-1]."""
    import torch
    device = model.get_input_embeddings().weight.device
    ids = task['input_ids']
    split = task['context_split']
    prefix, target = task['forced_prefix_ids'], task['target_ids']
    if device.type == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()

    def forward(part, offset, keep=1):
        return model(input_ids=torch.tensor([part], device=device),
            attention_mask=torch.ones((1, offset+len(part)), dtype=torch.long, device=device),
            cache_position=torch.arange(offset, offset+len(part), device=device),
            past_key_values=cache, use_cache=True, logits_to_keep=keep)

    with torch.inference_mode():
        output = forward(ids[:split], 0)
        del output
        output = forward(ids[split:], split)
        query_first = int(output.logits[0, -1].argmax())
        del output
        output = forward(prefix, len(ids))
        logits = [output.logits[0, -1].float()]
        del output
        if len(target) > 1:
            output = forward(target[:-1], len(ids)+len(prefix), keep=len(target)-1)
            logits.extend(output.logits[0].float().unbind())
            del output
        logits = torch.stack(logits)
        if not torch.isfinite(logits).all():
            raise ValueError('Неконечные logits teacher forcing')
        gold = torch.tensor(target, device=device)
        nll = -logits.log_softmax(-1).gather(-1, gold[:, None]).squeeze(-1)
        others = logits[0].clone()
        others[target[0]] = -torch.inf
        margin = float(logits[0, target[0]]-others.max())
        stats = {'gold_nll_per_token': nll.tolist(), 'gold_nll_mean': float(nll.mean()),
                 'gold_nll_sum': float(nll.sum()), 'first_number_token_margin': margin,
                 'first_number_token_top1': int(logits[0].argmax()),
                 'query_first_token': query_first, 'hqq_calls_by_layer': calls}
        if device.type == 'cuda':
            torch.cuda.synchronize()
        stats['probe_sec'] = time.perf_counter()-started
        return stats, logits


def parity(left, right):
    import torch
    if left.shape != right.shape or not torch.equal(left, right):
        difference = float((left-right).abs().max()) if left.shape == right.shape else None
        raise ValueError(f'Не пройдена точная сверка logits stock/cache: max_abs={difference}')
    return True


def validate_result(row, model, policy, config, phase, manifest, manifest_hash):
    p = row['provenance']
    tid = row['task_id']
    if (p.get('model_name') != model or p.get('policy') != policy
            or p.get('series') != config['series'] or p.get('phase') != phase
            or p.get('config_sha256') != digest_json(config) or p.get('manifest_sha256') != manifest_hash
            or p.get('versions') != manifest['versions'] or p.get('model_artifacts') != manifest['model_artifacts']
            or row.get('run_fingerprint') != digest_json(p) or row.get('policy') != policy
            or row.get('task_sha256') != manifest['task_hashes'].get(tid)):
        raise ValueError('Неверные условия/входы результата kv_probe')
    nll = row.get('gold_nll_per_token', [])
    if (not nll or len(nll) != len(row['target_ids'])
            or any(not math.isfinite(x) or x < 0 for x in nll)
            or not math.isclose(sum(nll), row['gold_nll_sum'], abs_tol=1e-5, rel_tol=1e-6)
            or not math.isclose(sum(nll)/len(nll), row['gold_nll_mean'], abs_tol=1e-5, rel_tol=1e-6)
            or not math.isfinite(row['first_number_token_margin'])
            or not math.isfinite(row['probe_sec']) or row['probe_sec'] < 0):
        raise ValueError('Неполные/повреждённые NLL')
    expected = [int(policy.startswith(('k_', 'kv_'))), int(policy.startswith(('v_', 'kv_')))]
    if row.get('hqq_calls_by_layer') != [expected for _ in range(manifest['hidden_layers'])]:
        raise ValueError('Квантизованы другие стороны или произошёл refresh')
    if phase == 'smoke' and policy in ('dynamic', 'kv_g32', 'kv_g64'):
        if row.get('stock_logits_match') is not True or row.get('stock_query_token_match') is not True:
            raise ValueError('Нет smoke-сверки с исходной реализацией')


def read_matrix(config, phase, prepared_root=PREPARED, results_root=RESULTS):
    matrix, common, task_sets = {}, None, []
    for model in config['models']:
        tasks, manifest, manifest_path = load_prepared(model, config, prepared_root)
        tasks = tasks[:config['smoke_n']] if phase == 'smoke' else tasks
        expected = {t['task_id']: t for t in tasks}
        task_sets.append({t: (v['context_sha256'], v['gold_number']) for t, v in expected.items()})
        for policy in config['policies']:
            path = result_path(Path(results_root)/phase, cell(model, policy, config))
            rows = {}
            for row in read_jsonl(path):
                tid = row['task_id']
                if tid in rows or tid not in expected:
                    raise ValueError(f'Повторная/чужая задача: {path}')
                validate_result(row, model, policy, config, phase, manifest, sha256_file(manifest_path))
                task = expected[tid]
                for field in ('context_sha256', 'target_ids', 'gold_number', 'input_ids_sha256'):
                    if row[field] != task[field]:
                        raise ValueError(f'Изменена задача: {tid}/{field}')
                env = {k: row['provenance'][k] for k in ('versions', 'source_hashes', 'gpu', 'cuda', 'config_sha256')}
                if common is not None and common != env:
                    raise ValueError('Между пробами изменились код/среда')
                common = env
                rows[tid] = row
            if set(rows) != set(expected) or len({r['run_fingerprint'] for r in rows.values()}) != 1:
                raise ValueError(f'Неполная/смешанная ячейка: {path}')
            matrix[model, policy] = rows
    if any(s != task_sets[0] for s in task_sets):
        raise ValueError('Модели исследуются на разных документах/эталонах')
    return matrix
