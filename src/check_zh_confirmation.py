"""CPU-проверка новой ZH-выборки и поэлементная сводка завершённого full; без GPU."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import zipfile

from src.audit_expansion import Catalog, audit, save_report
from src.download_data import sha256_file
from src.mixed_precision import PREPARED, ROOT, load_config, read_matrix, versions
from src.prepare_data import read_dataset_from_zip
from src.prepare_protocol import prepare_chat_task
from src.protocol import digest_json, model_signature
from src.score import score_prediction
from src.score_expansion import load_metrics

DATASET = 'passage_retrieval_zh'


def tokenization_summary(raw, indices, template, tokenizer):
    query, truncated, inputs = [], 0, []
    for i in indices:
        source = raw[i]
        if not source['answers'] or any(not re.fullmatch(r'段落\d+', answer) for answer in source['answers']):
            raise ValueError('ZH gold не соответствует официальному формату 段落N')
        row = prepare_chat_task(source, i, DATASET, template, 32, tokenizer, 16384)
        query.append(row['input_tokens']-row['context_split'])
        inputs.append(row['input_tokens'])
        truncated += row['middle_truncated']
    if not query:
        raise ValueError('Нет документов для tokenizer precheck')
    return {'n': len(query), 'max_query_tokens': max(query), 'max_query_plus_answer_tokens': max(query)+32,
        'query_plus_answer_ge_512_count': sum(q+32 >= 512 for q in query),
        'middle_truncated': truncated, 'mean_input_tokens': sum(inputs)/len(inputs),
        'max_input_tokens': max(inputs), 'minimum_residual_strict': max(query)+33}


def native_pair_details(matrix, metrics):
    result = []
    for (model, dataset, policy), rows in matrix.items():
        if policy != 'k8_v4':
            continue
        base = matrix[model, dataset, 'dynamic']
        if set(rows) != set(base):
            raise ValueError('Сравниваются разные документы')
        delta = {t: score_prediction(rows[t], metrics[dataset])-score_prediction(base[t], metrics[dataset]) for t in rows}
        result.append({'model': model, 'dataset': dataset, 'n': len(rows),
            'score_lower_count': sum(v < -1e-12 for v in delta.values()),
            'score_equal_count': sum(abs(v) <= 1e-12 for v in delta.values()),
            'score_higher_count': sum(v > 1e-12 for v in delta.values()),
            'output_ids_identical': sum(rows[t]['output_ids'] == base[t]['output_ids'] for t in rows),
            'mean_score_delta_pp': 100*sum(delta.values())/len(delta)})
    return result


def audit_ready(report, target):
    return (report.get('local_history_scan_passed') is True and not report.get('history_scan_errors')
            and len(report['datasets']) == 1 and report['datasets'][0]['preview_n'] == target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qwen17', required=True)
    parser.add_argument('--qwen4', required=True)
    parser.add_argument('--archive', default='data/raw/longbench-data.zip')
    parser.add_argument('--longbench-code', default='external/LongBench/LongBench')
    parser.add_argument('--output', default='results/zh_confirmation_preflight_2026-11-17.json')
    args = parser.parse_args()
    n, seed = 120, 2043
    published_path = ROOT/'docs/data_independence_audit_2026-11-03.json'
    published = json.loads(published_path.read_text())
    archive_hash = sha256_file(Path(args.archive))
    if archive_hash != published['archive_sha256_from_upstream_lfs']:
        raise ValueError('Архив отличается от зафиксированного LongBench')
    code = Path(args.longbench_code)
    official_hashes = {f: sha256_file(code/f) for f in ('config/dataset2prompt.json', 'config/dataset2maxlen.json', 'metrics.py')}
    templates = json.loads((code/'config/dataset2prompt.json').read_text())
    limits = json.loads((code/'config/dataset2maxlen.json').read_text())
    if limits[DATASET] != 32:
        raise ValueError('Изменён официальный предел ответа ZH retrieval')
    models = []
    for name, path in [('qwen3-1.7b', args.qwen17), ('qwen3-4b', args.qwen4)]:
        manifest_path = PREPARED/name/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if (manifest['model_artifacts'] != model_signature(path) or manifest['versions'] != versions()
                or manifest['longbench_files_sha256'] != official_hashes):
            raise ValueError('Модель/tokenizer/версии/официальный код отличаются от завершённой серии')
        models.append((name, path, manifest_path))
    current = read_matrix(load_config(), 'full')
    details = native_pair_details(current, load_metrics(code))
    spec = importlib.util.spec_from_file_location('official_zh_preflight_metrics', code/'metrics.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    metric = module.retrieval_zh_score
    if (metric('段落12', '段落12') != 1 or metric('段落2', '段落12') != 0
            or metric('段落12 段落2', '段落12') != .5):
        raise ValueError('Не пройдена sanity-сверка официальной ZH-метрики')
    with zipfile.ZipFile(args.archive) as archive:
        catalog = Catalog(archive)
        history = audit(catalog, published, ['data/prepared', 'results'], [DATASET], n, seed)
        indices = history['datasets'][0]['preview_source_indices']
        raw = read_dataset_from_zip(archive, DATASET)
        summaries = []
        if indices and not history['history_scan_errors']:
            from transformers import AutoTokenizer
            for name, path, manifest_path in models:
                tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                summaries.append({'model': name, 'parent_manifest_sha256': sha256_file(manifest_path),
                    'chat_template_sha256': digest_json(tokenizer.chat_template),
                    **tokenization_summary(raw, indices, templates[DATASET], tokenizer)})
        context_hashes = {str(i): hashlib.sha256(raw[i]['context'].encode()).hexdigest() for i in indices}
    report = {'purpose': 'Preview for an independent-document ZH confirmation; no inference or allocation frozen',
        'n': n, 'seed': seed, 'dataset': DATASET, 'archive_sha256': archive_hash,
        'source_revision': published['revision'], 'published_audit_sha256': sha256_file(published_path),
        'official_files_sha256': official_hashes, 'source_sha256': sha256_file(Path(__file__)),
        'audit_source_sha256': sha256_file(ROOT/'src/audit_expansion.py'),
        'history': history, 'preview_context_hashes': context_hashes, 'tokenizer_precheck': summaries,
        'current_k8v4_vs_dynamic': details, 'local_history_ready': audit_ready(history, n),
        'all_query_budgets_fit_residual512': len(summaries) == 2 and all(s['query_plus_answer_ge_512_count'] == 0 for s in summaries),
        'limitations': ['Preview only; not a prepared inference series.',
            'Exact context hashes do not detect semantic duplicates, translations or pretraining exposure.',
            'EN and ZH contain different documents; comparison cannot isolate a causal language effect.',
            'Existing full statistics are development-data diagnostics, not an independent confirmation.']}
    save_report(args.output, report)
    print('K8/V4 против Dynamic на завершённой EN-серии, поэлементная сводка:')
    for row in details:
        print(row)
    print('Свежие ZH-контексты:', {k: v for k, v in history['datasets'][0].items() if k not in ('member_sha256', 'preview_source_indices')})
    print('История:', history['history_coverage'], 'JSONL=', history['history_jsonl_files'],
          'записей=', history['recognized_history_rows'], 'ошибок=', len(history['history_scan_errors']))
    for row in summaries:
        print(row)
    print('Подходит 120 новых контекстов:', report['local_history_ready'])
    print('Все query+32 < residual512:', report['all_query_budgets_fit_residual512'])
    print('JSON:', args.output)
    print('Только CPU-preflight. GPU не запускается; исходные документы и результаты не изменяются.')
    if not report['local_history_ready'] or not report['all_query_budgets_fit_residual512']:
        for error in history['history_scan_errors'][:3]:
            print(error)
        parser.exit(2, 'Preflight требует разбора: доступность контекстов или residual. Выборку не фильтруем по результатам.\n')


if __name__ == '__main__':
    main()
