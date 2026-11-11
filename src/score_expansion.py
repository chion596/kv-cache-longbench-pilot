"""Официальные метрики и парные контрасты новой серии expansion_v1."""
from __future__ import annotations

import argparse
from itertools import combinations
import math
from pathlib import Path

from src.download_data import sha256_file
from src.expansion import load_config, read_matrix
from src.score import load_official_metrics, score_prediction, write_csv
from src.score_protocol import analyze as analyze_protocol, cluster_ci


def load_metrics(code):
    metrics = load_official_metrics(code)
    # Официальный eval.py использует qa_f1_score и для MultiFieldQA EN, и для HotpotQA.
    return {'passage_retrieval_en': metrics['passage_retrieval_en'], 'hotpotqa': metrics['multifieldqa_en']}


def runtime_estimate(matrix, config):
    estimates = []
    for (model, dataset, protocol, cache), rows in matrix.items():
        durations = []
        for row in rows.values():
            values = (row.get('prefill_sec'), row.get('decode_sec'))
            if all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values):
                durations.append(sum(values))
        estimates.append({'model': model, 'dataset': dataset, 'protocol': protocol, 'cache': cache,
            'answers': len(rows), 'timed_answers': len(durations),
            'observed_generation_hours': sum(durations)/3600 if durations else None,
            'estimated_full_generation_hours': config['n']*sum(durations)/len(durations)/3600 if durations else None})
    return estimates


def analyze(matrix, config, metrics):
    summaries, contrasts, examples = analyze_protocol(matrix, config, metrics)
    contrasts = [row for row in contrasts if not row['comparison'].startswith('g64_loss:')]
    for left, right in combinations(config['models'], 2):
        for dataset in config['datasets']:
            for protocol in config['protocols']:
                for cache in ('int4_g32', 'int4_g64'):
                    a, b = matrix[(left, dataset, protocol, 'dynamic')], matrix[(right, dataset, protocol, 'dynamic')]
                    aq, bq = matrix[(left, dataset, protocol, cache)], matrix[(right, dataset, protocol, cache)]
                    delta = {t: (score_prediction(bq[t], metrics[dataset])-score_prediction(b[t], metrics[dataset]))
                                  - (score_prediction(aq[t], metrics[dataset])-score_prediction(a[t], metrics[dataset])) for t in a}
                    groups = {t: a[t]['context_sha256'] for t in a}
                    low, high = cluster_ci(delta, groups)
                    contrasts.append({'model': right, 'dataset': dataset, 'protocol': protocol,
                        'comparison': f'{cache}_loss:{right}-{left}', 'n': len(delta), 'documents': len(set(groups.values())),
                        'delta_pp': 100*sum(delta.values())/len(delta),
                        'cluster_ci95_low_pp': 100*low, 'cluster_ci95_high_pp': 100*high})
    drift = []
    for model in config['models']:
        for dataset in config['datasets']:
            a, b = matrix[(model, dataset, 'single_prefill', 'dynamic')], matrix[(model, dataset, 'context_then_question', 'dynamic')]
            drift.append({'model': model, 'dataset': dataset, 'n': len(a),
                'different_output_ids': sum(a[t]['output_ids'] != b[t]['output_ids'] for t in a),
                'different_first_token': sum(a[t]['output_ids'][0] != b[t]['output_ids'][0] for t in a),
                'delta_pp': 100*sum(score_prediction(b[t], metrics[dataset])-score_prediction(a[t], metrics[dataset]) for t in a)/len(a)})
    return summaries, contrasts, examples, drift


def common_untruncated(matrix, config, metrics):
    counts, contrasts = [], []
    for dataset in config['datasets']:
        bases = [matrix[(m, dataset, 'single_prefill', 'dynamic')] for m in config['models']]
        ids = {t for t in bases[0] if all(not base[t]['middle_truncated'] for base in bases)}
        counts.append({'dataset': dataset, 'all_questions': len(bases[0]), 'common_untruncated_questions': len(ids)})
        if not ids or len(ids) == len(bases[0]):
            continue
        subset = {k: {t: r for t, r in rows.items() if t in ids} for k, rows in matrix.items() if k[1] == dataset}
        _, local_contrasts, _, _ = analyze(subset, {**config, 'datasets': [dataset]}, metrics)
        contrasts.extend(local_contrasts)
    return counts, contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--longbench-code', default='external/LongBench/LongBench')
    args = parser.parse_args()
    config = load_config()
    matrix = read_matrix(Path('results/expansion_v1') / args.phase, config, args.phase)
    print(f'Матрица expansion проверена: {len(matrix)} ячеек, {sum(len(v) for v in matrix.values())} ответов')
    print('Повторная квантизация отсутствует; single first-token invariants проверены')
    if args.check_only:
        return
    metric_hash = sha256_file(Path(args.longbench_code) / 'metrics.py')
    if any(r['provenance']['longbench_files_sha256']['metrics.py'] != metric_hash for rows in matrix.values() for r in rows.values()):
        raise ValueError('metrics.py отличается от подготовки серии')
    metrics = load_metrics(args.longbench_code)
    summaries, contrasts, examples, drift = analyze(matrix, config, metrics)
    counts, untruncated = common_untruncated(matrix, config, metrics)
    out = Path('results/expansion_v1/summary') / args.phase
    estimates = runtime_estimate(matrix, config)
    for filename, rows in (('dataset_scores.csv', summaries), ('paired_contrasts.csv', contrasts),
                           ('example_scores.csv', examples), ('dynamic_drift.csv', drift),
                           ('runtime_estimates.csv', estimates), ('untruncated_counts.csv', counts),
                           ('untruncated_contrasts.csv', untruncated)):
        write_csv(out / filename, rows)
    for row in summaries:
        print(row)
    print('Dynamic single→split:')
    for row in drift:
        print(row)
    print('Общая подвыборка без обрезания:')
    for row in counts:
        print(row)
    if all(r['timed_answers'] == r['answers'] for r in estimates):
        hours = sum(r['estimated_full_generation_hours'] for r in estimates)
        print(f'Грубая экстраполяция времени генерации full: {hours:.2f} GPU-часов; без загрузки весов, GC и ожидания.')
        if args.phase == 'smoke':
            print('Оценка по первым трём вопросам чувствительна к их длинам; это не точный бюджет Slurm.')
    print('CSV:', out, '; CI nominal, без коррекции множественных сравнений')
    if args.phase == 'smoke':
        print('Сверки model.generate пройдены; различия Dynamic single/split разбираем отдельно.')
        print('Качество на трёх вопросах не обобщаем и не используем для выбора cache.')


if __name__ == '__main__':
    main()
