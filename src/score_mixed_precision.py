"""Native качество, парные интервалы и равенство измеренных cache budgets."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.diagnose_family_retrieval import expected_numbers, first_int
from src.download_data import sha256_file
from src.mixed_precision import RESULTS, load_config, load_prepared, read_matrix, read_references
from src.score import score_prediction, write_csv
from src.score_expansion import load_metrics
from src.score_protocol import cluster_ci

COMPARISONS = [('k8_v4', 'k4_v8'), ('k8_v4', 'k4_v4'), ('k16_v4', 'k4_v4'),
               ('k8_v4', 'dynamic'), ('k4_v8', 'dynamic'), ('k16_v4', 'dynamic'), ('k4_v4', 'dynamic')]


def analyze(matrix, config, metrics):
    summaries, contrasts, budgets = [], [], []
    scores = {k: {t: score_prediction(r, metrics[k[1]]) for t, r in rows.items()} for k, rows in matrix.items()}
    if any(not 0 <= v <= 1 for values in scores.values() for v in values.values()):
        raise ValueError('Официальная метрика вне [0,1]')
    for (model, dataset, policy), rows in matrix.items():
        base = matrix[model, dataset, 'dynamic']
        mean = lambda field: sum(r[field] for r in rows.values())/len(rows)
        summaries.append({'model': model, 'dataset': dataset, 'policy': policy, 'n': len(rows),
            'score_pct': 100*sum(scores[model, dataset, policy].values())/len(rows),
            'eos_count': sum(r['stop_reason'] == 'eos' for r in rows.values()),
            'mean_output_tokens': mean('output_tokens'),
            'mean_cache_mib_after_query': mean('cache_gpu_bytes_after_query')/2**20,
            'mean_cache_ratio_vs_dynamic': sum(rows[t]['cache_gpu_bytes_after_query']/base[t]['cache_gpu_bytes_after_query'] for t in rows)/len(rows),
            'first_number_correct': sum(first_int(r['pred']) in expected_numbers(r)
                                        for r in rows.values()) if dataset == 'passage_retrieval_en' else None,
            'mean_prefill_sec': mean('prefill_sec'), 'mean_decode_sec': mean('decode_sec')})
    for model in config['models']:
        for dataset in config['datasets']:
            base = matrix[model, dataset, 'dynamic']
            a, b = matrix[model, dataset, 'k8_v4'], matrix[model, dataset, 'k4_v8']
            diff = [abs(a[t]['cache_gpu_bytes_after_query']-b[t]['cache_gpu_bytes_after_query']) for t in base]
            budgets.append({'model': model, 'dataset': dataset, 'n': len(base),
                'k8v4_k4v8_equal_bytes_count': sum(v == 0 for v in diff), 'max_absolute_bytes_difference': max(diff)})
            for left, right in COMPARISONS:
                values = {t: scores[model, dataset, left][t]-scores[model, dataset, right][t] for t in base}
                groups = {t: base[t]['context_sha256'] for t in base}
                low, high = cluster_ci(values, groups)
                contrasts.append({'model': model, 'dataset': dataset, 'comparison': f'{left}-{right}',
                    'n': len(values), 'documents': len(set(groups.values())),
                    'delta_pp': 100*sum(values.values())/len(values),
                    'cluster_ci95_low_pp': 100*low, 'cluster_ci95_high_pp': 100*high})
    return summaries, contrasts, budgets


def untruncated(matrix, config, metrics):
    counts, contrasts = [], []
    for dataset in config['datasets']:
        bases = [matrix[m, dataset, 'dynamic'] for m in config['models']]
        ids = {t for t in bases[0] if all(not rows[t]['middle_truncated'] for rows in bases)}
        counts.append({'dataset': dataset, 'n': len(bases[0]), 'common_untruncated_n': len(ids)})
        if ids and len(ids) < len(bases[0]):
            sub = {k: {t: r for t, r in rows.items() if t in ids} for k, rows in matrix.items() if k[1] == dataset}
            _, local, _ = analyze(sub, {**config, 'datasets': [dataset]}, metrics)
            contrasts.extend(local)
    return counts, contrasts


def reference_drift(matrix, config, metrics):
    drift = []
    for model in config['models']:
        for dataset in config['datasets']:
            tasks, manifest, _ = load_prepared(model, dataset, config)
            ids = set(matrix[model, dataset, 'dynamic'])
            refs = read_references(manifest, dataset, [t for t in tasks if t['task_id'] in ids])
            for policy, old in refs.items():
                rows = matrix[model, dataset, policy]
                drift.append({'model': model, 'dataset': dataset, 'policy': policy, 'n': len(rows),
                    'different_output_ids': sum(rows[t]['output_ids'] != old[t]['output_ids'] for t in rows),
                    'different_first_token': sum(rows[t]['output_ids'][0] != old[t]['output_ids'][0] for t in rows),
                    'score_delta_vs_expansion_pp': 100*sum(score_prediction(rows[t], metrics[dataset])-
                        score_prediction(old[t], metrics[dataset]) for t in rows)/len(rows)})
    return drift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--longbench-code', default='external/LongBench/LongBench')
    args = parser.parse_args()
    config = load_config()
    matrix = read_matrix(config, args.phase)
    print('Матрица mixed precision проверена:', len(matrix), 'ячеек,', sum(map(len, matrix.values())), 'ответов')
    print('Битность и отсутствие refresh проверены; K8/V4 и K4/V8 имеют равные GPU cache bytes на каждой задаче')
    if args.phase == 'smoke':
        print('Сверки logits stock HQQ 4/8 и exact с Dynamic пройдены')
    if args.check_only:
        return
    digest = sha256_file(Path(args.longbench_code)/'metrics.py')
    for model in config['models']:
        _, manifest, _ = load_prepared(model, config['datasets'][0], config)
        if manifest['longbench_files_sha256']['metrics.py'] != digest:
            raise ValueError('metrics.py отличается от исходной серии')
    metrics = load_metrics(args.longbench_code)
    summary, contrasts, budgets = analyze(matrix, config, metrics)
    counts, sensitivity = untruncated(matrix, config, metrics)
    drift = reference_drift(matrix, config, metrics)
    out = RESULTS/'summary'/args.phase
    for name, rows in [('dataset_scores.csv', summary), ('paired_contrasts.csv', contrasts),
                       ('memory_budget.csv', budgets), ('reference_drift.csv', drift),
                       ('untruncated_counts.csv', counts), ('untruncated_contrasts.csv', sensitivity)]:
        write_csv(out/name, rows)
    for row in summary:
        print(row)
    print('Source baseline drift:')
    for row in drift:
        print(row)
    print('Common untruncated:', counts)
    hours = sum((r['mean_prefill_sec']+r['mean_decode_sec'])*config['n'] for r in summary)/3600
    print('Основные native forward full по наблюдаемым длительностям:', round(hours, 3), 'GPU-часа')
    print('Включена CPU-проверка cache snapshot, без загрузки моделей/GC/smoke-сверок/очереди; не benchmark throughput')
    print('CSV:', out, '; CI nominal 95%, exploratory, без коррекции множественных сравнений')
    if args.phase == 'smoke':
        print('Качество и бюджет по трём документам не обобщаем; full отправляется отдельно')


if __name__ == '__main__':
    main()
