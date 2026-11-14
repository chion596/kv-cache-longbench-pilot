"""Парные NLL-контрасты K/V; самостоятельная диагностика, не LongBench score."""
from __future__ import annotations

import argparse

from src.kv_probe import RESULTS, load_config, read_matrix
from src.score import write_csv
from src.score_protocol import cluster_ci


def analyze(matrix, config):
    summaries, contrasts = [], []
    for model in config['models']:
        base = matrix[model, 'dynamic']
        for policy in config['policies']:
            rows = matrix[model, policy]
            mean = lambda field: sum(r[field] for r in rows.values())/len(rows)
            summaries.append({'model': model, 'policy': policy, 'n': len(rows),
                'gold_nll_mean': mean('gold_nll_mean'), 'gold_nll_sum_mean': mean('gold_nll_sum'),
                'mean_probe_sec': mean('probe_sec'),
                'first_number_token_margin_mean': mean('first_number_token_margin'),
                'first_number_token_top1_correct': sum(r['first_number_token_top1'] == r['target_ids'][0] for r in rows.values()),
                'query_first_token_changed_vs_dynamic': sum(rows[t]['query_first_token'] != base[t]['query_first_token'] for t in rows),
                'expansion_query_token_mismatches': sum(r.get('expansion_query_token_match') is False for r in rows.values())
                    if policy in ('dynamic', 'kv_g32', 'kv_g64') else None})
        for group in (32, 64):
            for metric in ('gold_nll_mean', 'first_number_token_margin'):
                values = {p: {t: r[metric] for t, r in matrix[model, p].items()} for p in config['policies']}
                k, v, both = (f'{s}_g{group}' for s in ('k', 'v', 'kv'))
                comparisons = {f'{p}-dynamic': {t: values[p][t]-values['dynamic'][t] for t in base}
                               for p in (k, v, both)}
                comparisons[f'{k}-{v}'] = {t: values[k][t]-values[v][t] for t in base}
                comparisons[f'nonadditivity_g{group}:kv-k-v+dynamic'] = {
                    t: values[both][t]-values[k][t]-values[v][t]+values['dynamic'][t] for t in base}
                for comparison, delta in comparisons.items():
                    groups = {t: base[t]['context_sha256'] for t in base}
                    low, high = cluster_ci(delta, groups)
                    contrasts.append({'model': model, 'metric': metric, 'comparison': comparison,
                        'n': len(base), 'documents': len(set(groups.values())), 'delta': sum(delta.values())/len(delta),
                        'cluster_ci95_low': low, 'cluster_ci95_high': high})
    return summaries, contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config = load_config()
    matrix = read_matrix(config, args.phase)
    print('Матрица KV probe проверена:', len(matrix), 'ячеек,', sum(map(len, matrix.values())), 'оценок')
    print('Общий prefix/target и селективные HQQ-вызовы проверены; refresh отсутствует')
    if args.phase == 'smoke':
        print('Точная сверка logits both со stock HQQ и exact с Dynamic пройдена')
    if args.check_only:
        return
    summaries, contrasts = analyze(matrix, config)
    out = RESULTS/'summary'/args.phase
    write_csv(out/'dataset_scores.csv', summaries)
    write_csv(out/'paired_contrasts.csv', contrasts)
    for row in summaries:
        print(row)
    if args.phase == 'full':
        print('NLL: положительная delta = ухудшение; margin: отрицательная delta = ухудшение')
        print('CI exploratory, nominal 95%, без поправки множественных сравнений; n=20 известных документов')
    else:
        print('Smoke проверяет совместимость; качество по трём документам не обобщаем')
    print('CSV:', out)
    hours = sum(r['mean_probe_sec']*config['n'] for r in summaries)/3600
    print('Оценка времени 280 основных проб по наблюдаемым forward:', round(hours, 3), 'GPU-часа')
    print('Без загрузки модели, GC, smoke-сверок и ожидания; это не замер throughput')


if __name__ == '__main__':
    main()
