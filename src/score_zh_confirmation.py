"""Официальный ZH score, парные контрасты и измеренные cache budgets."""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import random
import re

from src.diagnose_family_retrieval import first_int
from src.download_data import sha256_file
from src.score import score_prediction, write_csv
from src.zh_confirmation import DATASET, RESULTS, load_config, load_prepared, read_matrix


def load_metric(code):
    spec = importlib.util.spec_from_file_location('official_zh_confirmation_metrics', Path(code)/'metrics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metric = module.retrieval_zh_score
    if (metric('段落12', '段落12') != 1 or metric('段落2', '段落12') != 0
            or metric('段落12 段落2', '段落12') != .5):
        raise ValueError('Официальная ZH-метрика не прошла sanity check')
    return metric


def paired_intervals(values, config):
    if not values or any(not math.isfinite(v) or not -1 <= v <= 1 for v in values.values()):
        raise ValueError('Нужны конечные парные разности scores в [-1,1]')
    data = [values[t] for t in sorted(values)]
    rng = random.Random(config['bootstrap_seed'])
    iters = config['bootstrap_iterations']
    means = sorted(sum(data[rng.randrange(len(data))] for _ in data)/len(data) for _ in range(iters))
    result = {}
    for name, confidence in [('nominal95', config['secondary_ci_level']), ('primary97_5', config['primary_ci_level'])]:
        tail = (1-confidence)/2
        result[name] = (means[int(tail*iters)], means[min(iters-1, int((1-tail)*iters))])
    return result


def analyze(matrix, config, metric):
    summaries, contrasts, budgets, examples = [], [], [], []
    scores = {k: {t: score_prediction(r, metric) for t, r in rows.items()} for k, rows in matrix.items()}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for rows in scores.values() for v in rows.values()):
        raise ValueError('ZH score вне [0,1]')
    comparisons = ['k8_v4-k4_v8', *config['secondary_comparisons']]
    for model in config['models']:
        base = matrix[model, DATASET, 'dynamic']
        if len({r['context_sha256'] for r in base.values()}) != len(base):
            raise ValueError('ZH-серия требует один вопрос на context')
        for policy in config['policies']:
            rows, values = matrix[model, DATASET, policy], scores[model, DATASET, policy]
            delta = {t: values[t]-scores[model, DATASET, 'dynamic'][t] for t in base}
            mean = lambda field: sum(r[field] for r in rows.values())/len(rows)
            summaries.append({'model': model, 'dataset': DATASET, 'policy': policy, 'n': len(rows),
                'documents': len(rows), 'score_pct': 100*sum(values.values())/len(rows),
                'eos_count': sum(r['stop_reason'] == 'eos' for r in rows.values()),
                'first_number_correct': sum(first_int(r['pred']) in {int(re.fullmatch(r'段落(\d+)', a)[1])
                    for a in r['answers']} for r in rows.values()),
                'mean_output_tokens': mean('output_tokens'),
                'mean_cache_mib_after_query': mean('cache_gpu_bytes_after_query')/2**20,
                'mean_cache_ratio_vs_dynamic': sum(rows[t]['cache_gpu_bytes_after_query']/base[t]['cache_gpu_bytes_after_query'] for t in base)/len(base),
                'middle_truncated': sum(r['middle_truncated'] for r in rows.values()),
                'max_query_tokens': max(r['input_tokens']-r['context_split'] for r in rows.values()),
                'score_lower_vs_dynamic': sum(v < -1e-12 for v in delta.values()),
                'score_equal_vs_dynamic': sum(abs(v) <= 1e-12 for v in delta.values()),
                'score_higher_vs_dynamic': sum(v > 1e-12 for v in delta.values()),
                'identical_output_ids_vs_dynamic': sum(rows[t]['output_ids'] == base[t]['output_ids'] for t in base),
                'mean_prefill_sec': mean('prefill_sec'), 'mean_decode_sec': mean('decode_sec')})
            for tid, row in rows.items():
                examples.append({'model': model, 'policy': policy, 'task_id': tid, 'context_sha256': row['context_sha256'],
                    'score_pct': 100*values[tid], 'delta_vs_dynamic_pp': 100*delta[tid],
                    'output_ids_equal_dynamic': row['output_ids'] == base[tid]['output_ids'],
                    'cache_gpu_bytes_after_query': row['cache_gpu_bytes_after_query']})
        a, b = matrix[model, DATASET, 'k8_v4'], matrix[model, DATASET, 'k4_v8']
        diff = [abs(a[t]['cache_gpu_bytes_after_query']-b[t]['cache_gpu_bytes_after_query']) for t in base]
        if any(diff):
            raise ValueError('Primary не имеет равного измеренного бюджета памяти')
        budgets.append({'model': model, 'n': len(base), 'equal_bytes_count': sum(d == 0 for d in diff),
                        'max_absolute_bytes_difference': max(diff)})
        for comparison in comparisons:
            left, right = comparison.split('-')
            delta = {t: scores[model, DATASET, left][t]-scores[model, DATASET, right][t] for t in base}
            cis = paired_intervals(delta, config)
            primary = comparison == 'k8_v4-k4_v8'
            low, high = cis['primary97_5' if primary else 'nominal95']
            nominal = cis['nominal95']
            contrasts.append({'model': model, 'dataset': DATASET, 'comparison': comparison,
                'role': 'primary' if primary else 'secondary', 'n': len(delta), 'documents': len(delta),
                'delta_pp': 100*sum(delta.values())/len(delta),
                'ci_level': config['primary_ci_level'] if primary else config['secondary_ci_level'],
                'ci_low_pp': 100*low, 'ci_high_pp': 100*high,
                'nominal95_low_pp': 100*nominal[0], 'nominal95_high_pp': 100*nominal[1],
                'multiplicity': 'Bonferroni, 2 primary contrasts' if primary else 'exploratory, unadjusted'})
    return summaries, contrasts, budgets, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--longbench-code', default='external/LongBench/LongBench')
    args = parser.parse_args()
    config = load_config()
    matrix = read_matrix(config, args.phase)
    print('ZH-матрица проверена:', len(matrix), 'ячеек,', sum(map(len, matrix.values())), 'ответов')
    print('residual1024, нужная битность, нет refresh; K8/V4 и K4/V8 равны по GPU cache bytes на каждом входе')
    if args.phase == 'smoke':
        print('Самый длинный вопрос включён; stock4/8 и exact/Dynamic logits/output IDs совпали')
    if args.check_only:
        return
    metric_hash = sha256_file(Path(args.longbench_code)/'metrics.py')
    for model in config['models']:
        _, manifest, _ = load_prepared(model, config)
        if manifest['longbench_files_sha256']['metrics.py'] != metric_hash:
            raise ValueError('Официальная метрика изменилась после подготовки')
    summary, contrasts, budgets, examples = analyze(matrix, config, load_metric(args.longbench_code))
    out = RESULTS/'summary'/args.phase
    for name, rows in [('dataset_scores.csv', summary), ('paired_contrasts.csv', contrasts),
                       ('memory_budget.csv', budgets), ('example_scores.csv', examples)]:
        write_csv(out/name, rows)
    for row in summary:
        print(row)
    for row in contrasts:
        print(row)
    hours = sum((r['prefill_sec']+r['decode_sec']) for rows in matrix.values() for r in rows.values())/3600
    print('Основные forward:', round(hours, 3), 'GPU-часа; без загрузки/GC/очереди/сверок; не benchmark throughput')
    if args.phase == 'smoke':
        full = sum((r['mean_prefill_sec']+r['mean_decode_sec'])*config['n'] for r in summary)/3600
        print('Грубая full-экстраполяция по stress-smoke:', round(full, 2), 'GPU-часа; выборка неслучайная, оценка неточная')
        print('Smoke проверяет механику, качество по трём выбранным длинам не обобщаем; full отправляется отдельно')
    print('CSV:', out, '; primary 97.5% CI (Bonferroni для двух моделей), secondary nominal 95%')
    print('Нулевая эмпирическая разность не доказывает эквивалентность Dynamic или отсутствие будущих потерь')


if __name__ == '__main__':
    main()
