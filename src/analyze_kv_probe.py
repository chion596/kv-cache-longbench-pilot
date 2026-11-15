"""CPU-диагностика K/V: разброс, прямые контрасты и связь со свободным ответом."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from src.common import read_jsonl
from src.diagnose_family_retrieval import first_int
from src.download_data import sha256_file
from src.kv_probe import load_config, load_prepared, read_matrix
from src.score import write_csv
from src.score_protocol import cluster_ci


def interval(values, rows):
    groups = {t: rows[t]['context_sha256'] for t in values}
    low, high = cluster_ci(values, groups)
    return {'n': len(values), 'documents': len(set(groups.values())),
            'delta': sum(values.values())/len(values), 'cluster_ci95_low': low, 'cluster_ci95_high': high}


def load_native_references(matrix, config):
    native = {}
    for model in config['models']:
        tasks, manifest, _ = load_prepared(model, config)
        expected = {t['task_id']: t for t in tasks}
        for filename, digest in manifest['reference_results_sha256'].items():
            path = Path(filename)
            if sha256_file(path) != digest:
                raise ValueError(f'Изменён исходный JSONL: {path}')
            for row in read_jsonl(path):
                tid = row['task_id']
                if tid not in expected:
                    continue
                policy = row['cache'].replace('int4_', 'kv_')
                key = model, policy
                native.setdefault(key, {})
                if tid in native[key] or any(row[f] != expected[tid][f] for f in ('input_ids_sha256', 'context_sha256')):
                    raise ValueError('Повторные/несопоставимые native-ответы')
                native[key][tid] = row
        for policy in ('dynamic', 'kv_g32', 'kv_g64'):
            if set(native.get((model, policy), {})) != set(matrix[model, policy]):
                raise ValueError('Неполный native-набор на тех же документах')
    return native


def analyze_details(matrix, config, native):
    models, policies = config['models'], config['policies']
    sample = matrix[models[0], 'dynamic']
    common = {t for t in sample if all(matrix[m, 'dynamic'][t]['first_number_token_top1'] ==
              matrix[m, 'dynamic'][t]['target_ids'][0] for m in models)}
    spread, bridge, contrasts = [], [], []
    for model in models:
        base = matrix[model, 'dynamic']
        for policy in policies:
            rows = matrix[model, policy]
            if policy != 'dynamic':
                values = {t: rows[t]['gold_nll_mean']-base[t]['gold_nll_mean'] for t in rows}
                total = sum(values.values())
                loo = [(total-v)/(len(values)-1) for v in values.values()] if len(values) > 1 else []
                absolute = sum(abs(v) for v in values.values())
                spread.append({'model': model, 'policy': policy, 'n': len(rows),
                    'delta_nll': total/len(values),
                    'nll_increase_gt_1e_minus6_count': sum(v > 1e-6 for v in values.values()),
                    'nll_increase_ge_1_nat_count': sum(v >= 1 for v in values.values()),
                    'largest_absolute_delta_share': max(map(abs, values.values()))/absolute if absolute else 0,
                    'leave_one_out_min_delta': min(loo) if loo else None,
                    'leave_one_out_max_delta': max(loo) if loo else None})
            if policy in ('dynamic', 'kv_g32', 'kv_g64'):
                refs = native[model, policy]
                if set(refs) != set(rows):
                    raise ValueError('Native и forced сравниваются на разных задачах')
                free = {t: first_int(refs[t]['pred']) == rows[t]['gold_number'] for t in rows}
                # Достаточный диагностический порог p(gold token | gold prefix) > 1/2.
                # Не проверяет продолжение после номера, пунктуацию и EOS.
                high = {t: all(v < math.log(2) for v in rows[t]['gold_nll_per_token']) for t in rows}
                bridge.append({'model': model, 'policy': policy, 'n': len(rows),
                    'native_first_number_correct': sum(free.values()),
                    'forced_first_number_token_top1_correct': sum(r['first_number_token_top1'] == r['target_ids'][0] for r in rows.values()),
                    'all_gold_number_tokens_p_above_half_count': sum(high.values()),
                    'native_wrong_and_all_gold_tokens_p_above_half': sum(not free[t] and high[t] for t in rows),
                    'query_token_mismatch_with_native': sum(rows[t]['query_first_token'] != refs[t]['output_ids'][0] for t in rows)})
        for metric in ('gold_nll_mean', 'first_number_token_margin'):
            comparisons = []
            for group in (32, 64):
                comparisons.append((f'kv_g{group}-k_g{group}', 'all20',
                    {t: matrix[model, f'kv_g{group}'][t][metric]-matrix[model, f'k_g{group}'][t][metric] for t in base}))
                if common:
                    comparisons.append((f'k_g{group}-v_g{group}', 'common_dynamic_first_token_correct',
                        {t: matrix[model, f'k_g{group}'][t][metric]-matrix[model, f'v_g{group}'][t][metric] for t in common}))
            for side in ('k', 'v', 'kv'):
                comparisons.append((f'{side}_g64-{side}_g32', 'all20',
                    {t: matrix[model, f'{side}_g64'][t][metric]-matrix[model, f'{side}_g32'][t][metric] for t in base}))
            for comparison, scope, values in comparisons:
                contrasts.append({'model': model, 'metric': metric, 'comparison': comparison, 'scope': scope,
                                  **interval(values, base)})
    if len(models) == 2:
        left, right = models
        for metric in ('gold_nll_mean', 'first_number_token_margin'):
            for group in (32, 64):
                values = {t: (matrix[right, f'k_g{group}'][t][metric]-matrix[right, f'v_g{group}'][t][metric])-
                             (matrix[left, f'k_g{group}'][t][metric]-matrix[left, f'v_g{group}'][t][metric]) for t in sample}
                contrasts.append({'model': right, 'metric': metric,
                    'comparison': f'k_minus_v_g{group}:{right}-{left}', 'scope': 'all20', **interval(values, sample)})
    return spread, bridge, contrasts, len(common)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='results/kv_probe_v1/details/full')
    args = parser.parse_args()
    config = load_config()
    matrix = read_matrix(config, 'full')
    spread, bridge, contrasts, common_n = analyze_details(matrix, config, load_native_references(matrix, config))
    out = Path(args.out_dir)
    for name, rows in (('effect_spread.csv', spread), ('free_vs_forced.csv', bridge), ('additional_contrasts.csv', contrasts)):
        write_csv(out/name, rows)
    print('CPU-разбор 280 сохранённых проб. Общий Dynamic-first-token-correct subset:', common_n)
    print('Разброс NLL both относительно Dynamic:')
    for row in spread:
        if row['policy'].startswith('kv_'):
            print(row)
    print('\nNative первый номер и forced gold-токены:')
    for row in bridge:
        print(row)
    print('\nДополнительные парные NLL-контрасты:')
    for row in contrasts:
        if row['metric'] == 'gold_nll_mean' and ('kv_g' in row['comparison'] and '-k_g' in row['comparison']
                or row['scope'] == 'common_dynamic_first_token_correct' or row['comparison'].startswith('k_minus_v')):
            print(row)
    print('CSV:', out)
    print('LOO — чувствительность среднего, не CI. Subset зависит от Dynamic, это post-hoc диагностика.')
    print('p>1/2 для каждого gold-токена не проверяет конец номера/EOS и не заменяет native score.')


if __name__ == '__main__':
    main()
