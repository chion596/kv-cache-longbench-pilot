"""CPU-разбор полной серии: общий subset, формат QA и правильность первого номера."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.diagnose_family_qa import first_line
from src.diagnose_family_retrieval import expected_numbers, first_int
from src.download_data import sha256_file
from src.expansion import load_config, read_matrix
from src.score import score_prediction, write_csv
from src.score_expansion import load_metrics
from src.score_protocol import cluster_ci


def paired_interval(values, rows):
    groups = {t: rows[t]['context_sha256'] for t in values}
    low, high = cluster_ci(values, groups)
    return {'delta_pp': 100*sum(values.values())/len(values),
            'cluster_ci95_low_pp': 100*low, 'cluster_ci95_high_pp': 100*high}


def analyze_details(matrix, config, metrics):
    scores = {k: {t: score_prediction(r, metrics[k[1]]) for t, r in rows.items()} for k, rows in matrix.items()}
    if any(not 0 <= value <= 1 for values in scores.values() for value in values.values()):
        raise ValueError('Официальная метрика вне [0,1]')
    common = {}
    for dataset in config['datasets']:
        bases = [matrix[(m, dataset, 'single_prefill', 'dynamic')] for m in config['models']]
        common[dataset] = {t for t in bases[0] if all(not rows[t]['middle_truncated'] for rows in bases)}
    subset_scores, formats, retrieval, baseline_quality = [], [], [], []
    common_full_f1 = {}
    for protocol in config['protocols']:
        common_full_f1[protocol] = set.intersection(*[
            {t for t, value in scores[(m, 'hotpotqa', protocol, 'dynamic')].items() if value == 1}
            for m in config['models']])
    for (model, dataset, protocol, cache), rows in matrix.items():
        values = scores[(model, dataset, protocol, cache)]
        if dataset == 'hotpotqa':
            ids = common[dataset]
            if ids:
                subset_scores.append({'model': model, 'dataset': dataset, 'protocol': protocol, 'cache': cache,
                    'n': len(ids), 'documents': len({rows[t]['context_sha256'] for t in ids}),
                    'score_pct': 100*sum(values[t] for t in ids)/len(ids)})
            first = {t: score_prediction({**r, 'pred': first_line(r['pred'])}, metrics[dataset]) for t, r in rows.items()}
            gains = {t: first[t]-values[t] for t in rows}
            formats.append({'model': model, 'protocol': protocol, 'cache': cache, 'n': len(rows),
                'official_f1_pct': 100*sum(values.values())/len(rows),
                'first_line_f1_pct': 100*sum(first.values())/len(rows),
                'first_line_better_count': sum(gains[t] > 1e-12 for t in rows),
                'first_line_worse_count': sum(gains[t] < -1e-12 for t in rows),
                **paired_interval(gains, rows)})
            if cache == 'dynamic':
                baseline_quality.append({'model': model, 'protocol': protocol, 'n': len(rows),
                    'official_f1_pct': 100*sum(values.values())/len(rows),
                    'official_full_f1_count': sum(v == 1 for v in values.values()),
                    'official_zero_count': sum(v == 0 for v in values.values()),
                    'first_line_full_f1_count': sum(v == 1 for v in first.values()),
                    'all_models_dynamic_full_f1_count': len(common_full_f1[protocol]),
                    'common_untruncated_n': len(ids),
                    'common_untruncated_f1_pct': 100*sum(values[t] for t in ids)/len(ids) if ids else None})
        elif dataset == 'passage_retrieval_en':
            base = matrix[(model, dataset, protocol, 'dynamic')]
            gold = {t: expected_numbers(r) for t, r in rows.items()}
            if any(not v for v in gold.values()):
                raise ValueError('Не найден номер эталонного абзаца')
            first = {t: int(first_int(r['pred']) in gold[t]) for t, r in rows.items()}
            dynamic = {t: int(first_int(base[t]['pred']) in gold[t]) for t in rows}
            retrieval.append({'model': model, 'protocol': protocol, 'cache': cache, 'n': len(rows),
                'official_score_pct': 100*sum(values.values())/len(rows),
                'first_number_correct': sum(first.values()), 'dynamic_first_number_correct': sum(dynamic.values()),
                'baseline_correct_first_number_lost': sum(dynamic[t] and not first[t] for t in rows),
                'first_number_recovered': sum(not dynamic[t] and first[t] for t in rows),
                **paired_interval({t: first[t]-dynamic[t] for t in rows}, rows)})
    return subset_scores, formats, retrieval, baseline_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--longbench-code', default='external/LongBench/LongBench')
    parser.add_argument('--out-dir', default='results/expansion_v1/details/full')
    args = parser.parse_args()
    config = load_config()
    matrix = read_matrix('results/expansion_v1/full', config, 'full')
    metric_hash = sha256_file(Path(args.longbench_code)/'metrics.py')
    if any(r['provenance']['longbench_files_sha256']['metrics.py'] != metric_hash for rows in matrix.values() for r in rows.values()):
        raise ValueError('metrics.py отличается от подготовки')
    subset, formats, retrieval, quality = analyze_details(matrix, config, load_metrics(args.longbench_code))
    out = Path(args.out_dir)
    for name, rows in (('untruncated_scores.csv', subset), ('hotpot_format.csv', formats),
                       ('retrieval_first_number.csv', retrieval), ('baseline_quality.csv', quality)):
        write_csv(out/name, rows)
    print('HotpotQA: baseline, общий subset и число Dynamic-ответов с F1=1:')
    for row in quality:
        print(row)
    print('\nОбщий subset без обрезания: Dynamic/g64, split:')
    for row in subset:
        if row['cache'] in ('dynamic', 'int4_g64') and row['protocol'] == 'context_then_question':
            print(row)
    print('\nSmolLM3: диагностический F1 первой строки (официальный score не меняется):')
    for row in formats:
        if row['model'] == 'smollm3-3b':
            print(row)
    print('\nRetrieval: первые номера (Smol Dynamic/g16/g64 и Qwen Dynamic/g64 split):')
    for row in retrieval:
        if (row['model'] == 'smollm3-3b' and row['cache'] != 'int4_g32') or (row['protocol'] == 'context_then_question' and row['cache'] in ('dynamic', 'int4_g64') and row['model'] != 'smollm3-3b'):
            print(row)
    print('\nCSV:', out)
    print('Первая строка и первый номер — post-hoc диагностика, а не официальный результат или доказательство эквивалентности.')
    print('Общий набор Dynamic-ответов с F1=1 зависит от baseline и не является новой независимой выборкой.')


if __name__ == '__main__':
    main()
