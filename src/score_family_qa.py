"""Парное сравнение SmolLM3-3B на LongBench multifieldqa_en/zh.

Использует оригинальную QA F1 метрику LongBench; для каждого
модельного семейства отдельно проверяет одинаковые входные token IDs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.common import read_jsonl
from src.score import load_official_metrics, write_csv
from src.score_family_retrieval import (
    ALLOWED_CACHES, PAIRED_FIELDS, analyze_results,
)

DATASETS = ("multifieldqa_en", "multifieldqa_zh")


def read_qa_matrix(results_dir: str | Path, dataset: str,
                   caches: tuple[str, ...], expected_n: int):
    if dataset not in DATASETS:
        raise ValueError(f"Неизвестный QA датасет: {dataset}")
    if expected_n < 1 or not caches or caches[0] != "dynamic":
        raise ValueError("Нужны положительный expected_n и первым Dynamic")
    if len(set(caches)) != len(caches) or any(c not in ALLOWED_CACHES for c in caches):
        raise ValueError("Некорректный список конфигураций")
    matrix = {}
    for cache in caches:
        path = Path(results_dir) / f"{dataset}_{cache}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Нет результатов: {path}")
        records = {}
        for row in read_jsonl(path):
            task_id = row["task_id"]
            if row.get("dataset") != dataset or row.get("cache") != cache:
                raise ValueError(f"Чужая запись в {path}: {task_id}")
            if task_id != f"{dataset}:{row.get('source_index')}":
                raise ValueError(f"Не совпадает task_id/source_index: {task_id}")
            if task_id in records:
                raise ValueError(f"Повтор задания {task_id}")
            records[task_id] = row
        if len(records) != expected_n:
            raise ValueError(f"{cache}: ожидалось {expected_n}, найдено {len(records)}")
        matrix[cache] = records
    baseline = matrix["dynamic"]
    for cache in caches[1:]:
        if set(matrix[cache]) != set(baseline):
            raise ValueError(f"Dynamic и {cache}: различаются исходные задания")
        for task_id, first in baseline.items():
            second = matrix[cache][task_id]
            for field in PAIRED_FIELDS:
                if first.get(field) != second.get(field):
                    raise ValueError(f"{cache}/{task_id}: различается {field}")
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multifieldqa_en", choices=DATASETS)
    parser.add_argument("--results-dir",
                        default="results/families/smollm3-3b/qa_en80")
    parser.add_argument("--out-dir",
                        default="results/families/smollm3-3b/summary")
    parser.add_argument("--expected-n", type=int, default=80)
    parser.add_argument("--caches", nargs="+",
                        default=["dynamic", "int4_g16", "int4_g64"],
                        choices=ALLOWED_CACHES)
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    args = parser.parse_args()

    matrix = read_qa_matrix(args.results_dir, args.dataset,
                            tuple(args.caches), args.expected_n)
    metric = load_official_metrics(args.longbench_code)[args.dataset]
    summaries, contrasts = analyze_results(matrix, metric)
    output_dir = Path(args.out_dir)
    write_csv(output_dir / f"{args.dataset}_scores.csv", summaries)
    write_csv(output_dir / f"{args.dataset}_pairs.csv", contrasts)

    print(f"SmolLM3-3B: {args.dataset} ({args.expected_n} задач)")
    for row in summaries:
        print(f"{row['cache']:11} F1={row['quality_pct']:.3f}% "
              f"нулевых={row['zero_score_count']}/{row['n']} "
              f"лимит={row['generation_limit_count']}/{row['n']} "
              f"средняя длина={row['mean_output_tokens']:.1f}")
    for row in contrasts:
        print(f"{row['candidate']} − {row['reference']} = "
              f"{row['delta_pp']:+.3f} п.п. "
              f"95% bootstrap [{row['bootstrap_95_low_pp']:+.3f}; "
              f"{row['bootstrap_95_high_pp']:+.3f}]")
    print(f"CSV: {output_dir}")
    print("Это выборка LongBench, не официальный полный benchmark.")


if __name__ == "__main__":
    main()
