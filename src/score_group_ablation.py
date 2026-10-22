"""Оценка зависимости LongBench от q_group_size на пяти конфигурациях.

Этот скрипт намеренно не меняет src.score: старые результаты по
трём cache-конфигурациям остаются воспроизводимыми независимо.
Оригинальные функции метрик LongBench используются без изменений.

Запуск на login-узле после успешных 15 GPU-ячеек:
    python -m src.score_group_ablation
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.common import DATASETS, read_jsonl
from src.score import (
    bootstrap_paired_ci,
    load_official_metrics,
    score_prediction,
    write_csv,
)

GROUP_CACHES = (
    "dynamic",
    "int4_g16",
    "int4_g32",
    "int4_g64",
    "int4_g128",
)
COMPARE = (
    ("dynamic", "int4_g16"),
    ("dynamic", "int4_g32"),
    ("dynamic", "int4_g64"),
    ("dynamic", "int4_g128"),
    ("int4_g16", "int4_g32"),
    ("int4_g16", "int4_g64"),
    ("int4_g16", "int4_g128"),
    ("int4_g32", "int4_g64"),
    ("int4_g64", "int4_g128"),
)

REQUIRED_MATCH_FIELDS = (
    "input_ids_sha256",
    "max_new_tokens",
    "middle_truncated",
    "model",
    "dtype",
    "residual_length",
    "prompt_mode",
)


def read_matrix(results_dir: str | Path, expected_n: int = 80):
    """Проверить полную матрицу и идентичность входов во всех пяти случаях."""
    if expected_n < 1:
        raise ValueError("Размер выборки должен быть положительным")
    results_dir = Path(results_dir)
    matrix = {}
    for dataset in DATASETS:
        for cache in GROUP_CACHES:
            path = results_dir / f"{dataset}_{cache}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"Для полной матрицы не хватает: {path}")
            tasks = {}
            for row in read_jsonl(path):
                task_id = row["task_id"]
                if row.get("dataset") != dataset or row.get("cache") != cache:
                    raise ValueError(f"Чужая конфигурация в файле {path}: {task_id}")
                if task_id in tasks:
                    raise ValueError(f"Повтор task_id в {path}: {task_id}")
                if row.get("source_index") is not None and task_id != f"{dataset}:{row['source_index']}":
                    raise ValueError(f"Несовпадение task_id и source_index: {task_id}")
                tasks[task_id] = row
            if len(tasks) != expected_n:
                raise ValueError(f"{dataset}/{cache}: ожидалось {expected_n}, найдено {len(tasks)}")
            matrix[(dataset, cache)] = tasks

        baseline = matrix[(dataset, "dynamic")]
        for cache in GROUP_CACHES[1:]:
            quantized = matrix[(dataset, cache)]
            if set(quantized) != set(baseline):
                raise ValueError(f"Разные задачи в {dataset}: Dynamic и {cache}")
            for task_id, original in baseline.items():
                other = quantized[task_id]
                for field in REQUIRED_MATCH_FIELDS:
                    if original.get(field) != other.get(field):
                        raise ValueError(
                            f"{dataset}/{cache}/{task_id}: не совпадает {field}"
                        )
    return matrix


def analyze_matrix(matrix, metrics):
    """Получить оценки и прямые парные сравнения для каждого датасета."""
    datasets, comparisons, examples = [], [], []
    for dataset in DATASETS:
        scored = {}
        for cache in GROUP_CACHES:
            values = {}
            for task_id, row in sorted(matrix[(dataset, cache)].items()):
                score = score_prediction(row, metrics[dataset])
                if not 0 <= score <= 1:
                    raise ValueError(f"Некорректная оценка {dataset}/{cache}/{task_id}: {score}")
                values[task_id] = score
                examples.append({
                    "dataset": dataset,
                    "cache": cache,
                    "task_id": task_id,
                    "score_pct": round(score * 100, 6),
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "middle_truncated": row["middle_truncated"],
                    "pred": row["pred"],
                })
            scored[cache] = values
            datasets.append({
                "dataset": dataset,
                "cache": cache,
                "n": len(values),
                "score_pct": round(100 * sum(values.values()) / len(values), 3),
            })

        for reference, candidate in COMPARE:
            pairs = [
                (scored[reference][task_id], scored[candidate][task_id])
                for task_id in sorted(scored[reference])
            ]
            n = len(pairs)
            reference_pct = 100 * sum(a for a, _ in pairs) / n
            candidate_pct = 100 * sum(b for _, b in pairs) / n
            low, high = bootstrap_paired_ci(pairs)
            comparisons.append({
                "dataset": dataset,
                "reference": reference,
                "candidate": candidate,
                "n": n,
                "reference_pct": round(reference_pct, 3),
                "candidate_pct": round(candidate_pct, 3),
                "delta_pp": round(candidate_pct - reference_pct, 3),
                "bootstrap_95_low_pp": round(100 * low, 3),
                "bootstrap_95_high_pp": round(100 * high, 3),
            })
    return datasets, comparisons, examples


def main():
    parser = argparse.ArgumentParser(
        description="Качество Qwen3-1.7B на LongBench: DynamicCache и четыре q_group_size."
    )
    parser.add_argument("--results-dir", default="results/heldout/raw")
    parser.add_argument("--out-dir", default="results/heldout/group_size")
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    parser.add_argument("--expected-n", type=int, default=80)
    args = parser.parse_args()
    metrics = load_official_metrics(args.longbench_code)
    matrix = read_matrix(args.results_dir, expected_n=args.expected_n)
    datasets, comparisons, examples = analyze_matrix(matrix, metrics)

    out = Path(args.out_dir)
    write_csv(out / "dataset_scores.csv", datasets)
    write_csv(out / "paired_comparisons.csv", comparisons)
    write_csv(out / "example_scores.csv", examples)

    print("\nОфициальные оценки LongBench по конфигурациям:")
    for row in datasets:
        print(f"{row['dataset']:22} {row['cache']:11} n={row['n']} "
              f"score={row['score_pct']:.2f}%")
    print("\nПрямые парные контрасты относительно g16:")
    for row in comparisons:
        if row["reference"] != "int4_g16":
            continue
        print(f"{row['dataset']}: {row['candidate']} − g16 = "
              f"{row['delta_pp']:+.2f} п.п., "
              f"95% bootstrap [{row['bootstrap_95_low_pp']:+.2f}; "
              f"{row['bootstrap_95_high_pp']:+.2f}]")
    print("\nСоседние группы: g16→g32, g32→g64, g64→g128:")
    adjacent = {
        ("int4_g16", "int4_g32"),
        ("int4_g32", "int4_g64"),
        ("int4_g64", "int4_g128"),
    }
    for row in comparisons:
        if (row["reference"], row["candidate"]) not in adjacent:
            continue
        print(f"{row['dataset']}: {row['candidate']} − {row['reference']} = "
              f"{row['delta_pp']:+.2f} п.п., "
              f"95% bootstrap [{row['bootstrap_95_low_pp']:+.2f}; "
              f"{row['bootstrap_95_high_pp']:+.2f}]")
    print(f"\nТаблицы сохранены: {out}")
    print("Интервалы описывают вариацию выбранных заданий; выводы предварительные.")


if __name__ == "__main__":
    main()
