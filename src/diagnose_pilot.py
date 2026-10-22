"""Диагностика результатов первого пилота LongBench.

Для каждого датасета проверяет совпадение заданий в Dynamic, INT4 g16 и INT4 g64,
считает парную разницу g16−g64 и выводит самые показательные ошибки.
Используются оригинальные метрики LongBench без изменения их реализации.

Запуск на login-узле, без GPU:
    python -m src.diagnose_pilot
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.common import CACHES, DATASETS, read_jsonl
from src.score import bootstrap_paired_ci, load_official_metrics, score_prediction


def load_paired_data(results_dir: str | Path, dataset: str):
    """Загрузить строго одинаковый набор задач для трёх типов cache."""
    directory = Path(results_dir)
    by_cache = {}
    for cache in CACHES:
        filename = directory / f"{dataset}_{cache}.jsonl"
        if not filename.is_file():
            raise FileNotFoundError(f"Не найден файл: {filename}")
        indexed = {}
        for row in read_jsonl(filename):
            if row.get("dataset") != dataset or row.get("cache") != cache:
                raise ValueError(f"Неверный dataset/cache в {filename}: {row.get('task_id')}")
            key = row["task_id"]
            if key in indexed:
                raise ValueError(f"Повтор задачи {key} в {filename}")
            indexed[key] = row
        by_cache[cache] = indexed

    base_ids = set(by_cache["dynamic"])
    if not base_ids:
        raise ValueError(f"Нет заданий для {dataset}")
    for cache in CACHES:
        if set(by_cache[cache]) != base_ids:
            raise ValueError(f"Наборы задач отличаются: {dataset}, {cache}")
        for task_id in base_ids:
            baseline = by_cache["dynamic"][task_id]
            other = by_cache[cache][task_id]
            for field in ("input_ids_sha256", "max_new_tokens", "middle_truncated"):
                if baseline.get(field) != other.get(field):
                    raise ValueError(f"Не совпадает {field}: {dataset}, {cache}, {task_id}")
    return by_cache


def build_diagnostics(by_cache, metric):
    """Сформировать диагностику по заданиям, не изменяя официальные оценки."""
    result = []
    for task_id in sorted(by_cache["dynamic"]):
        rows = {cache: by_cache[cache][task_id] for cache in CACHES}
        scores = {cache: score_prediction(rows[cache], metric) for cache in CACHES}
        result.append({
            "task_id": task_id,
            "rows": rows,
            "scores": scores,
            "g16_minus_g64": scores["int4_g16"] - scores["int4_g64"],
            "dynamic_minus_g64": scores["dynamic"] - scores["int4_g64"],
        })
    return result


def describe_dataset(dataset, cases, show_examples=3):
    n = len(cases)
    if n < 1:
        raise ValueError("Нельзя анализировать пустую выборку")
    means = {
        cache: 100 * sum(case["scores"][cache] for case in cases) / n
        for cache in CACHES
    }
    direct_pairs = [
        (case["scores"]["int4_g64"], case["scores"]["int4_g16"])
        for case in cases
    ]
    low, high = bootstrap_paired_ci(direct_pairs)
    delta = means["int4_g16"] - means["int4_g64"]
    zero = {cache: sum(case["scores"][cache] == 0 for case in cases) for cache in CACHES}
    at_cap = {
        cache: sum(
            case["rows"][cache]["output_tokens"] >= case["rows"][cache]["max_new_tokens"]
            for case in cases
        )
        for cache in CACHES
    }
    print(f"\n{'=' * 70}\nДатасет: {dataset} | n={n}")
    print("Баллы: " + ", ".join(f"{cache}={means[cache]:.2f}%" for cache in CACHES))
    print(f"Главный контраст g16 − g64: {delta:+.2f} п.п.")
    print(f"Парный bootstrap 95%: [{100 * low:+.2f}; {100 * high:+.2f}] п.п.")
    print("Нулевой балл у примеров: " + ", ".join(f"{cache}={zero[cache]}/{n}" for cache in CACHES))
    print("Достигнут предел числа новых токенов: " +
          ", ".join(f"{cache}={at_cap[cache]}/{n}" for cache in CACHES))

    ranked = sorted(cases, key=lambda case: (
        -case["g16_minus_g64"], -case["dynamic_minus_g64"], case["task_id"]
    ))
    print(f"\nПримеры с наибольшим преимуществом g16 над g64 (первые {show_examples}):")
    for case in ranked[:show_examples]:
        print(f"  Задание: {case['task_id']}; "
              + ", ".join(f"{cache}={100*case['scores'][cache]:.1f}%"
                          for cache in CACHES))
        print(f"  Эталон: {case['rows']['dynamic'].get('answers', [])!r}"[:550])
        for cache in CACHES:
            row = case["rows"][cache]
            preview = str(row.get("pred", "")).replace("\n", "↵")
            print(f"  {cache}: {preview[:220]!r}; "
                  f"токенов ответа={row['output_tokens']}/{row['max_new_tokens']}")
            if dataset.startswith("passage_retrieval"):
                print(f"    Числа: {re.findall(r'[0-9]+', str(row.get('pred', '')))[:12]}")
        print()
    print("Примечание: интервалы описывают вариацию по выбранным заданиям; "
          "сравнение языков не является контролируемым.")


def main():
    parser = argparse.ArgumentParser(description="Разобрать ошибки пилота LongBench.")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--expected-n", type=int, default=40)
    args = parser.parse_args()
    if args.examples < 0 or args.expected_n < 1:
        parser.error("--examples >= 0 и --expected-n >= 1 обязательны")

    metrics = load_official_metrics(args.longbench_code)
    for dataset in DATASETS:
        data = load_paired_data(args.results_dir, dataset)
        cases = build_diagnostics(data, metrics[dataset])
        if len(cases) != args.expected_n:
            raise ValueError(
                f"{dataset}: ожидалось {args.expected_n} задач, найдено {len(cases)}"
            )
        describe_dataset(dataset, cases, show_examples=args.examples)


if __name__ == "__main__":
    main()
