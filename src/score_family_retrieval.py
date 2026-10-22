"""Парная оценка SmolLM3 на retrieval без требований к остальным датасетам.

Те же 80 source_index, что в Qwen3 held-out, но собственная токенизация
SmolLM3. Сравнения кэшей строго парные внутри одной модели.
Метрика retrieval_score загружается из оригинального LongBench.

На login-узле:
    python -m src.score_family_retrieval
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.common import read_jsonl
from src.score import bootstrap_paired_ci, load_official_metrics, score_prediction, write_csv

DATASET = "passage_retrieval_en"
ALLOWED_CACHES = ("dynamic", "int4_g16", "int4_g32", "int4_g64", "int4_g128")
PAIRED_FIELDS = (
    "input_ids_sha256",
    "max_new_tokens",
    "middle_truncated",
    "model",
    "dtype",
    "residual_length",
    "prompt_mode",
    "gpu",
)


def load_results(directory: str | Path, caches: tuple[str, ...], expected_n: int):
    """Прочитать результаты и проверить, что сравниваем одни и те же задания."""
    if expected_n < 1:
        raise ValueError("expected_n должен быть положительным")
    if not caches or len(set(caches)) != len(caches):
        raise ValueError("Нужен непустой набор неповторяющихся кэшей")
    if caches[0] != "dynamic":
        raise ValueError("Первой конфигурацией должен быть DynamicCache")
    if any(name not in ALLOWED_CACHES for name in caches):
        raise ValueError("Неизвестная конфигурация кэша")

    directory = Path(directory)
    matrix = {}
    for cache in caches:
        path = directory / f"{DATASET}_{cache}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Нет результатов: {path}")
        indexed = {}
        for row in read_jsonl(path):
            task_id = row["task_id"]
            if row.get("dataset") != DATASET or row.get("cache") != cache:
                raise ValueError(f"Неправильный dataset/cache в {path}: {task_id}")
            if task_id in indexed:
                raise ValueError(f"Повторный task_id в {path}: {task_id}")
            if task_id != f"{DATASET}:{row.get('source_index')}":
                raise ValueError(f"task_id не совпал с source_index: {task_id}")
            indexed[task_id] = row
        if len(indexed) != expected_n:
            raise ValueError(f"{cache}: ожидалось {expected_n}, найдено {len(indexed)}")
        matrix[cache] = indexed

    base = matrix["dynamic"]
    for cache in caches[1:]:
        if set(matrix[cache]) != set(base):
            raise ValueError(f"Между Dynamic и {cache} отличаются задания")
        for task_id, original in base.items():
            other = matrix[cache][task_id]
            for field in PAIRED_FIELDS:
                if original.get(field) != other.get(field):
                    raise ValueError(f"{cache}/{task_id}: не совпадает {field}")
    return matrix


def analyze_results(matrix, metric):
    """Баллы и интервалы по одинаковым заданиям."""
    caches = tuple(matrix)
    scores = {}
    summaries = []
    for cache in caches:
        by_id = matrix[cache]
        values = {task_id: score_prediction(row, metric)
                  for task_id, row in sorted(by_id.items())}
        if any(not 0 <= x <= 1 for x in values.values()):
            raise ValueError(f"Недопустимый балл LongBench для {cache}")
        scores[cache] = values
        capped = sum(
            row["output_tokens"] >= row["max_new_tokens"]
            for row in by_id.values()
        )
        summaries.append({
            "cache": cache,
            "n": len(values),
            "quality_pct": round(100 * sum(values.values()) / len(values), 3),
            "zero_score_count": sum(x == 0 for x in values.values()),
            "generation_limit_count": capped,
            "mean_output_tokens": round(
                sum(row["output_tokens"] for row in by_id.values()) / len(by_id), 3
            ),
        })

    contrasts = []
    pairs_to_check = [("dynamic", cache) for cache in caches[1:]]
    if "int4_g16" in caches:
        pairs_to_check.extend(("int4_g16", cache)
                              for cache in caches[1:] if cache != "int4_g16")
    if "int4_g32" in caches and "int4_g64" in caches:
        pairs_to_check.append(("int4_g32", "int4_g64"))
    for reference, candidate in pairs_to_check:
        pairs = [(scores[reference][task_id], scores[candidate][task_id])
                 for task_id in sorted(scores[reference])]
        low, high = bootstrap_paired_ci(pairs)
        contrasts.append({
            "reference": reference,
            "candidate": candidate,
            "n": len(pairs),
            "delta_pp": round(100 * sum(b-a for a, b in pairs) / len(pairs), 3),
            "bootstrap_95_low_pp": round(100 * low, 3),
            "bootstrap_95_high_pp": round(100 * high, 3),
        })
    return summaries, contrasts


def main():
    parser = argparse.ArgumentParser(
        description="Сравнить cache SmolLM3-3B на 80 retrieval-заданиях."
    )
    parser.add_argument("--results-dir",
                        default="results/families/smollm3-3b/retrieval80")
    parser.add_argument("--out-dir",
                        default="results/families/smollm3-3b/summary")
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    parser.add_argument("--expected-n", type=int, default=80)
    parser.add_argument("--caches", nargs="+",
                        default=["dynamic", "int4_g16", "int4_g64"],
                        choices=ALLOWED_CACHES)
    args = parser.parse_args()

    matrix = load_results(args.results_dir, tuple(args.caches), args.expected_n)
    metric = load_official_metrics(args.longbench_code)[DATASET]
    summaries, contrasts = analyze_results(matrix, metric)
    out = Path(args.out_dir)
    write_csv(out / "retrieval_scores.csv", summaries)
    write_csv(out / "retrieval_pairs.csv", contrasts)

    print("SmolLM3-3B, passage_retrieval_en:")
    for row in summaries:
        print(f"{row['cache']:11} качество={row['quality_pct']:6.2f}%, "
              f"нулевых={row['zero_score_count']}/{row['n']}, "
              f"лимит генерации={row['generation_limit_count']}/{row['n']}, "
              f"средняя длина={row['mean_output_tokens']:.1f} токенов")
    for row in contrasts:
        print(f"{row['candidate']} − {row['reference']} = "
              f"{row['delta_pp']:+.2f} п.п., "
              f"bootstrap95=[{row['bootstrap_95_low_pp']:+.2f};"
              f"{row['bootstrap_95_high_pp']:+.2f}]")
    print(f"Таблицы сохранены: {out}")
    print("Это пилот на фиксированных 80 заданиях, не весь LongBench.")


if __name__ == "__main__":
    main()
