"""Сопоставление длинного и короткого retrieval-ответа на одних задачах.

Не меняет официальную метрику: она применяется к полному ответу каждой серии.
Проверяет одинаковые input IDs, а также совпадение первого числа ответа.

Пример:
    python -m src.compare_answer_lengths
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.common import CACHES, read_jsonl
from src.score import bootstrap_paired_ci, load_official_metrics, score_prediction


DATASET = "passage_retrieval_en"


def first_number(prediction: str) -> str | None:
    match = re.search(r"\d+", prediction)
    return match.group(0) if match else None


def load_indexed(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл: {path}")
    rows = {}
    for row in read_jsonl(path):
        if row["task_id"] in rows:
            raise ValueError(f"Дубликат задания в {path}: {row['task_id']}")
        rows[row["task_id"]] = row
    if not rows:
        raise ValueError(f"Пустой файл: {path}")
    return rows


def compare_cache(old_rows, new_rows, metric, *, expected_n=40, old_limit=32, new_limit=8):
    old_ids = set(old_rows)
    if len(old_ids) != expected_n or old_ids != set(new_rows):
        raise ValueError(f"Наборы заданий отличаются или имеют неверный размер: {len(old_ids)} vs {len(new_rows)}")
    paired = []
    changed_first_number = 0
    examples = []
    for task_id in sorted(old_ids):
        old, new = old_rows[task_id], new_rows[task_id]
        if old["task_id"] != new["task_id"] or old["dataset"] != new["dataset"]:
            raise ValueError(f"Несовпадение идентификаторов задания: {task_id}")
        if old["cache"] != new["cache"]:
            raise ValueError(f"Несовпадение типа cache: {task_id}")
        for field in ("input_ids_sha256", "middle_truncated", "model"):
            if old.get(field) != new.get(field):
                raise ValueError(f"Несовпадение {field} для {task_id}")
        if old["max_new_tokens"] != old_limit or new["max_new_tokens"] != new_limit:
            raise ValueError(f"Неожиданные пределы генерации для {task_id}")
        first_old = first_number(old["pred"])
        first_new = first_number(new["pred"])
        if first_old != first_new:
            changed_first_number += 1
        score_old = score_prediction(old, metric)
        score_new = score_prediction(new, metric)
        paired.append((score_old, score_new))
        examples.append((task_id, score_old, score_new, old["pred"], new["pred"]))
    low, high = bootstrap_paired_ci(paired)
    n = len(paired)
    mean_old = sum(a for a, _ in paired) / n
    mean_new = sum(b for _, b in paired) / n
    return {
        "n": n, "old_score": 100 * mean_old, "new_score": 100 * mean_new,
        "delta_pp": 100 * (mean_new - mean_old),
        "ci_low_pp": 100 * low, "ci_high_pp": 100 * high,
        "changed_first_number": changed_first_number,
        "examples": sorted(examples, key=lambda x: -(x[2] - x[1])),
    }


def main():
    parser = argparse.ArgumentParser(description="Сравнение retrieval-ответов с лимитом 32 и 8 токенов.")
    parser.add_argument("--original-dir", default="results/raw")
    parser.add_argument("--short-dir", default="results/short_answer/8tokens")
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    parser.add_argument("--expected-n", type=int, default=40)
    args = parser.parse_args()
    metric = load_official_metrics(args.longbench_code)[DATASET]
    for cache in CACHES:
        filename = f"{DATASET}_{cache}.jsonl"
        old_rows = load_indexed(Path(args.original_dir) / filename)
        new_rows = load_indexed(Path(args.short_dir) / filename)
        result = compare_cache(old_rows, new_rows, metric, expected_n=args.expected_n)
        print(f"\n{cache}, n={result['n']}:")
        print(f"  Официальная оценка при лимите 32: {result['old_score']:.2f}%")
        print(f"  Официальная оценка при лимите 8:  {result['new_score']:.2f}%")
        print(f"  Изменение 8−32: {result['delta_pp']:+.2f} п.п.")
        print(f"  Парный bootstrap 95%: [{result['ci_low_pp']:+.2f}; {result['ci_high_pp']:+.2f}] п.п.")
        print(f"  Первое число изменилось: {result['changed_first_number']}/{result['n']}")
        if result["changed_first_number"]:
            print("  ВНИМАНИЕ: изменение первого числа требует проверки детерминированности!")

    print("\nЭто дополнительный контролируемый эксперимент на прежней выборке.")
    print("Он не заменяет исходный официальный LongBench pilot и не оценивает новые задачи.")


if __name__ == "__main__":
    main()
