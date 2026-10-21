"""Проверка ответов небольшого GPU smoke test по официальным метрикам LongBench.

Запуск на login-узле, без GPU:
    python -m src.check_smoke results/smoke/*.jsonl
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.common import read_jsonl
from src.score import load_official_metrics, score_prediction


def summarize(files, official_metrics):
    """Собрать проверенные записи, не путая разные датасеты и метрики."""
    rows = []
    task_keys = set()
    for filename in files:
        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"Файл не найден: {path}")
        for row in read_jsonl(path):
            dataset = row["dataset"]
            metric = official_metrics.get(dataset)
            if metric is None:
                raise ValueError(f"Неизвестный датасет: {dataset}")
            key = (dataset, row["task_id"], row["cache"])
            if key in task_keys:
                raise ValueError(f"Повтор задачи в проверяемых файлах: {key}")
            task_keys.add(key)
            score = score_prediction(row, metric)
            rows.append((row, score))
    if not rows:
        raise RuntimeError("В выбранных JSONL нет ни одной записи")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Проверить реальные ответы модели по официальным метрикам LongBench."
    )
    parser.add_argument("files", nargs="+", help="JSONL-файлы из results/smoke")
    parser.add_argument(
        "--longbench-code",
        default="external/LongBench/LongBench",
        help="Каталог с оригинальным metrics.py",
    )
    args = parser.parse_args()
    metrics = load_official_metrics(args.longbench_code)
    checked = summarize(args.files, metrics)

    print(f"Проверено ответов: {len(checked)}")
    for index, (row, score) in enumerate(checked, 1):
        prediction = str(row["pred"])
        references = row["answers"]
        print(f"\n--- Пример {index}: {row['task_id']} ---")
        print(f"Датасет: {row['dataset']}; cache: {row['cache']}")
        print(f"Длина входа: {row['input_tokens']} токенов")
        print(f"Правильный ответ: {references!r}")
        print(f"Ответ модели: {prediction[:350]!r}")
        if row["dataset"] == "passage_retrieval_en":
            print(f"Числа в ответе: {re.findall(r'[0-9]+', prediction)}")
        print(f"Официальная метрика: {100 * score:.1f}%")

    average = sum(score for _, score in checked) / len(checked)
    print(f"\nСредний балл этих {len(checked)} проверочных заданий: {100*average:.1f}%")
    print("Внимание: smoke test не является итоговым результатом LongBench.")


if __name__ == "__main__":
    main()
