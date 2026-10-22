"""Дополнительная диагностика retrieval: первый номер ответа против полной метрики.

Официальную оценку LongBench не изменяем. Проверяем, ошибается ли
модель уже в первом названном номере абзаца или дополнительные числа
портят оценку при продолжении генерации.

Запуск без GPU:
    python -m src.analyze_retrieval_prefix
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from src.diagnose_pilot import load_paired_data
from src.score import load_official_metrics, score_prediction

DATASET = "passage_retrieval_en"
CACHES = ("dynamic", "int4_g16", "int4_g64")


def first_number(text: str) -> str | None:
    """Первое непрерывное десятичное число в ответе."""
    matched = re.search(r"\d+", text)
    return matched.group(0) if matched else None


def reference_number(answers) -> str:
    """Номер абзаца из эталона в стандартном формате LongBench."""
    numbers = [re.search(r"Paragraph (\d+)", answer) for answer in answers]
    found = [match.group(1) for match in numbers if match is not None]
    if not found:
        raise ValueError(f"Не найден номер абзаца в эталоне: {answers!r}")
    if len(set(found)) != 1:
        raise ValueError(f"Несогласованные номера эталона: {answers!r}")
    return found[0]


def analyze(by_cache, official_metric):
    """Проверить входы, вычислить первое число и официальную оценку."""
    all_ids = sorted(by_cache["dynamic"])
    examples = []
    for task_id in all_ids:
        answers = by_cache["dynamic"][task_id]["answers"]
        gold = reference_number(answers)
        for cache in CACHES:
            record = by_cache[cache][task_id]
            pred = str(record["pred"])
            number = first_number(pred)
            official = score_prediction(record, official_metric)
            examples.append({
                "task_id": task_id,
                "cache": cache,
                "эталон": gold,
                "первое_число": number or "",
                "первое_число_верно": int(number == gold),
                "официальный_балл_проц": round(100 * official, 3),
                "чисел_в_ответе": len(re.findall(r"\d+", pred)),
                "предел_длины_достигнут": int(
                    record["output_tokens"] >= record["max_new_tokens"]
                ),
                "ответ": pred,
            })
    return examples


def summarize(examples):
    for cache in CACHES:
        subset = [r for r in examples if r["cache"] == cache]
        n = len(subset)
        correct = sum(r["первое_число_верно"] for r in subset)
        official = sum(r["официальный_балл_проц"] for r in subset) / n
        capped = sum(r["предел_длины_достигнут"] for r in subset)
        first_right_official_bad = sum(
            r["первое_число_верно"] and r["официальный_балл_проц"] < 100
            for r in subset
        )
        print(f"{cache}: первое число верно {correct}/{n} ({correct/n*100:.1f}%); "
              f"официальная оценка {official:.2f}%; "
              f"первое число верно, но итог <100: {first_right_official_bad}/{n}; "
              f"достигнут предел: {capped}/{n}")
    print("Внимание: точность первого числа — дополнительная диагностическая")
    print("метрика; она НЕ заменяет стандартный retrieval_score LongBench.")


def main():
    parser = argparse.ArgumentParser(description="Проверить первое число retrieval-ответа.")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    parser.add_argument("--out", default="results/diagnostics/retrieval_first_number.csv")
    parser.add_argument("--expected-n", type=int, default=40)
    args = parser.parse_args()

    by_cache = load_paired_data(args.results_dir, DATASET)
    if len(by_cache["dynamic"]) != args.expected_n:
        raise ValueError(f"Ожидалось n={args.expected_n}, найдено {len(by_cache['dynamic'])}")
    official = load_official_metrics(args.longbench_code)[DATASET]
    rows = analyze(by_cache, official)
    summarize(rows)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Подробные результаты: {target}")


if __name__ == "__main__":
    main()
