"""Диагностика влияния продолжений на LongBench QA F1 без GPU.

Считает два балла на одних сохранённых ответах:
- официальный LongBench F1 для неизменённого предсказания;
- тот же F1, если оставить только первую непустую строку.

Второй балл — диагностический постпроцессинг, а не новый
результат стандартного LongBench. Не меняет исходные JSONL.
"""
from __future__ import annotations

import argparse
from src.score import load_official_metrics, score_prediction
from src.score_family_qa import read_qa_matrix


def first_line(prediction: str):
    for line in prediction.splitlines():
        if line.strip():
            return line.strip()
    return ""


def diagnose(matrix, metric):
    results = []
    examples = []
    for cache, tasks in matrix.items():
        raw_sum = 0.0
        first_sum = 0.0
        improved = worsened = hit_limit = 0
        for task_id, row in sorted(tasks.items()):
            original = score_prediction(row, metric)
            first = score_prediction({**row, "pred": first_line(row["pred"])}, metric)
            raw_sum += original
            first_sum += first
            improved += first > original + 1e-9
            worsened += first < original - 1e-9
            hit_limit += row["output_tokens"] >= row["max_new_tokens"]
            if abs(first-original) > 1e-9:
                examples.append({
                    "cache": cache,
                    "task_id": task_id,
                    "raw_f1": round(100 * original, 2),
                    "first_line_f1": round(100 * first, 2),
                    "first_line": first_line(row["pred"])[:180],
                    "full_pred": row["pred"][:220],
                })
        n = len(tasks)
        results.append({
            "cache": cache,
            "n": n,
            "raw_f1_pct": round(100 * raw_sum / n, 3),
            "first_line_f1_pct": round(100 * first_sum / n, 3),
            "improved": improved,
            "worsened": worsened,
            "hit_limit": hit_limit,
        })
    return results, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",
                        default="results/families/smollm3-3b/qa_en_smoke")
    parser.add_argument("--expected-n", type=int, default=8)
    parser.add_argument("--dataset", default="multifieldqa_en",
                        choices=["multifieldqa_en", "multifieldqa_zh"])
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    args = parser.parse_args()

    caches = ("dynamic", "int4_g16", "int4_g64")
    matrix = read_qa_matrix(args.results_dir, args.dataset, caches, args.expected_n)
    metric = load_official_metrics(args.longbench_code)[args.dataset]
    results, examples = diagnose(matrix, metric)

    print(f"QA: {args.dataset}; n={args.expected_n} на cache")
    print("Первая строка — диагностический постпроцессинг, НЕ официальный результат.")
    for row in results:
        print(f"{row['cache']:11} исходный F1={row['raw_f1_pct']:6.2f}%, "
              f"только первая строка={row['first_line_f1_pct']:6.2f}%, "
              f"улучшений/ухудшений={row['improved']}/{row['worsened']}, "
              f"лимит={row['hit_limit']}/{row['n']}")
    print("\nОтветы, на которые влияет отсечение продолжения:")
    for row in examples:
        print(f"  {row['cache']}/{row['task_id']}: "
              f"{row['raw_f1']}% → {row['first_line_f1']}%, "
              f"первая строка={row['first_line']!r}")
    print("\nСохраняйте оригинальные предсказания: это диагностика, "
          "не изменение метрики или экспериментального протокола.")


if __name__ == "__main__":
    main()
