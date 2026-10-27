"""Парная декомпозиция QA F1: полный ответ против первой непустой строки.

Все сравнения делаются на тех же 80 task_id. «Первая строка» —
постпроцессинг сохранённого ответа, а НЕ официальный LongBench run.
Кроме F1 и парных CI считаем difference-in-differences:
(прибавка F1 от первой строки у quant) − (та же прибавка у Dynamic).

На login-узле, без GPU:
    python -m src.score_qa_continuations
"""
from __future__ import annotations

import argparse

from src.diagnose_family_qa import first_line
from src.score import bootstrap_paired_ci, load_official_metrics, score_prediction, write_csv
from src.score_family_qa import read_qa_matrix


def analyze_continuations(matrix, metric):
    """Проверенные пары raw / first-line и их разброс между задачами."""
    cache_scores = {}
    summary = []
    for cache, tasks in matrix.items():
        raw = {}
        first = {}
        for task_id, row in tasks.items():
            raw[task_id] = score_prediction(row, metric)
            first[task_id] = score_prediction({**row, "pred": first_line(row["pred"])}, metric)
            if not (0 <= raw[task_id] <= 1 and 0 <= first[task_id] <= 1):
                raise ValueError(f"Недопустимый балл F1: {task_id}")
        cache_scores[cache] = {"raw": raw, "first": first}
        ids = sorted(tasks)
        gains = [(raw[tid], first[tid]) for tid in ids]
        lo, hi = bootstrap_paired_ci(gains)
        summary.append({
            "cache": cache,
            "n": len(ids),
            "raw_f1_pct": round(100 * sum(raw.values()) / len(ids), 3),
            "first_line_f1_pct": round(100 * sum(first.values()) / len(ids), 3),
            "first_minus_raw_pp": round(
                100 * sum(first[tid] - raw[tid] for tid in ids) / len(ids), 3),
            "first_minus_raw_ci95_low_pp": round(100 * lo, 3),
            "first_minus_raw_ci95_high_pp": round(100 * hi, 3),
            "first_line_better": sum(first[tid] > raw[tid] + 1e-9 for tid in ids),
            "first_line_worse": sum(first[tid] < raw[tid] - 1e-9 for tid in ids),
        })

    base = cache_scores["dynamic"]
    contrasts = []
    for cache, values in cache_scores.items():
        if cache == "dynamic":
            continue
        tids = sorted(base["raw"])
        if set(tids) != set(values["raw"]):
            raise ValueError(f"Dynamic и {cache} сравниваются на разных заданиях")
        pairs = {
            "raw_delta": [(base["raw"][tid], values["raw"][tid]) for tid in tids],
            "first_line_delta": [(base["first"][tid], values["first"][tid]) for tid in tids],
            "difference_of_deltas": [
                (base["first"][tid] - base["raw"][tid],
                 values["first"][tid] - values["raw"][tid])
                for tid in tids
            ],
        }
        for contrast, scores in pairs.items():
            lo, hi = bootstrap_paired_ci(scores)
            contrasts.append({
                "cache": cache,
                "comparison": contrast,
                "n": len(tids),
                "delta_pp": round(
                    100 * sum(candidate - reference for reference, candidate in scores) /
                    len(tids), 3),
                "ci95_low_pp": round(100 * lo, 3),
                "ci95_high_pp": round(100 * hi, 3),
            })
    return summary, contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multifieldqa_en",
                        choices=["multifieldqa_en", "multifieldqa_zh"])
    parser.add_argument("--results-dir",
                        default="results/families/smollm3-3b/qa_en80")
    parser.add_argument("--out-dir",
                        default="results/families/smollm3-3b/summary")
    parser.add_argument("--expected-n", type=int, default=80)
    parser.add_argument("--caches", nargs="+",
                        default=["dynamic", "int4_g16", "int4_g64"])
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    args = parser.parse_args()

    matrix = read_qa_matrix(args.results_dir, args.dataset,
                            tuple(args.caches), args.expected_n)
    metric = load_official_metrics(args.longbench_code)[args.dataset]
    summary, contrasts = analyze_continuations(matrix, metric)

    from pathlib import Path
    out = Path(args.out_dir)
    write_csv(out / f"{args.dataset}_continuations_scores.csv", summary)
    write_csv(out / f"{args.dataset}_continuations_pairs.csv", contrasts)

    print(f"QA {args.dataset}: {args.expected_n} одних и тех же задач")
    print("RAW — официальный F1, FIRST — диагностический F1 первой строки.")
    for row in summary:
        print(f"{row['cache']:11} RAW={row['raw_f1_pct']:6.3f}% "
              f"FIRST={row['first_line_f1_pct']:6.3f}% "
              f"FIRST−RAW={row['first_minus_raw_pp']:+6.3f} п.п. "
              f"[{row['first_minus_raw_ci95_low_pp']:+.3f}; "
              f"{row['first_minus_raw_ci95_high_pp']:+.3f}], "
              f"улучшений/ухудшений={row['first_line_better']}/{row['first_line_worse']}")
    print("\nКонтрасты относительно Dynamic (candidate−Dynamic):")
    for row in contrasts:
        print(f"{row['cache']:11} {row['comparison']:20} "
              f"{row['delta_pp']:+.3f} п.п. "
              f"95% bootstrap [{row['ci95_low_pp']:+.3f}; {row['ci95_high_pp']:+.3f}]")
    print("\nРазность эффектов (difference_of_deltas) — насколько первая строка "
          "сильнее/слабее помогает quant по сравнению с Dynamic, не эффект квантизации "
          "на качество retrieval или доказательство эквивалентности.")


if __name__ == "__main__":
    main()
