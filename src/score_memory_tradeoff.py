"""Качество и пиковая GPU-память на одинаковых заданиях LongBench.

Использует уже записанные peak_gpu_gib (CUDA max_memory_allocated)
из результатов генерации. НЕ измеряет физический размер KV Cache
и не заменяет отдельный performance benchmark.

Запуск без GPU:
    python -m src.score_memory_tradeoff
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from src.common import DATASETS
from src.score import load_official_metrics, score_prediction, write_csv
from src.score_group_ablation import GROUP_CACHES, read_matrix


def summarize_memory(matrix, metrics):
    """Средние метрики и парные сравнения памяти на одинаковых входах."""
    summaries = []
    for dataset in DATASETS:
        baseline = matrix[(dataset, "dynamic")]
        baseline_scores = [
            score_prediction(baseline[tid], metrics[dataset])
            for tid in sorted(baseline)
        ]
        baseline_mean = sum(baseline_scores) / len(baseline_scores)
        for cache in GROUP_CACHES:
            tasks = matrix[(dataset, cache)]
            scores = [
                score_prediction(tasks[tid], metrics[dataset])
                for tid in sorted(tasks)
            ]
            baseline_peaks = []
            config_peaks = []
            savings = []
            for tid in sorted(baseline):
                reference, current = baseline[tid], tasks[tid]
                if reference.get("gpu") != current.get("gpu"):
                    raise ValueError(
                        f"Разные GPU в {dataset}/{cache}/{tid}: "
                        f"{reference.get('gpu')!r} и {current.get('gpu')!r}"
                    )
                a, b = float(reference["peak_gpu_gib"]), float(current["peak_gpu_gib"])
                if not (0 < a < 200 and 0 < b < 200):
                    raise ValueError(f"Недопустимое peak_gpu_gib для {tid}")
                baseline_peaks.append(a)
                config_peaks.append(b)
                savings.append(100 * (a - b) / a)

            score_pct = 100 * sum(scores) / len(scores)
            summaries.append({
                "dataset": dataset,
                "cache": cache,
                "n": len(tasks),
                "quality_pct": round(score_pct, 3),
                "delta_quality_from_dynamic_pp": round(
                    score_pct - 100 * baseline_mean, 3
                ),
                "median_peak_allocated_gib": round(statistics.median(config_peaks), 4),
                "mean_peak_allocated_gib": round(statistics.mean(config_peaks), 4),
                "mean_paired_peak_saving_pct": round(statistics.mean(savings), 3),
                "median_paired_peak_saving_pct": round(statistics.median(savings), 3),
                "median_reference_peak_gib": round(
                    statistics.median(baseline_peaks), 4
                ),
            })
    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Сопоставление качества и peak CUDA allocated memory LongBench."
    )
    parser.add_argument("--results-dir", default="results/heldout/raw")
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    parser.add_argument("--out", default="results/heldout/group_size/memory_quality.csv")
    parser.add_argument("--expected-n", type=int, default=80)
    args = parser.parse_args()
    matrix = read_matrix(args.results_dir, expected_n=args.expected_n)
    metrics = load_official_metrics(args.longbench_code)
    table = summarize_memory(matrix, metrics)
    write_csv(Path(args.out), table)
    print("Качество и пиковая выделенная память GPU на одинаковых заданиях:")
    for row in table:
        print(f"{row['dataset']:22} {row['cache']:10} "
              f"качество={row['quality_pct']:6.2f}%, "
              f"медиана peak={row['median_peak_allocated_gib']:.3f} GiB, "
              f"средняя парная экономия={row['mean_paired_peak_saving_pct']:+.2f}%")
    print("\nЭто peak CUDA memory_allocated, а НЕ физический размер KV Cache.")
    print("Средняя парная экономия считается отдельно для каждого задания.")
    print(f"Таблица: {args.out}")


if __name__ == "__main__":
    main()
