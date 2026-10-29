"""Одна модель: формат retrieval, первые числа, память и ошибки.

На login-узле кластера, без GPU:
    python -m src.diagnose_single_retrieval

По умолчанию анализирует Qwen3-4B на 80 заданиях.
Не изменяет исходные JSONL.
"""
from __future__ import annotations

import argparse

from src.diagnose_family_retrieval import analyze_model
from src.score import load_official_metrics
from src.score_family_retrieval import ALLOWED_CACHES, load_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/families/qwen3-4b/retrieval80")
    parser.add_argument("--expected-n", type=int, default=80)
    parser.add_argument("--label", default="Qwen3-4B")
    parser.add_argument("--caches", nargs="+",
                        default=["dynamic", "int4_g16", "int4_g64"],
                        choices=ALLOWED_CACHES)
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    args = parser.parse_args()

    metric = load_official_metrics(args.longbench_code)["passage_retrieval_en"]
    matrix = load_results(args.results_dir, tuple(args.caches), args.expected_n)
    summary, changed = analyze_model(matrix, metric)

    print(f"=== {args.label}: {args.expected_n} retrieval-заданий ===")
    for r in summary:
        print(f"{r['cache']:11} верное первое число={r['first_number_correct']}/{r['n']}, "
              f"2+ чисел={r['multiple_numbers']}/{r['n']}, "
              f"лимит={r['hit_generation_limit']}/{r['n']}, "
              f"median peak GPU GiB={r['median_peak_gpu_gib']}, "
              f"mean paired saving vs Dynamic={r['mean_paired_peak_memory_saving_pct']}%")

    for cache in args.caches[1:]:
        rows = [r for r in changed if r["cache"] == cache]
        worsened = sorted((r for r in rows if r["delta_pp"] < 0),
                          key=lambda x: x["delta_pp"])
        improved = [r for r in rows if r["delta_pp"] > 0]
        first_changed = sum(r["first_dynamic"] != r["first_cache"] for r in rows)
        print(f"\n{cache}: изменился официальный балл на {len(rows)} заданиях; "
              f"ухудшились {len(worsened)}, улучшились {len(improved)}, "
              f"среди них изменилось первое число {first_changed}")
        for row in worsened[:10]:
            print(f"  {row['task_id']}: Δ={row['delta_pp']:+.1f} п.п., "
                  f"эталон={row['gold']}, первые Dynamic/quant="
                  f"{row['first_dynamic']}/{row['first_cache']}, "
                  f"ответы={row['dynamic_pred']!r} / {row['cache_pred']!r}")

    print("\nПервое число — эвристика, не официальная метрика LongBench.")
    print("Пиковая GPU memory — torch.cuda.max_memory_allocated(), не объём KV Cache.")


if __name__ == "__main__":
    main()
