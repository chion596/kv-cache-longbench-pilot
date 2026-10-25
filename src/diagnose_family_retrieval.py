"""Сравнить механизмы ошибок retrieval у Qwen3-1.7B и SmolLM3-3B.

Один и тот же LongBench source_index, но свои токенизаторы.
Отчёт не требует GPU и не объявляет первое число официальной метрикой.

Пример:
    python -m src.diagnose_family_retrieval
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import median

from src.score import load_official_metrics, score_prediction
from src.score_family_retrieval import load_results

CACHES = ("dynamic", "int4_g16", "int4_g32", "int4_g64", "int4_g128")


def first_int(text: str):
    match = re.search(r"\d+", str(text))
    return int(match.group()) if match else None


def expected_numbers(row: dict):
    """Для passage_retrieval_en эталоны имеют вид Paragraph N."""
    return {number for answer in row["answers"]
            if (number := first_int(answer)) is not None}


def analyze_model(matrix, metric):
    base = matrix["dynamic"]
    summary = []
    changed = []
    for cache, records in matrix.items():
        correct_first = 0
        multiple_numbers = 0
        capped = 0
        memory_savings = []
        for task_id, row in records.items():
            gold = expected_numbers(row)
            if not gold:
                raise ValueError(f"Не найден номер эталонного абзаца: {task_id}")
            numbers = re.findall(r"\d+", str(row["pred"]))
            correct_first += bool(numbers and int(numbers[0]) in gold)
            multiple_numbers += len(numbers) > 1
            capped += row["output_tokens"] >= row["max_new_tokens"]
            baseline_peak = base[task_id].get("peak_gpu_gib")
            candidate_peak = row.get("peak_gpu_gib")
            if baseline_peak is not None and candidate_peak is not None:
                if baseline_peak <= 0:
                    raise ValueError(f"Некорректная память у {task_id}")
                memory_savings.append(
                    100 * (baseline_peak - candidate_peak) / baseline_peak
                )
            baseline_score = score_prediction(base[task_id], metric)
            candidate_score = score_prediction(row, metric)
            if cache != "dynamic" and abs(candidate_score - baseline_score) > 1e-9:
                changed.append({
                    "cache": cache,
                    "task_id": task_id,
                    "delta_pp": round(100 * (candidate_score - baseline_score), 3),
                    "first_dynamic": first_int(base[task_id]["pred"]),
                    "first_cache": first_int(row["pred"]),
                    "gold": sorted(gold),
                    "dynamic_pred": str(base[task_id]["pred"])[:130],
                    "cache_pred": str(row["pred"])[:130],
                })
        summary.append({
            "cache": cache,
            "n": len(records),
            "first_number_correct": correct_first,
            "multiple_numbers": multiple_numbers,
            "hit_generation_limit": capped,
            "median_peak_gpu_gib": round(
                median(row["peak_gpu_gib"] for row in records.values()), 3
            ) if all("peak_gpu_gib" in row for row in records.values()) else None,
            "mean_paired_peak_memory_saving_pct": round(
                sum(memory_savings) / len(memory_savings), 3
            ) if len(memory_savings) == len(records) else None,
        })
    return summary, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-dir", default="results/heldout/raw")
    parser.add_argument("--smol-dir",
                        default="results/families/smollm3-3b/retrieval80")
    parser.add_argument("--expected-n", type=int, default=80)
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    args = parser.parse_args()
    metric = load_official_metrics(args.longbench_code)["passage_retrieval_en"]
    qwen = load_results(Path(args.qwen_dir), CACHES, args.expected_n)
    smol = load_results(Path(args.smol_dir), CACHES, args.expected_n)
    if set(qwen["dynamic"]) != set(smol["dynamic"]):
        raise ValueError("Qwen и SmolLM используют разные исходные LongBench задания")
    for task_id in qwen["dynamic"]:
        a, b = qwen["dynamic"][task_id], smol["dynamic"][task_id]
        if a.get("source_index") != b.get("source_index"):
            raise ValueError(f"Разные исходные source_index: {task_id}")
        # Token-ID SHA между семействами заведомо отличаются — это ожидаемо.
        if a["answers"] != b["answers"]:
            raise ValueError(f"Разные эталонные ответы: {task_id}")

    for label, matrix in (("Qwen3-1.7B", qwen), ("SmolLM3-3B", smol)):
        summary, changed = analyze_model(matrix, metric)
        print(f"\n=== {label}: 80 retrieval-заданий ===")
        for r in summary:
            print(f"{r['cache']:11} первое число верно={r['first_number_correct']}/{r['n']}, "
                  f"2+ чисел={r['multiple_numbers']}/{r['n']}, "
                  f"достиг лимита={r['hit_generation_limit']}/{r['n']}, "
                  f"peak median GiB={r['median_peak_gpu_gib']}, "
                  f"экономия peak vs Dynamic={r['mean_paired_peak_memory_saving_pct']}%")
        for cache in ("int4_g64", "int4_g128"):
            rows = [r for r in changed if r["cache"] == cache]
            worsened = [r for r in rows if r["delta_pp"] < 0]
            improved = [r for r in rows if r["delta_pp"] > 0]
            print(f"\n{cache}: изменился официальный балл на {len(rows)} задачах; "
                  f"ухудшилось {len(worsened)}, улучшилось {len(improved)}")
            for row in sorted(worsened, key=lambda x: x["delta_pp"])[:8]:
                print(f"  {row['task_id']}: Δ={row['delta_pp']:+.1f} п.п., "
                      f"эталон={row['gold']}, первое Dynamic/g={row['first_dynamic']}/"
                      f"{row['first_cache']}, ответы={row['dynamic_pred']!r} / "
                      f"{row['cache_pred']!r}")
        print("\nПервое число — диагностическая эвристика, а не метрика LongBench.")
        print("Peak memory — torch.cuda.max_memory_allocated(), не размер KV Cache.")


if __name__ == "__main__":
    main()
