"""Разобрать сохранённые full JSONL: Dynamic drift, первые токены и квантизации.

CPU-анализ без повторной генерации; оригинальные ответы и метрики сохраняются.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from statistics import median

from src.diagnose_family_retrieval import expected_numbers, first_int
from src.download_data import sha256_file
from src.protocol import load_config
from src.score import load_official_metrics, score_prediction, write_csv
from src.score_protocol import cluster_ci, read_matrix


def first_difference(left, right):
    """Индекс с нуля; разная длина общего префикса также считается расхождением."""
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def interval(values, rows):
    lo, hi = cluster_ci(values, {tid: row["context_sha256"] for tid, row in rows.items()})
    return {"delta_pp": 100 * sum(values.values()) / len(values),
            "cluster_ci95_low_pp": 100 * lo, "cluster_ci95_high_pp": 100 * hi}


def analyze_details(matrix, config, metrics):
    scores = {key: {tid: score_prediction(row, metrics[key[1]]) for tid, row in rows.items()}
              for key, rows in matrix.items()}
    dynamic, cache_rows, comparisons, examples = [], [], [], []
    for model in config["models"]:
        for dataset in config["datasets"]:
            full_key = (model, dataset, "single_prefill", "dynamic")
            split_key = (model, dataset, "context_then_question", "dynamic")
            whole, split = matrix[full_key], matrix[split_key]
            drift = {tid: scores[split_key][tid] - scores[full_key][tid] for tid in whole}
            dynamic.append({"model": model, "dataset": dataset, "n": len(whole),
                "different_output_ids": sum(whole[t]["output_ids"] != split[t]["output_ids"] for t in whole),
                "different_text": sum(whole[t]["pred"] != split[t]["pred"] for t in whole),
                "different_score": sum(abs(drift[t]) > 1e-12 for t in whole),
                "different_first_token": sum(whole[t]["output_ids"][0] != split[t]["output_ids"][0] for t in whole),
                **interval(drift, whole)})
            for tid in whole:
                if whole[tid]["output_ids"] != split[tid]["output_ids"]:
                    examples.append({"model": model, "dataset": dataset, "protocol": "single_vs_split",
                        "cache": "dynamic", "task_id": tid, "context_sha256": whole[tid]["context_sha256"],
                        "first_difference_index": first_difference(whole[tid]["output_ids"], split[tid]["output_ids"]),
                        "reference_pred": whole[tid]["pred"], "candidate_pred": split[tid]["pred"]})

            for protocol in config["protocols"]:
                base_key = (model, dataset, protocol, "dynamic")
                base = matrix[base_key]
                for cache in config["caches"]:
                    key = (model, dataset, protocol, cache)
                    rows = matrix[key]
                    repeat_count = initial_count = missing_count = first_changed = correct_first = lost_first = 0
                    calls_max = []
                    query_lengths = []
                    for tid, row in rows.items():
                        calls = row.get("quantize_calls_by_layer")
                        if not isinstance(calls, list) or not calls or any(not isinstance(v, int) or v < 0 for v in calls):
                            missing_count += 1
                        else:
                            calls_max.append(max(calls))
                            repeat_count += max(calls) > 2
                            initial_count += all(v == 2 for v in calls)
                        query_lengths.append(row["input_tokens"] - row["context_split"])
                        changed = row["output_ids"][0] != base[tid]["output_ids"][0]
                        first_changed += changed
                        if dataset == "passage_retrieval_en":
                            gold = expected_numbers(row)
                            current_correct = first_int(row["pred"]) in gold
                            correct_first += current_correct
                            lost_first += first_int(base[tid]["pred"]) in gold and not current_correct
                        if cache != "dynamic" and row["output_ids"] != base[tid]["output_ids"]:
                            examples.append({"model": model, "dataset": dataset, "protocol": protocol,
                                "cache": cache, "task_id": tid, "context_sha256": row["context_sha256"],
                                "first_difference_index": first_difference(base[tid]["output_ids"], row["output_ids"]),
                                "reference_pred": base[tid]["pred"], "candidate_pred": row["pred"]})
                    cache_rows.append({"model": model, "dataset": dataset, "protocol": protocol,
                        "cache": cache, "n": len(rows), "score_pct": 100 * sum(scores[key].values()) / len(rows),
                        "first_token_changed_vs_dynamic": first_changed,
                        "single_prefill_first_token_invariant_violations": first_changed if protocol == "single_prefill" else None,
                        "first_number_correct": correct_first if dataset == "passage_retrieval_en" else None,
                        "baseline_correct_first_number_lost": lost_first if dataset == "passage_retrieval_en" else None,
                        "max_token_limit_count": sum(r["stop_reason"] == "max_new_tokens" for r in rows.values()),
                        "think_closing_text_count": sum("</think>" in r["pred"] for r in rows.values()),
                        "initial_quantization_only_count": initial_count,
                        "repeated_quantization_count": repeat_count,
                        "quantization_metadata_missing_count": missing_count,
                        "max_quantize_calls_per_layer": max(calls_max) if calls_max else None,
                        "query_tokens_min": min(query_lengths), "query_tokens_median": median(query_lengths),
                        "query_tokens_max": max(query_lengths)})

    for reference, candidate in combinations(config["models"], 2):
        for dataset in config["datasets"]:
            for protocol in config["protocols"]:
                a = scores[(reference, dataset, protocol, "dynamic")]
                aq = scores[(reference, dataset, protocol, "int4_g64")]
                b = scores[(candidate, dataset, protocol, "dynamic")]
                bq = scores[(candidate, dataset, protocol, "int4_g64")]
                delta = {tid: (bq[tid] - b[tid]) - (aq[tid] - a[tid]) for tid in a}
                comparisons.append({"reference_model": reference, "candidate_model": candidate,
                    "dataset": dataset, "protocol": protocol, "comparison": "candidate_g64_loss-reference_g64_loss",
                    "n": len(a), **interval(delta, matrix[(reference, dataset, protocol, "dynamic")])})
    return dynamic, cache_rows, comparisons, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/protocol_v1/full")
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    parser.add_argument("--out-dir", default="results/protocol_v1/details/full")
    args = parser.parse_args()
    config = load_config()
    matrix = read_matrix(args.results_dir, config, "full")
    metric_hash = sha256_file(Path(args.longbench_code) / "metrics.py")
    if any(row["provenance"]["longbench_files_sha256"]["metrics.py"] != metric_hash
           for rows in matrix.values() for row in rows.values()):
        raise ValueError("metrics.py отличается от версии при подготовке серии")
    dynamic, caches, comparisons, examples = analyze_details(matrix, config, load_official_metrics(args.longbench_code))
    out = Path(args.out_dir)
    for filename, rows in (("dynamic_drift.csv", dynamic), ("cache_diagnostics.csv", caches),
                           ("model_comparisons.csv", comparisons), ("changed_examples.csv", examples)):
        write_csv(out / filename, rows)
    print("Dynamic single vs split (delta = split - single):")
    for row in dynamic:
        print(row)
    print("\nДиагностика g64 (счётчик _quantize учитывает K и V отдельно):")
    for row in caches:
        if row["cache"] == "int4_g64":
            print(row)
    print("\nПрямой контраст SmolLM3-3B против Qwen3-4B:")
    for row in comparisons:
        if row["reference_model"] == "qwen3-4b" and row["candidate_model"] == "smollm3-3b":
            print(row)
    print("\nCSV:", out)
    print("Первое число и </think> — диагностические признаки. Они не заменяют LongBench score.")
    print("Повторная квантизация и расхождения Dynamic требуют отдельных контролей; эти счётчики не доказывают механизм.")


if __name__ == "__main__":
    main()
