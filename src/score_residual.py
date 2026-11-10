"""Парный анализ residual512−128: официальный retrieval_score и первое число."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.diagnose_family_retrieval import expected_numbers, first_int
from src.download_data import sha256_file
from src.residual import load_config, read_matrix, reference_matrices
from src.score import load_official_metrics, score_prediction, write_csv
from src.score_protocol import cluster_ci


def analyze(matrix, reference, config, metric):
    summaries, contrasts, examples, effects = [], [], [], {}

    def add(model, cache, comparison, values, rows, metric_name="retrieval_score"):
        groups = {tid: rows[tid]["context_sha256"] for tid in values}
        low, high = cluster_ci(values, groups)
        contrasts.append({"model": model, "cache": cache, "metric": metric_name,
            "comparison": comparison, "n": len(values), "documents": len(set(groups.values())),
            "delta_pp": 100 * sum(values.values()) / len(values),
            "cluster_ci95_low_pp": 100 * low, "cluster_ci95_high_pp": 100 * high})

    for (model, dataset, protocol, cache), rows in matrix.items():
        old = reference[(model, dataset, protocol, cache)]
        dynamic = reference[(model, dataset, protocol, "dynamic")]
        g16 = reference[(model, dataset, protocol, "int4_g16")]
        scores = {tid: score_prediction(row, metric) for tid, row in rows.items()}
        old_scores = {tid: score_prediction(old[tid], metric) for tid in rows}
        if any(not 0 <= value <= 1 for value in (*scores.values(), *old_scores.values())):
            raise ValueError("Официальная метрика вне [0,1]")
        first, old_first = {}, {}
        for tid, row in rows.items():
            gold = expected_numbers(row)
            if not gold:
                raise ValueError(f"Не найден номер эталонного абзаца: {tid}")
            first[tid] = int(first_int(row["pred"]) in gold)
            old_first[tid] = int(first_int(old[tid]["pred"]) in gold)
            examples.append({"model": model, "cache": cache, "task_id": tid,
                "context_sha256": row["context_sha256"], "score512_pct": 100 * scores[tid],
                "score128_pct": 100 * old_scores[tid], "first512_correct": first[tid],
                "first128_correct": old_first[tid], "pred512": row["pred"], "pred128": old[tid]["pred"]})
        delta = {tid: scores[tid] - old_scores[tid] for tid in rows}
        effects[(model, cache)] = delta
        summaries.append({"model": model, "dataset": dataset, "protocol": protocol, "cache": cache,
            "n": len(rows), "documents": len({r["context_sha256"] for r in rows.values()}),
            "residual512_score_pct": 100 * sum(scores.values()) / len(rows),
            "residual128_score_pct": 100 * sum(old_scores.values()) / len(rows),
            "first_number_correct512": sum(first.values()), "first_number_correct128": sum(old_first.values()),
            "eos_count512": sum(r["stop_reason"] == "eos" for r in rows.values()),
            "max_token_limit_count512": sum(r["stop_reason"] == "max_new_tokens" for r in rows.values()),
            "think_closing_text_count512": sum("</think>" in r["pred"] for r in rows.values()),
            "repeated_quantization_count512": sum(max(r["quantize_calls_by_layer"]) > 2 for r in rows.values()),
            "repeated_quantization_count128": sum(max(old[t]["quantize_calls_by_layer"]) > 2 for t in rows),
            "first_token_changed_vs128": sum(rows[t]["output_ids"][0] != old[t]["output_ids"][0] for t in rows),
            "identical_output_ids_vs128": sum(rows[t]["output_ids"] == old[t]["output_ids"] for t in rows)})
        add(model, cache, "residual512-residual128", delta, rows)
        add(model, cache, "residual512-residual128", {tid: first[tid] - old_first[tid] for tid in rows}, rows, "first_number")
        for label, baseline in (("dynamic", dynamic), ("g16_residual128", g16)):
            add(model, cache, f"residual512-{label}",
                {tid: scores[tid] - score_prediction(baseline[tid], metric) for tid in rows}, rows)
    for model in config["models"]:
        a, b = effects[(model, "int4_g32")], effects[(model, "int4_g64")]
        rows = matrix[(model, "passage_retrieval_en", "context_then_question", "int4_g32")]
        add(model, "interaction", "residual_effect:g64-g32", {t: b[t] - a[t] for t in a}, rows)
    left, right = config["models"]
    for cache in config["caches"]:
        a, b = effects[(left, cache)], effects[(right, cache)]
        rows = matrix[(right, "passage_retrieval_en", "context_then_question", cache)]
        add(right, cache, f"residual_effect:{right}-{left}", {t: b[t] - a[t] for t in a}, rows)
    return summaries, contrasts, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    args = parser.parse_args()
    config = load_config()
    reference = reference_matrices()
    matrix = read_matrix(Path("results/residual_v1") / args.phase, config, args.phase, reference)
    print(f"Матрица residual проверена: {len(matrix)} ячеек, {sum(len(r) for r in matrix.values())} ответов")
    print("Refresh подавлен во всех слоях; первые токены совпали с residual=128")
    if args.check_only:
        return
    metric_hash = sha256_file(Path(args.longbench_code) / "metrics.py")
    if any(r["provenance"]["longbench_files_sha256"]["metrics.py"] != metric_hash
           for rows in matrix.values() for r in rows.values()):
        raise ValueError("metrics.py отличается от baseline")
    summaries, contrasts, examples = analyze(matrix, reference, config,
        load_official_metrics(args.longbench_code)["passage_retrieval_en"])
    out = Path("results/residual_v1/summary") / args.phase
    for name, data in (("dataset_scores.csv", summaries), ("paired_contrasts.csv", contrasts), ("example_scores.csv", examples)):
        write_csv(out / name, data)
    for row in summaries:
        print(row)
    print("CSV:", out, "; CI exploratory, без поправки множественных сравнений")
    print("Первое число — вспомогательная метрика. Больший residual сохраняет больше точных состояний.")
    if args.phase == "smoke":
        print("Smoke проверяет механику; качество по трём вопросам не обобщаем")


if __name__ == "__main__":
    main()
