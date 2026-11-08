"""Парные сравнения g32 с неизменёнными Dynamic/g16/g64 из protocol_v1."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.download_data import sha256_file
from src.group_size import load_config, read_matrix, reference_matrix
from src.score import load_official_metrics, score_prediction, write_csv
from src.score_protocol import cluster_ci


def analyze(matrix, reference, config, metrics):
    summaries, contrasts, examples, losses = [], [], [], {}

    def add(model, dataset, protocol, comparison, values, rows):
        groups = {tid: rows[tid]["context_sha256"] for tid in values}
        low, high = cluster_ci(values, groups)
        contrasts.append({"model": model, "dataset": dataset, "protocol": protocol,
            "comparison": comparison, "n": len(values), "documents": len(set(groups.values())),
            "delta_pp": 100 * sum(values.values()) / len(values),
            "cluster_ci95_low_pp": 100 * low, "cluster_ci95_high_pp": 100 * high})

    for (model, dataset, protocol, cache), rows in matrix.items():
        scores = {tid: score_prediction(row, metrics[dataset]) for tid, row in rows.items()}
        if any(not 0 <= value <= 1 for value in scores.values()):
            raise ValueError("Официальная метрика вне [0,1]")
        base = reference[(model, dataset, protocol, "dynamic")]
        summaries.append({"model": model, "dataset": dataset, "protocol": protocol, "cache": cache,
            "n": len(rows), "documents": len({r["context_sha256"] for r in rows.values()}),
            "score_pct": 100 * sum(scores.values()) / len(rows),
            "eos_count": sum(r["stop_reason"] == "eos" for r in rows.values()),
            "first_token_changed_vs_dynamic": sum(r["output_ids"][0] != base[t]["output_ids"][0] for t, r in rows.items()),
            "repeated_quantization_count": sum(max(r["quantize_calls_by_layer"]) > 2 for r in rows.values())})
        for tid, row in rows.items():
            examples.append({"model": model, "dataset": dataset, "protocol": protocol,
                "task_id": tid, "context_sha256": row["context_sha256"], "score_pct": 100 * scores[tid],
                "stop_reason": row["stop_reason"], "pred": row["pred"]})
        for old_cache in ("dynamic", "int4_g16", "int4_g64"):
            old = reference[(model, dataset, protocol, old_cache)]
            delta = {tid: scores[tid] - score_prediction(old[tid], metrics[dataset]) for tid in rows}
            add(model, dataset, protocol, f"int4_g32-{old_cache}", delta, rows)
            if old_cache == "dynamic":
                losses[(model, dataset, protocol)] = delta
    for model in config["models"]:
        for dataset in config["datasets"]:
            whole = losses[(model, dataset, "single_prefill")]
            split = losses[(model, dataset, "context_then_question")]
            if set(whole) != set(split):
                raise ValueError("Для взаимодействия нужны одинаковые вопросы single/split")
            add(model, dataset, "interaction", "split_loss-single_loss:int4_g32",
                {t: split[t] - whole[t] for t in whole}, reference[(model, dataset, "single_prefill", "dynamic")])
    left, right = config["models"]
    for dataset in config["datasets"]:
        for protocol in config["protocols"]:
            a, b = losses[(left, dataset, protocol)], losses[(right, dataset, protocol)]
            add(right, dataset, protocol, f"g32_loss:{right}-{left}", {t: b[t] - a[t] for t in a},
                reference[(left, dataset, protocol, "dynamic")])
    return summaries, contrasts, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    args = parser.parse_args()
    config = load_config()
    reference = reference_matrix()
    matrix = read_matrix(Path("results/group_size_v1") / args.phase, config, args.phase, reference)
    print(f"Матрица g32 проверена: {len(matrix)} ячеек, {sum(len(v) for v in matrix.values())} ответов")
    if args.check_only:
        return
    metric_hash = sha256_file(Path(args.longbench_code) / "metrics.py")
    if any(r["provenance"]["longbench_files_sha256"]["metrics.py"] != metric_hash
           for rows in matrix.values() for r in rows.values()):
        raise ValueError("metrics.py отличается от baseline")
    summaries, contrasts, examples = analyze(matrix, reference, config, load_official_metrics(args.longbench_code))
    out = Path("results/group_size_v1/summary") / args.phase
    for filename, rows in (("dataset_scores.csv", summaries), ("paired_contrasts.csv", contrasts), ("example_scores.csv", examples)):
        write_csv(out / filename, rows)
    for row in summaries:
        print(row)
    print("CSV:", out, "; интервалы exploratory, без коррекции множественных сравнений")
    if args.phase == "smoke":
        print("Smoke проверяет совместимость; качество по трём вопросам не обобщаем")


if __name__ == "__main__":
    main()
