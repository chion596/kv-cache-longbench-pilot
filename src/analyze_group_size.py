"""CPU-разбор g32: первый токен, первое число и ошибки относительно Dynamic/g16/g64."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.diagnose_family_retrieval import expected_numbers, first_int
from src.download_data import sha256_file
from src.group_size import load_config, read_matrix, reference_matrix
from src.score import load_official_metrics, score_prediction, write_csv
from src.score_protocol import cluster_ci


def analyze_details(matrix, reference, metrics):
    summaries, number_contrasts, trajectories = [], [], []
    for (model, dataset, protocol, _), rows in matrix.items():
        sources = {cache: reference[(model, dataset, protocol, cache)] for cache in ("dynamic", "int4_g16", "int4_g64")}
        sources["int4_g32"] = rows
        scores = {cache: {tid: score_prediction(data[tid], metrics[dataset]) for tid in rows}
                  for cache, data in sources.items()}
        if any(not 0 <= v <= 1 for by_task in scores.values() for v in by_task.values()):
            raise ValueError("Официальная метрика вне [0,1]")
        base, quant = scores["dynamic"], scores["int4_g32"]
        changed = {tid: rows[tid]["output_ids"][0] != sources["dynamic"][tid]["output_ids"][0] for tid in rows}
        numbers = {}
        if dataset == "passage_retrieval_en":
            for tid in rows:
                if not expected_numbers(rows[tid]):
                    raise ValueError(f"Нет номера эталонного абзаца: {tid}")
            numbers = {cache: {tid: int(first_int(data[tid]["pred"]) in expected_numbers(data[tid])) for tid in rows}
                       for cache, data in sources.items()}
        lower = {tid: quant[tid] < base[tid] - 1e-12 for tid in rows}
        higher = {tid: quant[tid] > base[tid] + 1e-12 for tid in rows}
        summaries.append({"model": model, "dataset": dataset, "protocol": protocol, "n": len(rows),
            "documents": len({r["context_sha256"] for r in rows.values()}),
            "g32_score_pct": 100 * sum(quant.values()) / len(rows),
            "first_token_changed": sum(changed.values()),
            "changed_first_token_score_lower": sum(changed[t] and lower[t] for t in rows),
            "changed_first_token_score_same": sum(changed[t] and not lower[t] and not higher[t] for t in rows),
            "changed_first_token_score_higher": sum(changed[t] and higher[t] for t in rows),
            "changed_first_token_full_score": sum(changed[t] and quant[t] == 1 for t in rows),
            "same_first_token_score_lower": sum(not changed[t] and lower[t] for t in rows),
            "first_number_correct": sum(numbers["int4_g32"].values()) if numbers else None,
            "dynamic_first_number_correct": sum(numbers["dynamic"].values()) if numbers else None,
            "baseline_correct_first_number_lost": sum(numbers["dynamic"][t] and not numbers["int4_g32"][t] for t in rows) if numbers else None,
            "first_number_recovered": sum(not numbers["dynamic"][t] and numbers["int4_g32"][t] for t in rows) if numbers else None,
            "max_token_limit_count": sum(r["stop_reason"] == "max_new_tokens" for r in rows.values()),
            "think_closing_text_count": sum("</think>" in r["pred"] for r in rows.values())})
        if numbers:
            for cache in ("dynamic", "int4_g16", "int4_g64"):
                delta = {tid: numbers["int4_g32"][tid] - numbers[cache][tid] for tid in rows}
                groups = {tid: rows[tid]["context_sha256"] for tid in rows}
                low, high = cluster_ci(delta, groups)
                number_contrasts.append({"model": model, "dataset": dataset, "protocol": protocol,
                    "comparison": f"first_number:int4_g32-{cache}", "n": len(rows),
                    "documents": len(set(groups.values())), "delta_pp": 100 * sum(delta.values()) / len(rows),
                    "cluster_ci95_low_pp": 100 * low, "cluster_ci95_high_pp": 100 * high})
        for tid, row in rows.items():
            item = {"model": model, "dataset": dataset, "protocol": protocol, "task_id": tid,
                "context_sha256": row["context_sha256"], "input_tokens": row["input_tokens"],
                "first_token_changed_g32": changed[tid]}
            for cache in ("dynamic", "int4_g16", "int4_g32", "int4_g64"):
                item[f"{cache}_score_pct"] = 100 * scores[cache][tid]
                item[f"{cache}_first_number_correct"] = numbers[cache][tid] if numbers else None
                item[f"{cache}_stop_reason"] = sources[cache][tid]["stop_reason"]
            trajectories.append(item)
    return summaries, number_contrasts, trajectories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    parser.add_argument("--out-dir", default="results/group_size_v1/details/full")
    args = parser.parse_args()
    config = load_config()
    reference = reference_matrix()
    matrix = read_matrix("results/group_size_v1/full", config, "full", reference)
    metric_hash = sha256_file(Path(args.longbench_code) / "metrics.py")
    if any(r["provenance"]["longbench_files_sha256"]["metrics.py"] != metric_hash
           for data in matrix.values() for r in data.values()):
        raise ValueError("metrics.py отличается от серии")
    summaries, contrasts, trajectories = analyze_details(matrix, reference, load_official_metrics(args.longbench_code))
    out = Path(args.out_dir)
    for name, data in (("answer_diagnostics.csv", summaries), ("first_number_contrasts.csv", contrasts), ("trajectories.csv", trajectories)):
        write_csv(out / name, data)
    print("Первый токен и официальный score; сравнение с собственным Dynamic:")
    for row in summaries:
        print(row)
    print("\nПервое число retrieval — вспомогательная метрика:")
    for row in contrasts:
        print(row)
    print("\nCSV:", out)
    print("Разбиение по изменению первого токена описательное и не устанавливает причинность.")
    print("Официальная метрика LongBench не меняется. Повторная генерация не требуется.")


if __name__ == "__main__":
    main()
