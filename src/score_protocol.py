"""Проверка полной матрицы и парные интервалы по исходным документам."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random

from src.common import read_jsonl
from src.download_data import sha256_file
from src.protocol import cells, digest_json, load_config, result_path
from src.score import load_official_metrics, score_prediction, write_csv


def key(cell):
    return tuple(cell[field] for field in ("model_name", "dataset", "protocol", "cache"))


def read_matrix(root, config, phase):
    expected_n = config["smoke_n"] if phase == "smoke" else config["n"]
    matrix = {}
    for cell in cells(config):
        filename = result_path(root, cell)
        indexed = {}
        for row in read_jsonl(filename):
            tid = row["task_id"]
            if tid in indexed or tid != f"{cell['dataset']}:{row['source_index']}":
                raise ValueError(f"Повторная/чужая задача: {filename}/{tid}")
            if any(row.get(field) != value for field, value in cell.items()):
                raise ValueError(f"Неверная конфигурация: {filename}/{tid}")
            provenance = row["provenance"]
            if (provenance.get("phase") != phase or provenance.get("config_sha256") != digest_json(config)
                    or row.get("run_fingerprint") != digest_json(provenance)):
                raise ValueError(f"Некорректный fingerprint: {filename}/{tid}")
            if any(provenance.get(field) != value for field, value in cell.items()):
                raise ValueError(f"Ячейка не совпала с provenance: {filename}/{tid}")
            if not row.get("context_sha256") or not row.get("output_ids"):
                raise ValueError(f"Нет context hash/output IDs: {filename}/{tid}")
            if len(row["output_ids"]) != row["output_tokens"] or row["stop_reason"] not in ("eos", "max_new_tokens"):
                raise ValueError(f"Неполная генерация: {filename}/{tid}")
            indexed[tid] = row
        if len(indexed) != expected_n:
            raise ValueError(f"{filename}: ожидалось {expected_n}, получено {len(indexed)}")
        if len({row["run_fingerprint"] for row in indexed.values()}) != 1:
            raise ValueError(f"В одном файле смешаны запуски: {filename}")
        matrix[key(cell)] = indexed

    match_fields = ("input_ids_sha256", "context_sha256", "context_split", "answers", "max_new_tokens", "middle_truncated")
    for model in config["models"]:
        for dataset in config["datasets"]:
            base = matrix[(model, dataset, "single_prefill", "dynamic")]
            for protocol in config["protocols"]:
                for cache in config["caches"]:
                    other = matrix[(model, dataset, protocol, cache)]
                    if set(base) != set(other):
                        raise ValueError("Между cache/protocol сравниваются разные задачи")
                    for tid, a in base.items():
                        b = other[tid]
                        if any(a[field] != b[field] for field in match_fields):
                            raise ValueError(f"Изменились входы/эталоны: {model}/{tid}")
                        # Провенанс одинаков для cache/protocol, кроме самих этих осей.
                        for field in ("manifest_sha256", "longbench_files_sha256", "versions", "gpu", "cuda", "dtype", "attention", "residual_length", "source_hashes"):
                            if a["provenance"][field] != b["provenance"][field]:
                                raise ValueError(f"Несопоставимые условия: {model}/{tid}/{field}")
                        if phase == "smoke" and protocol == "single_prefill" and cache == "dynamic":
                            if a.get("reference_matches") is not True or a.get("reference_output_ids") != a["output_ids"]:
                                raise ValueError(f"Не пройдена сверка model.generate: {model}/{tid}")
                        if phase == "smoke" and protocol == "context_then_question" and cache == "dynamic":
                            if a["output_ids"] != b["output_ids"]:
                                raise ValueError(f"Dynamic изменился при split prefill: {model}/{tid}; нужен разбор")
                        if phase == "smoke" and protocol == "single_prefill" and cache != "dynamic":
                            if a["output_ids"][0] != b["output_ids"][0]:
                                raise ValueError(f"Первый токен изменился до чтения cache: {model}/{tid}")

    for dataset in config["datasets"]:
        base = matrix[(config["models"][0], dataset, "single_prefill", "dynamic")]
        for model in config["models"][1:]:
            other = matrix[(model, dataset, "single_prefill", "dynamic")]
            if set(base) != set(other) or any(base[t]["context_sha256"] != other[t]["context_sha256"]
                    or base[t]["answers"] != other[t]["answers"] for t in base):
                raise ValueError("Между моделями сравниваются разные исходные задачи")
            for tid in base:
                for field in ("longbench_files_sha256", "versions", "gpu", "cuda", "dtype", "attention", "residual_length", "source_hashes"):
                    if base[tid]["provenance"][field] != other[tid]["provenance"][field]:
                        raise ValueError(f"Между моделями различаются условия: {field}")
    return matrix


def cluster_ci(differences, groups, *, seed=2026, iters=2000):
    """Ресэмплируем документы; вместе переносим все вопросы документа.

    Точечная оценка и каждая bootstrap mean взвешены по вопросам.
    """
    if set(differences) != set(groups) or not differences or iters < 40:
        raise ValueError("Нужны совпадающие непустые задачи/группы и >=40 повторов")
    clustered = defaultdict(list)
    for tid in sorted(differences):
        clustered[groups[tid]].append(differences[tid])
    blocks = [(sum(v), len(v)) for _, v in sorted(clustered.items())]
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        sample = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        means.append(sum(s for s, _ in sample) / sum(n for _, n in sample))
    means.sort()
    return means[int(0.025 * iters)], means[min(iters - 1, int(0.975 * iters))]


def analyze(matrix, config, metrics):
    scored, summaries, examples = {}, [], []
    for (model, dataset, protocol, cache), rows in matrix.items():
        values = {tid: score_prediction(row, metrics[dataset]) for tid, row in rows.items()}
        if any(not 0 <= v <= 1 for v in values.values()):
            raise ValueError("Официальная метрика вне [0,1]")
        scored[(model, dataset, protocol, cache)] = values
        summaries.append({"model": model, "dataset": dataset, "protocol": protocol, "cache": cache,
            "n": len(rows), "documents": len({r["context_sha256"] for r in rows.values()}),
            "score_pct": 100 * sum(values.values()) / len(rows),
            "eos_count": sum(r["stop_reason"] == "eos" for r in rows.values()),
            "mean_output_tokens": sum(r["output_tokens"] for r in rows.values()) / len(rows)})
        for tid, row in rows.items():
            examples.append({"model": model, "dataset": dataset, "protocol": protocol, "cache": cache,
                "task_id": tid, "context_sha256": row["context_sha256"], "score_pct": values[tid] * 100,
                "output_tokens": row["output_tokens"], "stop_reason": row["stop_reason"], "pred": row["pred"]})

    contrasts = []
    def add(model, dataset, protocol, comparison, values):
        rows = matrix[(model, dataset, "single_prefill", "dynamic")]
        groups = {tid: row["context_sha256"] for tid, row in rows.items()}
        lo, hi = cluster_ci(values, groups)
        contrasts.append({"model": model, "dataset": dataset, "protocol": protocol,
            "comparison": comparison, "n": len(values), "documents": len(set(groups.values())),
            "delta_pp": 100 * sum(values.values()) / len(values),
            "cluster_ci95_low_pp": 100 * lo, "cluster_ci95_high_pp": 100 * hi})

    for model in config["models"]:
        for dataset in config["datasets"]:
            deltas = {}
            for protocol in config["protocols"]:
                base = scored[(model, dataset, protocol, "dynamic")]
                for cache in config["caches"][1:]:
                    candidate = scored[(model, dataset, protocol, cache)]
                    values = {tid: candidate[tid] - base[tid] for tid in base}
                    deltas[(protocol, cache)] = values
                    add(model, dataset, protocol, f"{cache}-dynamic", values)
            for cache in config["caches"][1:]:
                a, b = deltas[("single_prefill", cache)], deltas[("context_then_question", cache)]
                add(model, dataset, "interaction", f"split_loss-single_loss:{cache}",
                    {tid: b[tid] - a[tid] for tid in a})
    # Прямое различие внутримодельных потерь; общие task/document bootstrap units.
    reference_model = config["models"][0]
    for model in config["models"][1:]:
        for dataset in config["datasets"]:
            for protocol in config["protocols"]:
                a = scored[(reference_model, dataset, protocol, "dynamic")]
                aq = scored[(reference_model, dataset, protocol, "int4_g64")]
                b = scored[(model, dataset, protocol, "dynamic")]
                bq = scored[(model, dataset, protocol, "int4_g64")]
                add(model, dataset, protocol, f"g64_loss:{model}-{reference_model}",
                    {tid: (bq[tid] - b[tid]) - (aq[tid] - a[tid]) for tid in a})
    return summaries, contrasts, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    args = parser.parse_args()
    config = load_config()
    root = Path("results/protocol_v1") / args.phase
    matrix = read_matrix(root, config, args.phase)
    print(f"Матрица проверена: {len(matrix)} ячеек, {sum(len(v) for v in matrix.values())} ответов")
    if args.check_only:
        return
    metric_hash = sha256_file(Path(args.longbench_code) / "metrics.py")
    if any(row["provenance"]["longbench_files_sha256"]["metrics.py"] != metric_hash
           for rows in matrix.values() for row in rows.values()):
        raise ValueError("Официальный metrics.py отличается от версии при подготовке")
    summaries, contrasts, examples = analyze(matrix, config, load_official_metrics(args.longbench_code))
    out = Path("results/protocol_v1/summary") / args.phase
    write_csv(out / "dataset_scores.csv", summaries)
    write_csv(out / "paired_contrasts.csv", contrasts)
    write_csv(out / "example_scores.csv", examples)
    for row in summaries:
        print(row)
    print("CSV:", out, "; интервалы exploratory, без коррекции множественных сравнений")
    if args.phase == "smoke":
        print("Smoke проверяет совместимость; качество на трёх вопросах не обобщаем")


if __name__ == "__main__":
    main()
