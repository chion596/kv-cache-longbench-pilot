"""Контроль residual=512 на тех же split retrieval входах и весах."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from src.common import read_jsonl
from src.group_size import (ENV_FIELDS, MATCH_FIELDS, ROOT, load_config as group_config,
                            read_matrix as read_group, reference_fingerprints, reference_matrix)
from src.protocol import digest_json, result_path
from src.score_protocol import key

CONFIG = ROOT / "experiments/residual_v1.json"


def load_config(path=CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    group = group_config()
    expected = {"series": "residual_v1", "reference_series": ["protocol_v1", "group_size_v1"],
        "models": group["models"], "datasets": ["passage_retrieval_en"],
        "protocols": ["context_then_question"], "caches": ["int4_g32", "int4_g64"],
        "reference_residual_length": 128, "residual_length": 512, "n": group["n"], "smoke_n": group["smoke_n"]}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Матрица residual_v1 фиксирует split retrieval и residual 128→512")
    return config


def cells(config):
    return [dict(zip(("model_name", "dataset", "protocol", "cache"), values))
            for values in itertools.product(config["models"], config["datasets"], config["protocols"], config["caches"])]


def reference_matrices():
    original = reference_matrix()
    g32 = read_group("results/group_size_v1/full", group_config(), "full", original)
    return {**original, **g32}


def references_for(matrix, cell):
    return {cache: matrix[(cell["model_name"], cell["dataset"], cell["protocol"], cache)]
            for cache in ("dynamic", "int4_g16", "int4_g32", "int4_g64")}


def selected_ids(references, config, phase):
    baseline = references["dynamic"]
    ids = sorted(baseline, key=lambda tid: baseline[tid]["source_index"])
    return ids[:config["smoke_n"]] if phase == "smoke" else ids


def validate_budget(task, config):
    query = task["input_tokens"] - task["context_split"]
    if query <= 0 or query + task["max_new_tokens"] >= config["residual_length"]:
        raise ValueError(f"Residual не превышает query+answer: {task['task_id']}")


def validate_result(row, cell, config, phase, references, *, reference_hash=None):
    tid = row["task_id"]
    p = row["provenance"]
    if (any(row.get(k) != v or p.get(k) != v for k, v in cell.items())
            or p.get("series") != config["series"] or p.get("phase") != phase
            or p.get("config_sha256") != digest_json(config)
            or row.get("run_fingerprint") != digest_json(p)
            or p.get("reference_fingerprints") != reference_fingerprints(references)
            or p.get("reference_results_sha256") != (reference_hash or digest_json(references))):
        raise ValueError(f"Некорректный fingerprint residual: {tid}")
    if p.get("residual_length") != config["residual_length"] or row.get("residual_length") != config["residual_length"]:
        raise ValueError(f"Неверный residual_length: {tid}")
    old = references[cell["cache"]][tid]
    if old["provenance"]["residual_length"] != config["reference_residual_length"]:
        raise ValueError("Baseline должен использовать residual=128")
    if any(row.get(field) != old.get(field) for field in MATCH_FIELDS):
        raise ValueError(f"Изменились входы/эталоны residual: {tid}")
    for field in ENV_FIELDS:
        if field != "residual_length" and p.get(field) != old["provenance"].get(field):
            raise ValueError(f"Несопоставимые условия residual: {field}/{tid}")
    if (not row.get("output_ids") or len(row["output_ids"]) != row["output_tokens"]
            or not 1 <= row["output_tokens"] <= row["max_new_tokens"]
            or row["stop_reason"] not in ("eos", "max_new_tokens")):
        raise ValueError(f"Неполная генерация residual: {tid}")
    validate_budget(row, config)
    if row.get("prefill_split") != row["context_split"] or row.get("query_tokens") != row["input_tokens"] - row["context_split"]:
        raise ValueError(f"Не совпала граница prefill: {tid}")
    calls = row.get("quantize_calls_by_layer")
    old_calls = old.get("quantize_calls_by_layer")
    if (not isinstance(calls, list) or not calls or not isinstance(old_calls, list)
            or len(calls) != len(old_calls) or any(type(v) is not int or v != 2 for v in calls)):
        raise ValueError(f"Ожидалась только исходная квантизация K/V: {tid}; calls={calls}")
    if row["output_ids"][0] != old["output_ids"][0]:
        raise ValueError(f"Первый токен изменился относительно residual=128 до refresh: {tid}")


def read_matrix(root, config, phase, reference):
    matrix, control_sources = {}, None
    for cell in cells(config):
        path = result_path(root, cell)
        if not path.is_file():
            raise FileNotFoundError(f"Нет {path}; проверьте Slurm и logs/lb-residual-v1-*.err")
        references = references_for(reference, cell)
        reference_hash = digest_json(references)
        expected = selected_ids(references, config, phase)
        indexed = {}
        for row in read_jsonl(path):
            tid = row["task_id"]
            if tid not in expected or tid in indexed:
                raise ValueError(f"Повторная/чужая задача residual: {path}/{tid}")
            validate_result(row, cell, config, phase, references, reference_hash=reference_hash)
            sources = row["provenance"]["control_source_hashes"]
            if control_sources is None:
                control_sources = sources
            if sources != control_sources:
                raise ValueError("Смешаны разные версии runner residual")
            indexed[tid] = row
        if set(indexed) != set(expected):
            raise ValueError(f"Неполная ячейка residual: {path}: {len(indexed)}/{len(expected)}")
        if len({r["run_fingerprint"] for r in indexed.values()}) != 1:
            raise ValueError(f"Смешаны разные условия residual: {path}")
        matrix[key(cell)] = indexed
    return matrix
