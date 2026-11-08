"""Отдельная серия g32; исходные token IDs и baseline берём из protocol_v1."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from src.common import read_jsonl
from src.download_data import sha256_file
from src.protocol import digest_json, load_config as protocol_config, model_signature, result_path, validate_task
from src.score_protocol import key, read_matrix as read_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/group_size_v1.json"
REFERENCE_ROOT = Path("results/protocol_v1/full")
MATCH_FIELDS = ("input_ids_sha256", "context_sha256", "context_split", "answers", "all_classes",
                "max_new_tokens", "middle_truncated", "input_tokens")
ENV_FIELDS = ("manifest_sha256", "longbench_files_sha256", "model_artifacts", "versions", "gpu", "cuda",
              "dtype", "attention", "residual_length", "source_hashes")


def load_config(path=CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    original = protocol_config()
    if config["series"] != "group_size_v1" or config["reference_series"] != original["series"]:
        raise ValueError("Неверная серия g32 или ссылка на baseline")
    if config["models"] != ["qwen3-1.7b", "qwen3-4b"]:
        raise ValueError("Серия g32 фиксирует две модели Qwen")
    for field in ("datasets", "protocols", "n", "smoke_n", "residual_length"):
        if config[field] != original[field]:
            raise ValueError(f"g32 должен сохранять исходный {field}")
    if config["cache"] != "int4_g32":
        raise ValueError("Эта серия меняет только group size на 32")
    return config


def cells(config):
    return [{"model_name": model, "dataset": dataset, "protocol": protocol, "cache": config["cache"]}
            for model, dataset, protocol in itertools.product(config["models"], config["datasets"], config["protocols"])]


def reference_matrix():
    return read_protocol(REFERENCE_ROOT, protocol_config(), "full")


def references_for(matrix, cell):
    return {cache: matrix[(cell["model_name"], cell["dataset"], cell["protocol"], cache)]
            for cache in ("dynamic", "int4_g16", "int4_g64")}


def reference_fingerprints(references):
    return {cache: next(iter(rows.values()))["run_fingerprint"] for cache, rows in references.items()}


def selected_task_ids(reference, cell, config, phase):
    baseline = reference[(cell["model_name"], cell["dataset"], "single_prefill", "dynamic")]
    ids = sorted(baseline, key=lambda tid: baseline[tid]["source_index"])
    return ids[:config["smoke_n"]] if phase == "smoke" else ids


def validate_pair(row, baseline):
    if any(row.get(field) != baseline.get(field) for field in MATCH_FIELDS):
        raise ValueError(f"Изменились входы/эталоны: {row['task_id']}")
    for field in ENV_FIELDS:
        if row["provenance"].get(field) != baseline["provenance"].get(field):
            raise ValueError(f"Условия отличаются от protocol_v1: {field}")


def prepared_tasks(cell, model_path, references):
    folder = Path("data/prepared/protocol_v1") / cell["model_name"]
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline = references["dynamic"]
    example = next(iter(baseline.values()))
    if (manifest["config_sha256"] != digest_json(protocol_config())
            or manifest["model_name"] != cell["model_name"]
            or sha256_file(manifest_path) != example["provenance"]["manifest_sha256"]):
        raise ValueError("Manifest отличается от завершённой серии")
    if model_signature(model_path) != example["provenance"]["model_artifacts"]:
        raise ValueError("Модель/токенизатор отличаются от protocol_v1")
    filename = folder / f"{cell['dataset']}.jsonl"
    dataset = manifest["datasets"][cell["dataset"]]
    if sha256_file(filename) != dataset["file_sha256"]:
        raise ValueError("Подготовленные входы изменились")
    tasks = list(read_jsonl(filename))
    if len(tasks) != len(baseline) or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Неполный/повторный набор входов g32")
    if {t["task_id"]: t["input_ids_sha256"] for t in tasks} != dataset["task_hashes"]:
        raise ValueError("Входы не совпали с manifest")
    if {t["task_id"] for t in tasks} != set(baseline):
        raise ValueError("Входы не совпали с baseline")
    for task in tasks:
        validate_task(task)
        if any(task.get(field) != baseline[task["task_id"]].get(field) for field in MATCH_FIELDS):
            raise ValueError(f"Задача отличается от baseline: {task['task_id']}")
    return tasks


def validate_environment(provenance, *, versions, gpu=None, cuda=None):
    if versions != provenance["versions"]:
        raise ValueError("Версии пакетов изменились после protocol_v1; не обновляйте рабочее окружение")
    for name, digest in provenance["source_hashes"].items():
        if sha256_file(ROOT / "src" / name) != digest:
            raise ValueError(f"Изменился общий путь генерации: {name}")
    if gpu is not None and gpu != provenance["gpu"]:
        raise ValueError("Нужен тот же тип GPU, что в baseline")
    if cuda is not None and cuda != provenance["cuda"]:
        raise ValueError("Версия CUDA отличается от baseline")


def validate_result(row, cell, config, phase, references, *, reference_sha256=None):
    tid = row["task_id"]
    if any(row.get(field) != value for field, value in cell.items()):
        raise ValueError(f"Чужая ячейка g32: {tid}")
    p = row["provenance"]
    if (p.get("series") != config["series"] or p.get("phase") != phase
            or p.get("config_sha256") != digest_json(config)
            or row.get("run_fingerprint") != digest_json(p)
            or p.get("reference_fingerprints") != reference_fingerprints(references)
            or p.get("reference_results_sha256") != (reference_sha256 or digest_json(references))
            or any(p.get(field) != value for field, value in cell.items())):
        raise ValueError(f"Некорректный fingerprint g32: {tid}")
    if (not row.get("output_ids") or len(row["output_ids"]) != row["output_tokens"]
            or row["stop_reason"] not in ("eos", "max_new_tokens")):
        raise ValueError(f"Неполный результат g32: {tid}")
    calls = row.get("quantize_calls_by_layer")
    if not isinstance(calls, list) or not calls or any(not isinstance(v, int) or v < 2 or v % 2 for v in calls):
        raise ValueError(f"Нет счётчиков квантизации: {tid}")
    validate_pair(row, references["dynamic"][tid])
    if cell["protocol"] == "single_prefill" and row["output_ids"][0] != references["dynamic"][tid]["output_ids"][0]:
        raise ValueError(f"Первый токен изменился до чтения cache: {tid}")


def read_matrix(root, config, phase, reference):
    matrix = {}
    control_sources = None
    for cell in cells(config):
        filename = result_path(root, cell)
        if not filename.is_file():
            raise FileNotFoundError(f"Нет результата {filename}; проверьте Slurm и logs/lb-g32-v1-*.err")
        references = references_for(reference, cell)
        reference_hash = digest_json(references)
        expected = selected_task_ids(reference, cell, config, phase)
        indexed = {}
        for row in read_jsonl(filename):
            tid = row["task_id"]
            if tid not in expected or tid in indexed:
                raise ValueError(f"Повторная/чужая задача g32: {filename}/{tid}")
            validate_result(row, cell, config, phase, references, reference_sha256=reference_hash)
            sources = row["provenance"]["control_source_hashes"]
            if control_sources is None:
                control_sources = sources
            if sources != control_sources:
                raise ValueError("Смешаны разные версии кода g32")
            indexed[tid] = row
        if set(indexed) != set(expected):
            raise ValueError(f"Неполный результат {filename}: {len(indexed)}/{len(expected)}")
        if len({r["run_fingerprint"] for r in indexed.values()}) != 1:
            raise ValueError(f"Смешаны разные условия: {filename}")
        matrix[key(cell)] = indexed
    return matrix
