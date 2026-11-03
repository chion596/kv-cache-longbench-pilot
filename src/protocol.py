"""Фиксированная матрица новой серии; прежние пилоты остаются отдельными."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "experiments/protocol_v1.json"


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def load_config(path=CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["series"] != "protocol_v1":
        raise ValueError("Для другой серии нужен новый протокол")
    if config["prompt_mode"] != "native_chat_no_thinking":
        raise ValueError("Ожидался native chat без thinking")
    if config["dtype"] != "float16" or config["attention"] != "sdpa":
        raise ValueError("Эта серия фиксирует FP16/SDPA")
    for key in ("models", "datasets", "protocols", "caches"):
        values = config[key]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Пустая или повторная ось: {key}")
    return config


def cells(config):
    return [dict(zip(("model_name", "dataset", "protocol", "cache"), values))
            for values in itertools.product(config["models"], config["datasets"],
                                             config["protocols"], config["caches"])]


def result_path(root, cell):
    return Path(root) / cell["model_name"] / cell["protocol"] / (
        f"{cell['dataset']}_{cell['cache']}.jsonl")


def model_signature(path):
    """Configs/tokenizer по содержимому; веса по size/mtime, без чтения гигабайтов.

    Это fingerprint локальных артефактов, не криптографическая проверка весов.
    Источник/revision весов дополнительно сохраняется из model.config.
    """
    from src.download_data import sha256_file
    root = Path(path).resolve()
    files = {}
    for item in sorted(root.iterdir()):
        if item.is_file() and item.suffix in (".json", ".jinja", ".model", ".txt"):
            files[item.name] = {"sha256": sha256_file(item)}
        elif item.is_file() and item.suffix in (".safetensors", ".bin"):
            stat = item.stat()
            files[item.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if "config.json" not in files:
        raise FileNotFoundError(f"Нет config.json: {root}")
    return {"path": str(root), "files": files}


def validate_task(task):
    from src.common import digest_ids
    ids = task["input_ids"]
    if not ids or digest_ids(ids) != task["input_ids_sha256"]:
        raise ValueError(f"Повреждены token IDs: {task['task_id']}")
    if not 0 < task["context_split"] < len(ids):
        raise ValueError(f"Некорректная граница контекста: {task['task_id']}")
