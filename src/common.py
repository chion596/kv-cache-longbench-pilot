"""Small, deterministic helpers shared by data preparation and scoring."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

DATASETS = ("passage_retrieval_en", "multifieldqa_en", "multifieldqa_zh")
CACHES = ("dynamic", "int4_g16", "int4_g64")


def read_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON: {path}:{line_no}") from e


def write_jsonl(path: str | Path, records):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def deterministic_indices(population: int, sample_size: int, seed: int):
    """Same seed and n give the same task indices for every cache configuration."""
    if population < 0 or sample_size < 0:
        raise ValueError("population and sample_size must be >= 0")
    return sorted(random.Random(seed).sample(range(population), min(population, sample_size)))


def middle_truncate(ids: list[int], max_tokens: int):
    """Retain beginning and end, as in LongBench baseline's middle truncation."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if len(ids) <= max_tokens:
        return list(ids), False
    left = max_tokens // 2
    right = max_tokens - left
    return list(ids[:left]) + list(ids[-right:]), True


def digest_ids(ids: list[int]):
    encoded = json.dumps(ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_official_configs(longbench_code_dir: str | Path):
    root = Path(longbench_code_dir)
    prompt_path = root / "config" / "dataset2prompt.json"
    maxlen_path = root / "config" / "dataset2maxlen.json"
    if not prompt_path.is_file() or not maxlen_path.is_file():
        raise FileNotFoundError(
            f"Expected official LongBench files in {root}/config. "
            "Clone THUDM/LongBench and pass --longbench-code external/LongBench/LongBench"
        )
    with prompt_path.open(encoding="utf-8") as f:
        prompts = json.load(f)
    with maxlen_path.open(encoding="utf-8") as f:
        maxlens = json.load(f)
    for name in DATASETS:
        if name not in prompts or name not in maxlens:
            raise KeyError(f"Official LongBench config has no dataset {name}")
        if not isinstance(maxlens[name], int) or maxlens[name] <= 0:
            raise ValueError(f"Invalid max generation tokens for {name}")
    return prompts, maxlens
