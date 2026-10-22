"""Подготовить независимую контрольную выборку LongBench без пересечения с пилотом.

Новый набор из 80 задач на каждый датасет выбирается из дополнения
к исходным 40 задачам. Исходные файлы и manifest не перезаписываются.

На login-узле кластера:
    python -m src.prepare_heldout --model "$MODEL_PATH"
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from src.common import DATASETS, deterministic_indices, load_official_configs, read_jsonl, write_jsonl
from src.download_data import sha256_file
from src.prepare_data import prepare_records, read_dataset_from_zip


def select_heldout_indices(population: int, excluded, n: int, seed: int) -> list[int]:
    """Выбрать ровно n новых индексов из дополнения к исходной выборке."""
    excluded_list = list(excluded)
    excluded_set = set(excluded_list)
    if len(excluded_set) != len(excluded_list):
        raise ValueError("В исходной выборке встречаются повторные индексы")
    if population < 1 or n < 1:
        raise ValueError("Размер датасета и n должны быть положительными")
    if any(not isinstance(i, int) or i < 0 or i >= population for i in excluded_set):
        raise ValueError("Недопустимый индекс в исходной выборке")
    available = sorted(set(range(population)) - excluded_set)
    if n > len(available):
        raise ValueError(
            f"Недостаточно новых задач: доступно {len(available)}, требуется {n}"
        )
    return sorted(random.Random(seed).sample(available, n))


def load_original_selection(directory: Path, dataset: str, population: int,
                            original_n: int, original_seed: int):
    """Убедиться, что исключаем именно исходную заранее фиксированную выборку."""
    filename = directory / f"{dataset}.jsonl"
    if not filename.is_file():
        raise FileNotFoundError(f"Не найдены исходные подготовленные задания: {filename}")
    records = list(read_jsonl(filename))
    if len(records) != original_n:
        raise ValueError(
            f"{dataset}: в пилоте ожидалось {original_n} записей, найдено {len(records)}"
        )
    selected = []
    for record in records:
        index = record["source_index"]
        if record.get("dataset") != dataset or record.get("task_id") != f"{dataset}:{index}":
            raise ValueError(f"{dataset}: неверная идентичность исходного задания")
        if not isinstance(index, int):
            raise ValueError(f"{dataset}: source_index должен быть целым числом")
        selected.append(index)
    if len(set(selected)) != original_n:
        raise ValueError(f"{dataset}: повторный source_index в пилоте")
    expected = deterministic_indices(population, original_n, original_seed)
    if sorted(selected) != expected:
        raise ValueError(
            f"{dataset}: исходные задания не соответствуют зафиксированным "
            f"seed={original_seed}, n={original_n}. Не меняем выборку автоматически."
        )
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(
        description="Подготовить отдельную независимую выборку LongBench."
    )
    parser.add_argument("--model", required=True, help="Путь к локальным весам/токенизатору")
    parser.add_argument("--archive", default="data/raw/longbench-data.zip")
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    parser.add_argument("--original-dir", default="data/prepared")
    parser.add_argument("--out-dir", default="data/prepared/heldout")
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    args = parser.parse_args()

    if args.n < 1 or args.max_input_tokens < 1:
        parser.error("--n и --max-input-tokens должны быть >= 1")
    original_dir = Path(args.original_dir)
    out_dir = Path(args.out_dir)
    if original_dir.resolve() == out_dir.resolve():
        parser.error("Нельзя перезаписывать исходную выборку пилота")
    if out_dir.exists():
        raise FileExistsError(
            f"Каталог независимой выборки уже существует: {out_dir}. "
            "Не перезаписываем его; проверь существующие manifest/JSONL."
        )

    archive = Path(args.archive)
    manifest_path = original_dir / "manifest.json"
    if not archive.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Нет архива LongBench или исходного manifest.json")
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if original_manifest.get("sampling_seed") != 2026 or original_manifest.get("requested_n") != 40:
        raise ValueError("Исходный manifest не соответствует пилоту seed=2026, n=40")
    if original_manifest.get("max_input_tokens") != args.max_input_tokens:
        raise ValueError("Максимальная длина отличается от исходной серии")
    if original_manifest.get("tokenizer_path") != str(args.model):
        raise ValueError("Для held-out указан другой tokenizer/model path")
    archive_hash = sha256_file(archive)
    if original_manifest.get("archive_sha256") != archive_hash:
        raise ValueError("Исходный архив LongBench изменился")

    repo_root = Path(args.longbench_code).resolve().parent
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if commit != original_manifest.get("longbench_git_commit"):
        raise ValueError("Версия официального LongBench изменилась после пилота")

    from transformers import AutoTokenizer
    import transformers

    templates, max_outputs = load_official_configs(args.longbench_code)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if len(tokenizer) != original_manifest.get("tokenizer_vocab_size"):
        raise ValueError("Размер словаря токенизатора отличается от пилота")

    all_prepared = {}
    selection_manifest = {}
    with zipfile.ZipFile(archive) as zipped:
        for dataset in DATASETS:
            raw = read_dataset_from_zip(zipped, dataset)
            excluded = load_original_selection(original_dir, dataset, len(raw), 40, 2026)
            selected = select_heldout_indices(len(raw), excluded, args.n, args.seed)
            if set(selected) & set(excluded):
                raise RuntimeError("Ошибка: новая выборка пересекается с пилотом")
            prepared = prepare_records(
                raw, selected, dataset, templates[dataset],
                max_outputs[dataset], tokenizer, args.max_input_tokens,
            )
            if len(prepared) != args.n:
                raise RuntimeError(f"{dataset}: неверное число подготовленных заданий")
            all_prepared[dataset] = prepared
            selection_manifest[dataset] = {
                "source_count": len(raw),
                "original_excluded_count": len(excluded),
                "heldout_count": len(selected),
                "selected_source_indices": selected,
                "excluded_source_indices": excluded,
                "middle_truncated": sum(x["middle_truncated"] for x in prepared),
                "max_input_tokens": max(x["input_tokens"] for x in prepared),
                "input_ids_sha256": {x["task_id"]: x["input_ids_sha256"] for x in prepared},
            }
            print(f"{dataset}: новых={len(selected)}, исключено={len(excluded)}, "
                  f"усечено={selection_manifest[dataset]['middle_truncated']}",
                  flush=True)

    manifest = {
        "series": "LongBench v1 held-out 80; непересекающаяся выборка",
        "model_path": str(args.model),
        "tokenizer_vocab_size": len(tokenizer),
        "transformers_version": transformers.__version__,
        "longbench_git_commit": commit,
        "archive_sha256": archive_hash,
        "original_manifest_sha256": sha256_file(manifest_path),
        "sampling_seed": args.seed,
        "requested_n": args.n,
        "max_input_tokens": args.max_input_tokens,
        "prompt_mode": "официальный LongBench raw prompt, add_special_tokens=False",
        "truncation": "middle truncate token IDs: retain start and end",
        "datasets": selection_manifest,
    }

    # Подготовка во временном каталоге; исходный пилот остаётся неизменным.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".heldout_stage_", dir=out_dir.parent))
    try:
        for dataset in DATASETS:
            write_jsonl(staging / f"{dataset}.jsonl", all_prepared[dataset])
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if out_dir.exists():
            raise FileExistsError(f"Каталог уже создан параллельным процессом: {out_dir}")
        staging.rename(out_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Подготовлены новые задания: {out_dir}")
    print("Исходные 40 заданий и их результаты не изменены.")


if __name__ == "__main__":
    main()
