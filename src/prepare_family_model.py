"""Подготовка тех же LongBench-вопросов для модели с другим токенизатором.

Из held-out manifest берутся те же source_index, что у Qwen3.
Token IDs различаются между моделями, но в каждой отдельной
модели совпадают для всех сравниваемых конфигураций cache.

На login-узле:
    python -m src.prepare_family_model --model "$OTHER_MODEL_PATH" --name llama32-1b
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from src.common import DATASETS, load_official_configs, write_jsonl
from src.download_data import sha256_file
from src.prepare_data import prepare_records, read_dataset_from_zip


def original_heldout_indices(manifest: dict, dataset: str, population: int):
    block = manifest["datasets"][dataset]
    indices = block["selected_source_indices"]
    excluded = set(block["excluded_source_indices"])
    if len(indices) != manifest["requested_n"]:
        raise ValueError(f"{dataset}: размер выборки изменился")
    if len(indices) != len(set(indices)):
        raise ValueError(f"{dataset}: повторные индексы")
    if set(indices) & excluded:
        raise ValueError(f"{dataset}: пересечение с пилотом")
    if any(not isinstance(index, int) or index < 0 or index >= population for index in indices):
        raise ValueError(f"{dataset}: неверные индексы")
    return list(indices)


def main():
    parser = argparse.ArgumentParser(
        description="Те же LongBench-задачи для другого model family."
    )
    parser.add_argument("--model", required=True, help="Локальная директория модели и токенизатора")
    parser.add_argument("--name", required=True, help="Безопасное имя серии, например llama32-1b")
    parser.add_argument("--source-manifest", default="data/prepared/heldout/manifest.json")
    parser.add_argument("--archive", default="data/raw/longbench-data.zip")
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    parser.add_argument("--output-root", default="data/prepared/families")
    args = parser.parse_args()

    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.name):
        parser.error("--name должен содержать только a-z, 0-9, - и _")
    out_dir = Path(args.output_root) / args.name
    if out_dir.exists():
        raise FileExistsError(f"Серия уже существует: {out_dir}, перезапись запрещена")

    source = Path(args.source_manifest)
    archive = Path(args.archive)
    if not source.is_file() or not archive.is_file():
        raise FileNotFoundError("Не найден held-out manifest или исходный архив LongBench")
    original = json.loads(source.read_text(encoding="utf-8"))
    if original.get("requested_n") != 80 or original.get("sampling_seed") != 2027:
        raise ValueError("Источник должен быть фиксированной held-out серией n80/seed2027")
    if original.get("archive_sha256") != sha256_file(archive):
        raise ValueError("Архив LongBench не совпадает с исходной серией")

    repo_root = Path(args.longbench_code).resolve().parent
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if original.get("longbench_git_commit") != commit:
        raise ValueError("Исходный код LongBench изменился после Qwen3-эксперимента")

    from transformers import AutoTokenizer, AutoConfig
    import transformers
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    context_window = getattr(model_config, "max_position_embeddings", None)
    if not isinstance(context_window, int) or context_window < 1:
        raise ValueError("Не задан max_position_embeddings; проверь контекст модели")

    max_input = min(int(original["max_input_tokens"]), context_window - 256)
    if max_input < 4096:
        raise ValueError(f"У модели слишком короткий контекст ({context_window}) для этого протокола")
    templates, max_outputs = load_official_configs(args.longbench_code)

    prepared = {}
    details = {}
    with zipfile.ZipFile(archive) as z:
        for dataset in DATASETS:
            raw = read_dataset_from_zip(z, dataset)
            indices = original_heldout_indices(original, dataset, len(raw))
            examples = prepare_records(
                raw, indices, dataset, templates[dataset],
                max_outputs[dataset], tokenizer, max_input,
            )
            prepared[dataset] = examples
            details[dataset] = {
                "source_index": indices,
                "n": len(examples),
                "middle_truncated": sum(x["middle_truncated"] for x in examples),
                "input_tokens_max": max(x["input_tokens"] for x in examples),
                "input_ids_sha256": {x["task_id"]: x["input_ids_sha256"] for x in examples},
            }
            print(f"{dataset}: n={len(examples)}, "
                  f"усечено={details[dataset]['middle_truncated']}", flush=True)

    manifest = {
        "series_name": args.name,
        "model_path": str(args.model),
        "model_type": model_config.model_type,
        "transformers_version": transformers.__version__,
        "tokenizer_vocab_size": len(tokenizer),
        "context_window": context_window,
        "max_input_tokens": max_input,
        "source_manifest_sha256": sha256_file(source),
        "source_series": "Qwen3 held-out seed2027 n80",
        "archive_sha256": sha256_file(archive),
        "longbench_git_commit": commit,
        "prompt_mode": "официальный LongBench prompt, raw token IDs без chat template",
        "datasets": details,
    }

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".family_stage_", dir=out_dir.parent))
    try:
        for dataset, rows in prepared.items():
            write_jsonl(staging / f"{dataset}.jsonl", rows)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if out_dir.exists():
            raise FileExistsError(f"Папка появилась во время подготовки: {out_dir}")
        staging.rename(out_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Готова независимая токенизация: {out_dir}")


if __name__ == "__main__":
    main()
