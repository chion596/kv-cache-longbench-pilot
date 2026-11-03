"""Native chat/no-thinking, те же n80; одинаковые IDs для двух prefill-протоколов."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

from src.common import digest_ids, load_official_configs, middle_truncate, write_jsonl
from src.download_data import sha256_file
from src.prepare_data import build_prompt, read_dataset_from_zip
from src.prepare_family_model import original_heldout_indices
from src.protocol import digest_json, load_config, model_signature, validate_task


def prepare_chat_task(source, index, dataset, template, max_new, tokenizer, max_input):
    if not isinstance(tokenizer.chat_template, str) or "enable_thinking" not in tokenizer.chat_template:
        raise ValueError("Локальный chat template не поддерживает явный enable_thinking=False")
    prompt = build_prompt(template, source)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    context = source["context"]
    if not context or rendered.count(context) != 1:
        raise ValueError("Нельзя однозначно найти документ в chat prompt")
    boundary = rendered.index(context) + len(context)
    full_ids = tokenizer.encode(rendered, add_special_tokens=False)
    # Отдельная токенизация префикса нужна только для поиска границы. В inference
    # подаются части ЕДИНОЖДЫ токенизированного full_ids, без ретокенизации.
    prefix_ids = tokenizer.encode(rendered[:boundary], add_special_tokens=False)
    split = 0
    for a, b in zip(prefix_ids, full_ids):
        if a != b:
            break
        split += 1
    ids, truncated = middle_truncate(full_ids, max_input)
    if truncated:
        left = max_input // 2
        right_start = len(full_ids) - (max_input - left)
        split = split if split <= left else left + max(0, split - right_start)
    row = {
        "task_id": f"{dataset}:{index}", "dataset": dataset, "source_index": index,
        "source_id": str(source.get("_id", index)), "language": source.get("language"),
        "answers": source["answers"], "all_classes": source.get("all_classes"),
        "longbench_length": source.get("length"), "max_new_tokens": max_new,
        "full_prompt_tokens": len(full_ids), "input_tokens": len(ids),
        "middle_truncated": truncated, "input_ids": ids,
        "input_ids_sha256": digest_ids(ids), "context_split": split,
        "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
        "prompt_mode": "native_chat_no_thinking",
    }
    validate_task(row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-manifest", default="data/prepared/heldout/manifest.json")
    parser.add_argument("--archive", default="data/raw/longbench-data.zip")
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    parser.add_argument("--output-root", default="data/prepared/protocol_v1")
    args = parser.parse_args()
    config = load_config()
    if args.name not in config["models"]:
        parser.error("Модель отсутствует в зафиксированной матрице")
    destination = Path(args.output_root) / args.name
    if destination.exists():
        raise FileExistsError(f"Серия уже подготовлена: {destination}; перезапись запрещена")
    original = json.loads(Path(args.source_manifest).read_text())
    archive_hash = sha256_file(Path(args.archive))
    if archive_hash != original["archive_sha256"]:
        raise ValueError("Архив отличается от held-out серии")
    if (original["requested_n"], original["sampling_seed"]) != (config["n"], config["sampling_seed"]):
        raise ValueError("Не совпали n/seed исходной выборки")
    code = Path(args.longbench_code)
    commit = subprocess.check_output(["git", "-C", str(code.parent), "rev-parse", "HEAD"],
                                     text=True).strip()
    if commit != original["longbench_git_commit"]:
        raise ValueError("Изменилась версия LongBench")
    from transformers import AutoConfig, AutoTokenizer
    import transformers
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    templates, outputs = load_official_configs(code)
    if model_config.max_position_embeddings < config["max_input_tokens"] + max(outputs[d] for d in config["datasets"]):
        raise ValueError("Контекст модели короче зафиксированного протокола")
    prepared = {}
    with zipfile.ZipFile(args.archive) as archive:
        for dataset in config["datasets"]:
            raw = read_dataset_from_zip(archive, dataset)
            indices = original_heldout_indices(original, dataset, len(raw))
            prepared[dataset] = [prepare_chat_task(raw[i], i, dataset, templates[dataset],
                outputs[dataset], tokenizer, config["max_input_tokens"]) for i in indices]
    manifest = {
        "series": config["series"], "config_sha256": digest_json(config),
        "model_name": args.name, "model_artifacts": model_signature(args.model),
        "transformers": transformers.__version__, "archive_sha256": archive_hash,
        "longbench_git_commit": commit, "source_manifest_sha256": sha256_file(Path(args.source_manifest)),
        "longbench_files_sha256": {name: sha256_file(code / name) for name in
            ("config/dataset2prompt.json", "config/dataset2maxlen.json", "metrics.py")},
        "chat_template_sha256": digest_json(tokenizer.chat_template),
        "prompt_mode": config["prompt_mode"],
        "truncation": "middle truncation of full chat token IDs, keep start/end",
        "datasets": {dataset: {"n": len(rows), "middle_truncated": sum(r["middle_truncated"] for r in rows),
            "task_hashes": {r["task_id"]: r["input_ids_sha256"] for r in rows}}
            for dataset, rows in prepared.items()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".protocol_", dir=destination.parent))
    try:
        for dataset, rows in prepared.items():
            write_jsonl(staging / f"{dataset}.jsonl", rows)
            manifest["datasets"][dataset]["file_sha256"] = sha256_file(staging / f"{dataset}.jsonl")
            print(dataset, "n=", len(rows), "truncated=", manifest["datasets"][dataset]["middle_truncated"])
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print("Подготовлено:", destination)


if __name__ == "__main__":
    main()
