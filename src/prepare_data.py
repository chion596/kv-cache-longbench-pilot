"""Prepare fixed, identical LongBench instances for every cache configuration.

Reads official dataset JSONL files directly from the LongBench data.zip archive.
Prompt templates and output lengths are loaded from upstream LongBench code.
"""
from __future__ import annotations

import argparse
import json
import zipfile
import subprocess
from pathlib import Path

from src.common import DATASETS, deterministic_indices, digest_ids, load_official_configs, middle_truncate, write_jsonl
from src.download_data import sha256_file


def read_dataset_from_zip(archive: zipfile.ZipFile, dataset: str):
    suffix = "/" + dataset + ".jsonl"
    matches = [x for x in archive.namelist() if x.endswith(suffix) or x == dataset + ".jsonl"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {dataset}.jsonl inside data.zip, found {matches}")
    with archive.open(matches[0]) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(template: str, record: dict):
    # Formatting source text as values does not interpret braces inside documents.
    return template.format(context=record["context"], input=record["input"])


def prepare_records(raw_records, indices, dataset, prompt_template, max_output,
                    tokenizer, max_input_tokens):
    prepared = []
    for i in indices:
        source = raw_records[i]
        prompt = build_prompt(prompt_template, source)
        ids_full = tokenizer.encode(prompt, add_special_tokens=False)
        ids, truncated = middle_truncate(ids_full, max_input_tokens)
        if not ids:
            raise ValueError(f"Empty tokenized prompt: dataset={dataset} index={i}")
        prepared.append({
            "task_id": f"{dataset}:{i}",
            "dataset": dataset,
            "source_index": i,
            "source_id": str(source.get("_id", i)),
            "language": source.get("language"),
            "answers": source["answers"],
            "all_classes": source.get("all_classes"),
            "longbench_length": source.get("length"),
            "input_ids": ids,
            "full_prompt_tokens": len(ids_full),
            "input_tokens": len(ids),
            "middle_truncated": truncated,
            "input_ids_sha256": digest_ids(ids),
            "max_new_tokens": max_output,
        })
    return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local tokenizer/model path used for all runs")
    parser.add_argument("--archive", default="data/raw/longbench-data.zip")
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--out-dir", default="data/prepared")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be at least 1")
    if args.max_input_tokens < 1:
        parser.error("--max-input-tokens must be at least 1")

    import transformers
    from transformers import AutoTokenizer

    archive = Path(args.archive)
    if not archive.is_file():
        raise FileNotFoundError(f"Missing {archive}; run: python -m src.download_data")
    templates, max_outputs = load_official_configs(args.longbench_code)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    with zipfile.ZipFile(archive) as zipped:
        for dataset in args.datasets:
            records = read_dataset_from_zip(zipped, dataset)
            indices = deterministic_indices(len(records), args.n, args.seed)
            selected = prepare_records(records, indices, dataset, templates[dataset],
                                       max_outputs[dataset], tokenizer, args.max_input_tokens)
            output = out_dir / f"{dataset}.jsonl"
            write_jsonl(output, selected)
            counts[dataset] = {"source_count": len(records), "selected": len(selected),
                               "middle_truncated": sum(r["middle_truncated"] for r in selected),
                               "max_input_tokens": max(r["input_tokens"] for r in selected)}
            print(f"{dataset}: {counts[dataset]}; output={output}")
    repo_root = Path(args.longbench_code).resolve().parent
    sha = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False)
    manifest = {
        "transformers_version": transformers.__version__,
        "longbench_git_commit": sha.stdout.strip() if sha.returncode == 0 else None,
        "benchmark": "LongBench v1 (pilot subset)", "archive_sha256": sha256_file(archive),
        "tokenizer_path": str(args.model), "tokenizer_vocab_size": len(tokenizer),
        "longbench_code": str(args.longbench_code), "datasets": counts,
        "sampling_seed": args.seed, "requested_n": args.n,
        "max_input_tokens": args.max_input_tokens,
        "prompt_mode": "official LongBench prompt, raw token ids, no chat template, add_special_tokens=False",
        "truncation": "middle truncate at token-id level; retain start and end",
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
