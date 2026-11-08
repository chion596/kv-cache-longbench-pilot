"""Одна ячейка g32; исходные protocol_v1 и baseline не изменяются."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path

from src.common import read_jsonl
from src.download_data import sha256_file
from src.group_size import (ROOT, cells, load_config, prepared_tasks, read_matrix, reference_fingerprints,
                            reference_matrix, references_for, selected_task_ids, validate_environment, validate_result)
from src.protocol import code_revision, digest_json, result_path
from src.run_protocol import output_lock, validate_resume
from src.run_quality import greedy_generate
from src.score_protocol import key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    config = load_config()
    matrix = cells(config)
    if not 0 <= args.cell < len(matrix):
        parser.error("Индекс вне матрицы g32")
    cell = matrix[args.cell]
    reference = reference_matrix()
    references = references_for(reference, cell)
    baseline = next(iter(references["dynamic"].values()))["provenance"]
    tasks = prepared_tasks(cell, args.model, references)
    by_id = {t["task_id"]: t for t in tasks}
    tasks = [by_id[tid] for tid in selected_task_ids(reference, cell, config, args.phase)]
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "hqq")}
    validate_environment(baseline, versions=versions)
    if args.check_data:
        print("Входы, baseline, версии и общий код проверены:", cell, "n=", len(tasks))
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-задание выполняется через Slurm")
    validate_environment(baseline, versions=versions, gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
    reference_hash = digest_json(references)
    provenance = {**baseline, **cell, "series": config["series"], "phase": args.phase,
        "config_sha256": digest_json(config), "reference_fingerprints": reference_fingerprints(references),
        "reference_results_sha256": reference_hash,
        "control_source_hashes": {name: sha256_file(ROOT / "src" / name)
                                  for name in ("group_size.py", "run_group_size.py")}}
    fingerprint = digest_json(provenance)
    if args.phase == "full":
        smoke = read_matrix("results/group_size_v1/smoke", config, "smoke", reference)
        old = next(iter(smoke[key(cell)].values()))["provenance"]
        if {k: v for k, v in old.items() if k != "phase"} != {k: v for k, v in provenance.items() if k != "phase"}:
            raise ValueError("Условия изменились после smoke g32")
    output = result_path(Path("results/group_size_v1") / args.phase, cell)
    with output_lock(output):
        previous = list(read_jsonl(output)) if output.exists() else []
        completed = validate_resume(previous, tasks, fingerprint)
        for row in previous:
            validate_result(row, cell, config, args.phase, references, reference_sha256=reference_hash)
        if len(completed) == len(tasks):
            print("Уже завершено:", output)
            return
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
            attn_implementation=baseline["attention"], local_files_only=True).to("cuda").eval()
        with output.open("a", encoding="utf-8") as handle:
            for task in tasks:
                if task["task_id"] in completed:
                    continue
                gc.collect()
                torch.cuda.empty_cache()
                split = task["context_split"] if cell["protocol"] == "context_then_question" else None
                pred, stats = greedy_generate(model, tokenizer, task["input_ids"], cell["cache"],
                    config["residual_length"], task["max_new_tokens"], prefill_split=split, track_quantization=True)
                row = {k: v for k, v in task.items() if k != "input_ids"}
                row.update({**stats, **cell, "pred": pred, "provenance": provenance,
                    "run_fingerprint": fingerprint, "code_commit": code_revision(ROOT), "model": str(args.model),
                    "dtype": baseline["dtype"], "residual_length": config["residual_length"],
                    "gpu": baseline["gpu"], "transformers": versions["transformers"]})
                validate_result(row, cell, config, args.phase, references, reference_sha256=reference_hash)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(task["task_id"], cell, "stop=", stats["stop_reason"], "pred=", repr(pred[:120]), flush=True)
        print("Сохранено:", output)


if __name__ == "__main__":
    main()
