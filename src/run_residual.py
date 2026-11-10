"""Одна ячейка residual_v1: split retrieval, HQQ g32/g64, residual=512."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path

from src.common import read_jsonl
from src.download_data import sha256_file
from src.group_size import ROOT, prepared_tasks, reference_fingerprints, validate_environment
from src.protocol import code_revision, digest_json, result_path
from src.residual import (cells, load_config, read_matrix, reference_matrices, references_for,
                          selected_ids, validate_budget, validate_result)
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
        parser.error("Индекс ячейки residual вне матрицы")
    cell = matrix[args.cell]
    reference = reference_matrices()
    references = references_for(reference, cell)
    baseline = next(iter(references["dynamic"].values()))["provenance"]
    tasks = prepared_tasks(cell, args.model, references)
    # Проверяем весь n80 заранее, в том числе при отправке трёх smoke-вопросов.
    for task in tasks:
        validate_budget(task, config)
    by_id = {task["task_id"]: task for task in tasks}
    tasks = [by_id[tid] for tid in selected_ids(references, config, args.phase)]
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "hqq")}
    validate_environment(baseline, versions=versions)
    reference_hash = digest_json(references)
    provenance = {**baseline, **cell, "series": config["series"], "phase": args.phase,
        "config_sha256": digest_json(config), "residual_length": config["residual_length"],
        "reference_fingerprints": reference_fingerprints(references), "reference_results_sha256": reference_hash,
        "control_source_hashes": {name: sha256_file(ROOT / "src" / name)
                                  for name in ("residual.py", "run_residual.py", "group_size.py")}}
    fingerprint = digest_json(provenance)
    if args.phase == "full":
        smoke = read_matrix("results/residual_v1/smoke", config, "smoke", reference)
        old = next(iter(smoke[key(cell)].values()))["provenance"]
        if {k: v for k, v in old.items() if k != "phase"} != {k: v for k, v in provenance.items() if k != "phase"}:
            raise ValueError("Условия residual изменились после smoke")
    if args.check_data:
        print("Residual: входы, baselines, версии, код и запас длины проверены:", cell, "n=", len(tasks))
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-задание выполняется через Slurm")
    validate_environment(baseline, versions=versions, gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
    output = result_path(Path("results/residual_v1") / args.phase, cell)
    commit = code_revision(ROOT)
    with output_lock(output):
        previous = list(read_jsonl(output)) if output.exists() else []
        completed = validate_resume(previous, tasks, fingerprint)
        for row in previous:
            validate_result(row, cell, config, args.phase, references, reference_hash=reference_hash)
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
                pred, stats = greedy_generate(model, tokenizer, task["input_ids"], cell["cache"],
                    config["residual_length"], task["max_new_tokens"],
                    prefill_split=task["context_split"], track_quantization=True)
                row = {k: v for k, v in task.items() if k != "input_ids"}
                row.update({**stats, **cell, "pred": pred, "provenance": provenance,
                    "run_fingerprint": fingerprint, "code_commit": commit, "model": str(args.model),
                    "dtype": baseline["dtype"], "residual_length": config["residual_length"],
                    "gpu": baseline["gpu"], "transformers": versions["transformers"]})
                validate_result(row, cell, config, args.phase, references, reference_hash=reference_hash)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(task["task_id"], cell, "stop=", stats["stop_reason"], "pred=", repr(pred[:120]), flush=True)
        print("Сохранено:", output)


if __name__ == "__main__":
    main()
