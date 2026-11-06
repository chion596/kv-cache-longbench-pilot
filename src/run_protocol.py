"""Одна ячейка protocol_v1: native chat, полный или раздельный prefill."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import importlib.metadata
import json
from pathlib import Path

from src.common import read_jsonl
from src.download_data import sha256_file
from src.protocol import cells, code_revision, digest_json, load_config, model_signature, result_path, validate_task
from src.run_quality import eos_ids_for, greedy_generate, make_cache


@contextmanager
def output_lock(path):
    """Две копии Slurm-ячейки не должны одновременно дописывать один JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Этот результат уже пишет другой процесс: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_resume(rows, tasks, fingerprint):
    expected = {row["task_id"]: row for row in tasks}
    completed = set()
    for row in rows:
        task_id = row["task_id"]
        if task_id in completed or task_id not in expected:
            raise ValueError(f"Повторная или чужая задача при resume: {task_id}")
        if row.get("run_fingerprint") != fingerprint:
            raise ValueError(f"Условия запуска изменились: {task_id}; нужна новая серия")
        task = expected[task_id]
        for field in ("input_ids_sha256", "context_sha256", "context_split", "max_new_tokens"):
            if row.get(field) != task[field]:
                raise ValueError(f"Resume: не совпадает {field} у {task_id}")
        if not row.get("output_ids") or row.get("output_tokens") != len(row["output_ids"]):
            raise ValueError(f"Неполный результат: {task_id}")
        if row.get("reference_matches") is False:
            raise ValueError(f"Не прошла сверка model.generate: {task_id}")
        completed.add(task_id)
    return completed


def reference_generate(model, tokenizer, prompt_ids, limit):
    import torch
    from transformers import GenerationConfig
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([prompt_ids], device=device)
    eos = sorted(eos_ids_for(tokenizer, model))
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=limit,
        eos_token_id=eos or None, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (eos[0] if eos else 0))
    with torch.inference_mode():
        result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            generation_config=generation, past_key_values=make_cache(model, "dynamic", 128))
    return result[0, len(prompt_ids):].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--check-data", action="store_true", help="Только проверить подготовленные входы, без GPU")
    args = parser.parse_args()
    config = load_config()
    matrix = cells(config)
    if not 0 <= args.cell < len(matrix):
        parser.error("Индекс ячейки вне матрицы")
    cell = matrix[args.cell]
    folder = Path("data/prepared/protocol_v1") / cell["model_name"]
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_sha256"] != digest_json(config) or manifest["model_name"] != cell["model_name"]:
        raise ValueError("Данные подготовлены по другой матрице/модели")
    artifacts = model_signature(args.model)
    if artifacts != manifest["model_artifacts"]:
        raise ValueError("Модель/токенизатор изменились после подготовки")
    tasks_path = folder / f"{cell['dataset']}.jsonl"
    if sha256_file(tasks_path) != manifest["datasets"][cell["dataset"]]["file_sha256"]:
        raise ValueError("Подготовленные задачи изменились после manifest")
    tasks = list(read_jsonl(tasks_path))
    expected_hashes = manifest["datasets"][cell["dataset"]]["task_hashes"]
    if len(tasks) != config["n"] or len({r["task_id"] for r in tasks}) != len(tasks):
        raise ValueError("Неполный или повторный набор задач")
    if {r["task_id"]: r["input_ids_sha256"] for r in tasks} != expected_hashes:
        raise ValueError("Задачи не совпали с manifest")
    for task in tasks:
        validate_task(task)
    if args.check_data:
        print("Входы проверены:", cell["model_name"], cell["dataset"], "n=", len(tasks))
        return
    if args.phase == "smoke":
        tasks = tasks[:config["smoke_n"]]

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-запуск выполняется через Slurm")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "hqq")}
    if versions["transformers"] != manifest["transformers"]:
        raise ValueError("Transformers изменился после подготовки входов")
    root = Path(__file__).resolve().parents[1]
    code_commit = code_revision(root)
    if code_commit is None:
        print("Commit metadata недоступна; сохраняем хеши исходников runner", flush=True)
    provenance = {
        "config_sha256": digest_json(config), "manifest_sha256": sha256_file(manifest_path),
        "longbench_files_sha256": manifest["longbench_files_sha256"],
        "model_artifacts": artifacts, "versions": versions, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "phase": args.phase, **cell,
        "dtype": config["dtype"], "attention": config["attention"],
        "residual_length": config["residual_length"],
        "source_hashes": {name: sha256_file(root / "src" / name) for name in
            ("run_protocol.py", "run_quality.py", "protocol.py", "common.py")},
    }
    fingerprint = digest_json(provenance)
    if args.phase == "full":
        from src.score_protocol import read_matrix, key
        smoke = read_matrix("results/protocol_v1/smoke", config, "smoke")
        smoke_provenance = next(iter(smoke[key(cell)].values()))["provenance"]
        if {k: v for k, v in smoke_provenance.items() if k != "phase"} != {
                k: v for k, v in provenance.items() if k != "phase"}:
            raise ValueError("Условия изменились после smoke; нужен новый smoke")
    output = result_path(Path("results/protocol_v1") / args.phase, cell)
    with output_lock(output):
        completed = validate_resume(list(read_jsonl(output)) if output.exists() else [], tasks, fingerprint)
        if len(completed) == len(tasks):
            print("Уже завершено:", output)
            return
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
            attn_implementation=config["attention"], local_files_only=True).to("cuda").eval()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            for task in tasks:
                if task["task_id"] in completed:
                    continue
                gc.collect()
                torch.cuda.empty_cache()
                split = task["context_split"] if cell["protocol"] == "context_then_question" else None
                prediction, stats = greedy_generate(model, tokenizer, task["input_ids"], cell["cache"],
                    config["residual_length"], task["max_new_tokens"], prefill_split=split,
                    track_quantization=True)
                row = {k: v for k, v in task.items() if k != "input_ids"}
                row.update(stats)
                row.update({**cell, "pred": prediction, "model": str(args.model),
                    "run_fingerprint": fingerprint, "provenance": provenance, "code_commit": code_commit,
                    "model_revision": getattr(model.config, "_commit_hash", None),
                    "dtype": config["dtype"], "residual_length": config["residual_length"],
                    "gpu": provenance["gpu"], "transformers": transformers.__version__})
                if args.phase == "smoke" and cell["cache"] == "dynamic" and split is None:
                    reference = reference_generate(model, tokenizer, task["input_ids"], task["max_new_tokens"])
                    row["reference_output_ids"] = reference
                    row["reference_matches"] = reference == stats["output_ids"]
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(task["task_id"], cell, "stop=", row["stop_reason"], "pred=", repr(prediction[:120]), flush=True)
                if row.get("reference_matches") is False:
                    raise ValueError("Ручной decode не совпал с model.generate; результат сохранён для диагностики")
        print("Сохранено:", output)


if __name__ == "__main__":
    main()
