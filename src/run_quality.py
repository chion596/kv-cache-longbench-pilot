"""Run LongBench v1 tasks with the same stock HQQ QuantizedCache as previous work.

Intentionally uses manual greedy forward passes to isolate the cache path:
  - prefill entire prompt once
  - subsequent tokens use previous Key/Value states (decode)
This matches the former synthetic passkey research code path.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

from src.common import digest_ids, read_jsonl

# Варианты для генерации. src.common.CACHES содержит только исходные три
# конфигурации: не меняем старую пилотную матрицу и её анализ.
GENERATION_CACHES = ("dynamic", "int4_g16", "int4_g32", "int4_g64", "int4_g128")


def make_cache(model, cache_name: str, residual_length: int):
    from transformers import DynamicCache, QuantizedCache
    if cache_name == "dynamic":
        return DynamicCache(config=model.config)
    if cache_name in ("int4_g16", "int4_g32", "int4_g64", "int4_g128"):
        group = int(cache_name.split("_g", maxsplit=1)[1])
        return QuantizedCache(backend="hqq", config=model.config, nbits=4,
                              axis_key=1, axis_value=1, q_group_size=group,
                              residual_length=residual_length)
    raise ValueError(cache_name)


def effective_generation_limit(task_max_new_tokens: int, override: int | None) -> int:
    """Вернуть длину генерации для отдельной контрольной серии."""
    limit = task_max_new_tokens if override is None else override
    if limit < 1:
        raise ValueError("Число генерируемых токенов должно быть >= 1")
    return limit


def eos_ids_for(tokenizer, model):
    ids = set()
    for value in (tokenizer.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, (list, tuple)):
            ids.update(int(i) for i in value)
    return ids


def greedy_generate(model, tokenizer, prompt_ids, cache_name: str, residual_length: int,
                    max_new_tokens: int):
    import torch
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >=1")
    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    cache_position = torch.arange(input_ids.shape[-1], dtype=torch.long, device=device)
    cache = make_cache(model, cache_name, residual_length)
    # Timings are valid on the GPU (CUDA work is asynchronous otherwise).
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask,
                       cache_position=cache_position, past_key_values=cache,
                       use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [int(next_token.item())]
        del output
        torch.cuda.synchronize()
        t_prefill = time.perf_counter()
        current_position = torch.tensor([len(prompt_ids)], dtype=torch.long, device=device)
        eos_ids = eos_ids_for(tokenizer, model)
        for _ in range(max_new_tokens - 1):
            if generated[-1] in eos_ids:
                break
            attention_mask = torch.cat((attention_mask, attention_mask.new_ones((1, 1))), dim=-1)
            output = model(input_ids=next_token, attention_mask=attention_mask,
                           cache_position=current_position, past_key_values=cache,
                           use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
            current_position += 1
            del output
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        peak_gpu_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    stats = {
        "prefill_sec": round(t_prefill - t0, 6),
        "decode_sec": round(t_end - t_prefill, 6),
        "decode_steps": len(generated) - 1,
        "decode_tok_s": (len(generated)-1) / (t_end-t_prefill) if len(generated)>1 and t_end>t_prefill else None,
        "peak_gpu_gib": round(peak_gpu_gib, 6),
        "output_tokens": len(generated),
    }
    del cache
    return text, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local path to Qwen3-1.7B model")
    parser.add_argument("--tasks", required=True, help="Prepared dataset JSONL")
    parser.add_argument("--cache", required=True, choices=GENERATION_CACHES)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Smoke test: only first N prepared tasks")
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument("--max-new-tokens-override", type=int, default=None,
                        help="Предел генерации в контрольной серии, не меняет подготовленные входы")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Run through Slurm on the HSE cluster.")
    if args.residual_length < 1:
        parser.error("--residual-length must be >=1")
    if args.max_new_tokens_override is not None and args.max_new_tokens_override < 1:
        parser.error("--max-new-tokens-override должен быть >=1")
    tasks = list(read_jsonl(args.tasks))
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        raise RuntimeError("No prepared tasks to evaluate")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {out}. Use --resume or a different path.")
    completed = {}
    if out.exists():
        for row in read_jsonl(out):
            completed[row["task_id"]] = row
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, local_files_only=True
    ).to("cuda")
    model.eval()
    assert model.get_input_embeddings().weight.device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print("GPU:", gpu_name, "transformers:", transformers.__version__,
          "cache:", args.cache, "tasks:", len(tasks), flush=True)

    with out.open("a", encoding="utf-8") as output_file:
        for j, task in enumerate(tasks, 1):
            task_id = task["task_id"]
            output_limit = effective_generation_limit(task["max_new_tokens"], args.max_new_tokens_override)
            if task_id in completed:
                if (completed[task_id].get("cache") != args.cache
                        or completed[task_id].get("model") != str(args.model)
                        or completed[task_id].get("residual_length") != args.residual_length
                        or completed[task_id].get("max_new_tokens") != output_limit):
                    raise ValueError(f"Resume configuration mismatch: {task_id}")
                if completed[task_id]["input_ids_sha256"] != task["input_ids_sha256"]:
                    raise ValueError(f"Resume task hash mismatch: {task_id}")
                print(f"SKIP {task_id}", flush=True)
                continue
            if digest_ids(task["input_ids"]) != task["input_ids_sha256"]:
                raise ValueError(f"Prepared tokens modified: {task_id}")
            gc.collect()
            torch.cuda.empty_cache()
            pred, timing = greedy_generate(
                model, tokenizer, task["input_ids"], args.cache,
                args.residual_length, output_limit
            )
            record = {k: task[k] for k in (
                "task_id", "dataset", "source_index", "source_id", "language", "answers",
                "all_classes", "longbench_length", "full_prompt_tokens", "input_tokens",
                "middle_truncated", "input_ids_sha256", "max_new_tokens"
            )}
            record["max_new_tokens"] = output_limit
            record.update({"pred": pred, "cache": args.cache, "model": str(args.model),
                           "gpu": gpu_name, "transformers": transformers.__version__,
                           "torch": torch.__version__, "residual_length": args.residual_length,
                           "prompt_mode": "raw_longbench_official", "dtype": "float16", **timing})
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            print(f"[{j}/{len(tasks)}] {task_id} cache={args.cache} "
                  f"input={task['input_tokens']} output={timing['output_tokens']} "
                  f"time={timing['prefill_sec']+timing['decode_sec']:.1f}s "
                  f"pred={pred[:100]!r}", flush=True)
    print("Saved:", out, flush=True)


if __name__ == "__main__":
    main()
