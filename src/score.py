"""Use *official* LongBench metrics on a paired pilot set.

Imports metrics.py from a pinned local checkout of THUDM/LongBench.
No metrics are silently modified. Scores are 0..100 for reporting.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

from src.common import CACHES, DATASETS, read_jsonl

METRIC_NAMES = {
    "passage_retrieval_en": "retrieval_score",
    "multifieldqa_en": "qa_f1_score",
    "multifieldqa_zh": "qa_f1_zh_score",
}


def load_official_metrics(longbench_code: str | Path):
    path = Path(longbench_code) / "metrics.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing official LongBench metrics.py: {path}")
    spec = importlib.util.spec_from_file_location("official_longbench_metrics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {dataset: getattr(module, funcname) for dataset, funcname in METRIC_NAMES.items()}


def score_prediction(row: dict, metric):
    truths = row.get("answers", [])
    if not truths:
        raise ValueError(f"No ground-truth answers for {row.get('task_id')}")
    # LongBench official eval.py: score is max over all references.
    return max(float(metric(row["pred"], truth, all_classes=row.get("all_classes")))
               for truth in truths)


def bootstrap_paired_ci(pairs: list[tuple[float, float]], seed=2026, iters=2000):
    """Exploratory percentile bootstrap of paired mean differences (quant - dynamic)."""
    if not pairs:
        raise ValueError("No paired examples")
    differences = [b-a for a,b in pairs]
    rng = random.Random(seed)
    n = len(differences)
    means = [sum(differences[rng.randrange(n)] for _ in range(n))/n
             for _ in range(iters)]
    means.sort()
    return (means[int(0.025*iters)], means[min(iters-1, int(0.975*iters))])


def aggregate_rows(files, metrics):
    per_config = defaultdict(dict)
    for file in files:
        for row in read_jsonl(file):
            dataset = row["dataset"]
            if dataset not in DATASETS:
                raise ValueError(f"Unexpected dataset={dataset}: {file}")
            cache = row["cache"]
            key = row["task_id"]
            if key in per_config[(dataset, cache)]:
                raise ValueError(f"Duplicate task for {dataset}/{cache}: {key}")
            per_config[(dataset, cache)][key] = row
    # Missing configurations are also an error: do not report partial results as full pilot.
    missing = [(d, c) for d in DATASETS for c in CACHES if (d, c) not in per_config]
    if missing:
        raise ValueError(f"Incomplete pilot matrix; missing dataset/cache cells: {missing}")
    # Fail rather than quietly compare unequal task subsets.
    for dataset in DATASETS:
        keys_for_dataset = [(c, set(by_id)) for (d,c),by_id in per_config.items() if d == dataset]
        if not keys_for_dataset:
            continue
        baseline_cache, baseline_keys = keys_for_dataset[0]
        for cache, task_set in keys_for_dataset[1:]:
            if baseline_keys != task_set:
                raise ValueError(f"Unpaired tasks in {dataset}: {baseline_cache} vs {cache}: "
                                 f"{len(baseline_keys ^ task_set)} mismatched ids")
            for tid in baseline_keys:
                left = per_config[(dataset, baseline_cache)][tid]
                right = per_config[(dataset, cache)][tid]
                for field in ("input_ids_sha256", "max_new_tokens", "middle_truncated"):
                    if left[field] != right[field]:
                        raise ValueError(f"Input mismatch for {dataset}:{tid}: {field}")
    scored = defaultdict(dict)
    example_rows = []
    for (dataset, cache), by_id in sorted(per_config.items()):
        for tid, row in sorted(by_id.items()):
            score = score_prediction(row, metrics[dataset])
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Official score is outside [0,1]: {score}")
            scored[(dataset, cache)][tid] = score
            example_rows.append({
                "dataset":dataset, "cache":cache, "task_id":tid,
                "score_pct":round(score*100, 6), "input_tokens": row["input_tokens"],
                "middle_truncated":row["middle_truncated"], "output_tokens":row["output_tokens"],
                "prefill_sec":row["prefill_sec"], "decode_tok_s":row["decode_tok_s"],
                "peak_gpu_gib":row["peak_gpu_gib"], "pred":row["pred"],
            })
    dataset_rows = []
    for (dataset,cache), scores in sorted(scored.items()):
        dataset_rows.append({"dataset":dataset, "cache":cache, "n":len(scores),
                             "score_pct":round(100*sum(scores.values())/len(scores), 3)})
    paired_rows = []
    for dataset in DATASETS:
        base = scored.get((dataset,"dynamic"))
        if not base:
            continue
        for cache in ("int4_g16", "int4_g64"):
            quant = scored.get((dataset,cache))
            if not quant:
                continue
            pairs = [(base[k], quant[k]) for k in sorted(base)]
            lo,hi = bootstrap_paired_ci(pairs)
            paired_rows.append({"dataset":dataset,"cache":cache,"n":len(pairs),
                "dynamic_score_pct":round(100*sum(a for a,_ in pairs)/len(pairs),3),
                "quant_score_pct":round(100*sum(b for _,b in pairs)/len(pairs),3),
                "delta_pp":round(100*sum(b-a for a,b in pairs)/len(pairs),3),
                "bootstrap_delta_95_low_pp":round(lo*100,3),
                "bootstrap_delta_95_high_pp":round(hi*100,3)})
    return example_rows, dataset_rows, paired_rows


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--longbench-code", default="external/LongBench/LongBench")
    parser.add_argument("--out-dir", default="results/summary")
    args=parser.parse_args()
    files=sorted(Path(args.results_dir).glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *.jsonl files under {args.results_dir}")
    metrics=load_official_metrics(args.longbench_code)
    examples, datasets, paired=aggregate_rows(files, metrics)
    out=Path(args.out_dir)
    write_csv(out/"example_scores.csv", examples)
    write_csv(out/"dataset_scores.csv", datasets)
    write_csv(out/"paired_deltas.csv", paired)
    for r in datasets:
        print(f"{r['dataset']:23s} {r['cache']:11s} n={r['n']:3d} score={r['score_pct']:.2f}")
    for r in paired:
        print(f"PAIRED {r['dataset']} {r['cache']}: delta={r['delta_pp']:+.2f}pp "
              f"bootstrap95=[{r['bootstrap_delta_95_low_pp']:+.2f},{r['bootstrap_delta_95_high_pp']:+.2f}]")
    print("Wrote", out)


if __name__ == "__main__":
    main()
