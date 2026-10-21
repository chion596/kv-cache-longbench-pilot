"""Basic per-dataset plot; do not aggregate incompatible LongBench metrics."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", default="results/summary/dataset_scores.csv")
    ap.add_argument("--out", default="plots/pilot_scores.png")
    args=ap.parse_args()
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader(Path(args.input).open(encoding="utf-8")))
    if not rows:
        raise ValueError("No scores")
    by_dataset=defaultdict(dict)
    for r in rows:
        by_dataset[r["dataset"]][r["cache"]]=float(r["score_pct"])
    datasets=list(sorted(by_dataset))
    caches=["dynamic","int4_g16","int4_g64"]
    fig,ax=plt.subplots(figsize=(10,4.8))
    positions=list(range(len(datasets)))
    width=0.24
    for j,cache in enumerate(caches):
        values=[by_dataset[name].get(cache, float("nan")) for name in datasets]
        ax.bar([x+(j-1)*width for x in positions],values,width,label=cache)
    ax.set_xticks(positions,datasets)
    ax.set_ylim(0,100)
    ax.set_ylabel("Official LongBench score, %")
    ax.set_title("Per-dataset scores (pilot subset; different metrics)")
    ax.legend()
    ax.grid(axis="y",alpha=0.2)
    fig.tight_layout()
    out=Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out,dpi=180)
    print("Wrote",out)


if __name__ == "__main__":
    main()
