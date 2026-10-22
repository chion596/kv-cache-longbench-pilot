"""Графики качества LongBench для четырёх размеров группы.

Строится отдельная панель для каждого датасета; несопоставимые
LongBench-метрики не усредняются между собой.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

GROUPS = (16, 32, 64, 128)
DATASETS = (
    "passage_retrieval_en",
    "multifieldqa_en",
    "multifieldqa_zh",
)


def load_plot_data(csv_file: str | Path):
    scores = {}
    with Path(csv_file).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["cache"])
            if key in scores:
                raise ValueError(f"Две оценки для {key}")
            scores[key] = float(row["score_pct"])
    for dataset in DATASETS:
        for cache in ("dynamic",) + tuple(f"int4_g{g}" for g in GROUPS):
            if (dataset, cache) not in scores:
                raise ValueError(f"Нет оценки для {dataset}/{cache}")
    return scores


def main():
    parser = argparse.ArgumentParser(description="Графики LongBench: влияние размера группы HQQ.")
    parser.add_argument("--input", default="results/heldout/group_size/dataset_scores.csv")
    parser.add_argument("--out", default="plots/heldout_group_size.png")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    scores = load_plot_data(args.input)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharex=True)
    for ax, dataset in zip(axes, DATASETS):
        values = [scores[(dataset, f"int4_g{group}")] for group in GROUPS]
        baseline = scores[(dataset, "dynamic")]
        ax.plot(GROUPS, values, marker="o", linewidth=2, label="HQQ INT4")
        ax.axhline(baseline, linestyle="--", label="DynamicCache")
        ax.set_xticks(GROUPS)
        ax.set_xlabel("q_group_size")
        ax.set_ylabel("Оценка LongBench, %")
        ax.set_title(dataset)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle("LongBench: чувствительность качества к размеру группы (n=80)")
    fig.tight_layout()
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(f"График сохранён: {target}")


if __name__ == "__main__":
    main()
