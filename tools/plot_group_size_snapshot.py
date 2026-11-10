"""Рисунок g16/g32/g64 из опубликованных парных контрастов; нужен Matplotlib."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="reports/protocol_v1_2026-11-08/paired_contrasts.csv")
    parser.add_argument("--g32", default="reports/group_size_v1_2026-11-10/paired_contrasts.csv")
    parser.add_argument("--svg", default="reports/group_size_v1_2026-11-10/group_size_effect.svg")
    parser.add_argument("--png")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contrasts = {}
    for filename in (args.reference, args.g32):
        with Path(filename).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["comparison"] in ("int4_g16-dynamic", "int4_g32-dynamic", "int4_g64-dynamic"):
                    contrasts[(row["model"], row["dataset"], row["protocol"], row["comparison"])] = row
    matplotlib.rcParams.update({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "group-size-20261110"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.8), sharex=True, sharey=True)
    models = (("qwen3-1.7b", "Qwen3-1.7B"), ("qwen3-4b", "Qwen3-4B"))
    datasets = (("passage_retrieval_en", "Retrieval EN"), ("multifieldqa_en", "QA EN"))
    for row_index, (model, model_label) in enumerate(models):
        for column, (dataset, dataset_label) in enumerate(datasets):
            ax = axes[row_index, column]
            for protocol, label, color, offset, marker in (
                    ("single_prefill", "Whole prompt", "#ce701a", -0.035, "o"),
                    ("context_then_question", "Context then question", "#226eac", 0.035, "s")):
                rows = [contrasts[(model, dataset, protocol, f"int4_g{g}-dynamic")] for g in (16, 32, 64)]
                values = [float(r["delta_pp"]) for r in rows]
                errors = [[v - float(r["cluster_ci95_low_pp"]) for r, v in zip(rows, values)],
                          [float(r["cluster_ci95_high_pp"]) - v for r, v in zip(rows, values)]]
                ax.errorbar([i + offset for i in range(3)], values, yerr=errors, fmt=marker + "-",
                            color=color, label=label, linewidth=1.6, markersize=5, capsize=3)
            ax.axhline(0, color="#555555", linewidth=0.8)
            ax.set_title(f"{model_label} · {dataset_label}", fontsize=12)
            ax.set_xticks(range(3), ["16", "32", "64"])
            ax.set_xlim(-0.25, 2.25)
            ax.set_ylim(-108, 13)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if column == 0:
                ax.set_ylabel("Score change vs own Dynamic (pp)")
            if row_index == 1:
                ax.set_xlabel("HQQ INT4 group size")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("The group-size boundary depends on model, task and prefill", fontsize=14, y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False)
    fig.text(0.5, 0.025, "80 questions per cell · 95% paired document-bootstrap CIs · 2,000 resamples\n"
             "Native chat, FP16/SDPA, residual 128 · Exploratory intervals without multiplicity adjustment",
             ha="center", va="bottom", fontsize=9, color="#555555")
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.09, right=0.98, hspace=0.3, wspace=0.17)
    svg = Path(args.svg)
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg, metadata={"Date": None, "Creator": "tools/plot_group_size_snapshot.py"})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    if args.png:
        png = Path(args.png)
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=180)
    plt.close(fig)
    print(svg)


if __name__ == "__main__":
    main()
