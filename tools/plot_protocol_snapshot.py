"""Standalone research figure from published aggregate contrasts; no cluster data needed.

Requires matplotlib; run in a separate analysis environment if it is unavailable.
"""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="reports/protocol_v1_2026-11-08/paired_contrasts.csv")
    parser.add_argument("--svg", default="reports/protocol_v1_2026-11-08/g64_protocol_effect.svg")
    parser.add_argument("--png", help="Optional raster export")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    matplotlib.rcParams.update({"font.size": 11, "svg.fonttype": "none", "svg.hashsalt": "protocol-v1-20261108"})
    with Path(args.input).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = {(r["model"], r["dataset"], r["protocol"]): r for r in rows
            if r["comparison"] == "int4_g64-dynamic"}
    models = (("qwen3-1.7b", "Qwen3-1.7B"), ("qwen3-4b", "Qwen3-4B"), ("smollm3-3b", "SmolLM3-3B"))
    protocols = (("single_prefill", "Whole prompt", "#ce701a", -0.14, "o"),
                 ("context_then_question", "Context then question", "#226eac", 0.14, "s"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, dataset, title in zip(axes, ("passage_retrieval_en", "multifieldqa_en"),
                                  ("Passage retrieval EN", "MultiFieldQA EN")):
        for protocol, label, color, offset, marker in protocols:
            values, lower, upper = [], [], []
            for name, _ in models:
                row = rows[name, dataset, protocol]
                value = float(row["delta_pp"])
                values.append(value)
                lower.append(value - float(row["cluster_ci95_low_pp"]))
                upper.append(float(row["cluster_ci95_high_pp"]) - value)
            y = [i + offset for i in range(len(models))]
            ax.errorbar(values, y, xerr=[lower, upper], fmt=marker, color=color,
                        capsize=3, markersize=6, linewidth=1.7, label=label)
            for value, position in zip(values, y):
                ax.annotate(f"{value:+.1f}", (value, position), xytext=(0, -15 if offset > 0 else 9),
                            textcoords="offset points", ha="center", color=color, fontsize=9)
        ax.set_yticks(range(len(models)), [label for _, label in models])
        ax.set_ylim(2.55, -0.55)
        ax.set_xlim(-108, 12)
        ax.axvline(0, color="#555555", linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.grid(axis="x", alpha=0.22)
        ax.set_axisbelow(True)
        ax.set_xlabel("Score change: INT4 g64 − Dynamic (pp)")
        ax.set_title(title, fontsize=13, pad=12)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.91), ncol=2, frameon=False)
    fig.suptitle("HQQ INT4 g64: robustness depends on the prefill protocol", fontsize=15, y=0.99)
    fig.text(0.5, 0.025, "80 questions per cell · 95% paired document-bootstrap CIs · 2,000 resamples\n"
             "Native chat, thinking disabled · Exploratory intervals without multiplicity adjustment",
             ha="center", va="bottom", fontsize=9, color="#555555")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.76, bottom=0.23, wspace=0.45)
    svg = Path(args.svg)
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg, metadata={"Date": None, "Creator": "tools/plot_protocol_snapshot.py"})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8")
    if args.png:
        png = Path(args.png)
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=180)
    plt.close(fig)
    print(svg)


if __name__ == "__main__":
    main()
