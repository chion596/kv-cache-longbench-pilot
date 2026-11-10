"""Парные интервалы residual512−128 из опубликованного CSV; нужен Matplotlib."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', default='reports/residual_v1_2026-11-11')
    parser.add_argument('--png')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(args.input_dir)
    with (root / 'dataset_scores.csv').open(encoding='utf-8') as handle:
        scores = {(r['model'], r['cache']): r for r in csv.DictReader(handle)}
    with (root / 'paired_contrasts.csv').open(encoding='utf-8') as handle:
        rows = [r for r in csv.DictReader(handle) if r['comparison'] == 'residual512-residual128' and r['metric'] == 'retrieval_score']
    matplotlib.rcParams.update({'font.size': 11, 'svg.fonttype': 'none', 'svg.hashsalt': 'residual-20261111'})
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = []
    for i, r in enumerate(rows):
        score = scores[r['model'], r['cache']]
        value, low, high = (float(r[k]) for k in ('delta_pp', 'cluster_ci95_low_pp', 'cluster_ci95_high_pp'))
        ax.errorbar(value, i, xerr=[[value-low], [high-value]], fmt='o', color='#226eac', markersize=6, capsize=4)
        ax.annotate(f'{value:+.2f} [{low:+.2f}, {high:+.2f}]', (high, i), xytext=(9, -4),
                    textcoords='offset points', fontsize=10)
        model = 'Qwen3-1.7B' if r['model'] == 'qwen3-1.7b' else 'Qwen3-4B'
        labels.append(f"{model} · {r['cache'].split('_')[1]}\n{float(score['residual128_score_pct']):.2f}% → {float(score['residual512_score_pct']):.2f}%")
    ax.set_yticks(range(len(rows)), labels)
    ax.set_ylim(len(rows)-0.45, -0.55)
    ax.set_xlim(-12, 24)
    ax.axvline(0, color='#555555', linewidth=1)
    ax.grid(axis='x', alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Retrieval score change: residual 512 − 128 (pp)')
    ax.set_title('Suppressing refresh: score changes and remaining uncertainty', fontsize=14, pad=16)
    fig.text(0.5, 0.03, '80 paired documents per cell · Split prefill · HQQ INT4\n'
             '95% document-bootstrap CIs · Exploratory intervals without multiplicity adjustment',
             ha='center', fontsize=9, color='#555555')
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.23, top=0.85)
    svg = root / 'residual_effect.svg'
    fig.savefig(svg, metadata={'Date': None, 'Creator': 'tools/plot_residual_snapshot.py'})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    if args.png:
        png = Path(args.png)
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=180)
    plt.close(fig)
    print(svg)


if __name__ == '__main__':
    main()
