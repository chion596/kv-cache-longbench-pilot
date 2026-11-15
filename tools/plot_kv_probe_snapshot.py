"""NLL-контрасты K/V по опубликованному CSV; нужна Matplotlib."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', default='reports/kv_probe_full_2026-11-15')
    parser.add_argument('--png')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(args.input_dir)
    with (root/'paired_contrasts.csv').open() as handle:
        data = {(r['model'], r['comparison']): r for r in csv.DictReader(handle) if r['metric'] == 'gold_nll_mean'}
    matplotlib.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'svg.hashsalt': 'kv-probe-full-20261115'})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey=True)
    for i, (model, title) in enumerate([('qwen3-1.7b', 'Qwen3-1.7B'), ('qwen3-4b', 'Qwen3-4B')]):
        for j, group in enumerate((32, 64)):
            ax = axes[i, j]
            for y, (side, color) in enumerate([('k', '#c16c19'), ('v', '#2b806a'), ('kv', '#276eac')]):
                r = data[model, f'{side}_g{group}-dynamic']
                value, low, high = (float(r[k]) for k in ('delta', 'cluster_ci95_low', 'cluster_ci95_high'))
                ax.errorbar(value, y, xerr=[[value-low], [high-value]], fmt='o', color=color, capsize=4, markersize=6)
            ax.set_title(f'{title} · group {group}')
            ax.axvline(0, color='#555555', linewidth=0.8)
            ax.set_yticks([0, 1, 2], ['K only', 'V only', 'K + V'])
            ax.set_xlim(-0.4, 6.8)
            ax.set_ylim(2.5, -0.5)
            ax.grid(axis='x', alpha=0.2)
            ax.spines[['top', 'right']].set_visible(False)
            if i == 1:
                ax.set_xlabel('Gold-number NLL change vs Dynamic (nats/token)')
    fig.suptitle('Selective K/V quantization under a common forced answer prefix', fontsize=13, y=0.98)
    fig.text(0.5, 0.035, '20 previously studied retrieval contexts · FP16/SDPA, HQQ INT4, residual 512 · No refresh\n'
             '95% paired document-bootstrap CIs, unadjusted · Higher NLL means worse conditional prediction',
             ha='center', fontsize=9, color='#555555')
    fig.subplots_adjust(left=0.09, right=0.98, top=0.89, bottom=0.18, hspace=0.35, wspace=0.15)
    path = root/'kv_nll_effects.svg'
    fig.savefig(path, metadata={'Date': None, 'Creator': 'tools/plot_kv_probe_snapshot.py'})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().split('\n')))
    if args.png:
        target = Path(args.png)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180)
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    main()
