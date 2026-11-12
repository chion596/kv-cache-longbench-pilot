"""Рисунок полной expansion_v1 из опубликованных агрегатов; нужен Matplotlib."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', default='reports/expansion_v1_full_2026-11-12')
    parser.add_argument('--png')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(args.input_dir)
    with (root/'paired_contrasts.csv').open(encoding='utf-8') as handle:
        data = {(r['model'], r['dataset'], r['protocol'], r['comparison']): r for r in csv.DictReader(handle)
                if r['comparison'] in ('int4_g16-dynamic', 'int4_g32-dynamic', 'int4_g64-dynamic')}
    matplotlib.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'svg.hashsalt': 'expansion-full-20261112'})
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True, sharey=True)
    for i, (model, model_title) in enumerate((('qwen3-1.7b', 'Qwen3-1.7B'), ('qwen3-4b', 'Qwen3-4B'), ('smollm3-3b', 'SmolLM3-3B'))):
        for j, (dataset, title) in enumerate((('passage_retrieval_en', 'Retrieval EN'), ('hotpotqa', 'HotpotQA'))):
            ax = axes[i, j]
            for protocol, label, color, offset, marker in (
                    ('single_prefill', 'Whole prompt', '#ce701a', -0.035, 'o'),
                    ('context_then_question', 'Context then question', '#226eac', 0.035, 's')):
                rows = [data[(model, dataset, protocol, f'int4_g{g}-dynamic')] for g in (16, 32, 64)]
                values = [float(r['delta_pp']) for r in rows]
                error = [[v-float(r['cluster_ci95_low_pp']) for r, v in zip(rows, values)],
                         [float(r['cluster_ci95_high_pp'])-v for r, v in zip(rows, values)]]
                ax.errorbar([n+offset for n in range(3)], values, yerr=error, fmt=marker+'-',
                            label=label, color=color, capsize=3, markersize=5, linewidth=1.6)
            ax.axhline(0, color='#555555', linewidth=0.8)
            ax.set_title(f'{model_title} · {title}', fontsize=12)
            ax.set_xticks(range(3), ['16', '32', '64'])
            ax.set_xlim(-0.25, 2.25)
            ax.set_ylim(-104, 16)
            ax.grid(axis='y', alpha=0.2)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            if j == 0:
                ax.set_ylabel('Score change vs own Dynamic (pp)')
            if i == 2:
                ax.set_xlabel('HQQ INT4 group size')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle('New documents: prefill sensitivity repeats and transfers to QA', fontsize=14, y=0.985)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False)
    fig.text(0.5, 0.023, '80 new exact contexts per dataset · Residual 512, no refresh · Native chat, FP16/SDPA\n'
             '95% paired document-bootstrap CIs; unadjusted · SmolLM3 HotpotQA baseline: 18.35 F1',
             ha='center', va='bottom', fontsize=9, color='#555555')
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.12, hspace=0.32, wspace=0.16)
    path = root/'quality_effects.svg'
    fig.savefig(path, metadata={'Date': None, 'Creator': 'tools/plot_expansion_snapshot.py'})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    if args.png:
        png = Path(args.png)
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=180)
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    main()
