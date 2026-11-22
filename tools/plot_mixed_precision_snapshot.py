"""Quality–cache figure from published full aggregates; optional Matplotlib."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', default='reports/mixed_precision_full_2026-11-17')
    parser.add_argument('--png')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(args.input_dir)
    with (root/'dataset_scores.csv').open() as handle:
        scores = {(r['model'], r['dataset'], r['policy']): r for r in csv.DictReader(handle)}
    with (root/'paired_contrasts.csv').open() as handle:
        contrasts = {(r['model'], r['dataset'], r['comparison']): r for r in csv.DictReader(handle)}
    styles = [('dynamic', 'Dynamic', '#747474', 'o'), ('k4_v4', 'K4 / V4', '#a87c59', 's'),
              ('k8_v4', 'K8 / V4', '#176ea3', 'D'), ('k4_v8', 'K4 / V8', '#c14e39', '^'),
              ('k16_v4', 'K16 / V4', '#278774', 'P')]
    matplotlib.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'svg.hashsalt': 'mixed-full-20261117'})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for i, (model, title) in enumerate([('qwen3-1.7b', 'Qwen3-1.7B'), ('qwen3-4b', 'Qwen3-4B')]):
        for j, (dataset, task) in enumerate([('passage_retrieval_en', 'Поиск абзаца (англ.)'), ('hotpotqa', 'HotpotQA')]):
            ax = axes[i, j]
            for policy, label, color, marker in styles:
                x = 100*float(scores[model, dataset, policy]['mean_cache_ratio_vs_dynamic'])
                if policy == 'dynamic':
                    value = low = high = 0
                else:
                    r = contrasts[model, dataset, policy+'-dynamic']
                    value, low, high = (float(r[k]) for k in ('delta_pp', 'cluster_ci95_low_pp', 'cluster_ci95_high_pp'))
                ax.errorbar(x, value, yerr=[[value-low], [high-value]], fmt=marker, color=color,
                    label=label, capsize=4, markersize=6)
            ax.set_title(f'{title} · {task}')
            ax.axhline(0, color='#555555', linewidth=.8)
            ax.set_xlim(20, 105); ax.set_ylim(-105, 15)
            ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False)
            if i == 1:
                ax.set_xlabel('Объём KV-cache / Dynamic (%)')
            if j == 0:
                ax.set_ylabel('Изменение оценки (п.п.)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle('При равном объёме кэша точность K и V по-разному влияет на качество', fontsize=14, y=.985)
    fig.legend(handles, labels, ncol=5, loc='upper center', bbox_to_anchor=(.5, .945), frameon=False)
    fig.text(.5, .026, '80 контекстов из рабочей выборки на задачу · Подача документ → вопрос, HQQ, группа 64, residual 512\n'
        'Парные 95% bootstrap-интервалы без поправки · Кэш измерен после вопроса, без весов и временных тензоров',
        ha='center', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.085, right=.98, top=.83, bottom=.15, hspace=.3, wspace=.12)
    path = root/'quality_cache_tradeoff.svg'
    fig.savefig(path, metadata={'Date': None, 'Creator': 'tools/plot_mixed_precision_snapshot.py'})
    path.write_text('\n'.join(s.rstrip() for s in path.read_text().split('\n')))
    if args.png:
        target = Path(args.png); target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180)
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    main()
