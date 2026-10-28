"""Plot completed paired results without changing the frozen inference protocol."""
import numpy as np
from pathlib import Path
import os

from visual_evidence_probe.common import OUT, read


def main(output=None):
    os.environ.setdefault('MPLCONFIGDIR', str(OUT.parent / 'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(output) if output is not None else OUT
    result = read(out / 'summary.json')
    cropped = bool(read(out / 'protocol.json').get('cropping'))
    groups = [('2 observed', ('real2',)), ('Repeat to 8', ('repeat8',)),
              ('8 observed', ('real8',)), ('GT renders\n(oracle)', ('gt_render6',)),
              ('One donor\n(mean A/B)', ('donor_a6', 'donor_b6')),
              ('Mixed donors\n(mean AB/BA)', ('mixed_ab6', 'mixed_ba6'))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    rng = np.random.default_rng(20261025)
    for ax, key, title in zip(axes, ('cd', 'f02'), ('Surface Chamfer-L1 (lower is better)', 'F-score @ 0.02 (higher is better)')):
        values = np.array([np.mean([result['by_object'][a][key] for a in arms], 0) for _, arms in groups])
        means = values.mean(1)
        count = values.shape[1]
        bootstrap = values[:, rng.integers(count, size=(5000, count))].mean(-1)
        lo, hi = np.quantile(bootstrap, [.025, .975], axis=1)
        x = np.arange(len(groups))
        for i in range(count):
            ax.plot(x, values[:, i], color=plt.get_cmap('tab10')(i), alpha=.5, linewidth=.8, marker='.', markersize=4)
        ax.errorbar(x, means, yerr=np.maximum(np.stack((means-lo, hi-means)), 0), fmt='o',
                    color='black', markersize=6, capsize=4, linewidth=1.4, label='Object mean / exploratory CI')
        ax.set_xticks(x, [label for label, _ in groups], fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Native canonical coordinates; no alignment')
    condition = 'Training-crop control (post hoc)' if cropped else 'Full-frame exploratory run (crop mismatch)'
    fig.suptitle('Frozen 30K: observed vs synthetic-view inputs\n' + condition, fontsize=12)
    root = out / 'figures'
    root.mkdir(exist_ok=True)
    fig.savefig(root / 'paired_results.png', dpi=180)
    fig.savefig(root / 'paired_results.pdf')
    plt.close(fig)
    (root / 'NOTE.md').write_text(f"{result['objects']} previously inspected objects in one HSSD scene, two paired seeds. "
        'Lines represent objects; the bootstrap is descriptive within this scene. '
        'Source groups average complementary arms; these are not independent donor sources. '
        + condition + '\n')
    print(root / 'paired_results.png')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--output')
    main(p.parse_args().output)
