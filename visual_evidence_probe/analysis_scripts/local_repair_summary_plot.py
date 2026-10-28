"""Standalone scientific plot of actual mesh endpoint improvements."""
import numpy as np

from visual_evidence_probe.experiments.local_repair import config


def main():
    config.setup()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mesh = config.read(config.STORE / 'outputs/local_geometry_repair_cropped_20261026/summary.json')
    latent = config.read(config.STORE / 'outputs/local_slat_repair_20261026/summary.json')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    labels = ['Automatic', 'Random', 'GT region\n(diagnostic)', 'GT direction\n(diagnostic)']
    for ax, summary, title in zip(axes, (mesh, latent), ('Local mesh deformation', 'Fixed-support SLat feature update')):
        for offset, endpoint, color in [(-.17, 'proposed', '#7a9eb2'), (.17, 'accepted', '#244b66')]:
            means, lo, hi = [], [], []
            for method in config.METHODS:
                r = summary['methods'][method][endpoint]
                if summary is mesh:
                    r = r['surface/chamfer_l1']
                    mean, interval = r['mean_improvement'], r['object_bootstrap_ci95']
                else:
                    mean, interval = r['chamfer_l1_improvement'], r['cd_ci95']
                means.append(mean * 1.e4)
                lo.append(max(0., (mean - interval[0]) * 1.e4))
                hi.append(max(0., (interval[1] - mean) * 1.e4))
            x = np.arange(4) + offset
            ax.bar(x, means, width=.32, color=color, label='Full proposal' if endpoint == 'proposed' else 'Accepted by input gate')
            ax.errorbar(x, means, yerr=[lo, hi], fmt='none', ecolor='black', capsize=3, linewidth=1)
        ax.axhline(0, color='black', linewidth=.7)
        ax.set_xticks(np.arange(4), labels, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Chamfer-L1 improvement (x 1e-4)')
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Exploratory Omni8 x 2 seeds: actual decoded mesh endpoints\nHigher is better; object bootstrap 95% intervals; oracle arms use GT', fontsize=11)
    fig.tight_layout()
    root = config.STORE / 'outputs/local_slat_repair_20261026/figures'
    root.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(root / f'endpoint_summary.{extension}', dpi=180)
    plt.close(fig)
    print(root / 'endpoint_summary.png', flush=True)


if __name__ == '__main__': main()
