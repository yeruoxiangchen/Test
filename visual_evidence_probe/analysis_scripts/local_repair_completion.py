"""Finish reports and figures once all latent states are present; refuse partial success."""
import os
from pathlib import Path
import subprocess
import sys
import time

from visual_evidence_probe.experiments.local_repair import config


def main():
    config.setup()
    root = config.STORE / 'outputs/local_slat_repair_20261026'
    expected = [root / 'objects' / f'{i:03d}' / str(s) / m / 'result.json'
        for i in config.PARAMS['indices'] for s in config.PARAMS['seeds'] for m in config.METHODS]
    while not all(p.exists() for p in expected):
        status = subprocess.run(['systemctl', '--user', 'is-active', 'visual-evidence-local-repair-latent-batch-20261026.service'],
            text=True, capture_output=True)
        if status.stdout.strip() != 'active':
            raise RuntimeError('latent batch stopped before all64 states were written')
        time.sleep(10)
    for module, arguments in [('local_repair_latent_report', []), ('local_repair_summary_plot', []),
                              ('local_repair_final_notes', []), ('local_repair_latent_figures', ['--gpu', '0'])]:
        subprocess.run([sys.executable, '-B', '-u', '-m', 'visual_evidence_probe.analysis_scripts.' + module] + arguments,
            check=True)
    config.write(root / 'completion.json', dict(time=config.now(), states=64,
        all_expected_results=True, reports=True, figures=True, audit_appended=True))
    print(dict(stage='fourth_probe_complete', root=str(root)), flush=True)


if __name__ == '__main__': main()
