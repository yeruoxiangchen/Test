"""Process-local reuse of the tested generation/evidence stages."""
import runpy
import sys
from .config import bind_engine


def main():
    stage = sys.argv.pop(1)
    if stage not in ('generate', 'regions', 'depth'):
        raise ValueError(stage)
    bind_engine()
    runpy.run_module('visual_evidence_probe.experiments.dense_evidence.' + stage, run_name='__main__')


if __name__ == '__main__':
    main()
