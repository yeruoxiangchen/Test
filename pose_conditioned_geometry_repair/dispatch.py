"""Single entry point; applies the /data-only environment before importing torch."""
from __future__ import annotations
import os
import subprocess
import sys
from . import common as c


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit('Usage: run.sh {freeze|prepare|publish|train|evaluate|tests|launch|status} ...')
    action, *rest = args
    env = c.environment()
    if '--gpu' in rest:
        i = rest.index('--gpu')
        env['CUDA_VISIBLE_DEVICES'] = rest[i+1]
        rest = rest[:i] + rest[i+2:]
    mapping = {'freeze': ('prepare','freeze'), 'prepare': ('prepare','objects'),
               'publish': ('prepare','publish'), 'train': ('train',),
               'evaluate': ('evaluate',), 'infer': ('infer',), 'figures': ('figures',), 'launch': ('runner','launch'),
               'status': ('runner','status'), 'pipeline': ('runner','pipeline')}
    mapping.update({'baseline-audit':('baseline_audit',),'launch-bridge':('bridge_runner','launch'),
                    'bridge-pipeline':('bridge_runner','pipeline'),'status-bridge':('bridge_runner','status')})
    mapping.update({'launch-v1':('runner','launch'),'status-v1':('runner','status'),
                    'launch':('bridge_runner','launch'),'status':('bridge_runner','status')})
    mapping['inference-smoke']=('inference_smoke',)
    mapping['camera-audit']=('camera_audit',)
    if action == 'tests':
        cmd = [c.PYTHON, '-B', '-m', 'unittest', 'discover', '-s', str(c.HERE/'tests'), '-v'] + rest
    elif action in mapping:
        module, *prefix = mapping[action]
        cmd = [c.PYTHON,'-B','-m','pose_conditioned_geometry_repair.'+module,*prefix,*rest]
    else:
        raise SystemExit('Unknown action '+action)
    os.execve(cmd[0], cmd, env)


if __name__ == '__main__':
    main()
