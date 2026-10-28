"""Use /data storage and preserve logical bindings of frozen experiments."""
import importlib
import json
import os
from pathlib import Path
import sys

STORE=Path('/data/zjr/visual_evidence_probe')
SOURCE=Path('/home/zjr/Tracker/visual_evidence_probe')


def setup_io():
    receipt=json.loads((STORE/'io_relocation.json').read_text())
    cache=STORE/'cache'
    env=dict(TMPDIR=STORE/'tmp',TORCH_HOME=cache/'torch',HF_HOME=cache/'huggingface',
        HF_HUB_CACHE=cache/'huggingface/hub',TORCH_EXTENSIONS_DIR=cache/'torch_extensions',
        MPLCONFIGDIR=cache/'matplotlib',XDG_CACHE_HOME=cache,HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VISUAL_EVIDENCE_REPORT_ROOT=STORE)
    for key,value in env.items():os.environ[key]=str(value)
    sys.dont_write_bytecode=True
    from visual_evidence_probe import common
    aliases={str(Path(k)):str(Path(v)) for k,v in receipt['input_aliases'].items()}
    reverse={str(Path(v).resolve()):str(Path(k)) for k,v in aliases.items()}
    physical=STORE/'outputs';stage=STORE/'outputs_migrating';logical=SOURCE/'outputs'
    def actual(path):
        p=Path(path)
        if str(p) in aliases:return Path(aliases[str(p)])
        try:return physical/p.relative_to(logical)
        except ValueError:return p
    def binding(path):
        p=actual(path).resolve(strict=True)
        name=reverse.get(str(p))
        if name is None:
            for root in (physical,stage):
                try:name=str(logical/p.relative_to(root));break
                except ValueError:pass
            if name is None:name=str(p)
        return dict(path=name,sha256=common.sha(p))
    def verify(b):
        path=actual(b['path'])
        if common.sha(path)!=b['sha256']:raise ValueError(f'changed artifact: {path}')
        return path
    old_read,old_write=common.read,common.write
    def lock_json(path,value):
        p=actual(path)
        if p.exists():
            if old_read(p)!=value:raise ValueError(f'changed protocol: {p}')
        else:old_write(p,value)
    common.binding=binding;common.verify=verify
    common.read=lambda path:old_read(actual(path))
    common.write=lambda path,value:old_write(actual(path),value)
    common.lock_json=lock_json
    cfg=importlib.import_module('visual_evidence_probe.experiments.localization.config')
    cfg.OUT=physical/'visible_error_localization_20261026'
    docs=STORE/'experiments'
    if not docs.exists():docs.symlink_to(SOURCE/'experiments',target_is_directory=True)
    return receipt


def main():
    if len(sys.argv)<2:raise SystemExit('usage: data_entrypoint <analysis-module> [args]')
    module=sys.argv[1];sys.argv=[module]+sys.argv[2:]
    receipt=setup_io()
    if ('worker' in sys.argv or module=='localization_crop_control') and not receipt.get('cache_ready',False):
        raise SystemExit('Model cache migration is still running; no fallback to home caches is permitted.')
    importlib.import_module('visual_evidence_probe.analysis_scripts.'+module).main()


if __name__=='__main__':main()
