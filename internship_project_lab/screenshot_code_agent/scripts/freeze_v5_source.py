#!/usr/bin/env python3
"""Create an independent runtime checkout, leaving large read-only inputs linked."""
import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

from screenshot_agent.control import atomic_json


def freeze(project,output):
    output.mkdir(parents=True,exist_ok=False)
    directories = ['src','scripts','configs','tests'] + (['third_party'] if (project/'third_party').exists() else [])
    for name in directories:
        shutil.copytree(project/name,output/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['.venv','.model-env','vendor','datasets','.runtime','.cache']:
        (output/name).symlink_to(project/name,target_is_directory=True)
    metadata = ['pyproject.toml','requirements.lock.txt','requirements-benchmark.lock.txt','UPSTREAM.json','BENCHMARK_UPSTREAM.json']
    extras = [name for name in ['MINI_UPSTREAM.json','WORKFLOW_UPSTREAM.json','requirements-harness.lock.txt'] if (project/name).exists()]
    metadata += extras
    for name in metadata:
        shutil.copy2(project/name,output/name)
    hashes={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
        for name in directories for p in (output/name).rglob('*') if p.is_file()}
    hashes.update({name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in extras})
    atomic_json(output/'source_state.json',{'original_project':str(project),'sha256':hashes,
        'purpose':'Frozen supplementary evaluation runtime; never edit this checkout during a run.'})
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for name in dict.fromkeys([*hashes,'source_state.json',*metadata]):
            archive.write(output/name,name)
    return {'source':str(output),'source_files':len(hashes)}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('outputs/v5_eval_source'))
    a=p.parse_args()
    print(freeze(Path(__file__).resolve().parents[1],a.output.resolve()))
