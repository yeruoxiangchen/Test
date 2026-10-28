import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

STORE = Path('/data/zjr/visual_evidence_probe')
VENDOR = STORE / 'vendor/Depth-Anything-3'
MODEL = STORE / 'models/DA3-LARGE-1.1'
OUT = STORE / 'outputs/dense_evidence_20261027'
REVISION = '0e109ae307c5982f319a67cf6f9f99ccdc0ec97c'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--download', action='store_true'); a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.download:
        print(dict(stage='downloading', repo='depth-anything/DA3-LARGE-1.1', revision=REVISION), flush=True)
        MODEL.mkdir(parents=True, exist_ok=True)
        for name in ('config.json','README.md','model.safetensors'):
            final = MODEL/name
            if final.exists(): continue
            partial = MODEL/(name+'.partial')
            url = f'https://huggingface.co/depth-anything/DA3-LARGE-1.1/resolve/{REVISION}/{name}?download=true&receipt={int(time.time())}'
            subprocess.run(['curl','--fail','--location','--retry','4','--retry-all-errors',
                '--connect-timeout','30','--max-time','3600','--continue-at','-',
                '--output',str(partial),url],check=True)
            partial.replace(final)
        # Hash the checkpoint once, on /data; later phases consume this receipt.
        digest = hashlib.sha256()
        with (MODEL / 'model.safetensors').open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): digest.update(block)
        tree = json.loads((STORE / 'vendor/da3_large_1_1_tree.json').read_text())
        expected = next(row['lfs']['oid'] for row in tree if row['path'] == 'model.safetensors')
        if digest.hexdigest() != expected:
            raise ValueError('downloaded DA3 checkpoint differs from official pinned LFS identity')
        write(OUT / 'da3_weights.json', dict(repo='depth-anything/DA3-LARGE-1.1', revision=REVISION,
            path=str(MODEL), sha256=digest.hexdigest(), bytes=(MODEL / 'model.safetensors').stat().st_size,
            vendor_commit=subprocess.check_output(['git', '-C', str(VENDOR), 'rev-parse', 'HEAD'], text=True).strip()))
        print(dict(stage='download_complete', directory=str(MODEL)), flush=True)
    else:
        from depth_anything_3.api import DepthAnything3
        versions = {}
        for name in ('torch', 'torchvision', 'numpy', 'omegaconf', 'evo', 'e3nn', 'moviepy', 'pycolmap'):
            versions[name] = importlib.metadata.version(name)
        write(OUT / 'da3_environment.json', dict(time=time.time(), versions=versions,
            api_import_passed=True, isolated_environment=True, vendor=str(VENDOR)))
        print(dict(stage='environment_ready', versions=versions), flush=True)


if __name__ == '__main__': main()
