"""Pinned metadata acquisition and content-verified scene dependency downloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import time

import requests
from . import common as c

SESSION = requests.Session()


def request(url, **kwargs):
    for attempt in range(4):
        try:
            response = SESSION.request(timeout=(20, 90), url=url, **kwargs)
            response.raise_for_status(); return response
        except requests.RequestException:
            if attempt == 3: raise
            time.sleep(2 ** attempt)


def safe(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe upstream relative path')
    return c.RAW / name


def metadata_rows(names):
    names = sorted(set(names)); rows = []
    for start in range(0, len(names), 150):
        batch = names[start:start + 150]
        cache = c.DATASET / 'remote_metadata' / (c.digest(batch) + '.json')
        if cache.exists(): result = c.read(cache)
        else:
            result = request(f'https://huggingface.co/api/datasets/{c.REPO}/paths-info/{c.REVISION}',
                             method='POST', json={'paths': batch}).json()
            if not isinstance(result, list): raise ValueError('Invalid pinned paths-info response')
            c.write(cache, result)
        if {r['path'] for r in result} != set(batch) or any(r['type'] != 'file' for r in result):
            raise ValueError('Incomplete/changed remote metadata')
        rows.extend({'path': r['path'], 'size': int(r['size']), 'sha256': r.get('lfs', {}).get('oid'),
                     'git_oid': r.get('oid')} for r in result)
    return rows


def valid(path, row):
    if not path.is_file() or path.stat().st_size != row['size']: return False
    h = hashlib.sha256() if row.get('sha256') else hashlib.sha1(f"blob {row['size']}\0".encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''): h.update(chunk)
    return h.hexdigest() == (row.get('sha256') or row['git_oid'])


def download_one(row):
    target = c.data_path(safe(row['path'])); receipt = c.DATASET / 'downloads' / (c.digest(row['path']) + '.json')
    if receipt.exists():
        saved = c.read(receipt)
        if saved['remote'] != row: raise ValueError('Pinned file changed')
        if not target.is_file() or target.stat().st_size != row['size']: raise ValueError('Missing completed download')
        # Cached size/mtime checks avoid rehashing all raw assets at every resume.
        if target.stat().st_mtime_ns == saved['mtime_ns']: return target
        if not valid(target, row): raise ValueError('Modified completed asset: ' + str(target))
    elif target.exists() and not valid(target, row): raise ValueError('Unreceipted invalid asset: ' + str(target))
    elif not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True); tmp = target.with_suffix(target.suffix + '.partial')
        for attempt in range(4):
            try:
                response = request(f'https://huggingface.co/datasets/{c.REPO}/resolve/{c.REVISION}/{row["path"]}',
                                   method='GET', stream=True)
                count = 0
                with tmp.open('wb') as stream:
                    for chunk in response.iter_content(1 << 20):
                        count += len(chunk)
                        if count > row['size']: raise ValueError('Remote file exceeds frozen size')
                        stream.write(chunk)
                if not valid(tmp, row): raise ValueError('Download hash/size mismatch')
                tmp.replace(target); break
            except (requests.RequestException, ValueError):
                if attempt == 3: raise
                time.sleep(2 ** attempt)
    c.write(receipt, {'remote': row, 'path': str(target), 'mtime_ns': target.stat().st_mtime_ns,
                      'verified_on_download': True})
    return target


def download_many(rows, stage):
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download_one, row): row for row in rows}
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count == 1 or count % 20 == 0 or count == len(rows):
                c.event(stage, completed=count, total=len(rows), path=futures[future]['path'])


def inventory():
    c.environment()
    c.DATASET.mkdir(parents=True, exist_ok=True)
    source = c.LEGACY / 'metadata/HSSD-hab_repo.json'
    target = c.DATASET / 'pinned_repository.json'
    if not target.exists():
        # One bounded copy of the existing filename catalog; no raw scene scan.
        shutil.copyfile(source, target)
    repository = c.read(target)
    if repository['revision'] != c.REVISION or repository['repo'] != c.REPO: raise ValueError('Pinned repository mismatch')
    names = repository['files']
    scenes = sorted(x for x in names if x.startswith('scenes/') and x.endswith('.scene_instance.json'))
    wanted = scenes + ['README.md', 'scene_splits.yaml', 'metadata/fpmodels-with-decomposed.csv',
                      'metadata/hssd_obj_semantics_condensed.csv']
    rows = metadata_rows(wanted)
    if sum(r['size'] for r in rows) > 128 << 20: raise ValueError('Metadata exceeds bounded preparation budget')
    download_many(rows, 'catalog_download')
    value = {'repo': c.REPO, 'revision': c.REVISION, 'scenes': scenes, 'files': rows,
             'filename_catalog': c.binding(target), 'selection': 'all regular scenes in pinned public release'}
    c.locked(c.OUT / 'catalog.json', value)
    c.event('catalog_complete', scenes=len(scenes), files=len(rows), bytes=sum(r['size'] for r in rows))


def scene_dependencies(scene_ids, include_geometry=False):
    names = c.read(c.DATASET / 'pinned_repository.json')['files']; name_set = set(names)
    configs = {}
    for name in names:
        if name.startswith('objects/') and name.endswith('.object_config.json'):
            stem = PurePosixPath(name).name.removesuffix('.object_config.json')
            configs.setdefault(stem, []).append(name)
    wanted = set(); scenes = []
    for sid in scene_ids:
        scene_path = f'scenes/{sid}.scene_instance.json'; scene = c.read(safe(scene_path))
        config_paths = []
        for template in sorted({r['template_name'] for r in scene['object_instances']}):
            found = configs.get(template, [])
            if len(found) != 1: raise ValueError('Nonunique/missing object configuration: ' + template)
            config_paths.append(found[0]); wanted.add(found[0])
        stage = scene['stage_instance']['template_name'] + '.stage_config.json'
        semantic = f'semantics/scenes/{sid}.semantic_config.json'
        if stage not in name_set or semantic not in name_set: raise ValueError('Missing architecture/room annotations')
        wanted.update((stage, semantic))
        scenes.append({'scene': sid, 'scene_file': scene_path, 'configs': config_paths,
                       'stage_config': stage, 'semantic': semantic})
    download_many(metadata_rows(wanted), 'dependency_metadata')
    render_paths = set()
    for row in scenes:
        refs = {}
        for name in row['configs'] + [row['stage_config']]:
            cfg = c.read(safe(name)); relative = posixpath.normpath(posixpath.join(posixpath.dirname(name), cfg['render_asset']))
            if relative not in name_set:
                relative = cfg['render_asset']
                if relative not in name_set: raise ValueError('Missing render asset: ' + name)
            render_paths.add(relative); refs[name] = relative
        row['render_assets'] = refs
    remote = metadata_rows(render_paths)
    result = {'scenes': scenes, 'geometry_files': remote, 'download_bytes': sum(r['size'] for r in remote)}
    if include_geometry: download_many(remote, 'geometry_download')
    return result


def download_assets():
    p = c.plan(); deps = p['plan']['dependencies']
    if deps['download_bytes'] > 40 * 1024 ** 3: raise ValueError('Frozen raw dependency budget exceeds 40GiB')
    if shutil.disk_usage(c.DATA).free < 100 * 1024 ** 3: raise ValueError('/data needs at least 100GiB free')
    download_many(deps['geometry_files'], 'geometry_download')
    c.write(c.OUT / 'download_complete.json', {'time': c.now(), 'identity': p['identity'], 'success': True,
                                             'files': len(deps['geometry_files']), 'bytes': deps['download_bytes']})
    c.event('geometry_download_complete', files=len(deps['geometry_files']), bytes=deps['download_bytes'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['inventory', 'assets'])
    args = parser.parse_args()
    inventory() if args.action == 'inventory' else download_assets()
