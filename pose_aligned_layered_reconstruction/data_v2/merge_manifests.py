"""Mix existing clean 2K, occlusion variants and new scenes without rewriting caches."""
import argparse
import json
from pathlib import Path

import torch

from ..paired_dataset import sha256_file
from ..contracts import LAYERED_SAMPLE_MANIFEST_FORMAT, LAYERED_SAMPLE_FORMAT
from ..v2.dataset import SceneDataset, write_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests', nargs='+', required=True)
    p.add_argument('--exclude_objects_from', nargs='*', default=[])
    p.add_argument('--output', required=True)
    a = p.parse_args()
    excluded = set()
    for manifest in a.exclude_objects_from:
        excluded.update(SceneDataset(manifest, require_labels=False).identity()['object_uids'])
    rows, sources, seen = [], [], set()
    for manifest in a.manifests:
        dataset = SceneDataset(manifest, require_labels=False)
        payload = json.loads(Path(manifest).read_text())
        root = Path(payload['root'])
        sources.append(dataset.identity())
        for record in dataset.rows:
            uid = record['uid']
            if uid in seen:
                raise ValueError(f'duplicate sample UID while mixing: {uid}')
            if record.get('object_uid', uid) in excluded:
                raise ValueError(f'held-out physical object present in training sources: {uid}')
            seen.add(uid)
            path = (root / record['sample']).resolve()
            if sha256_file(path) != record['sha256']:
                raise ValueError('source sample hash mismatch')
            row = {**record, 'sample': str(path)}
            if payload['format'] == LAYERED_SAMPLE_MANIFEST_FORMAT:
                row['source_sample_format'] = LAYERED_SAMPLE_FORMAT
            rows.append(row)
    output = Path(a.output).resolve()
    write_manifest(output, rows, {'mixture_sources': sources, 'excluded_object_count': len(excluded)}, output.parent)


if __name__ == '__main__':
    main()
