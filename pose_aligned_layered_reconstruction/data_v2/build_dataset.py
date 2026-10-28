"""Batch asset -> paired clean/occluded scene -> SS target -> DINO cache.

The asset catalogue explicitly assigns train/val/test by physical object UID.
Example: [{"object_uid":"...","mesh":"/local/Aligned.obj","split":"train"}].
Run generation separately for each split; original 2K manifests can be mixed in.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..paired_dataset import sha256_file, canonical_json_sha256
from ..paired_encoder import build_frozen_dino_encoder
from ..pretrained import load_trellis_model
from ..v2.dataset import write_manifest
from .render_scenes import render_recipe, load_parts, canonicalize
from .encode_ss_targets import encode
from .build_feature_cache import cache_scene


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', required=True, help='JSON catalogue with explicit object_uid, mesh, split')
    p.add_argument('--split', choices=('train', 'val', 'test'), required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--variants', type=int, default=6)
    p.add_argument('--max_objects', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20261017)
    p.add_argument('--layers', type=int, default=5)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--min_roundtrip_iou', type=float, default=.9)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if min(a.variants, a.max_objects, a.layers) <= 0 or not 0 <= a.min_roundtrip_iou <= 1:
        raise ValueError('invalid generation counts/roundtrip threshold')
    catalogue = json.loads(Path(a.assets).read_text())
    # Check identity before augmentation; duplicate byte-identical assets cannot
    # enter different splits even if the catalogue gives them different aliases.
    splits, object_splits = {}, {}
    for record in catalogue:
        if record['split'] not in ('train', 'val', 'test') or not record['object_uid']:
            raise ValueError('catalogue requires explicit split and object UID')
        identity = sha256_file(record['mesh'])
        if identity in splits and splits[identity] != record['split']:
            raise ValueError('same source asset occurs in different splits')
        uid = record['object_uid']
        if uid in object_splits and object_splits[uid] != record['split']:
            raise ValueError('same physical object UID occurs in different splits')
        splits[identity], object_splits[uid] = record['split'], record['split']
    selected = [r for r in catalogue if r['split'] == a.split][:a.max_objects]
    if not selected:
        raise ValueError('no assets selected')
    if len({r['object_uid'] for r in selected}) != len(selected):
        raise ValueError('duplicate selected object UID')
    output = Path(a.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=a.resume)
    recipe = {'catalogue_sha256': sha256_file(a.assets), 'split': a.split, 'variants': a.variants,
              'max_objects': a.max_objects, 'seed': a.seed, 'layers': a.layers,
              'min_roundtrip_iou': a.min_roundtrip_iou, 'format': 'full_scene_dataset_recipe.v2'}
    recipe_file = output / 'recipe.json'
    if recipe_file.exists() and json.loads(recipe_file.read_text()) != recipe:
        raise ValueError('resume recipe mismatch')
    recipe_file.write_text(json.dumps(recipe, indent=2))
    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device.index or 0)
    encoder, enc_binding = load_trellis_model('microsoft/TRELLIS-image-large', 'ss_enc_conv3d_16l8_fp16')
    decoder, dec_binding = load_trellis_model('microsoft/TRELLIS-image-large', 'ss_dec_conv3d_16l8_fp16')
    runtime = (encoder.to(device).eval().requires_grad_(False), enc_binding,
               decoder.to(device).eval().requires_grad_(False), dec_binding)
    dino = build_frozen_dino_encoder(device=device)
    rows = []
    for i, asset in enumerate(selected):
        uid = asset['object_uid']
        if Path(uid).name != uid or uid in ('.', '..'):
            raise ValueError('object UID must be a safe directory name')
        folder = output / 'objects' / uid
        folder.mkdir(parents=True, exist_ok=a.resume)
        receipt_path = folder / 'receipt.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['asset'] != asset or receipt['recipe'] != recipe:
                raise ValueError('object receipt identity mismatch')
            for row in receipt['rows']:
                if sha256_file(row['sample']) != row['sha256']:
                    raise ValueError('cached object changed')
            rows.extend(receipt['rows'])
            continue
        object_rows = []
        ss_target = folder / 'ss_target.npz'
        for j in range(a.variants):
            scene = folder / f'scene_{j:03d}'
            scene_recipe = {'uid': f'{uid}:scene:{j}', 'object_uid': uid, 'target_mesh': asset['mesh'],
                'seed': a.seed + i * 1009 + j, 'views': 8, 'layers': a.layers,
                'trajectory': 'narrow' if j % 2 == 0 else 'wide', 'box_count': 1 + j % 3,
                'radius_range': [1.7, 4.8]}
            rng = np.random.default_rng(scene_recipe['seed'])
            scene_recipe['background_rgb'] = rng.uniform(.15, .9, 3).tolist()
            if len(selected) > 1:
                import trimesh
                index = (i + 1 + j) % len(selected)
                if index == i:
                    index = (index + 1) % len(selected)
                other = selected[index]
                canonical = canonicalize(load_parts(other['mesh']))
                placement = trimesh.transformations.random_rotation_matrix(rng.random(3))
                placement[:3, :3] *= rng.uniform(.25, .65)
                placement[:3, 3] = rng.uniform(-.7, .7, 3)
                scene_recipe['occluders'] = [{'mesh': other['mesh'], 'transform': (placement @ canonical).tolist()}]
            if not scene.exists():
                render_recipe(scene_recipe, scene, device)
            else:
                metadata = json.loads((scene / 'scene.json').read_text())
                if metadata['recipe'] != scene_recipe:
                    raise ValueError('existing scene recipe mismatch')
            if not ss_target.exists():
                info = encode(scene / 'target_canonical.obj', ss_target, device, runtime=runtime)
            else:
                info = json.loads(Path(str(ss_target) + '.json').read_text())
                if info['target_sha256'] != sha256_file(ss_target):
                    raise ValueError('SS target changed')
                if info['mesh_sha256'] != sha256_file(scene / 'target_canonical.obj'):
                    raise ValueError('canonical geometry differs across scene variants')
            if info['roundtrip_iou'] < a.min_roundtrip_iou:
                raise ValueError(f'SS roundtrip below threshold for {uid}: {info["roundtrip_iou"]}')
            for name in ('scene.json', 'clean_scene.json'):
                destination = scene / ('occluded.pt' if name == 'scene.json' else 'clean.pt')
                object_rows.append(cache_scene(scene / name, ss_target, destination, dino, resume=a.resume))
        receipt = {'asset': asset, 'recipe': recipe, 'rows': object_rows}
        receipt_path.write_text(json.dumps(receipt, indent=2))
        rows.extend(object_rows)
        print(json.dumps({'objects': i + 1, 'total': len(selected), 'samples': len(rows)}), flush=True)
    write_manifest(output / 'manifest.json', rows, recipe, output)


if __name__ == '__main__':
    main()
