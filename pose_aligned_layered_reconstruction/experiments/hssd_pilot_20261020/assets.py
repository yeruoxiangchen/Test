"""Freeze 16 distinct assets and independently materialize canonical geometry."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ...data_v2.render_scenes import load_parts
from ...mesh_metrics import structure_metrics
from .common import SOURCE, OUT, binding, identity, write_json, event, object_dir

C = np.array([[1., 0, 0, 0], [0, 0, -1., 0], [0, 1., 0, 0], [0, 0, 0, 1.]])
# Frozen before viewing latent/model scores. Includes chairs, thin legs,
# nonuniformly scaled tables and a reflected cabinet. One instance per asset.
INSTANCE_IDS = [76, 103, 83, 51, 36, 75, 80, 65, 57, 84, 97, 109, 56, 101, 108, 89]


def instance_matrix(instance):
    q = np.asarray(instance.get('rotation', [1, 0, 0, 0]), float)
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    m[:3, 3] = instance.get('translation', [0, 0, 0])
    scale = np.asarray(instance.get('non_uniform_scale', [1, 1, 1]), float)
    if np.any(np.abs(scale) < 1.e-8):
        raise ValueError('Singular instance scale')
    return m, np.diag([*scale, 1.])


def prepare():
    import trimesh
    scene_file = SOURCE / 'HSSD-hab/scenes/102344049.scene_instance.json'
    scene = json.loads(scene_file.read_text())
    meta_file = SOURCE / 'HSSD-hab/metadata/fpmodels-with-decomposed.csv'
    meta = {r['id']: r for r in csv.DictReader(meta_file.open())}
    configs = {p.name.removesuffix('.object_config.json'): p
               for p in (SOURCE / 'HSSD-hab/objects').rglob('*.object_config.json')}
    rows = []
    for index in INSTANCE_IDS:
        obj = scene['object_instances'][index]
        asset = obj['template_name']
        cfg_file = configs[asset]
        cfg = json.loads(cfg_file.read_text())
        original = (cfg_file.parent / cfg['render_asset']).resolve()
        converted = SOURCE / 'renders/hssd_102344049/converted_glb' / original.relative_to(SOURCE / 'HSSD-hab')
        rigid, deformation = instance_matrix(obj)
        parts = load_parts(converted, C @ deformation)
        vertices = np.concatenate([m.vertices for m in parts])
        lo, hi = vertices.min(0), vertices.max(0)
        scale = 1. / np.max(hi - lo)
        normalization = np.eye(4)
        normalization[:3, :3] *= scale
        normalization[:3, 3] = -(lo + hi) * .5 * scale
        for part in parts:
            part.apply_transform(normalization)
        mesh = trimesh.util.concatenate(parts)
        # This similarity acts on all scene geometry. Nonuniform scale and
        # reflection are already baked into the target, not into camera R.
        canonical_from_world = normalization @ C @ np.linalg.inv(rigid)
        uid = f'hssd_102344049_i{index:03d}'
        root = OUT / 'objects' / uid
        root.mkdir(parents=True, exist_ok=True)
        mesh_file = root / 'target_canonical.ply'
        payload = mesh.export(file_type='ply')
        if mesh_file.exists() and mesh_file.read_bytes() != payload:
            raise ValueError('Canonical mesh changed: ' + uid)
        if not mesh_file.exists():
            mesh_file.write_bytes(payload)
        recovered = trimesh.transform_points(mesh.vertices, np.linalg.inv(canonical_from_world))
        expected_parts = load_parts(converted, rigid @ deformation)
        expected = np.concatenate([m.vertices for m in expected_parts])
        roundtrip_error = float(np.max(np.abs(expected - recovered)))
        if roundtrip_error > 1.e-7:
            raise ValueError('Asset/canonical/world transform is inconsistent')
        info = meta.get(asset, {})
        row = {'uid': uid, 'scene': '102344049', 'instance': index, 'asset_uid': asset,
               'asset_family': info.get('decomposedFrom') or asset,
               'name': info.get('name'), 'category': info.get('main_category'),
               'source': binding(original), 'converted': binding(converted), 'config': binding(cfg_file),
               'mesh': binding(mesh_file), 'deformation': deformation.tolist(),
               'normalization': normalization.tolist(), 'canonical_from_world': canonical_from_world.tolist(),
               'rigid_world_from_asset': rigid.tolist(),
               'mirror': bool(np.linalg.det(deformation[:3, :3]) < 0),
               'nonuniform_scale': bool(np.ptp(np.abs(np.diag(deformation)[:3])) > 1.e-5),
               'transform_roundtrip_max_abs': roundtrip_error, 'structure': structure_metrics(mesh)}
        write_json(root / 'asset.json', row)
        rows.append(row)
        event(stage='assets', index=len(rows), total=len(INSTANCE_IDS), uid=uid)
    if len({r['asset_uid'] for r in rows}) != len(rows):
        raise ValueError('Duplicate targets in qualification cohort')
    frozen = {'version': 1, 'scope': 'training-set qualification only; no held-out/generalization claim',
              'source_scene': binding(scene_file), 'metadata': binding(meta_file), 'objects': rows,
              'target_views': 150, 'target_render_side': 512, 'input_views': 8,
              'target_render_samples': 128, 'surface_resolution': 64, 'seed': 20261020,
              'sampling_steps': 25, 'eval_seeds': [42, 43],
              'ss_roundtrip_min_iou': .90,
              'raw_gt_policy': 'native canonical raw mesh; no ICP, independent centering or scale fitting',
              'latent_policy': 'clean dense target renders; DINO x_prenorm without layernorm; posterior mean',
              'input_policy': 'original full architecture and original instance placement; paired clean views',
              'expansion_policy': 'no automatic expansion beyond these 16 targets'}
    path = OUT / 'plan.json'
    result = {'identity': identity(frozen), 'plan': frozen}
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('Existing pilot plan differs; create a new version')
    write_json(path, result)
    event(stage='assets_complete', objects=len(rows), identity=result['identity'])


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
