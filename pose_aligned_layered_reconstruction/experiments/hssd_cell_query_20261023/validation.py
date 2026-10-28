"""Lock 32 development assets from two unseen scenes, without model-based selection.

This is a selection manifest + native mesh coordinate audit, not a ready-to-train
SceneDataset. Full scene images, visibility labels and latent qualification are
explicitly separate, unfinished artifacts; no validation observations are used
by the HSSD16 training queue.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .common import PACKAGE, OUT, PILOT, binding, verify, write
from ..hssd_pilot_20261020.assets import instance_matrix, C

SOURCE = PACKAGE.parent / 'datasets/scene_occlusion_pilot_20261020/HSSD-hab'
SCENES = ('102343992', '102344022')
CATEGORIES = ('chair', 'table', 'couch', 'cabinet', 'bed', 'stool', 'shelves', 'stand', 'bench', 'chest_of_drawers')
SEED = 20261023


def family(asset, metadata):
    return metadata.get(asset, {}).get('decomposedFrom') or asset


def choose(candidates, excluded, limit=16):
    """Round-robin categories, hash order within category, unique asset families."""
    pools = {c: [] for c in CATEGORIES}
    for row in candidates:
        if row['asset_family'] not in excluded and row['category'] in pools:
            pools[row['category']].append(row)
    for rows in pools.values():
        rows.sort(key=lambda r: (hashlib.sha256(f'{SEED}:{r["scene"]}:{r["asset_family"]}'.encode()).hexdigest(), r['instance']))
    chosen, used = [], set(excluded)
    while len(chosen) < limit:
        progress = False
        for category in CATEGORIES:
            while pools[category] and pools[category][0]['asset_family'] in used:
                pools[category].pop(0)
            if pools[category] and len(chosen) < limit:
                row = pools[category].pop(0)
                chosen.append(row)
                used.add(row['asset_family'])
                progress = True
        if not progress:
            raise ValueError('Insufficient eligible distinct families; do not silently change selection')
    return chosen


def select():
    meta_file = SOURCE / 'metadata/fpmodels-with-decomposed.csv'
    metadata = {r['id']: r for r in csv.DictReader(meta_file.open())}
    train_scene = SOURCE / 'scenes/102344049.scene_instance.json'
    train = json.loads(train_scene.read_text())
    excluded = {family(r['template_name'], metadata) for r in train['object_instances']}
    configs = {}
    for p in (SOURCE / 'objects').rglob('*.object_config.json'):
        name = p.name.removesuffix('.object_config.json')
        if name in configs: raise ValueError('Ambiguous asset config')
        configs[name] = p
    selected, scene_bindings = [], []
    used = set(excluded)
    for scene_id in SCENES:
        path = SOURCE / 'scenes' / (scene_id + '.scene_instance.json')
        scene_bindings.append(binding(path))
        scene = json.loads(path.read_text())
        candidates = []
        for index, instance in enumerate(scene['object_instances']):
            asset = instance['template_name']
            m = metadata.get(asset, {})
            if m.get('hasMultipleObjects', '').lower() == 'true': continue
            cfg_path = configs[asset]
            cfg = json.loads(cfg_path.read_text())
            candidates.append({'uid': f'hssd_{scene_id}_i{index:03d}', 'scene': scene_id,
                'instance': index, 'asset_uid': asset, 'asset_family': family(asset, metadata),
                'category': m.get('main_category'), 'name': m.get('name'),
                'config_path': str(cfg_path), 'source_path': str((cfg_path.parent / cfg['render_asset']).resolve()),
                'instance_data': instance})
        chosen = choose(candidates, used)
        for row in chosen:
            row['config'] = binding(row.pop('config_path'))
            row['source'] = binding(row.pop('source_path'))
        selected.extend(chosen)
        used.update(r['asset_family'] for r in chosen)
    assert len(selected) == 32 and not excluded & {r['asset_family'] for r in selected}
    result = {'format': 'hssd_heldout_selection.v1', 'role': 'development_validation_not_final_test',
        'selection_seed': SEED, 'selection_rule': 'round-robin fixed categories; SHA order per family; lowest instance tie-break; 16 per scene',
        'train_scene': binding(train_scene), 'scenes': scene_bindings, 'metadata': binding(meta_file),
        'excluded_training_context_families': sorted(excluded), 'objects': selected,
        'qualification': {'source_assets_local': True, 'input_images_ready': False, 'latents_ready': False,
                          'training_manifest_ready': False, 'occlusion_bins_measured': False},
        'assumptions': 'Known pose, target mask and target canonicalization; family IDs prevent known family overlap, not all possible geometric near-duplicates.'}
    path = OUT / 'validation/selection.json'
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('Locked validation selection changed')
    write(path, result)
    print(json.dumps({'validation_objects': 32, 'scenes': SCENES, 'excluded_train_context_families': len(excluded), 'path': str(path)}))


def audit_meshes():
    import trimesh
    path = OUT / 'validation/selection.json'
    plan = json.loads(path.read_text())
    for b in [plan['train_scene'], plan['metadata'], *plan['scenes']]: verify(b)
    reports = []
    for row in plan['objects']:
        verify(row['config'])
        asset = verify(row['source'])
        scene = trimesh.load(asset, force='scene', process=False, skip_materials=True)
        rigid, deformation = instance_matrix(row['instance_data'])
        original, canonical = [], []
        for node in scene.graph.nodes_geometry:
            matrix, name = scene.graph[node]
            mesh = scene.geometry[name].copy()
            mesh.apply_transform(matrix)
            if not len(mesh.faces): continue
            original.append(mesh.copy())
            mesh.apply_transform(C @ deformation)
            canonical.append(mesh)
        if not canonical: raise ValueError('Empty heldout asset; keep row and report failure')
        mesh = trimesh.util.concatenate(canonical)
        lo, hi = mesh.bounds
        norm = np.eye(4)
        norm[:3, :3] /= np.max(hi-lo)
        norm[:3, 3] = -(lo+hi)*.5/np.max(hi-lo)
        mesh.apply_transform(norm)
        if not np.isfinite(mesh.vertices).all() or not len(mesh.faces): raise ValueError('Invalid mesh')
        world_to_canonical = norm @ C @ np.linalg.inv(rigid)
        expected = trimesh.transform_points(trimesh.util.concatenate(original).vertices, rigid @ deformation)
        recovered = trimesh.transform_points(mesh.vertices, np.linalg.inv(world_to_canonical))
        error = float(np.max(np.abs(expected - recovered)))
        if error > 1e-7: raise ValueError('Heldout coordinate roundtrip failed')
        root = OUT / 'validation/objects' / row['uid']
        root.mkdir(parents=True, exist_ok=True)
        output = root / 'target_canonical.ply'
        payload = mesh.export(file_type='ply')
        if output.exists() and output.read_bytes() != payload: raise ValueError('Changed heldout canonical mesh')
        if not output.exists(): output.write_bytes(payload)
        reread = trimesh.load(output, force='mesh', process=False)
        report = {'uid': row['uid'], 'mesh': binding(output), 'source': row['source'],
            'vertex_count': len(mesh.vertices), 'face_count': len(mesh.faces),
            'transform_roundtrip_max_abs': error, 'ply_max_abs': float(np.max(np.abs(mesh.vertices-reread.vertices))),
            'canonical_from_world': world_to_canonical.tolist(), 'normalization': norm.tolist(),
            'scope': 'geometry and coordinates only; no mesh-to-latent or scene-input qualification yet'}
        write(root / 'audit.json', report)
        reports.append(report)
        print(json.dumps({'uid': row['uid'], 'complete': len(reports), 'total': 32, 'roundtrip': error}), flush=True)
    write(OUT / 'validation/mesh_audit.json', {'selection': binding(path), 'passed': len(reports) == 32, 'objects': reports})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['select', 'audit'])
    a = p.parse_args()
    select() if a.action == 'select' else audit_meshes()
