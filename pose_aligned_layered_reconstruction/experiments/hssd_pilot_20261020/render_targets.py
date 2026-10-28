"""Blender entrypoint: official TRELLIS clean-target lighting/camera recipe."""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

HERE = Path(__file__).resolve().parent


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--asset', required=True)
    p.add_argument('--coordinate-only', action='store_true')
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    asset_file = Path(a.asset)
    row = json.loads(asset_file.read_text())
    root = asset_file.parent
    out = root / 'target_renders'
    out.mkdir(exist_ok=True)
    official = module('official_render', HERE / 'vendor/blender_render.py')
    utils = module('official_utils', HERE / 'vendor/utils.py')
    official.init_scene()
    # GLB importer already maps Habitat's Y-up to Blender's Z-up.
    bpy.ops.import_scene.gltf(filepath=row['converted']['path'], merge_vertices=False)
    c = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    transform = Matrix(row['normalization']) @ c @ Matrix(row['deformation']) @ c.inverted()
    for obj in list(bpy.context.scene.objects):
        if obj.parent is None:
            obj.matrix_world = transform @ obj.matrix_world
    bpy.context.view_layer.update()
    vertices = []
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            vertices.extend([list(obj.matrix_world @ v.co) for v in obj.data.vertices])
    vertices = np.asarray(vertices, np.float64)
    np.save(root / 'blender_canonical_vertices.npy', vertices)
    if not len(vertices) or not np.isfinite(vertices).all() or np.max(np.abs(vertices)) > .5001:
        raise ValueError('Blender geometry outside expected canonical frame')
    if a.coordinate_only:
        print('PILOT_COORDINATES ' + json.dumps({'uid': row['uid'], 'vertices': len(vertices)}), flush=True)
        return
    official.init_render(engine='CYCLES', resolution=512)
    scene = bpy.context.scene
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 4
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.get_devices()
    devices = [d for d in prefs.devices if d.type == 'CUDA']
    if len(devices) != 1:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one GPU')
    for d in prefs.devices:
        d.use = d.type == 'CUDA'
    cam = official.init_camera()
    official.init_lighting()
    rng = np.random.RandomState(20261020)
    offset = tuple(rng.rand(2))
    frames = []
    for i in range(150):
        yaw, pitch = utils.sphere_hammersley_sequence(i, 150, offset)
        cam.location = (2 * math.cos(yaw) * math.cos(pitch),
                        2 * math.sin(yaw) * math.cos(pitch), 2 * math.sin(pitch))
        fov = math.radians(40)
        cam.data.lens = 16 / math.tan(fov / 2)
        scene.render.filepath = str(out / f'{i:03d}.png')
        bpy.context.view_layer.update()
        if not Path(scene.render.filepath).exists():
            bpy.ops.render.render(write_still=True)
        frames.append({'file_path': f'{i:03d}.png', 'camera_angle_x': fov,
                       'transform_matrix': official.get_transform_matrix(cam)})
        print('PILOT_RENDER ' + json.dumps({'uid': row['uid'], 'view': i + 1, 'total': 150}), flush=True)
    (out / 'transforms.json').write_text(json.dumps({'frames': frames, 'asset': row,
        'recipe': 'official TRELLIS sphere_hammersley 150, radius2, fov40, Cycles128, RGBA512',
        'normalization': 'already applied, independently checked against trimesh'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
