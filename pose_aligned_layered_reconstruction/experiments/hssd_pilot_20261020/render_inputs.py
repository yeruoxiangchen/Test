"""Blender: original full-room lighting, paired input cameras, object-ID masks."""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--asset', required=True)
    a = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    root = Path(a.asset).parent
    row = json.loads(Path(a.asset).read_text())
    tracker = Path(__file__).resolve().parents[3]
    source = tracker / 'datasets/scene_occlusion_pilot_20261020/renders/hssd_102344049/complete_scene.blend'
    bpy.ops.wm.open_mainfile(filepath=str(source))
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 4
    scene.render.resolution_x = scene.render.resolution_y = 518
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    prefs.get_devices()
    devices = [d for d in prefs.devices if d.type == 'CUDA']
    if len(devices) != 1:
        raise RuntimeError('Exactly one visible CUDA device required')
    for d in prefs.devices:
        d.use = d.type == 'CUDA'
    # Enforce the same opaque geometry policy as the layered labels.
    for material in bpy.data.materials:
        material.blend_method = 'OPAQUE'
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    for name, value in [('Alpha', 1.), ('Transmission', 0.)]:
                        socket = node.inputs[name]
                        for link in list(socket.links):
                            material.node_tree.links.remove(link)
                        socket.default_value = value
    camera = bpy.data.objects.new('Qualification camera', bpy.data.cameras.new('Qualification camera'))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.sensor_width = 32
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.clip_start = .001
    camera.data.clip_end = 1000
    cameras = np.load(root / 'camera.npz')
    c = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]], float)
    world_from_canonical = np.linalg.inv(row['canonical_from_world'])
    cv_to_gl = np.diag([1., -1., -1., 1.])
    scene.view_layers[0].use_pass_object_index = True
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    render = nodes.new('CompositorNodeRLayers')
    mask = nodes.new('CompositorNodeIDMask')
    mask.index = row['instance'] + 2
    mask.use_antialiasing = False
    scene.node_tree.links.new(render.outputs['IndexOB'], mask.inputs[0])
    output = nodes.new('CompositorNodeOutputFile')
    output.format.file_format = 'PNG'
    output.format.color_mode = 'BW'
    output.format.color_depth = '8'
    composite = nodes.new('CompositorNodeComposite')
    scene.node_tree.links.new(render.outputs['Image'], composite.inputs['Image'])
    scene.node_tree.links.new(mask.outputs['Alpha'], output.inputs[0])
    folder = root / 'input_cycles'
    folder.mkdir(exist_ok=True)
    output.base_path = str(folder)
    frames = []
    for domain in ['clean', 'occluded']:
        for obj in scene.objects:
            if obj.type == 'MESH':
                obj.hide_render = domain == 'clean' and obj.pass_index != row['instance']+2
        for i, (k, w) in enumerate(zip(cameras['intrinsics'], cameras['extrinsics'])):
            pose = c @ world_from_canonical @ np.linalg.inv(w) @ cv_to_gl
            pose[:3, :3] /= np.linalg.norm(pose[:3, 0])
            camera.matrix_world = Matrix(pose.tolist())
            camera.data.lens = float(k[0, 0]) * 32 / 518
            scene.render.filepath = str(folder / f'{domain}_{i:02d}_rgba.png')
            output.file_slots[0].path = f'{domain}_{i:02d}_id_'
            scene.frame_set(1)
            bpy.context.view_layer.update()
            bpy.ops.render.render(write_still=True)
            mask_path = folder / f'{domain}_{i:02d}_mask.png'
            (folder / f'{domain}_{i:02d}_id_0001.png').replace(mask_path)
            frames.append({'domain': domain, 'index': i, 'world_blender_c2w': pose.tolist()})
            print('PILOT_INPUT ' + json.dumps({'uid': row['uid'], 'domain': domain, 'view': i+1}), flush=True)
    (folder / 'render_receipt.json').write_text(json.dumps({'frames': frames,
        'source': str(source), 'renderer': 'Cycles32 original preview lighting; all original geometry retained',
        'alpha_policy': 'opaque BSDF alpha1 transmission0 matching depth labels'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
