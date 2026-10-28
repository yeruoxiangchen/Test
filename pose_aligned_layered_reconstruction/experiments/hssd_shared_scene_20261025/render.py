"""Blender: render each shared RGB frame ONCE, all instance masks together."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import bpy
from mathutils import Matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', nargs='+', type=int)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    root = Path(args.output)
    doc = json.loads((root / 'plan.json').read_text())
    plan = doc['plan']
    bpy.ops.wm.open_mainfile(filepath=plan['blend']['path'])
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = plan['cycles_samples']
    scene.cycles.use_denoising = True
    scene.render.threads_mode, scene.render.threads = 'FIXED', 4
    scene.render.resolution_x = scene.render.resolution_y = plan['image_side']
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    prefs.get_devices()
    if len([d for d in prefs.devices if d.type == 'CUDA']) != 1:
        raise RuntimeError('Exactly one visible GPU is required')
    for d in prefs.devices:
        d.use = d.type == 'CUDA'
    for obj in scene.objects:
        if obj.type == 'MESH':
            obj.hide_render = False
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
    c = np.array([[1.,0,0,0],[0,0,-1.,0],[0,1.,0,0],[0,0,0,1.]])
    geometry_path = root / 'scene_geometry.npz'
    if not geometry_path.exists():
        vertices, faces, owners, offset = [], [], [], 0
        for obj in scene.objects:
            if obj.type != 'MESH' or not len(obj.data.polygons):
                continue
            mesh = obj.data
            mesh.calc_loop_triangles()
            xyz = np.empty(len(mesh.vertices) * 3, np.float32)
            mesh.vertices.foreach_get('co', xyz)
            xyz = xyz.reshape(-1, 3)
            matrix = np.linalg.inv(c) @ np.asarray(obj.matrix_world)
            xyz = xyz @ matrix[:3, :3].T + matrix[:3, 3]
            tri = np.empty(len(mesh.loop_triangles) * 3, np.int32)
            mesh.loop_triangles.foreach_get('vertices', tri)
            vertices.append(xyz.astype(np.float32)); faces.append(tri.reshape(-1, 3) + offset)
            owners.append(np.full(len(mesh.loop_triangles), obj.pass_index, np.int32))
            offset += len(xyz)
        with geometry_path.with_suffix('.tmp').open('wb') as stream:
            np.savez(stream, vertices=np.concatenate(vertices), faces=np.concatenate(faces), owners=np.concatenate(owners))
        geometry_path.with_suffix('.tmp').replace(geometry_path)
    camera = bpy.data.objects.new('Shared RGB capture', bpy.data.cameras.new('Shared RGB capture'))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.sensor_width = 32
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.clip_start, camera.data.clip_end = .001, 200
    scene.view_layers[0].use_pass_object_index = True
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    render = nodes.new('CompositorNodeRLayers')
    composite = nodes.new('CompositorNodeComposite')
    scene.node_tree.links.new(render.outputs['Image'], composite.inputs['Image'])
    outputs = []
    for row in plan['objects']:
        mask = nodes.new('CompositorNodeIDMask')
        mask.index = row['instance'] + 2
        mask.use_antialiasing = False
        scene.node_tree.links.new(render.outputs['IndexOB'], mask.inputs[0])
        out = nodes.new('CompositorNodeOutputFile')
        out.format.file_format, out.format.color_mode, out.format.color_depth = 'PNG', 'BW', '8'
        out.base_path = str(root / 'masks' / row['uid'])
        Path(out.base_path).mkdir(parents=True, exist_ok=True)
        scene.node_tree.links.new(mask.outputs['Alpha'], out.inputs[0])
        outputs.append((row, out))
    (root / 'rgb').mkdir(exist_ok=True)
    for index in args.frames if args.frames is not None else range(plan['frame_count']):
        receipt = root / 'frames' / f'{index:03d}.json'
        if receipt.exists():
            value = json.loads(receipt.read_text())
            if value['plan_identity'] != doc['identity']:
                raise ValueError('Changed completed frame')
            required = [root / 'rgb' / f'{index:03d}.png'] + [root / 'masks' / r['uid'] / f'{index:03d}.png' for r in plan['objects']]
            if not all(p.exists() for p in required):
                raise ValueError('Missing completed frame artifact')
            continue
        k, w = np.asarray(plan['intrinsics'][index]), np.asarray(plan['world_w2c'][index])
        pose = c @ np.linalg.inv(w) @ np.diag([1., -1., -1., 1.])
        camera.matrix_world = Matrix(pose.tolist())
        camera.data.lens = float(k[0, 0]) * 32 / plan['image_side']
        scene.render.filepath = str(root / 'rgb' / f'{index:03d}.png')
        for row, out in outputs:
            out.file_slots[0].path = f'{index:03d}_id_'
        scene.frame_set(1)
        bpy.context.view_layer.update()
        print('SHARED_RENDER ' + json.dumps({'frame': index, 'phase': 'rendering'}), flush=True)
        bpy.ops.render.render(write_still=True)
        for row, out in outputs:
            folder = Path(out.base_path)
            (folder / f'{index:03d}_id_0001.png').replace(folder / f'{index:03d}.png')
        receipt.parent.mkdir(exist_ok=True)
        receipt.write_text(json.dumps({'plan_identity': doc['identity'], 'frame': index,
            'world_blender_c2w': pose.tolist(), 'shared_rgb': str(root / 'rgb' / f'{index:03d}.png')}, indent=2) + '\n')
        print('SHARED_RENDER ' + json.dumps({'frame': index, 'phase': 'complete'}), flush=True)


if __name__ == '__main__':
    main()
