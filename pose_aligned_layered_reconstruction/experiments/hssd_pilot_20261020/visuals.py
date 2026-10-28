"""Independent camera/silhouette check and native-frame raw/decoded gallery."""
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw
import torch

from ...data_v2.render_scenes import pack_scene, rasterize
from .common import plan, object_dir, verify, write_json, binding, event


def shade(mesh):
    import trimesh
    result = mesh.copy()
    normals = np.asarray(result.vertex_normals)
    light = np.array([.4, -.3, .8])
    light /= np.linalg.norm(light)
    value = (.35 + .65 * np.maximum(normals @ light, 0))[:, None]
    colors = np.c_[np.repeat(value * 210, 3, 1), np.full(len(value), 255)].astype(np.uint8)
    result.visual = trimesh.visual.ColorVisuals(mesh=result, vertex_colors=colors)
    return result


@torch.no_grad()
def build(row):
    import trimesh
    root = object_dir(row)
    device = torch.device('cuda:0')
    source = trimesh.load(verify(row['mesh']), process=False, force='mesh')
    decoded = trimesh.load(root / 'decoded_target.ply', process=False, force='mesh')
    transforms = json.loads((root / 'target_renders/transforms.json').read_text())
    ids = [0, 37, 75, 112]
    k, w = [], []
    for index in ids:
        frame = transforms['frames'][index]
        focal = 256 / np.tan(frame['camera_angle_x'] * .5)
        k.append([[focal, 0, 255.5], [0, focal, 255.5], [0, 0, 1]])
        camera = np.array(frame['transform_matrix'])
        camera[:3, 1:3] *= -1
        w.append(np.linalg.inv(camera))
    kt, wt = torch.tensor(k, dtype=torch.float32, device=device), torch.tensor(np.array(w), dtype=torch.float32, device=device)
    panels = []
    masks = []
    for mesh in [source, decoded]:
        packed = pack_scene([shade(mesh)], [], device)
        result = rasterize(packed, kt, wt, image_size=(512, 512), raster_side=512, layers=1, rgb=True,
                           background_rgb=(1., 1., 1.))
        panels.append([Image.fromarray((rgb.numpy()*255).round().astype(np.uint8)) for rgb in result['rgb']])
        masks.append(result['visible_target_mask'].numpy())
        del packed, result
    canvas = Image.new('RGB', (3*384, 4*384+32), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), 'Target RGB / original mesh / encoded-decoded mesh (same cameras, no alignment)', fill='black')
    ious = []
    for j, index in enumerate(ids):
        path = root / 'target_renders' / f'{index:03d}.png'
        rgba = Image.open(path).convert('RGBA')
        alpha = np.asarray(rgba)[:, :, 3] > 127
        ious.append(float((alpha & masks[0][j]).sum() / max(1, (alpha | masks[0][j]).sum())))
        rgb = Image.new('RGB', rgba.size, 'white')
        rgb.paste(rgba, mask=rgba.getchannel('A'))
        for col, im in enumerate([rgb, panels[0][j], panels[1][j]]):
            canvas.paste(im.resize((384, 384)), (col*384, j*384+32))
    canvas.save(root / 'roundtrip_gallery.jpg', quality=94)
    report = {'uid': row['uid'], 'source_silhouette_iou': ious, 'min_source_silhouette_iou': min(ious),
              'views': ids, 'gallery': binding(root / 'roundtrip_gallery.jpg'),
              'comparison': 'Blender RGBA alpha vs independent nvdiffrast original-mesh projection',
              'raw_and_decoded_share_cameras_and_frame': True, 'fitted_transform': None}
    write_json(root / 'visual_audit.json', report)
    event(stage='visual_audit', uid=row['uid'], min_projection_iou=min(ious))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=int, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    build(plan()['plan']['objects'][args.index])
