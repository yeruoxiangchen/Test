"""Bounded CPU conversion, canonical targets and score-independent selection."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess

from . import common as c


def converted(name): return c.DATASET / 'converted_glb' / name


def convert():
    source = c.ROOT / 'datasets/scene_occlusion_pilot_20261020/render_tools/prepare_hssd_render.py'
    spec = importlib.util.spec_from_file_location('_frozen_glb_reader', source)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    p = c.plan(); records = []; texture_cache = c.DATASET / 'texture_cache'
    texture_cache.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(p['plan']['dependencies']['geometry_files']):
        original = c.RAW / row['path']; dest = converted(row['path'])
        receipt_file = c.DATASET / 'conversion_receipts' / (c.digest(row['path']) + '.json')
        if receipt_file.exists():
            receipt = c.read(receipt_file); c.verify(receipt['converted']); records.append(receipt); continue
        data, binary = helper.read_glb(original); jobs = {}
        meshes_before = c.digest(data.get('meshes', [])); accessors_before = c.digest(data.get('accessors', []))
        for image in data.get('images', []):
            if image.get('mimeType') != 'image/ktx2': continue
            view = data['bufferViews'][image['bufferView']]; offset = view.get('byteOffset', 0)
            payload = binary[offset:offset + view['byteLength']]; key = hashlib.sha256(payload).hexdigest()
            src, dst = texture_cache / (key + '.ktx2'), texture_cache / (key + '.png')
            if not src.exists(): src.write_bytes(payload)
            if not dst.exists(): jobs[key] = {'input': str(src), 'output': str(dst)}
        if jobs:
            job_file = texture_cache / 'current_jobs.json'; c.write(job_file, list(jobs.values()))
            subprocess.run(['node', str(source.with_name('decode_ktx2.cjs')), str(job_file)], check=True, env=c.environment())
        blob = bytearray(binary); count = 0
        for image in data.get('images', []):
            if image.get('mimeType') != 'image/ktx2': continue
            view = data['bufferViews'][image['bufferView']]; offset = view.get('byteOffset', 0)
            key = hashlib.sha256(binary[offset:offset + view['byteLength']]).hexdigest()
            png = (texture_cache / (key + '.png')).read_bytes(); blob.extend(b'\0' * (-len(blob) % 4))
            image.update(bufferView=len(data['bufferViews']), mimeType='image/png')
            data['bufferViews'].append({'buffer': 0, 'byteOffset': len(blob), 'byteLength': len(png)})
            blob.extend(png); count += 1
        for tex in data.get('textures', []):
            extensions = tex.get('extensions', {})
            if 'KHR_texture_basisu' in extensions: tex['source'] = extensions.pop('KHR_texture_basisu')['source']
            if not extensions: tex.pop('extensions', None)
        for key in ('extensionsUsed', 'extensionsRequired'):
            if key in data:
                data[key] = [x for x in data[key] if x != 'KHR_texture_basisu']
                if not data[key]: data.pop(key)
        blob.extend(b'\0' * (-len(blob) % 4)); data['buffers'][0]['byteLength'] = len(blob)
        if bytes(blob[:len(binary)]) != binary or c.digest(data.get('meshes', [])) != meshes_before or c.digest(data.get('accessors', [])) != accessors_before:
            raise ValueError('Texture conversion modified geometry')
        js = json.dumps(data, separators=(',', ':')).encode(); js += b' ' * (-len(js) % 4)
        payload = (struct.pack('<4sII', b'glTF', 2, 28 + len(js) + len(blob)) + struct.pack('<II', len(js), 0x4e4f534a)
                   + js + struct.pack('<II', len(blob), 0x004e4942) + blob)
        dest.parent.mkdir(parents=True, exist_ok=True); temp = dest.with_suffix('.tmp'); temp.write_bytes(payload); temp.replace(dest)
        receipt = {'source': c.binding(original), 'converted': c.binding(dest), 'png_images': count,
                   'geometry_binary_prefix_preserved': True, 'mesh_descriptors_sha256': meshes_before,
                   'accessors_sha256': accessors_before, 'original_render_asset': row['path']}
        c.locked(receipt_file, receipt); records.append(receipt)
        if (index + 1) % 10 == 0: c.event('convert_glb', completed=index + 1, total=len(p['plan']['dependencies']['geometry_files']))
    c.locked(c.DATASET / 'conversion_manifest.json', {'identity': p['identity'], 'assets': records, 'geometry_modified': False})
    c.event('convert_complete', assets=len(records))


def assets():
    import numpy as np
    import trimesh
    from ...data_v2.render_scenes import load_parts
    from ..hssd_pilot_20261020.assets import C, instance_matrix
    p = c.plan(); records = []
    for index, candidate in enumerate(p['plan']['candidates']):
        root = c.DATASET / 'objects' / candidate['uid']; receipt_file = root / 'asset.json'
        if receipt_file.exists():
            row = c.read(receipt_file); c.verify(row['mesh']); records.append(row); continue
        source = converted(candidate['source_asset']); rigid, deformation = instance_matrix(candidate['instance_data'])
        parts = load_parts(source, C @ deformation)
        if not parts: raise ValueError('Empty target asset: ' + candidate['uid'])
        vertices = np.concatenate([m.vertices for m in parts]); lo, hi = vertices.min(0), vertices.max(0)
        if not np.isfinite(vertices).all() or np.max(hi - lo) <= 1e-6: raise ValueError('Invalid target extent')
        scale = 1. / np.max(hi - lo); normalization = np.eye(4)
        normalization[:3, :3] *= scale; normalization[:3, 3] = -(lo + hi) * .5 * scale
        for part in parts: part.apply_transform(normalization)
        mesh = trimesh.util.concatenate(parts); cfw = normalization @ C @ np.linalg.inv(rigid)
        expected = np.concatenate([m.vertices for m in load_parts(source, rigid @ deformation)])
        recovered = trimesh.transform_points(mesh.vertices, np.linalg.inv(cfw))
        error = float(np.max(np.abs(recovered - expected)))
        if error > 1e-7: raise ValueError('Canonical/world mismatch: ' + candidate['uid'])
        root.mkdir(parents=True, exist_ok=True); target = root / 'target_canonical.ply'; mesh.export(target)
        # A translation/size-invariant triangle signature catches exact geometry
        # aliases among candidates beyond declared asset IDs/decomposition.
        tri = np.round(mesh.vertices[mesh.faces] * 1e6).astype(np.int64)
        vertex_order = np.lexsort((tri[:,:,2], tri[:,:,1], tri[:,:,0]), axis=1)
        tri = np.take_along_axis(tri, vertex_order[:,:,None], axis=1); flat = tri.reshape(len(tri), 9)
        order = np.lexsort(tuple(flat[:, k] for k in range(8, -1, -1)))
        geometry_hash = hashlib.sha256(flat[order].tobytes()).hexdigest()
        row = {**candidate, 'source': c.binding(c.RAW / candidate['source_asset']), 'converted': c.binding(source),
               'mesh': c.binding(target), 'deformation': deformation.tolist(), 'normalization': normalization.tolist(),
               'canonical_from_world': cfw.tolist(), 'rigid_world_from_asset': rigid.tolist(),
               'transform_roundtrip_max_abs': error, 'candidate_geometry_hash': geometry_hash,
               'vertex_count': len(mesh.vertices), 'face_count': len(mesh.faces),
               'mirror': bool(np.linalg.det(deformation[:3, :3]) < 0),
               'nonuniform_scale': bool(np.ptp(np.abs(np.diag(deformation)[:3])) > 1e-5)}
        c.locked(receipt_file, row); records.append(row)
        if (index + 1) % 10 == 0: c.event('canonical_assets', completed=index + 1, total=len(p['plan']['candidates']))
    c.locked(c.DATASET / 'candidate_assets.json', {'identity': p['identity'], 'objects': records})
    c.event('canonical_assets_complete', objects=len(records))


def select():
    import numpy as np
    from PIL import Image
    p = c.plan(); rows = c.read(c.DATASET / 'candidate_assets.json')['objects']
    hash_presence = defaultdict(set)
    for row in rows: hash_presence[row['candidate_geometry_hash']].add(row['split'])
    selected = []; exclusions = []; visual_rows = []; selected_families = set()
    for subset, scenes in p['plan']['scene_splits'].items():
        for sid in scenes:
            folder = c.DATASET / 'scenes' / sid; capture = c.read(folder / 'plan.json')
            pool = []
            for row in (r for r in rows if r['scene'] == sid):
                frames = []
                for frame in range(capture['plan']['frame_count']):
                    path = folder / 'masks' / row['uid'] / f'{frame:03d}.png'
                    with Image.open(path) as image: mask = np.asarray(image.resize((518,518), Image.Resampling.NEAREST)) > 127
                    pixels = int(mask.sum())
                    if pixels >= 128:
                        frames.append({'frame': frame, 'visible_pixels_518': pixels,
                                       'mask': c.binding(path), 'rgb': c.binding(folder / 'rgb' / f'{frame:03d}.png')})
                frames.sort(key=lambda r: (-r['visible_pixels_518'], r['frame']))
                chosen = []
                eyes = np.asarray(capture['plan']['world_eye'])
                for item in frames:
                    if any(np.linalg.norm(eyes[item['frame']] - eyes[r['frame']]) < .2 for r in chosen): continue
                    chosen.append(item)
                    if len(chosen) == 4: break
                reason = ('cross_split_exact_candidate_geometry' if len(hash_presence[row['candidate_geometry_hash']]) > 1
                          else 'fewer_than_two_visible_positions' if len(chosen) < 2 else None)
                if reason:
                    exclusions.append({'uid': row['uid'], 'scene': sid, 'reason': reason,
                                       'visible_frames': len(frames), 'distinct_positions': len(chosen)}); continue
                pool.append({**row, 'views': sorted(chosen, key=lambda r: r['frame']),
                             'capture_plan': c.binding(folder / 'plan.json')})
            pool = [r for r in pool if r['asset_family'] not in selected_families]
            pool.sort(key=lambda r: c.rank(r['uid']))
            if len(pool) < 8:
                c.write(c.OUT / 'selection_failure.json', {'time': c.now(), 'scene': sid, 'eligible': len(pool), 'required': 8,
                                                         'exclusions': exclusions, 'targets_not_silently_dropped': True})
                raise ValueError(f'{sid}: only {len(pool)} photograph-qualified targets; need8, stop before latent/model scores')
            selected.extend(pool[:8]); selected_families.update(r['asset_family'] for r in pool[:8]); visual_rows.extend(pool[:8])
    families = [r['asset_family'] for r in selected]
    if len(selected) != 192 or len(set(families)) != 192: raise ValueError('Frozen target denominator/family uniqueness failed')
    result = {'candidate_plan_identity': p['identity'], 'objects': selected,
              'split_counts': dict(Counter(r['split'] for r in selected)), 'excluded_before_latent': exclusions,
              'selection_stage': 'photograph coverage and candidate-geometry duplicates only, before latent or model scores'}
    result = {'identity': c.digest(result), 'plan': result}
    c.locked(c.DATASET / 'targets.json', result)
    c.event('targets_frozen', targets=len(selected), split_counts=result['plan']['split_counts'], identity=result['identity'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['convert', 'assets', 'select'])
    args = parser.parse_args(); {'convert': convert, 'assets': assets, 'select': select}[args.action]()
