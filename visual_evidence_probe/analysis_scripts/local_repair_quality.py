"""Post-hoc geometry quality checks prompted by visible roughness in proposals."""
import gc
from pathlib import Path
import subprocess
import time

import numpy as np

from visual_evidence_probe.experiments.local_repair import config


def orientation(base, displacement):
    faces = np.asarray(base.faces)
    tri = np.asarray(base.vertices)[faces]
    original = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    moved = tri + displacement[faces]
    changed = np.cross(moved[:, 1] - moved[:, 0], moved[:, 2] - moved[:, 0])
    relevant = np.linalg.norm(displacement[faces], axis=2).max(axis=1) > 1.e-8
    reversed_faces = (original * changed).sum(1) < 0
    return dict(affected_faces=int(relevant.sum()), orientation_reversal_faces=int(reversed_faces.sum()),
        orientation_reversal_fraction_affected=float(reversed_faces[relevant].mean()) if relevant.any() else 0.,
        affected_surface_area_fraction=float(np.linalg.norm(original[relevant], axis=1).sum() / max(np.linalg.norm(original, axis=1).sum(), 1.e-12)),
        changed_vertex_threshold=1.e-8,
        near_degenerate_faces=int((np.linalg.norm(changed, axis=1) < 1.e-14).sum()))


def main():
    import trimesh
    from pose_aligned_layered_reconstruction.mesh_metrics import structure_metrics
    root = config.STORE / 'outputs/local_slat_repair_20261026'
    mesh_root = config.STORE / 'outputs/local_geometry_repair_cropped_20261026'
    expected = [root / 'objects' / f'{i:03d}' / str(s) / 'automatic/result.json'
        for i in config.PARAMS['indices'] for s in config.PARAMS['seeds']]
    # Check all automatic latent endpoints; other arms remain covered by CD/depth endpoints.
    while not all(p.exists() for p in expected):
        status = subprocess.run(['systemctl', '--user', 'is-active', 'visual-evidence-local-repair-latent-batch-20261026.service'],
            text=True, capture_output=True)
        if status.stdout.strip() != 'active': raise RuntimeError('batch incomplete')
        time.sleep(10)
    rows = []
    for index in config.PARAMS['indices']:
        for seed in config.PARAMS['seeds']:
            baseline_path = config.BASE / 'objects' / f'{index:03d}' / f'fitted/{seed}.obj'
            base = trimesh.load(baseline_path, process=False, force='mesh', skip_materials=True)
            original_structure = structure_metrics(base)
            r = config.read(root / 'objects' / f'{index:03d}' / str(seed) / 'automatic/result.json')
            item = dict(index=index, seed=seed, baseline_structure=original_structure, mesh_deformation={})
            for method in config.METHODS:
                source = mesh_root / 'objects' / f'{index:03d}' / str(seed)
                with np.load(source / f'{method}_constraints.npz') as a:
                    displacement = a['vertex_displacement']
                amplitude = config.read(source / f'{method}.json')['selected_amplitude']
                item['mesh_deformation'][method] = dict(proposed=orientation(base, displacement),
                    accepted=orientation(base, displacement * amplitude))
            for endpoint, filename in [('proposed', 'step_020.ply'), ('accepted', 'accepted.obj')]:
                p = root / 'objects' / f'{index:03d}' / str(seed) / 'automatic' / filename
                if r['status'] == 'no_input_constraints' or (endpoint == 'accepted' and not r['edit']):
                    item[f'latent_{endpoint}_structure'] = original_structure
                else:
                    edited = trimesh.load(p, process=False, force='mesh', skip_materials=True)
                    item[f'latent_{endpoint}_structure'] = structure_metrics(edited)
                    del edited
            rows.append(item)
            config.write(root / 'quality' / f'{index:03d}_{seed}.json', item)
            print(dict(stage='geometry_quality', index=index, seed=seed), flush=True)
            del base; gc.collect()
    summary = dict(time=config.now(), records=rows, posthoc=True,
        trigger='Visible surface roughness in mesh proposal figures; do not infer quality from CD alone.',
        automatic_latent_proposals_with_more_components=sum(r['latent_proposed_structure']['component_count'] > r['baseline_structure']['component_count'] for r in rows),
        automatic_latent_accepted_with_more_components=sum(r['latent_accepted_structure']['component_count'] > r['baseline_structure']['component_count'] for r in rows),
        mesh_automatic_proposals_with_orientation_reversal=sum(r['mesh_deformation']['automatic']['proposed']['orientation_reversal_faces'] > 0 for r in rows),
        mean_automatic_affected_area_fraction=float(np.mean([r['mesh_deformation']['automatic']['proposed']['affected_surface_area_fraction'] for r in rows])),
        mean_random_affected_area_fraction=float(np.mean([r['mesh_deformation']['random']['proposed']['affected_surface_area_fraction'] for r in rows])),
        limitations='Face normal reversal is a local orientation diagnostic, not a complete self-intersection/manifold certificate. Component count may include pre-existing small fragments.')
    config.write(root / 'quality/summary.json', summary)
    marker = '## 第四轮几何质量补充检查'
    addition = '\n\n' + marker + '\n\n' + f'时间：{config.now()}\n\n' + \
        f'自动latent完整候选有 {summary["automatic_latent_proposals_with_more_components"]}/16 例连通分量数增加；接受后有 {summary["automatic_latent_accepted_with_more_components"]}/16 例增加。\n\n' + \
        f'自动mesh形变完整候选有 {summary["mesh_automatic_proposals_with_orientation_reversal"]}/16 例出现局部面朝向反转。这里只是翻面诊断，不是完整的自交检测。\n\n' + \
        f'尽管选区预算同为可见面面积的20%，实际受到位移影响的全表面面积，自动平均 {summary["mean_automatic_affected_area_fraction"]:.2%}、随机平均 {summary["mean_random_affected_area_fraction"]:.2%}；不能将两者解释为严格相同实际修改面积的对照。\n\n' + \
        '这是查看候选渲染后补充的检查，未据此重新选择结果。逐例结构、非流形边和翻面统计见 outputs/local_slat_repair_20261026/quality/summary.json。\n'
    # Wait for append-only final conclusions, then add the quality supplement without races.
    while not (root / 'completion.json').exists(): time.sleep(10)
    for path in (config.STORE / '第四次验证结论.md', config.STORE / '审计.md'):
        if marker not in path.read_text():
            with path.open('a') as stream: stream.write(addition)


if __name__ == '__main__': main()
