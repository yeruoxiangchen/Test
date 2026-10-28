"""Same-frame mesh endpoint comparisons, including latent proposals rejected by inputs."""
import argparse
import gc
import os

import numpy as np
import torch

from visual_evidence_probe.experiments.local_repair import config
from .local_repair_latent import OUT, MESH_SOURCE
from .local_repair_review import shaded


def main():
    p = argparse.ArgumentParser(); p.add_argument('--gpu', default='0'); a = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu; config.setup()
    from visual_evidence_probe.runtime import setup
    from visual_evidence_probe.experiments.local_repair.runner import raster, load_gt
    import nvdiffrast.torch as dr
    import trimesh
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    context = dr.RasterizeCudaContext(device=setup())
    for index in config.PARAMS['indices']:
        gt = load_gt(index)
        rec = config.read(config.BASE / 'objects' / f'{index:03d}' / 'observations.json')
        k = torch.tensor(rec['intrinsics'], device='cuda:0', dtype=torch.float32)
        w = torch.tensor(rec['extrinsics'], device='cuda:0', dtype=torch.float32)
        gt_pred = raster(gt.vertices, gt.faces, k, w, context, 256)
        for seed in config.PARAMS['seeds']:
            base = config.BASE / 'objects' / f'{index:03d}' / f'fitted/{seed}.obj'
            automatic = OUT / 'objects' / f'{index:03d}' / str(seed) / 'automatic'
            oracle = OUT / 'objects' / f'{index:03d}' / str(seed) / 'oracle_direction'
            rows = [('GT', None), ('Frozen30K + same global fit', base),
                ('Mesh automatic full proposal', MESH_SOURCE / 'objects' / f'{index:03d}' / str(seed) / 'automatic_proposal.ply'),
                ('SLat automatic accepted', automatic / 'accepted.obj'),
                ('SLat automatic step20 (before acceptance)', automatic / 'step_020.ply'),
                ('SLat GT-direction step20 (diagnostic)', oracle / 'step_020.ply')]
            fig, axes = plt.subplots(len(rows), 3, figsize=(9, 2.5*len(rows)))
            for row, (title, path) in enumerate(rows):
                if path is None:
                    pred = gt_pred
                else:
                    if not path.exists(): path = base
                    mesh = trimesh.load(path, force='mesh', process=False, skip_materials=True)
                    pred = raster(mesh.vertices, mesh.faces, k, w, context, 256)
                for view in range(3):
                    axes[row, view].imshow(shaded(pred, view)); axes[row, view].axis('off')
                    if view == 0: axes[row, view].set_title(title, loc='left', fontsize=9)
            fig.suptitle(f'{rec["uid"]}, seed{seed}; same frame/cameras; oracle row uses GT', fontsize=10)
            fig.tight_layout()
            dest = OUT / 'figures' / f'{index:03d}_{seed}.jpg'; dest.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(dest, dpi=130); plt.close(fig)
        del gt, gt_pred; gc.collect(); torch.cuda.empty_cache()
        print(dict(stage='latent_figures', index=index), flush=True)


if __name__ == '__main__': main()
