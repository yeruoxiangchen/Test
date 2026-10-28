import numpy as np
import torch
from scipy.spatial import cKDTree

from .config import PARAMS
from visual_evidence_probe.experiments.editor_support.objective import raster_tensor
from pose_aligned_layered_reconstruction.mesh_metrics import sample_surface, structure_metrics


@torch.no_grad()
def render_mesh(mesh, k, w, context):
    v = torch.as_tensor(np.array(mesh.vertices, dtype=np.float32), device=k.device)
    f = torch.as_tensor(np.array(mesh.faces, dtype=np.int32), device=k.device)
    return raster_tensor(v, f, k, w, context, PARAMS['side'])


def face_region(pred, selected_faces):
    import torch.nn.functional as F
    region = (selected_faces[pred['face_id'].clamp_min(0)] & pred['mask']).to(pred['depth'].dtype)
    return F.max_pool2d(region[:, None], 5, stride=1, padding=2)[:, 0]


def fixed_depth_error(pred, target):
    """Fixed target/background cases, including disappearing original rays."""
    error = (pred['depth'] - target['depth']).abs().clamp_max(PARAMS['missing_depth'])
    error = torch.where(pred['mask'], error, torch.full_like(error, PARAMS['missing_depth']))
    background = torch.where(pred['mask'], torch.full_like(error, PARAMS['missing_depth']), torch.zeros_like(error))
    return torch.where(target['mask'], error, background)


@torch.no_grad()
def visible_metrics(pred, target, original, region):
    error = fixed_depth_error(pred, target)
    local = region > .5
    target_domain = target['mask']
    union_domain = original['mask'] | target_domain
    common = original['mask'] & target_domain
    def mean(domain): return float(error[domain].mean()) if bool(domain.any()) else None
    common_now = original['mask'] & pred['mask']
    outside = common_now & ~local
    drift = (pred['depth'] - original['depth']).abs()
    union = pred['mask'] | original['mask']
    mismatch = (pred['mask'] ^ original['mask']).sum((1, 2)) / union.sum((1, 2)).clamp_min(1)
    original_error = fixed_depth_error(original, target)
    bad = local & target_domain & (original_error >= 1 / 256)
    return dict(local_target_depth_mae=mean(local & target_domain),
        local_union_depth_mae=mean(local & union_domain),
        global_target_depth_mae=mean(target_domain),
        global_union_depth_mae=mean(union_domain),
        common_depth_mae=mean(common), local_bad_depth_mae=mean(bad),
        local_target_pixels=int((local & target_domain).sum()),
        local_union_pixels=int((local & union_domain).sum()), local_bad_pixels=int(bad.sum()),
        target_pixels=int(target_domain.sum()),
        outside_depth_mae=float(drift[outside].mean()) if bool(outside.any()) else None,
        outside_depth_p95=float(torch.quantile(drift[outside], .95)) if bool(outside.any()) else None,
        lost_original_outside_fraction=float((~pred['mask'])[original['mask'] & ~local].float().mean())
            if bool((original['mask'] & ~local).any()) else None,
        mask_mismatch_by_view=mismatch.cpu().tolist(),
        local_depth_by_view=[float(error[i][local[i] & target_domain[i]].mean())
            if bool((local[i] & target_domain[i]).any()) else None for i in range(len(error))])


class Evaluator:
    def __init__(self, gt, original_mesh, input_original, input_gt, input_region,
                 novel_original, novel_gt, novel_region, k, w, nk, nw, context):
        self.gt_points, self.gt_normals = sample_surface(gt, PARAMS['surface_samples'], PARAMS['random_seed'])
        self.gt_tree = cKDTree(self.gt_points)
        self.original = original_mesh
        self.input_original, self.input_gt, self.input_region = input_original, input_gt, input_region
        self.novel_original, self.novel_gt, self.novel_region = novel_original, novel_gt, novel_region
        self.k, self.w, self.nk, self.nw, self.context = k, w, nk, nw, context
        faces = np.asarray(original_mesh.faces)
        areas = np.asarray(original_mesh.area_faces)
        self.vertex_area = np.bincount(faces.reshape(-1), np.repeat(areas / 3, 3), minlength=len(original_mesh.vertices))

    @torch.no_grad()
    def __call__(self, mesh, *, original=False, structure=None):
        p, pn = sample_surface(mesh, PARAMS['surface_samples'], PARAMS['random_seed'])
        tree = cKDTree(p)
        pd, pi = self.gt_tree.query(p, workers=2)
        gd, gi = tree.query(self.gt_points, workers=2)
        surface = dict(chamfer_l1=float(.5 * (pd.mean() + gd.mean())),
            normal_consistency=float(.5 * (np.abs((pn * self.gt_normals[pi]).sum(1)).mean() +
                np.abs((self.gt_normals * pn[gi]).sum(1)).mean())))
        for t in (.01, .02, .05):
            precision, recall = float((pd < t).mean()), float((gd < t).mean())
            surface[f'fscore_{str(t).replace(".", "p")}'] = 2 * precision * recall / max(precision + recall, 1.e-12)
        if original:
            ir, nr = self.input_original, self.novel_original
            distance = np.zeros(len(self.original.vertices))
        else:
            ir = render_mesh(mesh, self.k, self.w, self.context)
            nr = render_mesh(mesh, self.nk, self.nw, self.context)
            distance = cKDTree(np.asarray(mesh.vertices)).query(np.asarray(self.original.vertices), workers=2)[0]
        weight = self.vertex_area
        moved = dict(original_vertex_to_nearest_output_vertex_mean=float(np.average(distance, weights=weight)),
            original_vertex_to_nearest_output_vertex_p95=float(np.quantile(distance, .95)),
            approximate_area_changed_gt_1voxel=float(weight[distance > 1 / 256].sum() / max(weight.sum(), 1.e-12)),
            approximate_area_changed_gt_1quarter_voxel=float(weight[distance > .25 / 256].sum() / max(weight.sum(), 1.e-12)),
            measure='original vertex area weighting of nearest output VERTEX distance; not exact surface distance')
        return dict(surface=surface, input=visible_metrics(ir, self.input_gt, self.input_original, self.input_region),
            novel=visible_metrics(nr, self.novel_gt, self.novel_original, self.novel_region),
            structure=structure if structure is not None else structure_metrics(mesh), displacement=moved)
