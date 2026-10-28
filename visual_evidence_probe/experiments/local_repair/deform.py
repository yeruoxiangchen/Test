"""Area-matched region controls and a bounded, fixed-topology local solver."""
import numpy as np
from scipy.sparse import coo_matrix, vstack, eye
from scipy.sparse.linalg import lsqr

from .config import PARAMS


def area_region(mesh, face_ids, maps, method, seed):
    n = len(mesh.faces)
    visible = np.zeros(n, bool)
    score = np.full(n, -np.inf)
    for f, s in zip(face_ids, maps):
        valid = (f >= 0) & np.isfinite(s)
        np.maximum.at(score, f[valid], s[valid])
        visible[np.unique(f[valid])] = True
    ids = np.flatnonzero(visible)
    if method == 'random':
        rng = np.random.default_rng(seed)
        rank = ids[rng.permutation(len(ids))]
    else:
        rank = ids[np.argsort(-score[ids], kind='stable')]
    area = np.asarray(mesh.area_faces)
    budget = PARAMS['region_fraction'] * area[ids].sum()
    # Include the last crossing triangle; record actual area rather than claiming exact equality.
    end = min(len(rank), np.searchsorted(np.cumsum(area[rank]), budget) + 1) if len(rank) else 0
    selected = np.zeros(n, bool)
    selected[rank[:end]] = True
    return selected, dict(visible_faces=int(len(ids)), selected_faces=int(selected.sum()),
        eligible_area=float(area[ids].sum()), selected_area=float(area[selected].sum()), budget_area=float(budget))


def constraints(mesh, pred, proposals, selected, k, w, seed):
    import trimesh
    face = pred['face_id'][:3].cpu().numpy()
    ok = proposals['valid'] & (face >= 0)
    ok &= selected[np.maximum(face, 0)]
    ids = np.argwhere(ok)
    if len(ids) > PARAMS['max_constraints']:
        rng = np.random.default_rng(seed)
        ids = ids[np.sort(rng.choice(len(ids), PARAMS['max_constraints'], replace=False))]
    if not len(ids):
        return dict(face_ids=np.zeros(0, np.int64), bary=np.zeros((0, 3)),
            target=np.zeros((0, 3)), displacement=np.zeros((0, 3)), confidence=np.zeros(0), pixel=ids)
    vi, yi, xi = ids.T
    face_ids = face[vi, yi, xi]
    base = pred['xyz'].cpu().numpy()[vi, yi, xi]
    bary = trimesh.triangles.points_to_barycentric(np.asarray(mesh.triangles)[face_ids], base)
    # Raster interpolation/unprojection is accurate up to FP32; constrain actual surface barycentrics.
    bary = np.clip(bary, 0, 1)
    bary /= bary.sum(1, keepdims=True)
    actual = (np.asarray(mesh.vertices)[np.asarray(mesh.faces)[face_ids]] * bary[..., None]).sum(1)
    side = pred['depth'].shape[-1]
    uv = np.stack(((xi + .5) * 518 / side - .5, (yi + .5) * 518 / side - .5, np.ones(len(ids))), -1)
    rays = np.stack([uv[q] @ np.linalg.inv(k[int(v)]).T @ np.linalg.inv(w[int(v)])[:3, :3].T
        for q, v in enumerate(vi)])
    disp = rays * proposals['delta'][vi, yi, xi, None]
    length = np.linalg.norm(disp, axis=1)
    disp *= np.minimum(1., PARAMS['max_displacement'] / np.maximum(length, 1.e-12))[:, None]
    return dict(face_ids=face_ids, bary=bary, target=actual + disp, displacement=disp,
        confidence=proposals['confidence'][vi, yi, xi], pixel=ids)


def solve(mesh, c, selected):
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    region = np.zeros(len(vertices), bool)
    region[np.unique(faces[selected])] = True
    active = np.flatnonzero(region)
    result = np.zeros_like(vertices)
    if not len(c['face_ids']) or not len(active):
        return result, dict(active_vertices=len(active), constraints=0, finite=True, residual_norm=0.)
    lookup = np.full(len(vertices), -1)
    lookup[active] = np.arange(len(active))
    weights = np.sqrt(np.maximum(c['confidence'], 1.e-4))
    weights /= max(np.median(weights), 1.e-5)
    columns = lookup[faces[c['face_ids']]].reshape(-1)
    if (columns < 0).any():
        raise ValueError('constraint outside selected region')
    rows = np.repeat(np.arange(len(columns) // 3), 3)
    matrix = coo_matrix(((c['bary'] * weights[:, None]).reshape(-1), (rows, columns)),
        shape=(len(c['face_ids']), len(active))).tocsr()
    edges = np.unique(np.sort(np.concatenate((faces[:, :2], faces[:, 1:], faces[:, [2, 0]])), axis=1), axis=0)
    edges = edges[region[edges].any(1)]
    er, ec, ev = [], [], []
    for sign, endpoint in ((1., 0), (-1., 1)):
        cols = lookup[edges[:, endpoint]]
        valid = cols >= 0
        er.extend(np.flatnonzero(valid)); ec.extend(cols[valid]); ev.extend(np.full(valid.sum(), sign))
    smooth = coo_matrix((ev, (er, ec)), shape=(len(edges), len(active))).tocsr()
    # Normalize blocks by row count, so controls with different pixel density use the same weights.
    a = vstack((matrix / np.sqrt(len(c['face_ids'])),
        np.sqrt(PARAMS['smooth_weight'] / max(len(edges), 1)) * smooth,
        np.sqrt(PARAMS['ridge_weight'] / len(active)) * eye(len(active))), format='csr')
    b = np.vstack((c['displacement'] * weights[:, None] / np.sqrt(len(c['face_ids'])),
        np.zeros((len(edges) + len(active), 3))))
    solved = [lsqr(a, b[:, dim], atol=1.e-6, btol=1.e-6, iter_lim=200) for dim in range(3)]
    offset = np.stack([s[0] for s in solved], -1)
    length = np.linalg.norm(offset, axis=1)
    offset *= np.minimum(1., PARAMS['max_displacement'] / np.maximum(length, 1.e-12))[:, None]
    result[active] = offset
    if not np.isfinite(result).all() or np.any(result[~region] != 0):
        raise ValueError('nonfinite deformation or changed frozen vertex')
    return result, dict(active_vertices=len(active), constraints=len(c['face_ids']), finite=True,
        residual_norm=float(np.linalg.norm(a @ offset - b)), lsqr_istop=[int(s[1]) for s in solved],
        max_vertex_displacement=float(np.linalg.norm(result, axis=1).max()),
        frozen_vertices_exact=True)
