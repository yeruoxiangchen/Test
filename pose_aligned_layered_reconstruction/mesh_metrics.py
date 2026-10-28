"""Deterministic native-frame surface metrics; no ICP or per-mesh rescaling."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def sample_surface(mesh, count, seed):
    vertices, faces = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
        raise ValueError('cannot score empty/nonfinite mesh')
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(area) & (area > 1.e-15)
    if not valid.any() or count <= 0:
        raise ValueError('no non-degenerate triangles / invalid sample count')
    rng = np.random.default_rng(int(seed))
    ids = rng.choice(np.flatnonzero(valid), size=int(count), replace=True, p=area[valid] / area[valid].sum())
    u, v = rng.random(int(count)), rng.random(int(count))
    sqrt_u = np.sqrt(u)
    weights = np.stack((1.-sqrt_u, sqrt_u*(1.-v), sqrt_u*v), axis=1)
    points = np.sum(triangles[ids] * weights[:, :, None], axis=1)
    normals = cross[ids]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.e-15)
    return points, normals


def surface_metrics(predicted, target, *, count=20000, seed=42, thresholds=(.01, .02, .05)):
    p, pn = sample_surface(predicted, count, seed)
    g, gn = sample_surface(target, count, seed)
    pd, pi = cKDTree(g).query(p, workers=4)
    gd, gi = cKDTree(p).query(g, workers=4)
    normal_p = np.abs((pn * gn[pi]).sum(1)).mean()
    normal_g = np.abs((gn * pn[gi]).sum(1)).mean()
    result = dict(pred_to_gt_mean=float(pd.mean()), gt_to_pred_mean=float(gd.mean()),
        chamfer_l1=float(.5*(pd.mean()+gd.mean())), chamfer_l2=float(.5*((pd**2).mean()+(gd**2).mean())),
        normal_consistency=float(.5 * (normal_p + normal_g)))
    for threshold in thresholds:
        key = str(float(threshold)).replace('.', 'p')
        precision, recall = float((pd < threshold).mean()), float((gd < threshold).mean())
        result.update({f'precision_{key}': precision, f'recall_{key}': recall,
                       f'fscore_{key}': 2*precision*recall/(precision+recall) if precision+recall else 0.})
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('nonfinite mesh metric')
    return result


def structure_metrics(mesh):
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces, np.int64)
    if not len(v) or not len(f) or not np.isfinite(v).all():
        raise ValueError('invalid mesh')
    edges = np.sort(np.concatenate((f[:, :2], f[:, 1:], f[:, [2, 0]])), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    graph = coo_matrix((np.ones(len(edges), np.uint8), (edges[:, 0], edges[:, 1])), shape=(len(v), len(v)))
    _, labels = connected_components(graph, directed=False)
    _, sizes = np.unique(labels[np.unique(f)], return_counts=True)
    return dict(mesh_success=True, vertex_count=len(v), face_count=len(f),
        component_count=len(sizes), largest_component_ratio=float(sizes.max()/sizes.sum()),
        boundary_edge_count=int((counts == 1).sum()), nonmanifold_edge_count=int((counts > 2).sum()),
        bbox_extent=np.ptp(v, axis=0).tolist())


def aggregate_mesh_records(records, expected_uids, seeds=(42, 43)):
    """Refuse incomplete/mixed/duplicate pairs; bootstrap independent objects."""
    expected = {(uid, seed, mode) for uid in expected_uids for seed in seeds for mode in ('parent', 'candidate')}
    by_key = {(r['uid'], r['seed'], r['mode']): r for r in records}
    if len(by_key) != len(records) or set(by_key) != expected:
        raise ValueError('mesh evaluation is incomplete, duplicated, or contains unexpected cases')
    if not all(r['success'] for r in records):
        raise ValueError('mesh failures remain: no success-only aggregate is allowed')
    keys = tuple(records[0]['surface'])
    if not keys or any(set(r['surface']) != set(keys) or not all(np.isfinite(v) for v in r['surface'].values()) for r in records):
        raise ValueError('mesh metric schema differs or contains nonfinite values')
    summary = {mode: {key: float(np.mean([r['surface'][key] for r in records if r['mode'] == mode]))
                      for key in keys} for mode in ('parent', 'candidate')}
    paired = {}
    for key in keys:
        improvements = []
        by_seed = {str(seed): [] for seed in seeds}
        sign = -1. if key in ('pred_to_gt_mean', 'gt_to_pred_mean', 'chamfer_l1', 'chamfer_l2') else 1.
        for uid in expected_uids:
            values = [sign*(by_key[uid, seed, 'candidate']['surface'][key] - by_key[uid, seed, 'parent']['surface'][key]) for seed in seeds]
            improvements.append(float(np.mean(values)))
            for seed, value in zip(seeds, values):
                by_seed[str(seed)].append(value)
        gains = np.asarray(improvements)
        rng = np.random.default_rng(20260910)
        bootstrap = gains[rng.integers(len(gains), size=(5000, len(gains)))].mean(1)
        paired[key] = dict(mean_improvement=float(gains.mean()), positive_means_better=True,
            win_objects=int((gains > 1.e-12).sum()), loss_objects=int((gains < -1.e-12).sum()),
            tie_objects=int((np.abs(gains) <= 1.e-12).sum()), median_improvement=float(np.median(gains)),
            object_bootstrap_ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
            by_seed={seed: float(np.mean(values)) for seed, values in by_seed.items()})
    return dict(object_count=len(expected_uids), record_count=len(records), sample_pairs=len(expected)//2,
        summary=summary, paired=paired, all_meshes_successful=True)
