"""OpenCV cameras in the original Habitat Y-up world; no fitted GT alignment."""
import numpy as np

C = np.array([[1., 0, 0, 0], [0, 0, -1., 0], [0, 1., 0, 0], [0, 0, 0, 1.]])


def look_at(eye, target):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0., 1., 0.])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, y, z], axis=1)
    c2w[:3, 3] = eye
    return np.linalg.inv(c2w)


def trajectory():
    # One continuous room-level walk, fixed before rendering or model outputs.
    anchors = np.array([[.75, 1.55, -3.7], [1.35, 1.55, -4.1],
                        [3.55, 1.55, -4.1], [4.45, 1.55, -3.15]])
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(anchors, axis=0), axis=1))]
    eyes = np.stack([np.interp(np.linspace(0, distance[-1], 12), distance, anchors[:, j])
                     for j in range(3)], axis=1)
    target = [2.7, .55, -1.5]
    k = np.array([[.5 * 518 / np.tan(np.deg2rad(75 / 2)), 0, 258.5],
                  [0, .5 * 518 / np.tan(np.deg2rad(75 / 2)), 258.5], [0, 0, 1.]])
    return eyes, np.repeat(k[None], len(eyes), axis=0), np.stack([look_at(e, target) for e in eyes])


def canonical_extrinsics(world_w2c, canonical_from_world):
    # Remove uniform scale from the camera rotation, while scaling translation
    # into canonical units. Leaving a scaled rotation changes projected depths.
    a = np.asarray(canonical_from_world, dtype=float)
    scale = np.linalg.norm(a[:3, 0])
    if scale <= 0 or not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3) * scale**2, atol=1.e-7):
        raise ValueError('Expected a similarity object frame')
    result = np.asarray(world_w2c) @ np.linalg.inv(a)
    result[:, :3, :] *= scale
    return result


def project(points, k, w):
    points = np.asarray(points)
    camera = np.c_[points, np.ones(len(points))] @ w.T
    uvw = camera[:, :3] @ k.T
    return uvw[:, :2] / uvw[:, 2:], camera[:, 2]


def contract_checks(rows):
    eyes, k, world = trajectory()
    result = []
    for row in rows:
        a = np.asarray(row['canonical_from_world'])
        local = canonical_extrinsics(world, a)
        points = np.array([[0, 0, 0], [.2, -.1, .1], [-.1, .15, -.2]])
        global_points = (np.c_[points, np.ones(len(points))] @ np.linalg.inv(a).T)[:, :3]
        error = max(float(np.abs(project(points, kk, ww)[0] - project(global_points, kk, gw)[0]).max())
                    for kk, ww, gw in zip(k, local, world))
        rotations = local[:, :3, :3]
        if error > 1.e-8 or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1.e-7):
            raise ValueError('World/canonical camera projection mismatch')
        result.append({'uid': row['uid'], 'max_projection_error_pixels': error,
                       'det_rotation': np.linalg.det(rotations).tolist()})
    return result
