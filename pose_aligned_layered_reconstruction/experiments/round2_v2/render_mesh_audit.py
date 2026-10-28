"""Deterministic triangle-surface renders in the untouched native mesh frame.

No per-object fit, normalization, recentering, rotation, smoothing, decimation,
component removal, or texture is applied.  All objects share three orthographic
world cameras, scale, material, lights and background.  Optional explicit 4x4
transforms are supported only by ``render_file`` for separately labelled audits.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import numba
from numba import njit
from PIL import Image
import PIL
import trimesh

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/round2_v2_diagnostics_20261018/mesh"
OUTPUT = ROOT / "outputs/round2_v2_mesh_visual_audit_20261018/visuals"
SIZE = 512
SUPERSAMPLE = 2
HALF_SPAN = 0.9
DISTANCE = 3.0
NEAR, FAR = 1.5, 4.5
AZIMUTHS = (-45.0, 75.0, 195.0)
ELEVATION = 25.0


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cameras() -> list[dict]:
    result = []
    for i, azimuth in enumerate(AZIMUTHS):
        a, e = np.deg2rad([azimuth, ELEVATION])
        back = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        right = np.cross([0.0, 0.0, 1.0], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        eye = DISTANCE * back
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = np.stack([right, up, back])
        world_to_camera[:3, 3] = -world_to_camera[:3, :3] @ eye
        projection = np.eye(4)
        projection[0, 0] = projection[1, 1] = 1.0 / HALF_SPAN
        projection[2, 2] = -2.0 / (FAR - NEAR)
        projection[2, 3] = -(FAR + NEAR) / (FAR - NEAR)
        result.append({
            "view": i, "azimuth_degrees": azimuth, "elevation_degrees": ELEVATION,
            "eye_world": eye.tolist(), "look_at_world": [0.0, 0.0, 0.0],
            "up_world": [0.0, 0.0, 1.0], "world_to_camera": world_to_camera.tolist(),
            "camera_to_clip": projection.tolist(),
            "convention": "column vectors; camera looks along -Z; pixel X right, Y down",
        })
    return result


def parameters() -> dict:
    return {
        "renderer": "native_mesh_numba_triangle_zbuffer_v1",
        "size": [SIZE, SIZE], "supersample": SUPERSAMPLE,
        "orthographic_full_span": 2 * HALF_SPAN, "near": NEAR, "far": FAR,
        "material_srgb": [0.61, 0.68, 0.75], "background_srgb": [0.975, 0.98, 0.985],
        "key_light_world_direction": [1.0, -2.0, 4.0],
        "fill_light_world_direction": [-3.0, -1.0, 1.0],
        "ambient": 0.34, "key_diffuse": 0.57, "fill_diffuse": 0.16,
        "specular_weight": 0.08, "specular_exponent": 32,
        "shading": "area-weighted interpolated vertex normals; two-sided; no shadow map",
        "depth": "orthographic barycentric interpolation with strict nearest-surface z-buffer",
        "antialias": "2x rasterization, Lanczos reduction",
        "mesh_transform": "identity; no per-object normalization or fitting",
        "mesh_processing": "none; normals computed without changing vertices or faces",
        "cameras": cameras(),
        "versions": {"numpy": np.__version__, "numba": numba.__version__,
                     "trimesh": trimesh.__version__, "pillow": PIL.__version__},
        "script_sha256": sha256(Path(__file__)),
    }


@njit(cache=True, nogil=True)
def vertex_normals(vertices, faces):
    normals = np.zeros_like(vertices)
    for f in faces:
        i, j, k = f[0], f[1], f[2]
        ax, ay, az = vertices[j] - vertices[i]
        bx, by, bz = vertices[k] - vertices[i]
        nx, ny, nz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
        for v in (i, j, k):
            normals[v, 0] += nx
            normals[v, 1] += ny
            normals[v, 2] += nz
    for i in range(len(normals)):
        length = math.sqrt(np.dot(normals[i], normals[i]))
        if length > 1e-20:
            normals[i] /= length
    return normals


@njit(cache=True, nogil=True)
def rasterize(projected, normals, faces, width, height):
    depth = np.full((height, width), np.inf, dtype=np.float32)
    normal_buffer = np.zeros((height, width, 3), dtype=np.float32)
    for face in faces:
        i, j, k = face[0], face[1], face[2]
        ax, ay, az = projected[i]
        bx, by, bz = projected[j]
        cx, cy, cz = projected[k]
        denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denominator) < 1e-12:
            continue
        xmin = max(0, int(math.ceil(min(ax, bx, cx) - 0.5)))
        xmax = min(width - 1, int(math.floor(max(ax, bx, cx) - 0.5)))
        ymin = max(0, int(math.ceil(min(ay, by, cy) - 0.5)))
        ymax = min(height - 1, int(math.floor(max(ay, by, cy) - 0.5)))
        for y in range(ymin, ymax + 1):
            py = y + 0.5
            for x in range(xmin, xmax + 1):
                px = x + 0.5
                a = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denominator
                b = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denominator
                c = 1.0 - a - b
                if a < -1e-8 or b < -1e-8 or c < -1e-8:
                    continue
                z = a * az + b * bz + c * cz
                if z < NEAR or z > FAR or z >= depth[y, x]:
                    continue
                depth[y, x] = z
                for d in range(3):
                    normal_buffer[y, x, d] = a * normals[i, d] + b * normals[j, d] + c * normals[k, d]
    return depth, normal_buffer


def render_arrays(vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray,
                  camera: dict) -> tuple[Image.Image, dict]:
    resolution = SIZE * SUPERSAMPLE
    world_to_camera = np.asarray(camera["world_to_camera"])
    camera_vertices = vertices @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    projected = np.empty_like(camera_vertices)
    projected[:, 0] = (camera_vertices[:, 0] / HALF_SPAN + 1) * resolution / 2
    projected[:, 1] = (1 - camera_vertices[:, 1] / HALF_SPAN) * resolution / 2
    projected[:, 2] = -camera_vertices[:, 2]
    depth, n = rasterize(projected, normals, faces, resolution, resolution)
    foreground = np.isfinite(depth)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-20)
    view_direction = world_to_camera[2, :3]
    n *= np.where((n @ view_direction) < 0, -1.0, 1.0)[..., None]
    key = np.array([1.0, -2.0, 4.0]); key /= np.linalg.norm(key)
    fill = np.array([-3.0, -1.0, 1.0]); fill /= np.linalg.norm(fill)
    half_vector = key + view_direction; half_vector /= np.linalg.norm(half_vector)
    diffuse = 0.34 + 0.57 * np.maximum(n @ key, 0) + 0.16 * np.maximum(n @ fill, 0)
    specular = 0.08 * np.maximum(n @ half_vector, 0) ** 32
    rgb = np.clip(diffuse[..., None] * np.array([0.61, 0.68, 0.75]) + specular[..., None], 0, 1)
    rgb[~foreground] = [0.975, 0.98, 0.985]
    image = Image.fromarray(np.uint8(np.rint(rgb * 255)), "RGB")
    image = image.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    outside_xy = ((np.abs(camera_vertices[:, :2]) > HALF_SPAN).any(axis=1))
    outside_depth = ((projected[:, 2] < NEAR) | (projected[:, 2] > FAR))
    border = np.concatenate([foreground[0], foreground[-1], foreground[:, 0], foreground[:, -1]])
    y, x = np.nonzero(foreground)
    stats = {
        "view": camera["view"], "foreground_pixels_supersampled": int(foreground.sum()),
        "foreground_fraction": float(foreground.mean()),
        "foreground_bbox_supersampled_xyxy": [int(x.min()), int(y.min()), int(x.max()), int(y.max())] if len(x) else None,
        "foreground_depth_min_max": [float(depth[foreground].min()), float(depth[foreground].max())] if len(x) else None,
        "vertices_outside_xy": int(outside_xy.sum()), "vertices_outside_depth": int(outside_depth.sum()),
        "triangles_with_any_vertex_outside": int((outside_xy | outside_depth)[faces].any(axis=1).sum()),
        "foreground_touches_image_border": bool(border.any()),
        "projected_vertex_extent_xy": [[float(projected[:, 0].min()), float(projected[:, 1].min())],
                                      [float(projected[:, 0].max()), float(projected[:, 1].max())]],
    }
    return image, stats


def render_file(mesh_path: Path | str, prefix: str, output: Path = OUTPUT,
                matrix: np.ndarray | None = None, expected_sha256: str | None = None,
                identity: dict | None = None) -> dict:
    """Render all three views; any explicit transform is recorded and never fitted here."""
    mesh_path = Path(mesh_path)
    source_hash = sha256(mesh_path)
    if expected_sha256 is not None and source_hash != expected_sha256:
        raise ValueError(f"Source mesh hash mismatch: {mesh_path}")
    params = parameters()
    transformation = np.eye(4) if matrix is None else np.asarray(matrix, dtype=np.float64)
    if transformation.shape != (4, 4) or not np.isfinite(transformation).all():
        raise ValueError("Transform must be a finite explicit 4x4 matrix")
    binding = {"mesh_sha256": source_hash, "parameters": params, "transform": transformation.tolist()}
    binding_hash = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    metadata_path = output / "records" / f"{prefix}.json"
    if metadata_path.exists():
        cached = json.loads(metadata_path.read_text())
        if cached["binding_hash"] != binding_hash:
            raise ValueError(f"Render binding differs: {metadata_path}")
        if all(Path(v["png"]).exists() and sha256(v["png"]) == v["png_sha256"] for v in cached["views"]):
            return cached
    started = time.perf_counter()
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0 or not np.isfinite(vertices).all():
        raise ValueError(f"Invalid or empty triangle mesh: {mesh_path}")
    source_bounds = [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]
    if matrix is not None:
        vertices = np.ascontiguousarray(vertices @ transformation[:3, :3].T + transformation[:3, 3])
    normals = vertex_normals(vertices, faces)
    record = {
        "binding_hash": binding_hash, "prefix": prefix, "identity": identity or {},
        "mesh": {"path": str(mesh_path.resolve()), "sha256": source_hash},
        "vertex_count": len(vertices), "face_count": len(faces), "source_bounds": source_bounds,
        "rendered_bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "matrix_original_to_rendered": transformation.tolist(), "views": [],
    }
    (output / "renders").mkdir(parents=True, exist_ok=True)
    for camera in params["cameras"]:
        image, stats = render_arrays(vertices, faces, normals, camera)
        path = output / "renders" / f"{prefix}_v{camera['view']}.png"
        image.save(path)
        stats.update(png=str(path.resolve()), png_sha256=sha256(path))
        record["views"].append(stats)
    record["elapsed_seconds"] = time.perf_counter() - started
    save_json(metadata_path, record)
    return record


def jobs(preview: bool = False) -> list[dict]:
    result = []
    for path in sorted((SOURCE / "targets").glob("*.json")):
        object_index = int(path.stem)
        if preview and object_index != 0:
            continue
        meta = json.loads(path.read_text())
        result.append({"mesh_path": meta["mesh"]["path"], "prefix": f"gt_{object_index:03d}",
                       "expected_sha256": meta["mesh"]["sha256"],
                       "identity": {"mode": "gt", "object_index": object_index}})
    for path in sorted((SOURCE / "records").glob("*.json")):
        record = json.loads(path.read_text())
        if preview and (record["index"] != 0 or record["seed"] != 42):
            continue
        result.append({"mesh_path": record["mesh"]["path"],
                       "prefix": f"{record['index']:03d}_{record['seed']}_{record['mode']}",
                       "expected_sha256": record["mesh"]["sha256"],
                       "identity": {k: record[k] for k in ("index", "object_index", "object_uid", "domain", "seed", "mode")}})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use between one and four CPU workers")
    task_list = jobs(args.preview)
    assert len(task_list) == (4 if args.preview else 416), len(task_list)
    # Compile outside workers to avoid duplicate JIT work and cache contention.
    triangle = np.array([[0., 0., 0.], [.2, 0., 0.], [0., .2, 0.]])
    f = np.array([[0, 1, 2]], dtype=np.int64)
    vertex_normals(triangle, f)
    rasterize(np.array([[1., 1., 3.], [2., 1., 3.], [1., 2., 3.]]), triangle, f, 4, 4)
    records = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(render_file, **job, output=args.output) for job in task_list]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps({"complete": len(records), "total": len(task_list), "prefix": record["prefix"],
                              "mesh_seconds": record["elapsed_seconds"], "elapsed_seconds": time.perf_counter() - start}), flush=True)
    records.sort(key=lambda r: r["prefix"])
    all_views = [v for record in records for v in record["views"]]
    clipped = [{"prefix": r["prefix"], "view": v["view"], "vertices_outside_xy": v["vertices_outside_xy"],
                "vertices_outside_depth": v["vertices_outside_depth"]}
               for r in records for v in r["views"] if v["vertices_outside_xy"] or v["vertices_outside_depth"]]
    manifest = {"complete": True, "preview_only": args.preview, "mesh_count": len(records),
                "image_count": len(all_views), "parameters": parameters(), "records": records,
                "clipped_mesh_views": clipped,
                "empty_images": [v["png"] for v in all_views if v["foreground_pixels_supersampled"] == 0],
                "border_touching_images": [v["png"] for v in all_views if v["foreground_touches_image_border"]],
                "elapsed_seconds": time.perf_counter() - start}
    save_json(args.output / ("preview_manifest.json" if args.preview else "manifest.json"), manifest)
    print(json.dumps({k: manifest[k] for k in ("mesh_count", "image_count", "clipped_mesh_views", "empty_images", "border_touching_images", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
