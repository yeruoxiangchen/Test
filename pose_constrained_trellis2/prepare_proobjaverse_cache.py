"""Build the independent TRELLIS.2 cache from the frozen ProObjaverse 30K set.

The source archive contains posed RGBA renders and TRELLIS-v1 8-channel
latents.  This builder consumes only the renders.  ``finalize`` requires
separately generated native TRELLIS.2 SS and Shape-512 targets and rejects the
old latent directory explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from typing import Any

import numpy as np
from PIL import Image
import torch

from .constants import (
    CACHE_FORMAT,
    DINO_FEATURE_DIM,
    DINO_FULL_TOKENS,
    DINO_IMAGE_SIZE,
    DINO_MODEL,
    SAMPLE_FORMAT,
)
from .data import (
    PoseTrellis2Dataset,
    canonical_json_sha256,
    load_pose_payload,
    load_shape_target,
    load_ss_target,
    sha256_file,
)
from .prepare_pose_cache import PREPROCESS_FORMAT, prepare_rgba_view
from .runtime import load_dinov3


SELECTION_FORMAT = "reconviagen.proobjaverse_paired_combined.v1"
PROTOCOL_FORMAT = "pose_point_depth_mv.proobjaverse_official_slat_protocol.v1"
OBJECT_RECORD_FORMAT = "pose_constrained_trellis2.proobjaverse_pose_object.v1"
SOURCE_ERROR_FORMAT = "pose_constrained_trellis2.proobjaverse_source_error.v1"
WORKER_REPORT_FORMAT = "pose_constrained_trellis2.proobjaverse_worker.v1"

DEFAULT_SELECTION = (
    "/data/zjr/ProObjaverse-300K-ReconViaGen-30K-state/combined_audit/"
    "combined_selection_30k.json"
)
DEFAULT_PROTOCOL = (
    "/data/zjr/proobjaverse_official_slat_protocol30k_seed20260813_f39_frozen_v1/"
    "protocol.json"
)
SPLIT_NAMES = ("decoder_audit", "predicted_support_bridge", "dev", "train")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _canonical_unicode_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_selection_row(row: Any, index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"selection row {index} is not an object")
    uid = str(row.get("uid", ""))
    shard = str(row.get("shard", ""))
    render = row.get("render")
    old_slat = row.get("slat")
    if not uid or not shard.startswith("shard-"):
        raise ValueError(f"selection row {index} has invalid UID/shard")
    if not isinstance(render, dict) or not isinstance(old_slat, dict):
        raise ValueError(f"uid={uid} lacks its frozen render/SLat identities")
    expected_render = f"renders_random_env/{shard}/{uid}.tar"
    expected_slat = f"lh-slats/{shard}/{uid}.npz"
    if render.get("path") != expected_render or old_slat.get("path") != expected_slat:
        raise ValueError(f"uid={uid} selection path binding differs")
    if int(render.get("size", 0)) <= 0 or int(old_slat.get("size", 0)) <= 0:
        raise ValueError(f"uid={uid} has a non-positive frozen payload size")
    result = dict(row)
    result["selection_rank"] = int(index)
    return result


def _partition_rows(
    rows: list[dict[str, Any]], counts: dict[str, Any], seed: int
) -> dict[str, list[dict[str, Any]]]:
    expected = {name: int(counts[name]) for name in SPLIT_NAMES}
    if sum(expected.values()) != len(rows):
        raise ValueError("protocol split counts do not cover the selection exactly")
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{int(seed)}:{row['uid']}".encode("ascii")
        ).hexdigest(),
    )
    result: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for name in SPLIT_NAMES:
        result[name] = ordered[cursor : cursor + expected[name]]
        cursor += expected[name]
    return result


def load_frozen_rows(
    selection_path: str | Path,
    *,
    protocol_path: str | Path | None,
    data_root: str | Path | None,
    split: str,
    max_samples: int,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    """Validate frozen identities and return one deterministic protocol split."""

    selection_file = Path(selection_path).expanduser().resolve()
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    if selection.get("format") != SELECTION_FORMAT:
        raise ValueError(f"unexpected selection format={selection.get('format')!r}")
    body = {
        key: value
        for key, value in selection.items()
        if key != "combined_selection_sha256"
    }
    selection_identity = _canonical_unicode_sha256(body)
    if selection.get("combined_selection_sha256") != selection_identity:
        raise ValueError("combined 30K selection canonical identity is invalid")
    source_rows = selection.get("selected")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("frozen selection is empty")
    rows = [_validate_selection_row(row, index) for index, row in enumerate(source_rows)]
    uids = [str(row["uid"]) for row in rows]
    if len(rows) != int(selection.get("pair_count", -1)) or len(set(uids)) != len(uids):
        raise ValueError("frozen selection count/UID identity is invalid")
    target_contract = selection.get("target_contract")
    if (
        not isinstance(target_contract, dict)
        or target_contract.get("trellis2_shape_slat_compatible") is not False
    ):
        raise ValueError("selection lacks the explicit TRELLIS-v1 incompatibility guard")
    root = Path(data_root or selection["data_root"]).expanduser().resolve()
    protocol_identity = ""
    if protocol_path:
        protocol_file = Path(protocol_path).expanduser().resolve()
        protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
        if protocol.get("format") != PROTOCOL_FORMAT:
            raise ValueError(f"unexpected protocol format={protocol.get('format')!r}")
        protocol_body = {
            key: value for key, value in protocol.items() if key != "protocol_sha256"
        }
        protocol_identity = canonical_json_sha256(protocol_body)
        if protocol.get("protocol_sha256") != protocol_identity:
            raise ValueError("30K protocol canonical identity is invalid")
        if protocol.get("selection_file_sha256") != sha256_file(selection_file):
            raise ValueError("protocol is bound to a different selection file")
        if protocol.get("combined_selection_sha256") != selection_identity:
            raise ValueError("protocol is bound to a different selection identity")
        if int(protocol.get("pair_count", -1)) != len(rows):
            raise ValueError("protocol pair count differs from the frozen selection")
        partitions = _partition_rows(rows, protocol["split_counts"], int(protocol["seed"]))
        for name, part in partitions.items():
            actual = canonical_json_sha256([row["uid"] for row in part])
            if protocol["split_uid_sha256"].get(name) != actual:
                raise ValueError(f"protocol UID identity differs for split={name}")
        if split == "all":
            selected_rows = rows
        elif split in partitions:
            selected_rows = partitions[split]
        else:
            raise ValueError(f"unsupported split={split!r}")
    elif split == "all":
        selected_rows = rows
    else:
        raise ValueError("a protocol is required for any split other than 'all'")
    if int(max_samples) < 0:
        raise ValueError("max-samples must be non-negative")
    if int(max_samples) > 0:
        selected_rows = selected_rows[: int(max_samples)]
    if not selected_rows:
        raise ValueError("selected ProObjaverse split is empty")
    identity = {
        "selection_file": str(selection_file),
        "selection_file_sha256": sha256_file(selection_file),
        "selection_identity": selection_identity,
        "protocol_file": str(Path(protocol_path).expanduser().resolve())
        if protocol_path
        else "",
        "protocol_identity": protocol_identity,
        "split": split,
        "selected_count": len(selected_rows),
        "selected_uid_sha256": canonical_json_sha256(
            [row["uid"] for row in selected_rows]
        ),
    }
    return selected_rows, root, identity


def load_render_views(
    path: str | Path, uid: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read complete JSON/RGBA pairs, recovering valid views before tar damage."""

    archive_path = Path(path)
    views: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r") as archive:
        members: dict[str, tarfile.TarInfo] = {}
        read_error = ""
        while True:
            try:
                member = archive.next()
            except tarfile.ReadError as error:
                read_error = str(error)
                break
            if member is None:
                break
            if member.name in members:
                raise RuntimeError(f"duplicate tar member {member.name}: {archive_path}")
            members[member.name] = member
        for metadata_name in sorted(name for name in members if name.endswith(".json")):
            image_name = metadata_name[:-5] + ".rgba.webp"
            if image_name not in members:
                continue
            try:
                metadata_handle = archive.extractfile(members[metadata_name])
                image_handle = archive.extractfile(members[image_name])
                if metadata_handle is None or image_handle is None:
                    continue
                metadata_bytes = metadata_handle.read()
                image_bytes = image_handle.read()
            except tarfile.ReadError:
                continue
            metadata = json.loads(metadata_bytes)
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
            camera_to_world = np.asarray(metadata["extrinsic"], dtype=np.float32)
            intrinsic = np.asarray(metadata["intrinsic"], dtype=np.float32)
            if camera_to_world.shape != (4, 4) or intrinsic.shape != (3, 3):
                raise ValueError(f"uid={uid} invalid camera matrix in {metadata_name}")
            if not np.isfinite(camera_to_world).all() or not np.isfinite(intrinsic).all():
                raise ValueError(f"uid={uid} non-finite camera in {metadata_name}")
            rotation = camera_to_world[:3, :3]
            center = camera_to_world[:3, 3]
            world_to_camera = np.eye(4, dtype=np.float32)
            world_to_camera[:3, :3] = rotation.T
            world_to_camera[:3, 3] = -(rotation.T @ center)
            views.append(
                {
                    "id": int(metadata.get("image_index", Path(metadata_name).stem)),
                    "rgba": rgba,
                    "intrinsic": intrinsic,
                    "extrinsic": world_to_camera,
                    "center": center.astype(np.float64),
                }
            )
    views.sort(key=lambda row: int(row["id"]))
    ids = [int(row["id"]) for row in views]
    if not views or len(ids) != len(set(ids)):
        raise ValueError(f"uid={uid} contains no views or duplicate view IDs")
    return views, {
        "archive_complete": not bool(read_error),
        "archive_recovered": bool(read_error),
        "archive_read_error": read_error,
        "complete_view_count": len(views),
        "member_count_before_error": len(members),
    }


def pose_diverse_indices(
    views: list[dict[str, Any]], count: int, uid: str
) -> list[int]:
    """Match the old method's deterministic spherical farthest-view policy."""

    count = min(int(count), len(views))
    if count <= 0:
        raise ValueError("selected view count must be positive")
    centers = np.stack([np.asarray(row["center"], dtype=np.float64) for row in views])
    radii = np.linalg.norm(centers, axis=1, keepdims=True)
    directions = centers / np.maximum(radii, 1.0e-12)
    hashes = [
        canonical_json_sha256({"uid": uid, "view": int(row["id"])})
        for row in views
    ]
    selected = [min(range(len(views)), key=lambda index: hashes[index])]
    while len(selected) < count:
        candidates = [index for index in range(len(views)) if index not in selected]

        def score(index: int) -> tuple[float, str]:
            cosine = directions[selected] @ directions[index]
            angle = float(np.arccos(np.clip(cosine, -1.0, 1.0)).min())
            inverse_hash = "".join(chr(255 - ord(char)) for char in hashes[index])
            return angle, inverse_hash

        selected.append(max(candidates, key=score))
    return selected


def camera_forward_sign(extrinsics: np.ndarray) -> float:
    origin_depth = np.asarray(extrinsics)[:, 2, 3]
    positive = int(np.sum(origin_depth > 0.0))
    negative = int(np.sum(origin_depth < 0.0))
    if positive == negative:
        raise RuntimeError("official cameras have an ambiguous forward-axis sign")
    sign = 1.0 if positive > negative else -1.0
    if float(np.mean(origin_depth * sign > 0.0)) < 0.95:
        raise RuntimeError("official cameras do not consistently face the origin")
    return sign


def _object_paths(output: Path, uid: str) -> dict[str, Path]:
    root = output / "objects" / uid[:2] / uid
    return {
        "pose": root / "pose.pt",
        "record": root / "record.json",
        "error": root / "source_error.json",
    }


def _build_config(args: argparse.Namespace, identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_identity": identity,
        "dino_model": str(args.dino_model),
        "selected_views": int(args.selected_views),
        "preprocessing": {
            "format": PREPROCESS_FORMAT,
            "image_size": DINO_IMAGE_SIZE,
            "crop_margin": float(args.crop_margin),
            "mask_threshold": float(args.mask_threshold),
            "background": "black_after_resized_alpha_threshold",
            "rgb_and_mask_resampling": "bilinear",
            "intrinsics_rule": "K_feature=source_to_feature_affine@K_source",
        },
        "camera_contract": {
            "source_field": "extrinsic",
            "source_semantics": "camera_to_world",
            "runtime_semantics": "world_to_camera",
            "conversion": "R_w2c=R_c2w.T;t_w2c=-R_c2w.T@C_world",
            "source_axes": "OpenGL/Blender: camera -Z forward and +Y up",
            "pixel_axes": "image +v down; projection multiplies camera Y by -1",
        },
        "view_policy": "deterministic_camera_center_spherical_farthest_v1",
    }


def _check_existing(
    paths: dict[str, Path],
    *,
    uid: str,
    config_hash: str,
    verify_hash: bool,
) -> bool:
    pose_exists = paths["pose"].is_file()
    record_exists = paths["record"].is_file()
    if not pose_exists and not record_exists:
        return False
    if not pose_exists or not record_exists:
        raise RuntimeError(f"uid={uid} has an incomplete pose cache")
    record = json.loads(paths["record"].read_text(encoding="utf-8"))
    if (
        record.get("format") != OBJECT_RECORD_FORMAT
        or record.get("uid") != uid
        or record.get("config_hash") != config_hash
        or int(record.get("pose_bytes", -1)) != paths["pose"].stat().st_size
    ):
        raise RuntimeError(f"uid={uid} cached pose identity differs")
    if verify_hash and record.get("pose_sha256") != sha256_file(paths["pose"]):
        raise RuntimeError(f"uid={uid} cached pose hash differs")
    return True


@torch.no_grad()
def materialize(args: argparse.Namespace) -> dict[str, Any]:
    rows, source_root, identity = load_frozen_rows(
        args.selection,
        protocol_path=args.protocol,
        data_root=args.data_root,
        split=args.split,
        max_samples=int(args.max_samples),
    )
    if not 1 <= int(args.selected_views) <= 40:
        raise ValueError("selected-views must be in [1,40]")
    if not 0.0 <= float(args.crop_margin) <= 1.0:
        raise ValueError("crop-margin must be in [0,1]")
    if not 0.0 <= float(args.mask_threshold) <= 1.0:
        raise ValueError("mask-threshold must be in [0,1]")
    if not 0 <= int(args.worker_index) < int(args.worker_count):
        raise ValueError("worker-index must be in [0,worker-count)")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = _build_config(args, identity)
    config_hash = canonical_json_sha256(config)
    assigned = rows[int(args.worker_index) :: int(args.worker_count)]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ProObjaverse DINOv3 materialization requires CUDA")
    if device.index is not None:
        torch.cuda.set_device(device.index)
    extractor = load_dinov3(device=device, model_name=str(args.dino_model))
    completed = 0
    reused = 0
    failed = 0
    for local_index, row in enumerate(assigned, start=1):
        uid = str(row["uid"])
        paths = _object_paths(output, uid)
        if _check_existing(
            paths,
            uid=uid,
            config_hash=config_hash,
            verify_hash=bool(args.verify_existing_hashes),
        ):
            reused += 1
            print(
                json.dumps({"uid": uid, "worker_item": local_index, "reused": True}),
                flush=True,
            )
            continue
        error_exists = paths["error"].is_file()
        if error_exists and not bool(args.retry_source_errors):
            error = json.loads(paths["error"].read_text(encoding="utf-8"))
            if error.get("config_hash") != config_hash:
                raise RuntimeError(f"uid={uid} stale source-error identity differs")
            if not bool(args.record_source_errors):
                raise RuntimeError(f"uid={uid} has a recorded source error")
            failed += 1
            continue
        render = source_root / str(row["render"]["path"])
        try:
            if not render.is_file() or render.stat().st_size != int(row["render"]["size"]):
                raise FileNotFoundError(f"frozen render size/path differs: {render}")
            views, archive_audit = load_render_views(render, uid)
            if len(views) < int(args.selected_views):
                raise RuntimeError(
                    f"only {len(views)} complete views; need {args.selected_views}"
                )
            selected = [
                views[index]
                for index in pose_diverse_indices(views, int(args.selected_views), uid)
            ]
            prepared: list[Image.Image] = []
            transformed_intrinsics: list[torch.Tensor] = []
            view_metadata: list[dict[str, Any]] = []
            for view in selected:
                image, intrinsic, metadata = prepare_rgba_view(
                    view["rgba"],
                    torch.from_numpy(view["intrinsic"]),
                    margin=float(args.crop_margin),
                    threshold=float(args.mask_threshold),
                    require_foreground=True,
                )
                prepared.append(image)
                transformed_intrinsics.append(intrinsic)
                view_metadata.append(metadata)
            features = extractor(prepared).detach().to(dtype=torch.float16, device="cpu")
            expected = (len(selected), DINO_FULL_TOKENS, DINO_FEATURE_DIM)
            if tuple(features.shape) != expected or not bool(
                torch.isfinite(features).all()
            ):
                raise RuntimeError(
                    f"DINOv3 output={tuple(features.shape)}, expected={expected}"
                )
            extrinsics = np.stack([view["extrinsic"] for view in selected])
            forward_sign = camera_forward_sign(extrinsics)
            selected_ids = [int(view["id"]) for view in selected]
            payload: dict[str, Any] = {
                "format": SAMPLE_FORMAT,
                "uid": uid,
                "dino_features": features.contiguous(),
                "intrinsics": torch.stack(transformed_intrinsics).float(),
                "extrinsics": torch.from_numpy(extrinsics).float(),
                "extrinsics_type": "w2c",
                "camera_forward_sign": forward_sign,
                "camera_image_y_sign": -1.0,
                "grid_transform": "identity",
                "image_size": [DINO_IMAGE_SIZE, DINO_IMAGE_SIZE],
                "source_images": [f"{render}#{view_id:03d}" for view_id in selected_ids],
                "selected_view_ids": selected_ids,
                "view_preprocessing": view_metadata,
                "preprocessing": config["preprocessing"],
                "cache_config_hash": config_hash,
            }
            _atomic_torch_save(paths["pose"], payload)
            record = {
                "format": OBJECT_RECORD_FORMAT,
                "uid": uid,
                "selection_rank": int(row["selection_rank"]),
                "config_hash": config_hash,
                "pose_file": str(paths["pose"].resolve()),
                "pose_bytes": paths["pose"].stat().st_size,
                "pose_sha256": sha256_file(paths["pose"]),
                "render_file": str(render.resolve()),
                "render_bytes": render.stat().st_size,
                "candidate_view_count": len(views),
                "selected_view_ids": selected_ids,
                "camera_forward_sign": forward_sign,
                "camera_image_y_sign": -1.0,
                "archive_audit": archive_audit,
            }
            _atomic_json(paths["record"], record)
            paths["error"].unlink(missing_ok=True)
            completed += 1
            print(
                json.dumps(
                    {"uid": uid, "worker_item": local_index, "views": selected_ids}
                ),
                flush=True,
            )
        except Exception as error:
            if not bool(args.record_source_errors):
                raise
            failure = {
                "format": SOURCE_ERROR_FORMAT,
                "uid": uid,
                "selection_rank": int(row["selection_rank"]),
                "config_hash": config_hash,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _atomic_json(paths["error"], failure)
            failed += 1
            print(json.dumps({"source_error": failure}), flush=True)
    del extractor
    torch.cuda.empty_cache()
    report = {
        "format": WORKER_REPORT_FORMAT,
        "config": config,
        "config_hash": config_hash,
        "worker_index": int(args.worker_index),
        "worker_count": int(args.worker_count),
        "assigned": len(assigned),
        "completed": completed,
        "reused": reused,
        "failed": failed,
    }
    report_path = (
        output
        / "workers"
        / f"worker_{int(args.worker_index):03d}_of_{int(args.worker_count):03d}.json"
    )
    _atomic_json(report_path, report)
    return report


def _native_target_path(root: Path, template: str, row: dict[str, Any]) -> Path:
    relative = template.format(uid=row["uid"], shard=row["shard"])
    return (root / relative).resolve()


def _reject_old_targets(path: Path, source_root: Path, uid: str) -> None:
    old_root = (source_root / "lh-slats").resolve()
    if path == old_root or old_root in path.parents:
        raise ValueError(
            f"uid={uid} points at TRELLIS-v1 lh-slats; generate native TRELLIS.2 targets"
        )


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    rows, source_root, identity = load_frozen_rows(
        args.selection,
        protocol_path=args.protocol,
        data_root=args.data_root,
        split=args.split,
        max_samples=int(args.max_samples),
    )
    output = Path(args.output_dir).expanduser().resolve()
    config = _build_config(args, identity)
    config_hash = canonical_json_sha256(config)
    ss_root = Path(args.ss_target_dir).expanduser().resolve()
    shape_root = Path(args.shape_target_dir).expanduser().resolve()
    if not ss_root.is_dir() or not shape_root.is_dir():
        raise FileNotFoundError("both native TRELLIS.2 target directories must exist")
    manifest_rows: list[dict[str, Any]] = []
    source_errors: list[dict[str, Any]] = []
    target_errors: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        uid = str(row["uid"])
        paths = _object_paths(output, uid)
        if not paths["pose"].is_file() or not paths["record"].is_file():
            if paths["error"].is_file():
                error = json.loads(paths["error"].read_text(encoding="utf-8"))
                if error.get("config_hash") != config_hash:
                    raise RuntimeError(f"uid={uid} stale source-error identity differs")
                source_errors.append(error)
            else:
                missing.append(uid)
            continue
        record = json.loads(paths["record"].read_text(encoding="utf-8"))
        if (
            record.get("format") != OBJECT_RECORD_FORMAT
            or record.get("uid") != uid
            or record.get("config_hash") != config_hash
            or int(record.get("pose_bytes", -1)) != paths["pose"].stat().st_size
        ):
            raise RuntimeError(f"uid={uid} pose sidecar identity differs")
        if bool(args.deep_validate):
            pose = load_pose_payload(paths["pose"], uid)
            if pose.get("cache_config_hash") != config_hash:
                raise RuntimeError(f"uid={uid} pose payload config differs")
        try:
            ss_target = _native_target_path(
                ss_root, str(args.ss_target_template), row
            )
            shape_target = _native_target_path(
                shape_root, str(args.shape_target_template), row
            )
            _reject_old_targets(ss_target, source_root, uid)
            _reject_old_targets(shape_target, source_root, uid)
            if not ss_target.is_file() or not shape_target.is_file():
                raise FileNotFoundError(
                    f"native targets do not both exist: {ss_target}, {shape_target}"
                )
            load_ss_target(ss_target, uid)
            load_shape_target(shape_target, uid)
        except Exception as error:
            target_errors.append(
                {
                    "uid": uid,
                    "selection_rank": int(row["selection_rank"]),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        manifest_row: dict[str, Any] = {
            "uid": uid,
            "pose_file": str(paths["pose"].relative_to(output)),
            "ss_target_file": str(ss_target),
            "shape_target_file": str(shape_target),
            "view_count": int(args.selected_views),
            "selection_rank": int(row["selection_rank"]),
        }
        if not bool(args.skip_hashes):
            pose_hash = sha256_file(paths["pose"])
            if pose_hash != record.get("pose_sha256"):
                raise RuntimeError(f"uid={uid} pose hash differs from its sidecar")
            manifest_row.update(
                {
                    "pose_file_sha256": pose_hash,
                    "ss_target_file_sha256": sha256_file(ss_target),
                    "shape_target_file_sha256": sha256_file(shape_target),
                }
            )
        manifest_rows.append(manifest_row)
    if missing:
        raise RuntimeError(
            f"pose materialization is incomplete: {len(missing)} missing; first={missing[:8]}"
        )
    if len(source_errors) > int(args.max_source_errors):
        raise RuntimeError(
            f"recorded source errors={len(source_errors)} exceed allowed "
            f"max-source-errors={int(args.max_source_errors)}"
        )
    if len(target_errors) > int(args.max_target_errors):
        raise RuntimeError(
            f"native target errors={len(target_errors)} exceed allowed "
            f"max-target-errors={int(args.max_target_errors)}; "
            f"first={target_errors[:4]}"
        )
    if not manifest_rows:
        raise RuntimeError("no valid samples remain after source validation")
    manifest = {
        "format": CACHE_FORMAT,
        "root": ".",
        "source": "frozen_proobjaverse_30k_posed_rgba",
        "source_identity": identity,
        "cache_config": config,
        "cache_config_hash": config_hash,
        "dino_model": str(args.dino_model),
        "dino_image_size": DINO_IMAGE_SIZE,
        "dino_tokens": DINO_FULL_TOKENS,
        "geometry_targets": "native_trellis2_geometry",
        "old_trellis1_lh_slats_loaded": False,
        "requested_sample_count": len(rows),
        "excluded_source_error_count": len(source_errors),
        "excluded_source_errors": source_errors,
        "excluded_target_error_count": len(target_errors),
        "excluded_target_errors": target_errors,
        "samples": manifest_rows,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(f"refusing to overwrite a changed manifest: {manifest_path}")
    else:
        _atomic_json(manifest_path, manifest)
    PoseTrellis2Dataset(manifest_path, stage="inference")
    PoseTrellis2Dataset(manifest_path, stage="ss")
    PoseTrellis2Dataset(manifest_path, stage="shape")
    return {
        "manifest": str(manifest_path),
        "samples": len(manifest_rows),
        "excluded_source_errors": len(source_errors),
        "excluded_target_errors": len(target_errors),
        "cache_config_hash": config_hash,
    }


def export_instances(args: argparse.Namespace) -> dict[str, Any]:
    """Export the exact selected SHA256 list for the TRELLIS.2 data toolkit."""

    rows, _, identity = load_frozen_rows(
        args.selection,
        protocol_path=args.protocol,
        data_root=args.data_root,
        split=args.split,
        max_samples=int(args.max_samples),
    )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = (
        Path(args.instances_file).expanduser().resolve()
        if args.instances_file
        else output / f"instances_{args.split}.txt"
    )
    contents = "".join(f"{row['uid']}\n" for row in rows)
    if destination.exists():
        if destination.read_text(encoding="ascii") != contents:
            raise RuntimeError(f"refusing to overwrite changed UID list: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        temporary.write_text(contents, encoding="ascii")
        os.replace(temporary, destination)
    binding = {
        "format": "pose_constrained_trellis2.proobjaverse_instances.v1",
        "instances_file": str(destination),
        "instances_file_sha256": sha256_file(destination),
        "source_identity": identity,
        "count": len(rows),
    }
    binding_path = destination.with_suffix(destination.suffix + ".json")
    if binding_path.exists():
        if json.loads(binding_path.read_text(encoding="utf-8")) != binding:
            raise RuntimeError(f"refusing to overwrite changed binding: {binding_path}")
    else:
        _atomic_json(binding_path, binding)
    return binding


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--selection", default=DEFAULT_SELECTION)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--split", choices=(*SPLIT_NAMES, "all"), default="train")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dino-model", default=DINO_MODEL)
    parser.add_argument("--selected-views", type=int, default=8)
    parser.add_argument("--crop-margin", type=float, default=0.10)
    parser.add_argument("--mask-threshold", type=float, default=0.80)
    return parser


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    instances_parser = commands.add_parser("instances", parents=[_common_parser()])
    instances_parser.add_argument("--instances-file", default="")
    materialize_parser = commands.add_parser(
        "materialize", parents=[_common_parser()]
    )
    materialize_parser.add_argument("--device", default="cuda")
    materialize_parser.add_argument("--worker-index", type=int, default=0)
    materialize_parser.add_argument("--worker-count", type=int, default=1)
    materialize_parser.add_argument("--verify-existing-hashes", action="store_true")
    materialize_parser.add_argument("--record-source-errors", action="store_true")
    materialize_parser.add_argument("--retry-source-errors", action="store_true")
    finalize_parser = commands.add_parser("finalize", parents=[_common_parser()])
    finalize_parser.add_argument("--ss-target-dir", required=True)
    finalize_parser.add_argument("--shape-target-dir", required=True)
    finalize_parser.add_argument("--ss-target-template", default="{uid}.npz")
    finalize_parser.add_argument("--shape-target-template", default="{uid}.npz")
    finalize_parser.add_argument("--max-source-errors", type=int, default=0)
    finalize_parser.add_argument("--max-target-errors", type=int, default=0)
    finalize_parser.add_argument("--skip-hashes", action="store_true")
    finalize_parser.add_argument("--deep-validate", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "instances":
        report = export_instances(args)
    elif args.command == "materialize":
        report = materialize(args)
    else:
        report = finalize(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "camera_forward_sign",
    "export_instances",
    "finalize",
    "load_frozen_rows",
    "load_render_views",
    "materialize",
    "pose_diverse_indices",
]
