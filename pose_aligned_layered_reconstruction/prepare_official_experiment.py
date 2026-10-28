"""Independent, immutable ProObjaverse cache bridge for the first real experiment.

The old cache files are data inputs only: no old project code is imported.
SS targets are real frozen-encoder latents; layered supervision is rendered
from the official SLat target decoded by the frozen stock mesh decoder.
It is target-VAE geometry, NOT an original uncompressed source mesh.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .contracts import LAYERED_SAMPLE_FORMAT, validate_paired_sample
from .geometry_layers import peel_mesh
from .paired_dataset import canonical_json_sha256, sha256_file, write_manifest
from .paired_encoder import build_frozen_dino_encoder, _encode_patch_tokens
from .pretrained import DEFAULT_PRETRAINED, load_trellis_model


DATA = Path("/data/zjr")
PROTOCOL = DATA / "proobjaverse_official_slat_protocol30k_seed20260813_f39_frozen_v1"
TRAIN = DATA / "proobjaverse_official_native_ss_train2000_20260815_v1/cache_train2000_official_ss_v1/lifting_manifest.json"
DEV = DATA / "proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1"
RECIPE = {
    "version": "official_paired_target_vae_layers.v2", "views": 8, "layers": 5,
    "layer_camera_y_sign": -1.0,
    "parent_appearance_camera_unchanged": True,
    "raw": "source_RGBA_alpha_composited_on_black_after_shared_crop",
    "masked": "byte_identical_existing_SS30K_DINO_cache",
    "raw_has_external_occluders": False,
    "labels": "depth_peeling_frozen_mesh_decoder_of_official_SLat_target",
    "labels_are_original_source_mesh": False,
    "ss_target": "existing_audited_frozen_SS_encoder_latent_of_official_support",
    "target_coords": "official_SLat_support_64_cubed",
    "parent_training_overlap_allowed": True,
}
QA_POLICY = "record_mesh_alpha_disagreement_without_filtering_thin_or_alpha_textured_objects"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def source_rows(split):
    train_uids = {r["uid"] for r in read_json(PROTOCOL / "train_usable29861_runtime_v2.json")["rows"]}
    dev_uids = {r["uid"] for r in read_json(PROTOCOL / "dev.json")["rows"]}
    if train_uids & dev_uids:
        raise RuntimeError("parent train/dev leakage")
    if split.startswith("train"):
        original = read_json(TRAIN)["samples"]
        rows = []
        for row in original:
            uid = row["uid"]
            if uid not in train_uids or uid in dev_uids:
                raise RuntimeError("first-round training does not respect parent split")
            slat = Path(row["official_lh_slat"])
            rows.append({"uid": uid, "cache": row["cache_file"],
                         "cache_sha256": row["cache_file_sha256"],
                         "ss": row["ss_latent"], "ss_sha256": row["ss_latent_sha256"],
                         "slat": str(slat), "slat_sha256": row["official_lh_slat_sha256"],
                         "render": str(slat.parents[2] / "renders_random_env" / slat.parent.name / (uid + ".tar"))})
        if split == "train32":
            rows = rows[:32]
        expected = 32 if split == "train32" else 2000
    else:
        ss = read_json(DEV / "ss_targets_dev64/lifting_manifest.json")
        slat_manifest = read_json(DEV / "cache_dev64_compact_v2/slat_manifest.json")
        slats = {r["uid"]: r for r in slat_manifest["samples"]}
        protocol_rows = {r["uid"]: r for r in read_json(PROTOCOL / "dev.json")["rows"]}
        rows = []
        for row in ss["samples"]:
            uid = row["uid"]
            sr = slats[uid]
            if uid in train_uids or uid not in dev_uids:
                raise RuntimeError("development data overlaps parent training")
            rows.append({"uid": uid,
                         "cache": str(DEV / "cache_dev64_compact_v2" / sr["compact_file"]),
                         "cache_sha256": sr["compact_file_sha256"],
                         "ss": str(DEV / "ss_targets_dev64" / row["ss_latent"]),
                         "ss_sha256": row["ss_latent_sha256"],
                         "slat": sr["source_lh_slat"], "slat_sha256": sr["source_lh_slat_sha256"],
                         "render": str(DATA / "ProObjaverse-300K-ReconViaGen-30K/renders_random_env" / protocol_rows[uid]["shard"] / (uid + ".tar"))})
        expected = 64
    if len(rows) != expected or len({r["uid"] for r in rows}) != expected:
        raise RuntimeError("unexpected experiment object count")
    return rows


def read_selected_render_members(path, names):
    """Read exactly the required camera/image members, not unrelated views.

    Some inherited archives have incomplete *later, unused* views. This is not
    an archive-integrity certificate: every selected member must still be
    present, regular, fully readable and independently recorded by SHA-256.
    """
    wanted = set(names)
    found = {}
    with tarfile.open(path) as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            if not member.isfile() or member.name in found:
                raise ValueError("invalid/duplicate selected render member")
            payload = archive.extractfile(member).read()
            if len(payload) != member.size:
                raise ValueError("incomplete selected render member")
            found[member.name] = payload
            if set(found) == wanted:
                break
    if set(found) != wanted:
        raise ValueError(f"missing selected render members: {sorted(wanted - set(found))}")
    return found


def paired_pixels(row, cached):
    raw, masked, masks = [], [], []
    camera_error = 0.
    selected = read_selected_render_members(row["render"], [
        f"{row['uid']}/{view_id:03d}{suffix}"
        for view_id in cached["view_ids"].tolist() for suffix in (".rgba.webp", ".json")])
    for i, view_id in enumerate(cached["view_ids"].tolist()):
        base = f"{row['uid']}/{view_id:03d}"
        rgba = Image.open(io.BytesIO(selected[base + ".rgba.webp"])).convert("RGBA")
        camera = json.loads(selected[base + ".json"])
        affine = cached["source_to_feature_affines"][i].numpy()
        scale = float(affine[0, 0])
        side = int(round(518 / scale))
        left, top = np.rint(-affine[:2, 2] / scale).astype(int)
        # PIL RGBA resize matches inherited masked-image preprocessing.
        resized = np.asarray(rgba.crop((left, top, left + side, top + side)).resize((518, 518), Image.Resampling.BILINEAR))
        alpha = resized[..., 3].astype(np.float32) / 255
        raw.append(Image.fromarray(np.rint(resized[..., :3] * alpha[..., None]).astype(np.uint8)))
        masked.append(Image.fromarray(resized[..., :3] * (resized[..., 3:4] > .8 * 255).astype(np.uint8)))
        masks.append(alpha)
        transformed_k = affine @ np.asarray(camera["intrinsic"], dtype=np.float32)
        w2c = np.linalg.inv(np.asarray(camera["extrinsic"], dtype=np.float32))
        camera_error = max(camera_error, float(np.max(np.abs(transformed_k - cached["intrinsics"][i].numpy()))),
                           float(np.max(np.abs(w2c - cached["extrinsics"][i].numpy()))))
    if camera_error > 1.e-3:
        raise RuntimeError(f"source/cache camera mismatch {camera_error}")
    masks = torch.from_numpy(np.stack(masks)).half()
    mask_error = float((masks - cached["masks"]).abs().max()) if "masks" in cached else None
    if mask_error is not None and mask_error > 1.e-3:
        raise RuntimeError(f"source/cache crop mismatch {mask_error}")
    return raw, masked, masks, {"camera_max_abs_error": camera_error, "mask_max_abs_error": mask_error,
        "render_audit_scope": "selected_members_only_not_full_archive_integrity",
        "selected_render_member_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in selected.items()}}


@torch.no_grad()
def materialize(row, output, encoder, decoder, decoder_binding, device, *, audit_reencode=False):
    from trellis.modules import sparse as sp
    uid = row["uid"]
    directory = output / "objects" / uid
    record_path = directory / "record.json"
    if record_path.is_file():
        record = read_json(record_path)
        if record["recipe_hash"] != canonical_json_sha256(RECIPE) or record["source"] != row:
            raise RuntimeError("resume recipe/source mismatch")
        if record["sha256"] != sha256_file(output / record["sample"]):
            raise RuntimeError("resume sample hash mismatch")
        return record
    for path_key, hash_key in (("cache", "cache_sha256"), ("ss", "ss_sha256"), ("slat", "slat_sha256")):
        if sha256_file(row[path_key]) != row[hash_key]:
            raise RuntimeError(f"changed source {row[path_key]}")
    cached = torch.load(row["cache"], map_location="cpu", weights_only=False)
    if cached["uid"] != uid or cached["grid_transform"] != "identity" or cached["extrinsics_type"] != "w2c":
        raise RuntimeError("unsupported source identity/coordinate contract")
    with np.load(row["ss"], allow_pickle=False) as target:
        z = torch.from_numpy(np.asarray(target["z"], dtype=np.float32).reshape(8, 16, 16, 16))
        # Legacy target_coords stores the frozen SS decoder's round-trip,
        # which can differ by a voxel. Score against original official GT.
        coords = torch.from_numpy(np.asarray(target["official_coords"], dtype=np.int32))
        if str(target["uid"].item()) != uid or float(target["roundtrip_iou"]) < .9:
            raise RuntimeError("invalid true SS target")
        roundtrip_iou = float(target["roundtrip_iou"])
    with np.load(row["slat"], allow_pickle=False) as target:
        slat_coords = torch.from_numpy(np.asarray(target["coords"], dtype=np.int32))
        feats = torch.from_numpy(np.asarray(target["feats"], dtype=np.float32))
    if not torch.equal(coords, slat_coords):
        if not np.array_equal(np.unique(coords.numpy(), axis=0), np.unique(slat_coords.numpy(), axis=0)):
            raise RuntimeError("SS/SLat target support mismatch")
    raw, masked, masks, pixel_audit = paired_pixels(row, cached)
    raw_features = _encode_patch_tokens(encoder, raw).half().cpu()
    cached_features = cached["visual_patch_features"].half()
    if audit_reencode:
        reencoded = _encode_patch_tokens(encoder, masked).half().cpu()
        pixel_audit["masked_DINO_reencode_max_abs"] = float((reencoded - cached_features).abs().max())
        pixel_audit["masked_DINO_reencode_rmse"] = float((reencoded.float() - cached_features.float()).square().mean().sqrt())
        if pixel_audit["masked_DINO_reencode_rmse"] > .02:
            raise RuntimeError(f"masked DINO preprocessing regression: {pixel_audit}")
    sparse = sp.SparseTensor(feats=feats.to(device), coords=torch.cat((torch.zeros_like(slat_coords[:, :1]), slat_coords), dim=1).to(device))
    mesh = decoder(sparse)[0]
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError("empty target-VAE mesh")
    layer_intrinsics = cached["intrinsics"].clone()
    layer_intrinsics[:, 1, 1] *= RECIPE["layer_camera_y_sign"]
    # nvdiffrast allocates outside PyTorch's caching allocator.
    torch.cuda.empty_cache()
    labels = peel_mesh(mesh.vertices, mesh.faces, layer_intrinsics.to(device), cached["extrinsics"].to(device), camera_forward_sign=float(cached["camera_forward_sign"]))
    silhouette = labels["ray_stop_index_gt"] > 0
    alpha = F.interpolate(masks[:, None].float(), (37, 37), mode="bilinear", align_corners=False)[:, 0] > .5
    intersections = (silhouette & alpha).sum((1, 2)).float()
    unions = (silhouette | alpha).sum((1, 2)).clamp_min(1)
    silhouette_iou = intersections / unions
    silhouette_qa_low = float(silhouette_iou.mean()) < .8
    if silhouette_qa_low:
        directory.mkdir(parents=True, exist_ok=True)
        raw[0].save(directory / "qa_raw_view0.png")
        masked[0].save(directory / "qa_masked_view0.png")
        Image.fromarray((silhouette[0].numpy() * 255).astype(np.uint8)).resize((518, 518), Image.Resampling.NEAREST).save(directory / "qa_mesh_silhouette_view0.png")
        # The official SS target includes opaque geometric surfaces even when
        # a source texture is transparent. Thin VAE geometry can also thicken.
        # Camera/crop identities remain strict; silhouette IoU is a diagnostic,
        # not a way to discard difficult objects from the frozen 2000 UIDs.
        print(json.dumps({"uid": uid, "quality_warning": QA_POLICY,
                          "silhouette_iou": silhouette_iou.tolist()}), flush=True)
    sample = {
        "format": LAYERED_SAMPLE_FORMAT, "uid": uid, "object_uid": uid,
        "raw_visual_patch_features": raw_features,
        "masked_visual_patch_features": cached_features,
        "visible_target_mask": masks, "intrinsics": cached["intrinsics"],
        "extrinsics": cached["extrinsics"], "image_size": (518, 518),
        "grid_transform": "identity", "extrinsics_type": "w2c",
        "camera_forward_sign": float(cached["camera_forward_sign"]),
        "layer_camera_y_sign": RECIPE["layer_camera_y_sign"],
        "view_ids": cached["view_ids"], "target": z, "target_coords": coords,
        "preprocessing": cached["preprocessing"], "source_binding": row,
        "paired_encoding_contract": {"same_frozen_dino_encoder": True, "same_crop_and_raster": True},
        **labels,
    }
    validate_paired_sample(sample, require_labels=True, require_ss_target=True, expected_layers=5)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "sample.pt"
    torch.save(sample, destination)
    # A few input/label previews for direct inspection, no AI image generation.
    if audit_reencode:
        raw[0].save(directory / "raw_view0.png")
        masked[0].save(directory / "masked_view0.png")
        Image.fromarray((silhouette[0].numpy() * 255).astype(np.uint8)).resize((518, 518), Image.Resampling.NEAREST).save(directory / "target_mesh_silhouette_view0.png")
    record = {
        "uid": uid, "object_uid": uid, "sample": str(destination.relative_to(output)),
        "sha256": sha256_file(destination), "recipe_hash": canonical_json_sha256(RECIPE),
        "source": row, "decoder_binding": decoder_binding, "pixel_audit": pixel_audit,
        "official_ss_roundtrip_iou": roundtrip_iou,
        "mesh_vs_alpha_silhouette_iou_by_view": silhouette_iou.tolist(),
        "target_mesh_vertices": len(mesh.vertices), "target_mesh_faces": len(mesh.faces),
        "mesh_alpha_disagreement_flag": silhouette_qa_low,
        "mesh_labels_ignore_texture_alpha": True, "qa_policy": QA_POLICY,
        "has_ss_target": True, "has_layer_labels": True, "view_count": 8,
    }
    atomic_json(record_path, record)
    return record


def finalize(output, split):
    source = source_rows(split)
    records = []
    for row in source:
        record = read_json(output / "objects" / row["uid"] / "record.json")
        if record["source"] != row or record["recipe_hash"] != canonical_json_sha256(RECIPE):
            raise RuntimeError("materialized source/recipe changed")
        if sha256_file(output / record["sample"]) != record["sha256"]:
            raise RuntimeError("materialized object changed")
        records.append(record)
    config = {**RECIPE, "split": split, "quality_policy": QA_POLICY, "source_uid_hash": canonical_json_sha256([r["uid"] for r in source]),
              "parent_protocol_sha256": sha256_file(PROTOCOL / "protocol.json")}
    path = output / (split + ".json")
    write_manifest(path, rows=records, config=config, root=output)
    print(json.dumps({"completed_manifest": str(path), "objects": len(records)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=("train32", "train2000", "dev64"), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--finalize_only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output_dir).resolve()
    if args.finalize_only:
        finalize(output, args.split)
        return
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    encoder = build_frozen_dino_encoder(device=device)
    decoder, binding = load_trellis_model(DEFAULT_PRETRAINED, "slat_dec_mesh_swin8_B_64l8m256c_fp16")
    decoder.to(device).eval().requires_grad_(False)
    rows = source_rows(args.split)
    end = args.end or len(rows)
    for i in range(args.start, end):
        begin = time.time()
        try:
            record = materialize(rows[i], output, encoder, decoder, binding, device, audit_reencode=i == args.start)
        except Exception as error:
            failure = {"index": i, "source": rows[i], "error_type": type(error).__name__, "error": str(error)}
            atomic_json(output / "failures" / (rows[i]["uid"] + ".json"), failure)
            print(json.dumps(failure), flush=True)
            raise
        print(json.dumps({"index": i, "uid": rows[i]["uid"], "elapsed_seconds": time.time() - begin,
                          "silhouette_iou": float(np.mean(record["mesh_vs_alpha_silhouette_iou_by_view"])),
                          "pixel_audit": record["pixel_audit"]}), flush=True)
    if args.start == 0 and end == len(rows):
        finalize(output, args.split)


if __name__ == "__main__":
    main()
