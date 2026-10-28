from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

from pose_constrained_trellis2.checkpoint import (
    initialize_ema,
    save_checkpoint,
    update_checkpoint_alias,
)
from pose_constrained_trellis2.conditioning import (
    EveryBlockCondition,
    MultiviewAggregator,
    build_2d_context,
)
from pose_constrained_trellis2.constants import (
    CACHE_FORMAT,
    DINO_FEATURE_DIM,
    DINO_FULL_TOKENS,
    SAMPLE_FORMAT,
    SHAPE_LATENT_MEAN,
    SHAPE_MAX_TOKENS,
    SHAPE_LATENT_STD,
)
from pose_constrained_trellis2.data import (
    PoseTrellis2Dataset,
    canonical_json_sha256,
)
from pose_constrained_trellis2.geometry import (
    canonical_points,
    dense_coords,
    project_pose_features,
)
from pose_constrained_trellis2.infer import parse_views
from pose_constrained_trellis2.models import (
    BoundPoseCFGFlow,
    DensePoseConditionedFlow,
    SparsePoseConditionedFlow,
)
from pose_constrained_trellis2.prepare_pose_cache import _prepare_view
from pose_constrained_trellis2.prepare_proobjaverse_cache import (
    camera_forward_sign,
    load_frozen_rows,
    load_render_views,
    pose_diverse_indices,
)
from pose_constrained_trellis2.runtime import ss_latent_to_shape_coords
from pose_constrained_trellis2.trellis2_toolkit_compat.datasets import (
    ObjaverseXL,
)


def camera_sample(views: int = 1) -> dict[str, object]:
    features = torch.zeros(views, DINO_FULL_TOKENS, DINO_FEATURE_DIM)
    patch_x = torch.arange(32, dtype=torch.float32)[None].expand(32, 32)
    features[:, 5:, 0] = patch_x.reshape(-1)
    intrinsics = torch.tensor(
        [[[100.0, 0.0, 255.5], [0.0, 100.0, 255.5], [0.0, 0.0, 1.0]]]
    ).expand(views, -1, -1).clone()
    extrinsics = torch.eye(4)[None].expand(views, -1, -1).clone()
    extrinsics[:, 2, 3] = 1.0
    return {
        "dino_features": features,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "extrinsics_type": "w2c",
        "camera_forward_sign": 1.0,
        "camera_image_y_sign": 1.0,
        "grid_transform": "identity",
        "image_size": [512, 512],
    }


class GeometryTests(unittest.TestCase):
    def test_dense_order_and_canonical_centres(self) -> None:
        coords = dense_coords(2, with_batch=False)
        self.assertEqual(coords.tolist()[0], [0, 0, 0])
        self.assertEqual(coords.tolist()[-1], [1, 1, 1])
        points = canonical_points(coords, resolution=2)
        self.assertTrue(torch.equal(points[0], torch.tensor([-0.25, -0.25, -0.25])))
        self.assertTrue(torch.equal(points[-1], torch.tensor([0.25, 0.25, 0.25])))

    def test_projection_samples_spatial_dino_tokens(self) -> None:
        projected, valid, stats = project_pose_features(
            camera_sample(),
            torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
            resolution=1,
            device="cpu",
        )
        self.assertEqual(tuple(projected.shape), (1, 1, DINO_FEATURE_DIM))
        self.assertTrue(bool(valid.item()))
        self.assertAlmostEqual(float(projected[0, 0, 0]), 15.5, places=5)
        self.assertEqual(float(stats["supported_fraction"]), 1.0)

    def test_opengl_vertical_axis_is_explicitly_flipped(self) -> None:
        sample = camera_sample()
        patch_y = torch.arange(32, dtype=torch.float32)[:, None].expand(32, 32)
        sample["dino_features"][:, 5:, 1] = patch_y.reshape(-1)
        sample["intrinsics"] = torch.tensor(
            [[[100.0, 0.0, 255.5], [0.0, 100.0, 255.5], [0.0, 0.0, 1.0]]]
        )
        sample["camera_image_y_sign"] = -1.0
        projected, valid, _ = project_pose_features(
            sample,
            torch.tensor([[1, 1, 1]], dtype=torch.int32),
            resolution=2,
            device="cpu",
        )
        self.assertTrue(bool(valid.item()))
        self.assertGreater(float(projected[0, 0, 0]), 15.5)
        self.assertLess(float(projected[0, 0, 1]), 15.5)


class ConditioningTests(unittest.TestCase):
    def test_context_keeps_native_prefix_tokens(self) -> None:
        features = torch.randn(3, DINO_FULL_TOKENS, DINO_FEATURE_DIM)
        indices = torch.tensor([2, 0])
        first = build_2d_context(features, indices, mode="first")
        concat = build_2d_context(features, indices, mode="concat")
        self.assertTrue(torch.equal(first, features[2:3]))
        self.assertEqual(tuple(concat.shape), (1, 2 * DINO_FULL_TOKENS, DINO_FEATURE_DIM))
        self.assertTrue(torch.equal(concat[:, :DINO_FULL_TOKENS], features[2:3]))

    def test_aggregator_is_permutation_invariant(self) -> None:
        torch.manual_seed(3)
        aggregator = MultiviewAggregator(4)
        features = torch.randn(3, 7, 4)
        valid = torch.tensor(
            [[True, True, False, True, True, False, True],
             [True, False, True, True, True, False, False],
             [False, True, True, True, False, False, True]]
        )
        direct, _ = aggregator(features, valid)
        order = torch.tensor([2, 0, 1])
        permuted, _ = aggregator(features[order], valid[order])
        self.assertTrue(torch.allclose(direct, permuted, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(direct[5], torch.zeros(4)))

    def test_every_block_projection_starts_exactly_zero(self) -> None:
        projector = EveryBlockCondition(4, 6, 3)
        self.assertTrue(projector.exact_zero())
        values = torch.randn(5, 4)
        for index in range(3):
            self.assertEqual(int(torch.count_nonzero(projector(index, values))), 0)


class _TimeEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(1, channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value[:, None].float())


class _DenseBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(channels, channels, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        rope: object,
    ) -> torch.Tensor:
        del timestep, context, rope
        return hidden + 0.1 * self.linear(hidden)


class _DenseFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resolution = 2
        self.in_channels = 2
        self.out_channels = 2
        self.model_channels = 4
        self.dtype = torch.float32
        self.pe_mode = "rope"
        self.rope_phases = None
        self.share_mod = False
        self.input_layer = nn.Linear(2, 4)
        self.t_embedder = _TimeEmbedding(4)
        self.blocks = nn.ModuleList([_DenseBlock(4), _DenseBlock(4)])
        self.out_layer = nn.Linear(4, 2)
        self.forward_calls = 0

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        self.forward_calls += 1
        hidden = x.view(1, 2, -1).permute(0, 2, 1).contiguous()
        hidden = self.input_layer(hidden)
        timestep = self.t_embedder(t)
        for block in self.blocks:
            hidden = block(hidden, timestep, context, self.rope_phases)
        hidden = torch.nn.functional.layer_norm(hidden, hidden.shape[-1:])
        hidden = self.out_layer(hidden)
        return hidden.permute(0, 2, 1).view(1, 2, 2, 2, 2).contiguous()


class _SparseValue:
    def __init__(self, feats: torch.Tensor, coords: torch.Tensor) -> None:
        self.feats = feats
        self.coords = coords

    @property
    def shape(self) -> tuple[int, int, int]:
        return (1, int(self.feats.shape[0]), int(self.feats.shape[1]))

    @property
    def device(self) -> torch.device:
        return self.feats.device

    @property
    def dtype(self) -> torch.dtype:
        return self.feats.dtype

    def type(self, dtype: torch.dtype) -> "_SparseValue":
        return self.replace(self.feats.to(dtype=dtype))

    def replace(self, feats: torch.Tensor) -> "_SparseValue":
        return _SparseValue(feats, self.coords)


class _SparseLinear(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_channels, output_channels)

    def forward(self, value: _SparseValue) -> _SparseValue:
        return value.replace(self.linear(value.feats))


class _SparseBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(channels, channels, bias=False)

    def forward(
        self, hidden: _SparseValue, timestep: torch.Tensor, context: torch.Tensor
    ) -> _SparseValue:
        del timestep, context
        return hidden.replace(hidden.feats + 0.1 * self.linear(hidden.feats))


class _SparseFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resolution = 2
        self.in_channels = 2
        self.out_channels = 2
        self.model_channels = 4
        self.dtype = torch.float32
        self.pe_mode = "rope"
        self.share_mod = False
        self.input_layer = _SparseLinear(2, 4)
        self.t_embedder = _TimeEmbedding(4)
        self.blocks = nn.ModuleList([_SparseBlock(4), _SparseBlock(4)])
        self.out_layer = _SparseLinear(4, 2)

    def forward(
        self, x: _SparseValue, t: torch.Tensor, context: torch.Tensor
    ) -> _SparseValue:
        hidden = self.input_layer(x)
        timestep = self.t_embedder(t)
        for block in self.blocks:
            hidden = block(hidden, timestep, context)
        hidden = hidden.replace(
            torch.nn.functional.layer_norm(hidden.feats, hidden.feats.shape[-1:])
        )
        return self.out_layer(hidden)


class AdapterForwardTests(unittest.TestCase):
    def _assert_live_projection_gradients(self, model: nn.Module) -> None:
        gradients = [
            projection.weight.grad
            for projection in model.block_condition.projections
        ]
        self.assertTrue(all(value is not None for value in gradients))
        self.assertTrue(all(int(torch.count_nonzero(value)) > 0 for value in gradients if value is not None))

    def test_dense_zero_init_matches_stock_with_live_gradients(self) -> None:
        flow = _DenseFlow()
        for parameter in flow.parameters():
            parameter.requires_grad_(False)
        model = DensePoseConditionedFlow(flow, strict_schema=False)
        x = torch.randn(1, 2, 2, 2, 2)
        t = torch.tensor([500.0])
        context = torch.randn(1, DINO_FULL_TOKENS, DINO_FEATURE_DIM)
        stock = model.stock_prediction(x, t, context)
        prediction, stats = model.adapted_prediction(
            x, t, context, camera_sample()
        )
        self.assertTrue(torch.equal(prediction, stock))
        self.assertEqual(float(stats["zero_init_stock_anchor"]), 1.0)
        prediction.sum().backward()
        self._assert_live_projection_gradients(model)

    def test_sparse_zero_init_matches_stock_with_live_gradients(self) -> None:
        flow = _SparseFlow()
        for parameter in flow.parameters():
            parameter.requires_grad_(False)
        model = SparsePoseConditionedFlow(flow, strict_schema=False)
        x = _SparseValue(
            torch.randn(2, 2),
            torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.int32),
        )
        t = torch.tensor([500.0])
        context = torch.randn(1, DINO_FULL_TOKENS, DINO_FEATURE_DIM)
        stock = model.stock_prediction(x, t, context)
        prediction, stats = model.adapted_prediction(
            x, t, context, camera_sample()
        )
        self.assertTrue(torch.equal(prediction.coords, stock.coords))
        self.assertTrue(torch.equal(prediction.feats, stock.feats))
        self.assertEqual(float(stats["zero_init_stock_anchor"]), 1.0)
        prediction.feats.sum().backward()
        self._assert_live_projection_gradients(model)

    def test_inference_caches_pose_condition_and_skips_stock_forward(self) -> None:
        flow = _DenseFlow()
        for parameter in flow.parameters():
            parameter.requires_grad_(False)
        model = DensePoseConditionedFlow(flow, strict_schema=False)
        with torch.no_grad():
            model.block_condition.projections[0].weight.fill_(0.01)
        x = torch.randn(1, 2, 2, 2, 2)
        t = torch.tensor([500.0])
        context = torch.randn(1, DINO_FULL_TOKENS, DINO_FEATURE_DIM)
        sample = camera_sample()
        bound = BoundPoseCFGFlow(
            model, positive_context=context, sample=sample
        )
        bound(x, t, context)
        bound(x, t, context)
        bound(x, t, torch.zeros_like(context))
        self.assertEqual(bound.condition_builds, 1)
        self.assertEqual(bound.positive_calls, 2)
        self.assertEqual(bound.negative_calls, 1)
        self.assertEqual(flow.forward_calls, 0)


class DataAndRuntimeTests(unittest.TestCase):
    def test_mask_crop_updates_intrinsics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgba = np.zeros((80, 100, 4), dtype=np.uint8)
            rgba[10:50, 20:60, :3] = 200
            rgba[10:50, 20:60, 3] = 255
            image_path = root / "view.png"
            Image.fromarray(rgba, mode="RGBA").save(image_path)
            intrinsic = torch.tensor(
                [[50.0, 0.0, 30.0], [0.0, 50.0, 20.0], [0.0, 0.0, 1.0]]
            )
            image, transformed, metadata = _prepare_view(
                image_path, None, intrinsic, margin=0.0, threshold=0.5
            )
            self.assertEqual(image.size, (512, 512))
            self.assertEqual(metadata["crop_xywh"], [20, 10, 40, 40])
            self.assertAlmostEqual(float(transformed[0, 2]), 128.0, places=4)
            self.assertAlmostEqual(float(transformed[1, 2]), 128.0, places=4)

    def test_manifest_loads_native_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = camera_sample()
            sample.update({"format": SAMPLE_FORMAT, "uid": "one"})
            torch.save(sample, root / "pose.pt")
            np.savez_compressed(root / "ss.npz", z=np.zeros((8, 16, 16, 16), np.float32))
            coords = np.array([[0, 0, 0], [31, 31, 31]], dtype=np.uint8)
            feats = np.stack(
                [np.asarray(SHAPE_LATENT_MEAN), np.asarray(SHAPE_LATENT_MEAN)], axis=0
            ).astype(np.float32)
            np.savez_compressed(root / "shape.npz", coords=coords, feats=feats)
            manifest = {
                "format": CACHE_FORMAT,
                "root": ".",
                "samples": [{
                    "uid": "one",
                    "pose_file": "pose.pt",
                    "ss_target_file": "ss.npz",
                    "shape_target_file": "shape.npz",
                }],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            ss = PoseTrellis2Dataset(root / "manifest.json", stage="ss")[0]
            shape = PoseTrellis2Dataset(root / "manifest.json", stage="shape")[0]
            self.assertEqual(tuple(ss["target"].shape), (8, 16, 16, 16))
            self.assertEqual(tuple(shape["target_feats"].shape), (2, 32))
            self.assertTrue(torch.allclose(shape["target_feats"], torch.zeros(2, 32), atol=1e-6))
            self.assertEqual(len(SHAPE_LATENT_STD), 32)

    def test_shape_target_above_official_token_limit_is_rejected(self) -> None:
        from pose_constrained_trellis2.data import load_shape_target

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "too_many.npz"
            count = SHAPE_MAX_TOKENS + 1
            flat = np.arange(count, dtype=np.int32)
            coords = np.stack(
                (flat // (32 * 32), (flat // 32) % 32, flat % 32), axis=1
            )
            feats = np.zeros((count, 32), dtype=np.float32)
            np.savez_compressed(path, coords=coords, feats=feats)
            with self.assertRaisesRegex(ValueError, "maximum"):
                load_shape_target(path, "too-many")

    def test_ss_decode_downsamples_to_shape_support(self) -> None:
        class Decoder(nn.Module):
            def forward(self, latent: torch.Tensor) -> torch.Tensor:
                del latent
                result = torch.zeros(1, 1, 64, 64, 64)
                result[0, 0, 2, 4, 6] = 1.0
                return result

        coords = ss_latent_to_shape_coords(
            torch.zeros(1, 8, 16, 16, 16), Decoder(), shape_resolution=32
        )
        self.assertEqual(coords.tolist(), [[0, 1, 2, 3]])

    def test_ordered_view_parser(self) -> None:
        self.assertEqual(parse_views("3,1,0", 4).tolist(), [3, 1, 0])
        self.assertEqual(parse_views("all", 3).tolist(), [0, 1, 2])
        with self.assertRaises(ValueError):
            parse_views("1,1", 3)

    def test_frozen_selection_protocol_and_tar_camera_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index in range(4):
                uid = f"{index:064x}"
                shard = "shard-0001"
                render_relative = f"renders_random_env/{shard}/{uid}.tar"
                slat_relative = f"lh-slats/{shard}/{uid}.npz"
                rows.append({
                    "uid": uid,
                    "shard": shard,
                    "render": {"path": render_relative, "size": 1},
                    "slat": {"path": slat_relative, "size": 1},
                })
            selection = {
                "format": "reconviagen.proobjaverse_paired_combined.v1",
                "data_root": str(root),
                "repository": {"repo_id": "fixture", "revision": "one"},
                "pair_count": 4,
                "target_contract": {
                    "trellis2_shape_slat_compatible": False,
                },
                "selected": rows,
            }
            unicode_body = json.dumps(
                selection,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            selection["combined_selection_sha256"] = hashlib.sha256(
                unicode_body
            ).hexdigest()
            selection_path = root / "selection.json"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            ordered = sorted(
                rows,
                key=lambda row: hashlib.sha256(
                    f"7:{row['uid']}".encode("ascii")
                ).hexdigest(),
            )
            protocol = {
                "format": (
                    "pose_point_depth_mv.proobjaverse_official_slat_protocol.v1"
                ),
                "selection_file_sha256": hashlib.sha256(
                    selection_path.read_bytes()
                ).hexdigest(),
                "combined_selection_sha256": selection[
                    "combined_selection_sha256"
                ],
                "pair_count": 4,
                "seed": 7,
                "split_counts": {
                    "decoder_audit": 1,
                    "predicted_support_bridge": 1,
                    "dev": 1,
                    "train": 1,
                },
                "split_uid_sha256": {
                    name: canonical_json_sha256([ordered[index]["uid"]])
                    for index, name in enumerate((
                        "decoder_audit",
                        "predicted_support_bridge",
                        "dev",
                        "train",
                    ))
                },
            }
            protocol["protocol_sha256"] = canonical_json_sha256(protocol)
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            selected, selected_root, identity = load_frozen_rows(
                selection_path,
                protocol_path=protocol_path,
                data_root=None,
                split="train",
                max_samples=0,
            )
            self.assertEqual([selected[0]["uid"]], [ordered[3]["uid"]])
            self.assertEqual(selected_root, root.resolve())
            self.assertEqual(identity["selected_count"], 1)

            uid = selected[0]["uid"]
            tar_path = root / "one.tar"
            rgba = np.zeros((8, 8, 4), dtype=np.uint8)
            rgba[2:6, 2:6, :3] = 127
            rgba[2:6, 2:6, 3] = 255
            image_buffer = io.BytesIO()
            Image.fromarray(rgba, mode="RGBA").save(image_buffer, format="WEBP")
            metadata = {
                "image_index": 3,
                "intrinsic": [[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0, 0, 1]],
                "extrinsic": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 2],
                    [0, 0, 0, 1],
                ],
            }
            with tarfile.open(tar_path, "w") as archive:
                for name, value in (
                    (f"{uid}/003.json", json.dumps(metadata).encode("utf-8")),
                    (f"{uid}/003.rgba.webp", image_buffer.getvalue()),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(value)
                    archive.addfile(info, io.BytesIO(value))
            views, audit = load_render_views(tar_path, uid)
            self.assertEqual([view["id"] for view in views], [3])
            self.assertTrue(audit["archive_complete"])
            self.assertEqual(float(views[0]["extrinsic"][2, 3]), -2.0)
            self.assertEqual(camera_forward_sign(np.stack([
                views[0]["extrinsic"]
            ])), -1.0)
            self.assertEqual(pose_diverse_indices(views, 1, uid), [0])


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_keeps_per_rank_rng_and_hardlink_alias(self) -> None:
        class Scaler:
            @staticmethod
            def state_dict() -> dict[str, object]:
                return {}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            checkpoint = root / "step_000001.pt"
            rng_by_rank = [{"rank": 0}, {"rank": 1}]
            save_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scaler=Scaler(),
                ema=initialize_ema(model),
                stage="ss",
                step=1,
                micro_step=1,
                args={},
                data_identity={},
                model_identity={},
                history=[],
                rng_by_rank=rng_by_rank,
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["rng_by_rank"], rng_by_rank)
            alias = root / "last.pt"
            update_checkpoint_alias(checkpoint, alias)
            self.assertEqual(checkpoint.stat().st_ino, alias.stat().st_ino)


class ToolkitCompatibilityTests(unittest.TestCase):
    def test_metadata_is_filtered_and_reordered_by_frozen_uid_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requested = ["b" * 64, "a" * 64]
            instances = root / "instances.txt"
            instances.write_text("".join(f"{uid}\n" for uid in requested))
            sketchfab = pd.DataFrame(
                {
                    "sha256": ["a" * 64],
                    "file_identifier": ["sketchfab-a"],
                }
            )
            github = pd.DataFrame(
                {
                    "sha256": ["b" * 64],
                    "file_identifier": ["github-b"],
                }
            )
            with mock.patch.object(
                ObjaverseXL.pd,
                "read_csv",
                side_effect=[sketchfab, github],
            ):
                metadata = ObjaverseXL.get_metadata(
                    source="all", metadata_instances=str(instances)
                )
            self.assertEqual(metadata["sha256"].tolist(), requested)

    def test_toolkit_adapter_handles_zip_member_and_no_file_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "raw" / "github" / "repos" / "org" / "repo.zip"
            archive.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("models/object.glb", b"mesh-bytes")
            resolved_root = root / "resolved"
            resolved_root.mkdir()
            with tempfile.TemporaryDirectory(dir=resolved_root) as extraction:
                path = ObjaverseXL._resolved_file(
                    root,
                    "raw/github/repos/org/repo.zip/models/object.glb",
                    Path(extraction),
                )
                self.assertEqual(path.read_bytes(), b"mesh-bytes")

            row = {"sha256": "c" * 64}

            def callback(file: object, metadatum: dict[str, str]) -> dict[str, object]:
                return {
                    "sha256": metadatum["sha256"],
                    "file_is_none": file is None,
                }

            record = ObjaverseXL._process_instance((row, None, callback, True))
            self.assertTrue(record["file_is_none"])


if __name__ == "__main__":
    unittest.main()
