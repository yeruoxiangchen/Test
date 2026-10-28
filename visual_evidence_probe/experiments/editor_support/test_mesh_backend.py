"""Small CPU checks; no checkpoint/model load, GPU use, or output files."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch

from . import mesh_backend as backend


ROOT = Path(__file__).resolve().parents[3]
STOCK = ROOT / "ReconViaGen/trellis/representations/mesh"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _EDict(dict):
    def __getattr__(self, name):
        return self[name]


class _MeshResult:
    def __init__(self, vertices, faces, vertex_attrs=None, res=64):
        self.vertices, self.faces = vertices, faces.long()
        self.vertex_attrs, self.res = vertex_attrs, res
        self.success = bool(vertices.shape[0] and faces.shape[0])


def _cpu_stock():
    """Load stock geometry source without importing the sparse GPU package."""
    utils = _load("_editor_cpu_utils", STOCK / "utils_cube.py")
    package = types.ModuleType("_editor_cpu_mesh")
    package.__path__ = [str(STOCK)]
    with mock.patch.dict(sys.modules, {package.__name__: package}):
        flexi = _load(package.__name__ + ".flexicube", STOCK / "flexicube.py")
    # Compile the unchanged extractor class, replacing only unrelated imports
    # and the mesh-result container (normals do not feed geometry extraction).
    tree = ast.parse((STOCK / "cube2mesh.py").read_text())
    definition = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == "SparseFeatures2Mesh")
    namespace = types.ModuleType("_editor_cpu_cube2mesh")
    namespace.__dict__.update(utils.__dict__)
    namespace.__dict__.update(SparseTensor=object, edict=_EDict,
                              FlexiCubes=flexi.FlexiCubes, MeshExtractResult=_MeshResult)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(STOCK / "cube2mesh.py"), "exec"),
         namespace.__dict__)
    return utils, namespace


class ScatterTests(unittest.TestCase):
    def setUp(self):
        self.utils = _load("_editor_original_utils", STOCK / "utils_cube.py")
        self.cubes = torch.tensor([[0, 1, 1, 2, 4, 5, 6, 7],
                                   [1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int64)
        self.value = torch.randn(2, 8, 7, generator=torch.Generator().manual_seed(23))

    def test_forward_and_analytic_gradient_match_original(self):
        reference = self.value.clone().requires_grad_()
        target = self.utils.cubes_to_verts(11, self.cubes, reference)
        weights = torch.randn(target.shape, generator=torch.Generator().manual_seed(24))
        expected = torch.autograd.grad((target.sin() * weights).sum(), reference)[0]
        for chunk in (1, 2, 5, 7, 11):
            value = self.value.clone().requires_grad_()
            observed = backend.differentiable_cubes_to_verts(11, self.cubes, value,
                                                            chunk_channels=chunk)
            torch.testing.assert_close(observed, target, rtol=0, atol=0)
            derivative = torch.autograd.grad((observed.sin() * weights).sum(), value)[0]
            torch.testing.assert_close(derivative, expected, rtol=0, atol=0)
            self.assertEqual(observed[9:].count_nonzero().item(), 0)

    def test_empty_inputs_and_unused_vertices(self):
        value = torch.empty(0, 8, 3, requires_grad=True)
        result = backend.differentiable_cubes_to_verts(2, torch.empty(0, 8, dtype=torch.int64), value)
        torch.testing.assert_close(result, torch.zeros(2, 3), rtol=0, atol=0)
        self.assertEqual(torch.autograd.grad(result.sum(), value)[0].shape, value.shape)

    def test_gradcheck_and_second_derivatives(self):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            value = self.value.double().requires_grad_()
            function = lambda x: backend.differentiable_cubes_to_verts(
                11, self.cubes, x, chunk_channels=3)
            self.assertTrue(torch.autograd.gradcheck(function, (value,), fast_mode=True))
            self.assertTrue(torch.autograd.gradgradcheck(function, (value,), fast_mode=True))
        finally:
            torch.set_default_dtype(previous)

    def test_rejects_silent_precision_and_reduction_changes(self):
        with self.assertRaises(ValueError):
            backend.differentiable_cubes_to_verts(11, self.cubes, self.value.half())
        with self.assertRaises(ValueError):
            backend.differentiable_cubes_to_verts(11, self.cubes, self.value, reduce="sum")


class StockGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.utils, cls.mesh = _cpu_stock()

    def _inputs(self):
        generator = torch.Generator().manual_seed(43)
        coords = torch.cartesian_prod(*(torch.arange(3) for _ in range(3)))
        positions = coords[:, None, :] + self.utils.cube_corners.long()[None]
        sdf = (positions.float() - 1.5).norm(dim=-1) - 1.05
        sdf = sdf + 1 / 3  # stock extractor adds its -1/res bias
        features = torch.cat((sdf, torch.randn(27, 24, generator=generator) * .1,
                              torch.randn(27, 21, generator=generator) * .2,
                              torch.randn(27, 48, generator=generator) * .1), dim=-1)
        batch_coords = torch.cat((torch.zeros(27, 1, dtype=torch.int64), coords), dim=-1)
        return batch_coords, features

    def _compare(self, training):
        coords, features = self._inputs()
        extractor = self.mesh.SparseFeatures2Mesh(device="cpu", res=3, use_color=True)
        reference_leaf = features.clone().requires_grad_()
        # Original in-place SDF bias cannot operate directly on a leaf view.
        reference = extractor(types.SimpleNamespace(coords=coords, feats=reference_leaf * 1),
                              training=training)
        generator = torch.Generator().manual_seed(44)
        vertex_weights = torch.randn(reference.vertices.shape, generator=generator)
        color_weights = torch.randn(reference.vertex_attrs.shape, generator=generator)
        loss = (reference.vertices * vertex_weights).sum() + (reference.vertex_attrs * color_weights).sum()
        expected_gradient = torch.autograd.grad(loss, reference_leaf)[0]
        self.assertTrue(reference.success)
        original_reducer = self.utils.cubes_to_verts
        try:
            self.utils.cubes_to_verts = backend.differentiable_cubes_to_verts
            observed_leaf = features.clone().requires_grad_()
            observed = backend._safe_features_to_mesh(
                extractor, types.SimpleNamespace(coords=coords, feats=observed_leaf),
                training=training, _mesh_module=self.mesh)
            torch.testing.assert_close(observed_leaf, features, rtol=0, atol=0)
            torch.testing.assert_close(observed.vertices, reference.vertices, rtol=0, atol=0)
            torch.testing.assert_close(observed.faces, reference.faces, rtol=0, atol=0)
            torch.testing.assert_close(observed.vertex_attrs, reference.vertex_attrs, rtol=0, atol=0)
            observed_loss = ((observed.vertices * vertex_weights).sum()
                             + (observed.vertex_attrs * color_weights).sum())
            observed_gradient = torch.autograd.grad(observed_loss, observed_leaf)[0]
            torch.testing.assert_close(observed_gradient, expected_gradient, rtol=0, atol=0)
        finally:
            self.utils.cubes_to_verts = original_reducer
        for start, stop in ((0, 8), (8, 32), (32, 44), (44, 52), (53, 101)):
            self.assertGreater(observed_gradient[:, start:stop].abs().sum().item(), 0)
        gamma_magnitude = observed_gradient[:, 52].abs().sum().item()
        if training:
            self.assertGreater(gamma_magnitude, 0)
        else:
            self.assertEqual(gamma_magnitude, 0)

    def test_complete_eval_geometry_color_and_gradient_parity(self):
        self._compare(training=False)

    def test_training_gamma_gradient_and_geometry_parity(self):
        self._compare(training=True)

    def test_context_restores_preexisting_bindings_and_exception(self):
        old_reducer, old_call = self.utils.cubes_to_verts, self.mesh.SparseFeatures2Mesh.__call__
        with mock.patch.object(backend, "_mesh_modules", return_value=(self.utils, self.mesh)):
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                with backend.differentiable_mesh_backend():
                    outer = self.utils.cubes_to_verts
                    with backend.differentiable_mesh_backend(chunk_channels=2):
                        self.assertIsNot(self.utils.cubes_to_verts, outer)
                    self.assertIs(self.utils.cubes_to_verts, outer)
                    raise RuntimeError("deliberate")
        self.assertIs(self.utils.cubes_to_verts, old_reducer)
        self.assertIs(self.mesh.SparseFeatures2Mesh.__call__, old_call)

    def test_tensor_decode_bypasses_old_inference_wrapper_and_preserves_gradient(self):
        extractor = self.mesh.SparseFeatures2Mesh(device="cpu", res=3, use_color=True)

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mesh_extractor = extractor

            def forward(self, value):
                return [self.mesh_extractor(value, training=self.training)]

        decoder = Decoder().eval()

        def historical_endpoint(*args, **kwargs):
            raise RuntimeError("historical endpoint must not be called with gradients")

        decoder.forward = historical_endpoint
        coords, features = self._inputs()
        features.requires_grad_()
        with mock.patch.object(backend, "_mesh_modules", return_value=(self.utils, self.mesh)):
            vertices, faces = backend.decode_tensor_mesh(
                decoder, types.SimpleNamespace(coords=coords, feats=features))
        self.assertTrue(vertices.requires_grad)
        self.assertGreater(faces.shape[0], 0)
        self.assertGreater(torch.autograd.grad(vertices.square().sum(), features)[0].abs().sum().item(), 0)
        self.assertIs(decoder.forward, historical_endpoint)
        decoder.train()
        with self.assertRaisesRegex(ValueError, "triangulation"):
            backend.decode_tensor_mesh(decoder, None)


class RuntimePolicyTests(unittest.TestCase):
    def test_full_native_tuner_covers_forward_and_backward_scope(self):
        class FixedNative:
            def __enter__(self):
                def fixed():
                    return self
                return fixed

        policy = FixedNative()
        policy.original, policy.old_algo, policy.old_ops = (lambda: None), object(), object()
        fixed = policy.__enter__()
        ops = types.SimpleNamespace(indice_conv=fixed, GEMM_CPP=object())
        algo = types.SimpleNamespace(GEMM_CPP=object())
        previous = (ops.indice_conv, algo.GEMM_CPP, ops.GEMM_CPP)
        with mock.patch.dict(sys.modules, {"spconv.pytorch.ops": ops, "spconv.algo": algo}):
            with self.assertRaisesRegex(RuntimeError, "inference-only"):
                backend._ensure_full_native_tuner()
            with backend.full_native_tuner() as binding:
                self.assertEqual(binding["unwrapped_fixed_native"], 1)
                self.assertIs(ops.indice_conv, policy.original)
                self.assertIs(algo.GEMM_CPP, policy.old_algo)
                self.assertIs(ops.GEMM_CPP, policy.old_ops)
                backend._ensure_full_native_tuner()
            self.assertEqual((ops.indice_conv, algo.GEMM_CPP, ops.GEMM_CPP), previous)

    def test_checkpoint_preserves_tensor_forward_and_latent_gradient(self):
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.use_checkpoint = False
                self.upsample = torch.nn.ModuleList([torch.nn.Linear(3, 3)])

        decoder = Decoder().eval().requires_grad_(False)
        value = torch.randn(2, 3, requires_grad=True)
        reference = decoder.upsample[0](value)
        expected = torch.autograd.grad(reference.square().sum(), value)[0]
        backend.enable_checkpoint(decoder)
        wrapper = decoder.upsample[0].forward
        backend.enable_checkpoint(decoder)
        self.assertIs(decoder.upsample[0].forward, wrapper)
        observed = decoder.upsample[0](value)
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        derivative = torch.autograd.grad(observed.square().sum(), value)[0]
        torch.testing.assert_close(derivative, expected, rtol=0, atol=0)
        self.assertFalse(decoder.training)


if __name__ == "__main__":
    unittest.main()
