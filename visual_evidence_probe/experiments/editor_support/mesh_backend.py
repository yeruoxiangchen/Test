"""Reversible, differentiable access to the frozen stock TRELLIS mesh decoder.

No installed source, model parameter, precision, topology rule, deformation,
FlexiCubes weight, or color channel is changed.  The scatter forward retains
the original ``mean, include_self=False`` operation and contribution order;
its analytic backward only needs vertex indices and incidence counts.

Evaluation triangulation is differentiable in SDF, deformation, alpha and
beta while topology stays fixed.  Gamma still selects the original discrete
diagonal, so eval vertices have no continuous gamma derivative.  Setting the
decoder to train changes triangulation and is NOT an inference parity path.

CUDA forward parity and sparse-convolution backward must be checked on the
actual frozen latent separately; the CPU tests are not a CUDA certificate.
All imports of the stock GPU implementation are lazy.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial
import importlib
import inspect
import sys
from threading import RLock

import torch


_PATCH_LOCK = RLock()


def _validate_mean_inputs(num_verts, cubes, value, reduce, chunk_channels):
    if reduce != "mean":
        raise ValueError("differentiable mesh reduction only supports mean")
    if (not isinstance(num_verts, int) or isinstance(num_verts, bool) or num_verts < 0
            or not isinstance(chunk_channels, int) or isinstance(chunk_channels, bool)
            or chunk_channels < 1):
        raise ValueError("num_verts must be nonnegative and chunk_channels positive integers")
    if (cubes.ndim != 2 or value.ndim != 3 or cubes.shape[1] != 8
            or tuple(cubes.shape) != tuple(value.shape[:2]) or value.shape[2] < 1
            or cubes.dtype != torch.int64 or cubes.device != value.device):
        raise ValueError("expected int64 indices [N,8] and matching features [N,8,C]")
    if value.dtype != torch.get_default_dtype():
        raise ValueError("features must match the stock default-dtype scatter tensor")


class _CubeMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, num_verts, cubes, value, chunk_channels):
        channels = value.shape[-1]
        output = torch.zeros(num_verts, channels, device=cubes.device)
        # Do not row-chunk or sum first and divide afterward: retain the stock
        # operation, including the untouched zero rows and floating-point sum.
        for start in range(0, channels, chunk_channels):
            stop = min(start + chunk_channels, channels)
            width = stop - start
            initial = torch.zeros(num_verts, width, device=cubes.device)
            index = cubes.unsqueeze(-1).expand(-1, -1, width).flatten(0, 1)
            source = value[..., start:stop].flatten(0, 1)
            output[:, start:stop] = torch.scatter_reduce(
                initial, 0, index, source, reduce="mean", include_self=False)
        if ctx.needs_input_grad[2]:
            index = cubes.reshape(-1)
            counts = torch.zeros(num_verts, dtype=torch.int64, device=cubes.device)
            counts.scatter_add_(0, index, torch.ones_like(index))
            ctx.save_for_backward(cubes, counts)
            ctx.value_shape = tuple(value.shape)
            ctx.chunk_channels = chunk_channels
        return output

    @staticmethod
    def backward(ctx, grad_output):
        cubes, counts = ctx.saved_tensors
        index = cubes.reshape(-1)
        # Every source contribution sees the same derivative of its vertex
        # mean.  Unreferenced output vertices therefore cannot affect inputs.
        denominator = counts.index_select(0, index).to(grad_output.dtype).unsqueeze(-1)
        gradient = grad_output.new_empty(ctx.value_shape)
        flat_gradient = gradient.reshape(-1, ctx.value_shape[-1])
        for start in range(0, ctx.value_shape[-1], ctx.chunk_channels):
            stop = min(start + ctx.chunk_channels, ctx.value_shape[-1])
            flat_gradient[:, start:stop] = (
                grad_output[:, start:stop].index_select(0, index) / denominator)
        return None, None, gradient, None


def differentiable_cubes_to_verts(num_verts, cubes, value, reduce="mean", *, chunk_channels=1):
    """Original channelwise scatter forward with a compact analytic backward.

    Unlike the historical inference helper, this accepts a grad-bearing value.
    The stock implementation only accepts the current default floating dtype;
    reject mismatches instead of silently changing model precision.
    """
    _validate_mean_inputs(num_verts, cubes, value, reduce, chunk_channels)
    return _CubeMean.apply(num_verts, cubes, value, chunk_channels)


def _mesh_modules():
    return (importlib.import_module("trellis.representations.mesh.utils_cube"),
            importlib.import_module("trellis.representations.mesh.cube2mesh"))


def _safe_features_to_mesh(self, cubefeats, training=False, *, _mesh_module=None):
    """Stock extractor operations, with out-of-place SDF bias on its view.

    Keep the complete color path.  Its removal is a separate optimization
    requiring actual decoded geometry parity, not assumed here.
    """
    mesh_module = _mesh_module if _mesh_module is not None else _mesh_modules()[1]
    coords, feats = cubefeats.coords[:, 1:], cubefeats.feats
    sdf, deform, color, weights = [
        self.get_layout(feats, name) for name in ("sdf", "deform", "color", "weights")]
    sdf = sdf + self.sdf_bias
    attributes = [sdf, deform, color] if self.use_color else [sdf, deform]
    v_pos, v_attrs, reg_loss = mesh_module.sparse_cube2verts(
        coords, torch.cat(attributes, dim=-1), training=training)
    v_attrs_d = mesh_module.get_dense_attrs(v_pos, v_attrs, res=self.res + 1, sdf_init=True)
    weights_d = mesh_module.get_dense_attrs(coords, weights, res=self.res, sdf_init=False)
    sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
    colors_d = v_attrs_d[..., 4:] if self.use_color else None
    x_nx3 = mesh_module.get_defomed_verts(self.reg_v, deform_d, self.res)
    vertices, faces, L_dev, colors = self.mesh_extractor(
        voxelgrid_vertices=x_nx3, scalar_field=sdf_d, cube_idx=self.reg_c,
        resolution=self.res, beta=weights_d[:, :12], alpha=weights_d[:, 12:20],
        gamma_f=weights_d[:, 20], voxelgrid_colors=colors_d, training=training)
    mesh = mesh_module.MeshExtractResult(
        vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res)
    if training:
        if mesh.success:
            reg_loss = reg_loss + L_dev.mean() * 0.5
        reg_loss = reg_loss + weights[:, :20].abs().mean() * 0.2
        mesh.reg_loss = reg_loss
        mesh.tsdf_v = mesh_module.get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
        mesh.tsdf_s = v_attrs[:, 0]
    return mesh


@contextmanager
def differentiable_mesh_backend(decoder=None, *, chunk_channels=1):
    """Patch both reduction bindings and SDF bias only inside this process.

    Nesting restores the exact previous functions, including a pre-existing
    inference reduction patch.  The lock serializes contexts created here;
    callers must not concurrently decode outside these contexts in one process.
    Model weights and train/eval mode are untouched.
    """
    if not isinstance(chunk_channels, int) or isinstance(chunk_channels, bool) or chunk_channels < 1:
        raise ValueError("chunk_channels must be a positive integer")
    with _PATCH_LOCK:
        utils, mesh = _mesh_modules()
        extractor_class = mesh.SparseFeatures2Mesh
        if decoder is not None and not isinstance(decoder.mesh_extractor, extractor_class):
            raise TypeError("this backend requires the stock FlexiCubes SparseFeatures2Mesh extractor")
        if mesh.sparse_cube2verts.__globals__ is not utils.__dict__:
            raise RuntimeError("stock sparse_cube2verts no longer resolves utils_cube globals")
        old_utils, old_mesh = utils.cubes_to_verts, mesh.cubes_to_verts
        old_call = extractor_class.__call__
        patched = partial(differentiable_cubes_to_verts, chunk_channels=chunk_channels)
        utils.cubes_to_verts = mesh.cubes_to_verts = patched
        extractor_class.__call__ = _safe_features_to_mesh
        try:
            yield {"reduce": "mean", "include_self": False,
                   "chunk_channels": chunk_channels, "analytic_backward": True,
                   "color_enabled": bool(decoder.mesh_extractor.use_color) if decoder is not None else None,
                   "geometry_only": False, "weights_unchanged": True,
                   "sdf_bias_out_of_place": True}
        finally:
            utils.cubes_to_verts, mesh.cubes_to_verts = old_utils, old_mesh
            extractor_class.__call__ = old_call


def _fixed_policy(function):
    if not inspect.isfunction(function) or "FixedNative.__enter__." not in function.__qualname__:
        return None
    policy = inspect.getclosurevars(function).nonlocals.get("self")
    if policy is None or not all(hasattr(policy, key) for key in ("original", "old_algo", "old_ops")):
        raise RuntimeError("could not recover the active FixedNative policy")
    return policy


def _ensure_full_native_tuner():
    ops = sys.modules.get("spconv.pytorch.ops")
    if torch.is_grad_enabled() and ops is not None and _fixed_policy(ops.indice_conv) is not None:
        raise RuntimeError("FixedNative is inference-only: leave its context before mesh gradients, "
                           "or wrap BOTH decode and backward in full_native_tuner()")


@contextmanager
def full_native_tuner():
    """Temporarily unwrap active FixedNative policies for forward AND backward.

    Normally end FixedNative before optimization and this is unnecessary.
    If nesting is unavoidable, keep this context open until backward finishes;
    spconv backward looks up the process-global full GEMM tuner at that time.
    """
    with _PATCH_LOCK:
        ops, algo = sys.modules.get("spconv.pytorch.ops"), sys.modules.get("spconv.algo")
        if ops is None or algo is None:
            yield {"unwrapped_fixed_native": 0}
            return
        previous = ops.indice_conv, algo.GEMM_CPP, ops.GEMM_CPP
        function, algo_tuner, ops_tuner = previous
        depth = 0
        while (policy := _fixed_policy(function)) is not None:
            function, algo_tuner, ops_tuner = policy.original, policy.old_algo, policy.old_ops
            depth += 1
        if depth:
            ops.indice_conv, algo.GEMM_CPP, ops.GEMM_CPP = function, algo_tuner, ops_tuner
        try:
            yield {"unwrapped_fixed_native": depth}
        finally:
            if depth:
                ops.indice_conv, algo.GEMM_CPP, ops.GEMM_CPP = previous


def enable_checkpoint(decoder, *, checkpoint_upsample=True):
    """Enable stock non-reentrant transformer and optional upsample checkpoints.

    Parameter values, precision and eval mode remain unchanged.  Upsample
    wrappers are idempotent and run their original function directly in no_grad.
    """
    from torch.utils.checkpoint import checkpoint
    transformer = []
    for name, module in decoder.named_modules():
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = True
            transformer.append(name or "<decoder>")
    upsample = []
    if checkpoint_upsample:
        for index, block in enumerate(decoder.upsample):
            if not getattr(block, "_editor_mesh_checkpoint", False):
                original = block.forward

                def wrapped(*args, _original=original, **kwargs):
                    if not torch.is_grad_enabled():
                        return _original(*args, **kwargs)
                    return checkpoint(_original, *args, use_reentrant=False, **kwargs)

                block.forward = wrapped
                block._editor_mesh_checkpoint = True
            upsample.append(index)
    return {"stock_checkpoint_modules": transformer, "upsample_checkpoint_blocks": upsample,
            "use_reentrant": False}


def decode_tensor_mesh(decoder, z, *, batch_index=0, chunk_channels=1):
    """Return stock eval mesh ``(vertices, faces)`` with latent gradients.

    The class forward bypasses the historical instance-level inference-only
    endpoint wrapper.  Decoder training mode is rejected because it changes
    triangulation.  Do not put this call or its backward in FixedNative.
    """
    if decoder.training:
        raise ValueError("call decoder.eval(): training triangulation changes the stock inference mesh")
    _ensure_full_native_tuner()
    with differentiable_mesh_backend(decoder, chunk_channels=chunk_channels):
        meshes = type(decoder).forward(decoder, z)
    if not 0 <= batch_index < len(meshes):
        raise IndexError("mesh batch_index is out of range")
    mesh = meshes[batch_index]
    return mesh.vertices, mesh.faces


__all__ = ["differentiable_cubes_to_verts", "differentiable_mesh_backend",
           "decode_tensor_mesh", "enable_checkpoint", "full_native_tuner"]
