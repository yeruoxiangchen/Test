"""Explicit runtime-only channel chunking for the frozen mesh decoder.

Each channel retains the original scatter_reduce(mean, include_self=False)
and the complete ordered set of cube-corner contributions. No rows, geometry,
weights, dtypes, resolutions, or deterministic settings are changed. CUDA
bitwise equivalence must be validated in the actual runtime before deployment;
CPU equality alone is not a CUDA equivalence certificate.
"""
from __future__ import annotations

from functools import partial
import importlib
import importlib.util
from pathlib import Path

import torch

from ..._bootstrap import TRACKER_ROOT, bootstrap_external_paths
from ...paired_dataset import sha256_file


ORIGINAL_SOURCE_SHA256 = {
    'ReconViaGen/trellis/representations/mesh/utils_cube.py':
        '3153e11dab5f0794caf3541ec6ba8b06e041dac26e4193ecf821d4e146336c35',
    'ReconViaGen/trellis/representations/mesh/cube2mesh.py':
        '18e69140f1421a1ee6d2709d6313b5b996a4f25078fcf99f79669dc8c5065060',
    'ReconViaGen/trellis/models/structured_latent_vae/decoder_mesh.py':
        '4c2a399f5b42037222462560eec17b3ac2cc299914db8ae95b5b742986ebbc8b',
}
TORCH_GIT_VERSION = 'e4ee3be4063b7c430974252fdf7db42273388d86'
_installation = None


def original_source_bindings():
    result = {}
    for relative, expected in ORIGINAL_SOURCE_SHA256.items():
        path = TRACKER_ROOT / relative
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError('unvalidated original mesh implementation: ' + str(path))
        result[relative] = {'path': str(path), 'sha256': observed}
    return result


def load_original_cubes_to_verts():
    """Read the exact original function without importing GPU model classes."""
    bindings = original_source_bindings()
    path = bindings['ReconViaGen/trellis/representations/mesh/utils_cube.py']['path']
    spec = importlib.util.spec_from_file_location('_palr_original_utils_cube', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cubes_to_verts


def channelwise_cubes_to_verts(num_verts, cubes, value, reduce='mean', *, chunk_channels=1):
    """Bound scatter workspace by processing independent feature channels.

The original initializes a default-dtype zero tensor. Preserve that behavior;
do not implicitly cast a mismatched input or split a vertex's contributions.
This helper is deliberately inference-only and restricted to the used mean.
"""
    if reduce != 'mean' or not isinstance(chunk_channels, int) or chunk_channels < 1:
        raise ValueError('validated mesh reduction requires mean and a positive integer channel chunk')
    if (cubes.ndim != 2 or value.ndim != 3 or tuple(cubes.shape) != tuple(value.shape[:2])
            or cubes.shape[1] != 8 or cubes.dtype != torch.int64
            or cubes.device != value.device or value.shape[2] < 1):
        raise ValueError('expected int64 cube indices [N,8] and matching features [N,8,C]')
    if value.dtype != torch.get_default_dtype():
        raise ValueError('feature dtype must match the original default-dtype reduction tensor')
    if torch.is_grad_enabled() and value.requires_grad:
        raise ValueError('runtime mesh memory optimization is inference-only')
    if value.is_cuda and not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError('the frozen decoder requires deterministic CUDA reduction')
    channels = value.shape[2]
    result = torch.zeros(num_verts, channels, device=cubes.device)
    for start in range(0, channels, chunk_channels):
        stop = min(start + chunk_channels, channels)
        width = stop - start
        initial = torch.zeros(num_verts, width, device=cubes.device)
        index = cubes.unsqueeze(-1).expand(-1, -1, width).flatten(0, 1)
        source = value[..., start:stop].flatten(0, 1)
        reduced = torch.scatter_reduce(initial, 0, index, source,
                                       reduce=reduce, include_self=False)
        result[:, start:stop] = reduced
        del initial, index, source, reduced
    return result


def install_memory_efficient_mesh(chunk_channels=1):
    """Install only the checked reduction function; return auditable bindings."""
    global _installation
    if not isinstance(chunk_channels, int) or chunk_channels < 1:
        raise ValueError('chunk_channels must be a positive integer')
    if torch.__version__.split('+')[0] != '2.4.0' or torch.version.git_version != TORCH_GIT_VERSION:
        raise RuntimeError('runtime patch has only been reviewed for the installed PyTorch 2.4.0 build')
    sources = original_source_bindings()
    patch_binding = {'path': str(Path(__file__).resolve()), 'sha256': sha256_file(__file__)}
    bootstrap_external_paths()
    utils = importlib.import_module('trellis.representations.mesh.utils_cube')
    mesh = importlib.import_module('trellis.representations.mesh.cube2mesh')
    if _installation is not None:
        patched, binding = _installation
        if (binding['chunk_channels'] != chunk_channels or binding['patch'] != patch_binding
                or utils.cubes_to_verts is not patched or mesh.cubes_to_verts is not patched
                or mesh.sparse_cube2verts.__globals__['cubes_to_verts'] is not patched):
            raise RuntimeError('another mesh reduction patch changed this process')
        return binding
    original = utils.cubes_to_verts
    if (Path(original.__code__.co_filename).resolve() != Path(utils.__file__).resolve()
            or original.__name__ != 'cubes_to_verts'
            or mesh.cubes_to_verts is not original
            or mesh.sparse_cube2verts is not utils.sparse_cube2verts
            or mesh.sparse_cube2verts.__globals__['cubes_to_verts'] is not original):
        raise RuntimeError('original mesh reduction call path was already modified')
    patched = partial(channelwise_cubes_to_verts, chunk_channels=chunk_channels)
    utils.cubes_to_verts = patched
    mesh.cubes_to_verts = patched
    if mesh.sparse_cube2verts.__globals__['cubes_to_verts'] is not patched:
        raise RuntimeError('runtime patch did not reach the decoder call path')
    binding = {'kind': 'channelwise_original_scatter_reduce_mean_v1',
        'chunk_channels': chunk_channels, 'original_sources': sources, 'patch': patch_binding,
        'torch': torch.__version__, 'torch_git_version': torch.version.git_version,
        'reduce': 'mean', 'include_self': False, 'row_chunking': False,
        'dtype_or_geometry_changed': False, 'cuda_deterministic_required': True,
        'call_path_checked': True, 'cuda_equivalence_requires_separate_validation': True}
    _installation = (patched, binding)
    return binding


__all__ = ['channelwise_cubes_to_verts', 'install_memory_efficient_mesh',
           'load_original_cubes_to_verts', 'original_source_bindings']
