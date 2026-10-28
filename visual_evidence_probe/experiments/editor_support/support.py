"""Deterministic, input-only coarse64 support expansion.

This module only plans coordinates and interpolates features.  It does not claim
that adding tokens preserves the decoded mesh; that requires a separate full
decoder initialization check.  Importing the NumPy planner needs no TRELLIS or
CUDA dependencies.
"""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np


RESOLUTION = 64
PATH_FRACTIONS = (1.0, 0.5, 0.0)


def _points(value, name):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'{name} must have shape [N, 3]')
    return points


def _inside(points):
    # The upper boundary belongs to cell64, outside the coarse64 support.
    return np.isfinite(points).all(axis=1) & (points >= -.5).all(axis=1) & (points < .5).all(axis=1)


def _lex_order(coords):
    return np.lexsort((coords[:, 3], coords[:, 2], coords[:, 1], coords[:, 0]))


def _cells(points):
    # Widen before translating: float64's nearest value below .5 must remain
    # cell63, rather than rounding to 1.0 during the addition.  This is
    # quantization of already validated coordinates, never boundary clipping.
    return np.floor((np.asarray(points, dtype=np.longdouble) + np.longdouble(.5)) * RESOLUTION).astype(np.int64)


def plan_expansion(coords, features, source, targets, confidence,
                   max_added_tokens=128, max_fraction=.2):
    """Return ``(coords, features, added_mask, receipt)`` as CPU NumPy arrays.

    Inputs are batch0 ``coords[Nold,4]``, floating ``features[Nold,8]`` and
    matching point/confidence arrays.  Every valid target is considered before
    path points.  Within each stage, confidence descending and original point
    order break ties; path points run midpoint then source.  The complete
    candidate set is deduplicated before applying the dual token budget.

    Invalid/out-of-cube targets remain in receipt diagnostics and generate no
    candidates, including along their paths.  No input point is clipped.  The
    three uniformly spaced segment samples have t=(1, .5, 0).
    """
    coords = np.asarray(coords)
    features = np.asarray(features)
    if coords.ndim != 2 or coords.shape[1] != 4 or not np.issubdtype(coords.dtype, np.integer):
        raise ValueError('coords must be integer [Nold, 4]')
    if not len(coords):
        raise ValueError('original support must be nonempty')
    if (coords[:, 0] != 0).any() or (coords[:, 1:] < 0).any() or (coords[:, 1:] >= RESOLUTION).any():
        raise ValueError('original coordinates must be batch0 coarse64 cells')
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError('original coordinates must be unique')
    if features.shape != (len(coords), 8) or not np.issubdtype(features.dtype, np.floating):
        raise ValueError('features must be floating [Nold, 8]')
    if not np.isfinite(features).all():
        raise ValueError('original features must be finite')
    source, targets = _points(source, 'source'), _points(targets, 'targets')
    confidence = np.asarray(confidence, dtype=np.float64)
    if source.shape != targets.shape or confidence.shape != (len(targets),):
        raise ValueError('source, targets and confidence must have matching point counts')
    if isinstance(max_added_tokens, (bool, np.bool_)) or not isinstance(max_added_tokens, Integral) or max_added_tokens < 0:
        raise ValueError('max_added_tokens must be a nonnegative integer')
    if not np.isscalar(max_fraction) or not np.isfinite(max_fraction) or not 0 <= max_fraction <= 1:
        raise ValueError('max_fraction must be finite and between zero and one')

    budget = min(int(max_added_tokens), math.floor(float(max_fraction) * len(coords)))
    finite_targets = np.isfinite(targets).all(axis=1)
    valid_targets = _inside(targets)
    valid_confidence = np.isfinite(confidence)
    eligible = valid_targets & valid_confidence
    eligible_ids = np.flatnonzero(eligible)
    ranking = eligible_ids[np.argsort(-confidence[eligible_ids], kind='stable')]
    original_cells = {tuple(int(v) for v in row[1:]) for row in coords}
    candidate_cells, candidate_inputs, candidate_kind = [], [], []
    seen_candidates = set()
    raw_candidates, original_hits, duplicate_hits, invalid_path_samples = 0, 0, 0, 0

    def add(point, point_id, kind):
        nonlocal raw_candidates, original_hits, duplicate_hits, invalid_path_samples
        if not _inside(np.asarray(point)[None])[0]:
            invalid_path_samples += 1
            return
        raw_candidates += 1
        cell = tuple(int(v) for v in _cells(point))
        if cell in original_cells:
            original_hits += 1
        elif cell in seen_candidates:
            duplicate_hits += 1
        else:
            seen_candidates.add(cell)
            candidate_cells.append(cell)
            candidate_inputs.append(int(point_id))
            candidate_kind.append(kind)

    for point_id in ranking:
        add(targets[point_id], point_id, 'target')
    for point_id in ranking:
        if not np.isfinite(source[point_id]).all():
            invalid_path_samples += 2
            continue
        for fraction in PATH_FRACTIONS[1:]:
            point = (1 - fraction) * source[point_id] + fraction * targets[point_id]
            add(point, point_id, 'path')

    chosen_cells = candidate_cells[:budget]
    added_count = len(chosen_cells)
    old_count = len(coords)
    if added_count:
        new_coords = np.column_stack((np.zeros(added_count, dtype=coords.dtype),
                                      np.asarray(chosen_cells, dtype=coords.dtype)))
        # Distances are in integer cell-center coordinates, so squared-distance
        # ties are exact.  Lexicographic old coordinates break ties independent
        # of the caller's original row order.
        original_order = _lex_order(coords)
        original_cells_sorted = coords[original_order, 1:].astype(np.float64)
        original_features_sorted = features[original_order].astype(np.float64)
        interpolated = []
        for cell in new_coords[:, 1:]:
            squared = ((original_cells_sorted - cell) ** 2).sum(axis=1)
            neighbors = np.argsort(squared, kind='stable')[:min(4, old_count)]
            weights = 1 / np.sqrt(squared[neighbors])
            weights /= weights.sum()
            interpolated.append((original_features_sorted[neighbors] * weights[:, None]).sum(axis=0))
        merged_coords = np.concatenate((coords, new_coords))
        merged_features = np.concatenate((features, np.asarray(interpolated, dtype=features.dtype)))
        order = _lex_order(merged_coords)
        expanded_coords, expanded_features = merged_coords[order].copy(), merged_features[order].copy()
        added_mask = order >= old_count
        inverse_order = np.empty(len(order), dtype=np.int64)
        inverse_order[order] = np.arange(len(order))
        old_to_expanded = inverse_order[:old_count]
        added_to_expanded = inverse_order[old_count:]
    else:
        # A true no-op clone preserves even an unsorted caller's row order.
        expanded_coords, expanded_features = coords.copy(), features.copy()
        added_mask = np.zeros(old_count, dtype=bool)
        old_to_expanded = np.arange(old_count, dtype=np.int64)
        added_to_expanded = np.empty(0, dtype=np.int64)

    target_cells = np.full((len(targets), 3), -1, dtype=np.int64)
    target_cells[valid_targets] = _cells(targets[valid_targets])
    expanded_cell_set = original_cells | set(chosen_cells)
    old_target_cell_present = np.asarray([bool(valid_targets[i] and tuple(row) in original_cells)
                                         for i, row in enumerate(target_cells)], dtype=bool)
    expanded_target_cell_present = np.asarray([bool(valid_targets[i] and tuple(row) in expanded_cell_set)
                                              for i, row in enumerate(target_cells)], dtype=bool)
    receipt = dict(
        format='coarse64_support_expansion.v1',
        status='expanded' if added_count else ('budget_zero' if budget == 0 else 'no_new_candidates'),
        resolution=RESOLUTION, old_tokens=old_count, added_tokens=added_count,
        expanded_tokens=len(expanded_coords), max_added_tokens=int(max_added_tokens),
        max_fraction=float(max_fraction), token_budget=budget,
        input_points=len(targets), eligible_input_points=int(eligible.sum()),
        unsupported_target_count=int((~valid_targets).sum()),
        out_of_bounds_target_count=int((finite_targets & ~valid_targets).sum()),
        nonfinite_target_count=int((~finite_targets).sum()),
        nonfinite_source_count=int((~np.isfinite(source).all(axis=1)).sum()),
        invalid_confidence_count=int((~valid_confidence).sum()),
        unsupported_target_indices=np.flatnonzero(~valid_targets).tolist(),
        raw_valid_candidate_samples=raw_candidates, candidate_hits_original=original_hits,
        duplicate_new_candidate_samples=duplicate_hits, invalid_path_samples=invalid_path_samples,
        unique_new_candidates=len(candidate_cells), candidates_dropped_by_budget=len(candidate_cells) - added_count,
        path_fractions=list(PATH_FRACTIONS), candidate_priority='all targets, then midpoint/source paths; confidence descending, stable input order',
        interpolation='original-only inverse-distance kNN4; exact cell-distance ties use lexicographic original coordinates',
        old_to_expanded=old_to_expanded.tolist(), added_to_expanded=added_to_expanded.tolist(),
        added_coordinates=[list(cell) for cell in chosen_cells],
        added_input_indices=candidate_inputs[:added_count], added_candidate_kind=candidate_kind[:added_count],
        old_target_cell_present_count=int(old_target_cell_present.sum()),
        expanded_target_cell_present_count=int(expanded_target_cell_present.sum()),
        target_cell_presence_is_decoder_coverage=False,
        original_coordinates_exact=bool(np.array_equal(expanded_coords[old_to_expanded], coords)),
        original_features_exact=bool(np.array_equal(expanded_features[old_to_expanded], features)),
        mesh_preservation_checked=False,
    )
    return expanded_coords, expanded_features, added_mask, receipt


def expand_support(z, native_source_points, native_targets, confidence,
                   max_added_tokens=128, max_fraction=.2):
    """Return a fresh sparse tensor, torch.bool added mask and JSON-safe receipt.

    The fresh constructor deliberately discards cached sparse neighborhoods.
    Neither TRELLIS imports nor a decoder call are needed here.
    """
    import torch

    def numpy(value):
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.dtype == torch.bfloat16:
                value = value.float()
            return value.numpy()
        return np.asarray(value)

    coords, features, added, receipt = plan_expansion(
        numpy(z.coords), numpy(z.feats), numpy(native_source_points), numpy(native_targets), numpy(confidence),
        max_added_tokens=max_added_tokens, max_fraction=max_fraction)
    expanded_coords = torch.as_tensor(coords, device=z.coords.device, dtype=z.coords.dtype).clone()
    expanded_features = torch.as_tensor(features, device=z.feats.device, dtype=z.feats.dtype).clone()
    original_rows = torch.tensor(receipt['old_to_expanded'], device=z.feats.device, dtype=torch.long)
    expanded_features.index_copy_(0, original_rows, z.feats.detach())
    # SparseTensor's feature constructor forwards unknown kwargs to spconv;
    # wrapper-only scale metadata must be assigned after backend construction.
    expanded = type(z)(feats=expanded_features, coords=expanded_coords)
    expanded._scale = getattr(z, '_scale', (1, 1, 1))
    expanded._spatial_cache = {}
    return expanded, torch.as_tensor(added, device=z.feats.device, dtype=torch.bool), receipt
