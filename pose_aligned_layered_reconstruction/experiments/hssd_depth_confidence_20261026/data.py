"""Bound archived image inputs, physically loaded only through data mappings."""
import copy
import torch

from .common import ARMS, VARIANT, logical_cases, read, verify
from ...v2.contracts import validate_sample
from ...v2.dataset import select_views
from ..hssd_cell_query_20261023.model import observations


def load_case(obj, arm, cache):
    if arm not in ARMS:
        raise ValueError('Only the new depth/confidence arms load training inputs')
    cases = [c for c in logical_cases() if c['uid'] == obj['uid'] and c['arm'] == arm and not c['is_reference']]
    if not cases or any(c['condition_record'] != cases[0]['condition_record'] for c in cases):
        # Records can differ with sampling seed while retaining identical inputs.
        records = [read(verify(c['condition_record'])) for c in cases]
        if not records or any(r['inputs'] != records[0]['inputs'] for r in records):
            raise ValueError('Conflicting archived image conditions')
        expected = records[0]
    else:
        expected = read(verify(cases[0]['condition_record']))
    inputs = copy.deepcopy(expected['inputs'])
    selected = list(obj['visible_indices'])
    if (inputs['selected_indices'] != selected
            or inputs['selected_frames'] != [[0, 4, 7, 11][i] for i in selected]):
        raise ValueError('Archived visible-view selection changed')
    key = (VARIANT, obj['uid'])
    if key not in cache:
        value = torch.load(verify(inputs['sample']), map_location='cpu', weights_only=True)
        validate_sample(value, require_labels=False, require_ss_target=True)
        if value['object_uid'] != obj['uid']:
            raise ValueError('Object/input mismatch')
        # The bound, preencoded sample is the actual input. Its source_files
        # retain provenance; no raw scene assets are opened or rehashed here.
        verify(obj['ss'])
        cache[key] = value, inputs
    value, cached_inputs = cache[key]
    if cached_inputs != inputs:
        raise ValueError('Input cache binding changed')
    sample = select_views(observations(value), torch.tensor(selected, dtype=torch.long))
    if len(sample['intrinsics']) != len(selected):
        raise ValueError('Context/view mismatch')
    from .evidence import load_evidence
    evidence, evidence_binding = load_evidence(obj, arm)
    return value, sample, inputs, evidence, evidence_binding
