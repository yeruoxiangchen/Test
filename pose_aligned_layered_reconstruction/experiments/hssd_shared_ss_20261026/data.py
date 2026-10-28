"""The same selected image observations for both trainable SS conditions."""
import torch
from .common import *
from ..hssd_input_adaptation_20261025.evaluate import load_input


def load_case(obj,arm,cache):
    source,sample,inputs=load_input(obj,VARIANT,cache)
    expected=read(verify(next(c['condition_record'] for c in logical_cases() if c['uid']==obj['uid'])))
    if inputs!=expected['inputs']:raise ValueError('Current image observations differ from frozen comparison')
    evidence=evidence_binding=None
    if arm=='geometry_observed':
        from .evidence import load_evidence
        evidence,evidence_binding=load_evidence(obj)
    elif arm not in ('appearance_control','frozen'):
        raise ValueError('Unknown observation arm')
    return source,sample,inputs,evidence,evidence_binding
