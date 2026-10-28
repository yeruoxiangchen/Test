"""Finalize completed rigid audit using its frozen NumPy aggregation functions.

No Torch import, no ICP, no transformed mesh output. This entry point exists to
finish source-hash checks and aggregation under shared-host library I/O pressure.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import ast
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / 'pose_aligned_layered_reconstruction'
OUT = PACKAGE / 'outputs/round2_v2_mesh_visual_audit_20261018/alignment'
FROZEN = Path(__file__).with_name('mesh_pose_audit.py')
LEGACY = ROOT / 'experiments/rigid_aligned_mesh_eval_20260911.py'


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    namespace = dict(np=np, json=json, Path=Path, hashlib=hashlib, os=os, OUT=OUT,
                     VERSION='palr.round2.mesh_pose_audit.v1', MODES=('parent', 'v1', 'raw'))
    wanted = {'sha', 'dump', 'binding', 'pose_checks', 'paired', 'aggregate'}
    tree = ast.parse(FROZEN.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise ValueError('frozen aggregation functions missing')
    legacy_nodes = [n for n in ast.parse(LEGACY.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'rotation_angle_degrees']
    exec(compile(ast.Module(body=legacy_nodes + nodes, type_ignores=[]), str(FROZEN), 'exec'), namespace)
    original_sha = namespace['sha']
    cache = {}
    started = time.time()

    def checked_sha(path):
        path = Path(path).resolve()
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        old = cache.get(str(path))
        if old is not None and old['signature'] == signature:
            return old['sha256']
        digest = original_sha(path)
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != signature:
            raise ValueError(f'source changed while hashing: {path}')
        cache[str(path)] = dict(signature=signature, sha256=digest)
        if len(cache) % 40 == 0:
            print(json.dumps(dict(stage='source_hash_verification', unique_files=len(cache),
                                  elapsed_seconds=time.time() - started)), flush=True)
        return digest

    namespace['sha'] = checked_sha
    for bound in [protocol['source_plan'], *protocol['code']]:
        if checked_sha(bound['path']) != bound['sha256']:
            raise ValueError(f"frozen code or source plan changed: {bound['path']}")
    expected = {case['id']: case for case in protocol['cases']}
    actual = {p.stem for p in (OUT / 'records').glob('*.json')}
    if actual != set(expected) or len(actual) != 384:
        raise ValueError('incomplete, duplicate, or unexpected audit record cohort')
    records = []
    for case in protocol['cases']:
        record = json.loads((OUT / 'records' / (case['id'] + '.json')).read_text())
        if record['id'] != case['id'] or any(record[k] != case[k] for k in ('source_record', 'mesh', 'target')):
            raise ValueError('case source identity mismatch')
        old = json.loads(Path(record['source_record']['path']).read_text())
        if record['native'] != old['surface']:
            raise ValueError('native values changed during aggregation')
        records.append(record)
    result = namespace['aggregate'](protocol['cases'], protocol)
    for path, cached in cache.items():
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != cached['signature']:
            raise ValueError(f'source changed during final aggregation: {path}')
    verification = dict(passed=True, finalizer=namespace['binding'](__file__),
                        frozen_functions_extracted_verbatim=sorted(wanted | {'rotation_angle_degrees'}),
                        source_hashes_unique_files=len(cache), source_hashes_all_match=True,
                        source_stats_unchanged_through_aggregation=True,
                        record_count=len(records), native_reproduction_max_abs_error=result['native_reproduction_max_abs_error'],
                        max_detR_abs_error=max(abs(r['alignment']['detR'] - 1) for r in records),
                        max_singular_value_abs_error=max(abs(v - 1) for r in records for v in r['alignment']['singular_values']),
                        max_orthogonality_error=max(r['alignment']['orthogonality_error'] for r in records),
                        identity_selected=sum(r['alignment']['selected_identity_on_surface_metric'] for r in records),
                        no_icp_restarted=True, elapsed_seconds=time.time() - started)
    namespace['dump'](OUT / 'verification.json', verification)
    print(json.dumps(dict(complete=True, verification=verification, summary=result['groups']['all']['summary'])), flush=True)


if __name__ == '__main__':
    main()
