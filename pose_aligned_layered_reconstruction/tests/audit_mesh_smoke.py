"""Check all paired smoke artifacts and offload-vs-resident equivalence."""
from pathlib import Path

import numpy as np
import trimesh

from pose_aligned_layered_reconstruction.evaluate_mesh import DEFAULT_OUTPUT, EXPERIMENT_ROOT, load_plan, read_completed
from pose_aligned_layered_reconstruction.mesh_metrics import surface_metrics, aggregate_mesh_records
from pose_aligned_layered_reconstruction.prepare_official_experiment import atomic_json


def main():
    plan, identity=load_plan(DEFAULT_OUTPUT)
    records=[]
    rechecks=[]
    for index in (0,15):
        for case in plan['rows'][index]['cases']:
            row=read_completed(DEFAULT_OUTPUT/'records'/(case['name']+'.json'),identity)
            mesh=trimesh.load(row['mesh']['path'],process=False,force='mesh')
            target=trimesh.load(row['target_mesh']['path'],process=False,force='mesh')
            surface=surface_metrics(mesh,target,count=20000,seed=case['seed']*1009+index*9173)
            error=max(abs(surface[k]-row['surface'][k]) for k in surface)
            if error>1.e-12:
                raise RuntimeError('saved mesh does not reproduce its metrics')
            detail=dict(name=case['name'],reloaded_metrics_max_abs=error)
            if index==0:
                with np.load(EXPERIMENT_ROOT/'latents'/(case['name']+'.npz')) as old, np.load(row['latent']['path']) as new:
                    exact=all(np.array_equal(old[key],new[key]) for key in ('coords','feats'))
                old_mesh=trimesh.load(EXPERIMENT_ROOT/'meshes'/(case['name']+'.ply'),process=False,force='mesh')
                mesh_exact=np.array_equal(mesh.vertices,old_mesh.vertices) and np.array_equal(mesh.faces,old_mesh.faces)
                detail.update(offload_latent_bit_exact=exact,offload_mesh_bit_exact=mesh_exact)
                if not exact or not mesh_exact:
                    raise RuntimeError('CPU offload changed the small-case mesh or latent')
            rechecks.append(detail)
            records.append(row)
    report=aggregate_mesh_records(records,[plan['rows'][i]['uid'] for i in (0,15)])
    report.update(passed=True,identity_hash=identity,scope='smoke_only_not_Dev64_generalization',
                  rechecks=rechecks,max_peak_allocated_gib=max(r['gpu_peak_allocated_bytes'] for r in records)/1024**3)
    atomic_json(DEFAULT_OUTPUT/'smoke_report.json',report)
    print(report,flush=True)


if __name__=='__main__':
    main()
