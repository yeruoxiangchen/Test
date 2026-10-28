"""Input-only candidate decisions. No GT quantities are accepted."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .config import PARAMS


def component_summary(vertices,faces):
    v,f=np.asarray(vertices),np.asarray(faces,np.int64)
    edges=np.sort(np.concatenate((f[:,:2],f[:,1:],f[:,[2,0]])),axis=1)
    edges,counts=np.unique(edges,axis=0,return_counts=True)
    graph=coo_matrix((np.ones(len(edges),np.uint8),(edges[:,0],edges[:,1])),shape=(len(v),len(v)))
    count,labels=connected_components(graph,directed=False)
    tri=v[f];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1)*.5
    total=float(areas.sum())
    comp_area=np.bincount(labels[f[:,0]],weights=areas,minlength=count)
    sizes=np.bincount(labels[np.unique(f)],minlength=count)
    present=sizes>0
    fractions=comp_area[present]/max(total,1.e-12)
    return dict(mesh_success=True,vertex_count=len(v),face_count=len(f),component_count=int(present.sum()),
        largest_component_ratio=float(sizes.max()/max(sizes.sum(),1)),
        boundary_edge_count=int((counts==1).sum()),nonmanifold_edge_count=int((counts>2).sum()),
        bbox_extent=np.ptp(v,axis=0).tolist(),total_surface_area=total,
        large_component_count=int((fractions>=PARAMS['small_component_area_fraction']).sum()),
        small_component_area_fraction=float(fractions[fractions<PARAMS['small_component_area_fraction']].sum()),
        component_area_fractions=sorted(fractions.tolist(),reverse=True))


def legacy_safe(row,base,original_structure):
    outside=row['outside'];structure=row['structure']
    safe=min(row['iou_original'])>=.995
    safe &= outside['depth_mae'] is not None and outside['depth_mae']<=PARAMS['max_outside_depth_mae']
    safe &= outside['depth_p95'] is not None and outside['depth_p95']<=PARAMS['max_outside_depth_p95']
    safe &= structure['component_count']<=original_structure['component_count']
    safe &= structure['nonmanifold_edge_count']<=original_structure['nonmanifold_edge_count']
    score,initial=row['validation_cost'],base['validation_cost']
    improved=score is not None and initial is not None and score<=initial-PARAMS['validation_min_gain']
    return bool(safe and improved)


def observed_decision(row,base,original_structure,query_count,heldout_pixels):
    outside=row['outside'];structure=row['structure']
    deltas=np.asarray(row['iou_observed'])-np.asarray(base['iou_observed'])
    silhouette_safe=bool(deltas.min()>=-PARAMS['observed_mask_tolerance'])
    outside_safe=outside['depth_mae'] is not None and outside['depth_mae']<=PARAMS['max_outside_depth_mae']
    outside_safe &= outside['depth_p95'] is not None and outside['depth_p95']<=PARAMS['max_outside_depth_p95']
    structure_safe=structure['nonmanifold_edge_count']<=original_structure['nonmanifold_edge_count']
    structure_safe &= structure['large_component_count']<=original_structure['large_component_count']
    structure_safe &= structure['small_component_area_fraction']<=original_structure['small_component_area_fraction']+PARAMS['max_added_small_component_area_fraction']
    area_ratio=structure['total_surface_area']/max(original_structure['total_surface_area'],1.e-12)
    structure_safe &= .8<=area_ratio<=1.25
    photo,initial=row['validation_cost'],base['validation_cost']
    photo_available=photo is not None and initial is not None
    photo_safe=(photo<=initial+.002) if photo_available else heldout_pixels>=PARAMS['min_validation_pixels']
    original_depth=base['local_depth']
    depth_gain=(original_depth-row['local_depth'])/max(original_depth,1.e-8)
    enough=query_count>=PARAMS['min_valid_depth_pixels'] and heldout_pixels>=PARAMS['min_validation_pixels']
    evidence_improved=depth_gain>=.10
    # Learned depth improvement is accepted only with heldout input non-regression.
    okay=bool(silhouette_safe and outside_safe and structure_safe and photo_safe and enough and evidence_improved)
    score=(photo-initial if photo_available else 0.)-.5*float(deltas.mean())
    return dict(accepted=okay,score=float(score),silhouette_safe=silhouette_safe,
        outside_safe=bool(outside_safe),structure_safe=bool(structure_safe),heldout_nonregression=bool(photo_safe),
        validation_available=bool(photo_available),enough_evidence=bool(enough),depth_relative_gain=float(depth_gain),
        observed_iou_delta=deltas.tolist(),uses_gt=False)


def select(snapshots,original_structure,query_count,heldout_pixels):
    base=snapshots[0]
    own=min(snapshots,key=lambda r:(r['loss'],r['step']))
    legacy=own['step'] if own['step']>0 and legacy_safe(own,base,original_structure) else 0
    decisions=[]
    for row in snapshots[1:]:
        decision=observed_decision(row,base,original_structure,query_count,heldout_pixels)
        decisions.append(dict(step=row['step'],**decision))
    eligible=[r for r in decisions if r['accepted']]
    revised=min(eligible,key=lambda r:(r['score'],-r['depth_relative_gain'],r['step']))['step'] if eligible else 0
    return dict(legacy_step=legacy,observed_step=revised,own_training_step=own['step'],
        observed_decisions=decisions,input_only=True,
        selection='legacy: own-training best then old gate; observed: all checkpoints, holdout non-regression and trusted target improvement')
