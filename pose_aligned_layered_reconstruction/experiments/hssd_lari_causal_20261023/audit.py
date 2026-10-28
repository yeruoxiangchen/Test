"""Audit all original32 LaRI predictions on CPU, including label/coordinate checks."""
import json
import numpy as np
import torch
from scipy.spatial import cKDTree
from ...v2.dataset import SceneDataset
from ...v2.losses import resized_labels, layer_loss
from ...v2.ray_geometry import CameraBundle, make_patch_rays
from ..hssd_full_diagnostic_20261021.diagnose import load_prediction, depth_metrics, query_sets
from .common import OUT, CELL, PILOT, OLD, read, binding, verify, write, freeze, bind_predictions, locked_identity, tensor_hash


def mean(x, mask):
    return float(x[mask].float().mean()) if mask.any() else None


def audit_sample(sample, pred):
    d, own, count, valid, supervision = resized_labels(sample, pred)
    ranks = torch.arange(pred.layer_count)[None, :, None, None]
    exists = ranks < count[:, None]
    target = valid & (own > .5)
    other = valid & (own <= .5)
    front = target & (ranks == 0)
    before = torch.cat([torch.zeros_like(other[:, :1]), other[:, :-1]], 1).cumsum(1) > 0
    hidden = target & before
    cameras = CameraBundle.from_sample(sample, 'cpu')
    near = (cameras.center_depth - 2.).clamp_min(1.e-4)[:, None, None, None]
    far = torch.maximum(cameras.center_depth + 2., near[:, 0, 0, 0] + .1)[:, None, None, None]
    outside = (d < near) | (d > far)
    err = (pred.depths - d).abs()
    signed = pred.depths - d
    confidence = pred.confidence_logits.sigmoid()
    ownership = pred.target_logits.sigmoid()
    soft_target = torch.exp(-err / .05)
    parts = {}
    for key, mask in [('all_scene', valid), ('target', target), ('front_target', front), ('hidden_behind_non_target', hidden), ('non_target', other)]:
        parts[key] = {'count': int(mask.sum()), 'mae': mean(err,mask), 'bias': mean(signed,mask),
            'fraction_error_lt_002':mean((err < .02).float(),mask),
            'fraction_error_lt_005':mean((err < .05).float(),mask),
            'outside_output_range_fraction':mean(outside.float(),mask),
            'mean_confidence':mean(confidence,mask), 'confidence_training_target':mean(soft_target,mask),
            'confidence_soft_target_mse':mean((confidence-soft_target).square(),mask),
            'mean_ownership':mean(ownership,mask), 'owner_positive_fraction':mean((ownership>.5).float(),mask),
            'mean_existence':mean(pred.valid_probabilities(),mask)}
    ray_mask = supervision.all(1)
    estimated = pred.stop_logits.argmax(1)
    stop_confusion = torch.bincount((count[ray_mask]*(pred.layer_count+1)+estimated[ray_mask]).flatten(),
        minlength=(pred.layer_count+1)**2).reshape(pred.layer_count+1,pred.layer_count+1)
    loss, metrics = layer_loss(pred, sample)
    _, p64, _, _ = query_sets(sample, 'cpu')
    tree = cKDTree(p64.numpy())
    rays = make_patch_rays(cameras, pred.spatial_size)
    xyz, pixel_errors, z_errors = [], [], []
    h,w = pred.spatial_size
    for view in range(len(cameras.w2c)):
        ids = torch.nonzero(target[view],as_tuple=False)
        if not len(ids):
            continue
        layer, yy, xx = ids.T
        pixel = yy*w+xx
        depth = d[view,layer,yy,xx]
        pts = rays.origins[view,pixel]+depth[:,None]*rays.axial_directions[view,pixel]
        projected = cameras.select(slice(view,view+1)).project(pts,w)
        ih,iw = cameras.image_size
        screen = (projected['image_grid'][0]+1)*torch.tensor([iw-1,ih-1])/2
        pixel_errors.append(float((screen-rays.pixels[pixel]).abs().max()))
        z_errors.append(float((projected['camera_depth'][0]-depth).abs().max()))
        xyz.append(pts.numpy())
    distances = tree.query(np.concatenate(xyz))[0]
    if max(pixel_errors)>1.e-3 or max(z_errors)>1.e-4:
        raise ValueError('Camera ray/depth roundtrip failed')
    return {'parts':parts,'target_fraction_of_valid_scene_layers':float(target.sum()/valid.sum().clamp_min(1)),
        'target_supervised_fraction':float(target.sum()/((exists&(own>.5)).sum().clamp_min(1))),
        'stop':{'accuracy':mean((estimated==count).float(),ray_mask),'count_mae':mean((estimated-count).abs().float(),ray_mask),
            'gt_capped_fraction':mean((count==pred.layer_count).float(),ray_mask),'confusion_gt_rows_pred_columns':stop_confusion.tolist()},
        'loss':{'total_layer_loss':float(loss),**{k:float(v) for k,v in metrics.items()}},
        'coordinates':{'pixel_roundtrip_max_abs':max(pixel_errors),'forward_z_roundtrip_max_abs':max(z_errors),
            'target_label_point_to_gt64_mean':float(distances.mean()),'target_label_point_to_gt64_p95':float(np.quantile(distances,.95)),
            'target_label_point_fraction_distance_le_003':float(np.mean(distances<=.03)),
            'note':'algebraic camera roundtrip plus independent GT voxel-center proximity; neither fits an alignment nor proves complete mesh/latent fidelity'}}


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    contract = freeze()
    summary = {'scope':'final trained LaRI1000 on its original16 training objects; not convergence or capacity proof','domains':{}}
    for domain in ['clean','occluded']:
        prediction_bindings = bind_predictions(domain)
        lari_path = PILOT / f'{domain}_lari/checkpoints/last.pt'
        base_path = PILOT / f'{domain}_ss_no_layer/checkpoints/last.pt'
        lari = torch.load(lari_path,map_location='cpu',weights_only=False)
        base = torch.load(base_path,map_location='cpu',weights_only=False)
        raw = {k.removeprefix('target_lari.'):v for k,v in base['model_state'].items() if k.startswith('target_lari.')}
        assert raw.keys()==lari['model_state'].keys()
        assert all(torch.equal(raw[k],v) for k,v in lari['model_state'].items())
        assert lari['geometry_config']['scene_margin']==2.
        ident = {'protocol':contract,'lari':binding(lari_path),'base':binding(base_path),'predictions':prediction_bindings,
                 'geometry_weights_exactly_equal':True,'geometry_sha':tensor_hash(raw)}
        locked_identity(OUT/'audit'/domain/'identity.json',ident)
        del lari,base,raw
        dataset = SceneDataset(PILOT/f'{domain}16.json',require_labels=True)
        rows=[]
        for index in range(16):
            path = OUT/'audit'/domain/f'{index:03d}.json'
            if path.exists():
                row=read(path)
            else:
                sample=dataset[index]
                pred=load_prediction(verify(prediction_bindings[index]['prediction']),'cpu')
                row={'index':index,'uid':sample['uid'],'sample':dataset.rows[index],**audit_sample(sample,pred)}
                write(path,row)
            assert row['uid']==dataset.rows[index]['uid']
            rows.append(row)
        def avg(values):
            values=[v for v in values if v is not None]
            return sum(values)/len(values) if values else None
        parts={group:{k:avg([r['parts'][group][k] for r in rows]) for k in rows[0]['parts'][group]} for group in rows[0]['parts']}
        history=[json.loads(line) for line in (PILOT/f'{domain}_lari/history.jsonl').read_text().splitlines()]
        assert [r['step'] for r in history]==list(range(1,1001))
        summary['domains'][domain]={'parts':parts,'target_fraction_of_valid_scene_layers':avg([r['target_fraction_of_valid_scene_layers'] for r in rows]),
            'target_supervised_fraction':avg([r['target_supervised_fraction'] for r in rows]),
            'stop_accuracy':avg([r['stop']['accuracy'] for r in rows]),
            'coordinates':{k:avg([r['coordinates'][k] for r in rows]) for k in rows[0]['coordinates'] if k!='note'},
            'history_windows':{str(start+1)+'_'+str(start+200):{key:avg([r[key] for r in history[start:start+200]]) for key in ['depth_mae','layer_loss','loss']} for start in [0,600,800]},
            'records':[binding(OUT/'audit'/domain/f'{i:03d}.json') for i in range(16)]}
        print(json.dumps({'domain':domain,**{k:v for k,v in summary['domains'][domain].items() if k!='records'}}),flush=True)
    write(OUT/'audit_summary.json',summary)


if __name__=='__main__':main()
