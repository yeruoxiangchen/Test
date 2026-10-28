"""GT-only qualification of input-mask pair proposals; never model conditions."""
import os
from . import common as c


def main():
    os.environ.update(c.environment())
    import numpy as np
    import torch,trimesh
    from PIL import Image
    from trellis2.renderers.mesh_renderer import MeshRenderer
    from trellis2.representations.mesh import Mesh
    doc=c.read(c.DATA/'pair_proposals_v2.json');records=[]
    renderer=MeshRenderer({'resolution':518,'near':.01,'far':100.,'ssaa':1,'antialias':False})
    for index,row in enumerate(doc['groups']):
        folder=c.native.SOURCE/'scenes'/row['scene'];capture=c.read(folder/'plan.json')['plan']
        meshes={}
        for uid in (row['a'],row['b']):
            asset=doc['endpoint_assets'][uid]
            m=trimesh.load(asset['mesh']['path'],force='mesh',process=False)
            m.apply_transform(np.linalg.inv(asset['canonical_from_world']))
            meshes[uid]=Mesh(torch.tensor(np.array(m.vertices),device='cuda',dtype=torch.float32),
                              torch.tensor(np.array(m.faces),device='cuda',dtype=torch.int32))
        frames=[]
        for view in row['selected_frames']:
            frame=view['frame'];E=torch.tensor(capture['world_w2c'][frame],device='cuda',dtype=torch.float32)
            K=np.asarray(capture['intrinsics'][frame],dtype=np.float32).copy()
            # Raster coordinates use pixel boundaries; original K uses centres.
            K[0,2]+=.5;K[1,2]+=.5;K[:2]/=capture['image_side']
            intr=torch.tensor(K,device='cuda');rendered={};visible={};hit={}
            with torch.no_grad():
                for uid,m in meshes.items():
                    out=renderer.render(m,E,intr,return_types=['mask','depth'])
                    rendered[uid]=(out['mask'].cpu().numpy()>.5,out['depth'].cpu().numpy())
                    with Image.open(folder/'masks'/uid/f'{frame:03d}.png') as image:
                        visible[uid]=np.asarray(image.resize((518,518),Image.Resampling.NEAREST))>127
                    mask,depth=rendered[uid]
                    hit[uid]=float((visible[uid]&mask).sum()/max(visible[uid].sum(),1))
            a,b=row['a'],row['b'];ma,da=rendered[a];mb,db=rendered[b]
            ab=ma&mb&visible[b]&(db+.002<da)
            ba=ma&mb&visible[a]&(da+.002<db)
            calibrated=min(hit.values())>=.85
            frames.append({'frame':frame,'visible_to_amodal_hit':hit,'projection_qualified':calibrated,
                'a_hidden_by_b_pixels':int(ab.sum()),'b_hidden_by_a_pixels':int(ba.sum()),
                'a_hidden_by_b_fraction':float(ab.sum()/max(ma.sum(),1)),
                'b_hidden_by_a_fraction':float(ba.sum()/max(mb.sum(),1))})
        occluded=any(f['projection_qualified'] and (
            (f['a_hidden_by_b_pixels']>=32 and f['a_hidden_by_b_fraction']>=.01) or
            (f['b_hidden_by_a_pixels']>=32 and f['b_hidden_by_a_fraction']>=.01)) for f in frames)
        records.append({**row,'gt_qualification':frames,'verified_occlusion':occluded,
                        'native_training_ready':occluded and row['both_native_ready']})
        print(f'PAIR_QUALIFIED {index+1}/{len(doc["groups"])} occlusion={occluded}',flush=True)
        del meshes;torch.cuda.empty_cache()
    c.write(c.DATA/'pair_qualification_gt_only.json',{'time':c.now(),'groups':records,
        'input_proposals':str(c.DATA/'pair_proposals_v2.json'),
        'role':'GT geometry used ONLY for dataset qualification/labels; forbidden as test-time relation features',
        'verified_occlusion_pairs':sum(r['verified_occlusion'] for r in records),
        'native_training_ready_pairs':sum(r['native_training_ready'] for r in records),
        'thresholds':{'depth_margin_m':.002,'minimum_hidden_pixels':32,'minimum_fraction':.01,'minimum_projection_hit':.85}})


if __name__=='__main__':main()
