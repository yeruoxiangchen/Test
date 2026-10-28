"""Freeze the baseline and inspect existing shared masks; no target substitution."""
import itertools
import os
from . import common as c


def prepare():
    os.environ.update(c.environment())
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy.ndimage import binary_dilation
    c.OUT.mkdir(parents=True,exist_ok=True); c.DATA.mkdir(parents=True,exist_ok=True)
    native = c.native.plan()
    old = c.read(c.native.DATA/'all.json')
    train = [r for r in old['samples'] if r['split']=='train']
    val = [r for r in old['samples'] if r['split']=='val']
    assert len(train)==8 and len(val)==4
    trsource = [r for r in native['plan']['objects'] if r['split']=='train']
    vasource = [r for r in native['plan']['objects'] if r['split']=='val']
    for field in ('scene','asset_family','candidate_geometry_hash'):
        assert not ({r[field] for r in trsource}&{r[field] for r in vasource}), field
    for subset, rows in [('all',old['samples']),('train',train),('val',val)]:
        c.immutable(c.DATA/f'{subset}.json',{**old,'samples':rows})
    max_row=max(train,key=lambda r:c.read(c.native.DATA/'objects'/r['uid']/'encode_512.json')['tokens'])
    payload={'native_plan_identity':native['identity'], 'config':c.CONFIG,
             'manifests':{k:{'path':str(c.DATA/f'{k}.json'),'sha256':c.sha(c.DATA/f'{k}.json')} for k in ('all','train','val')},
             'objects':native['plan']['objects'],'smoke_train_index':train.index(max_row),
             'smoke_eval_uid':val[0]['uid'], 'smoke_train_uid':max_row['uid']}
    c.immutable(c.OUT/'plan.json',{'identity':c.digest(payload),'plan':payload})
    # Hash only small native conditions/labels once at preparation, not all weights.
    for row in old['samples']:
        for key in ('pose_file','ss_target_file','shape_target_file'):
            assert c.sha(row[key])==row[key+'_sha256'],row[key]
    if (c.DATA/'pair_proposals_v2.json').exists():
        assert c.read(c.DATA/'pair_proposals_v2.json')['native_plan_identity']==native['identity']
        return
    proposals=[]; coverage=[]; native_uids={r['uid'] for r in old['samples']}
    # Native12 anchors with every eligible already-rendered scene candidate.
    # Original random eight targets per scene omit many actual occluders.
    candidates=c.read(c.native.SOURCE/'candidate_assets.json')['objects']
    family_splits={};geometry_splits={}
    for r in candidates:
        family_splits.setdefault(r['asset_family'],set()).add(r['split'])
        geometry_splits.setdefault(r['candidate_geometry_hash'],set()).add(r['split'])
    source=[r for r in candidates if len(family_splits[r['asset_family']])==1 and len(geometry_splits[r['candidate_geometry_hash']])==1]
    scenes=sorted({r['scene'] for r in native['plan']['objects']})
    for sid in scenes:
        rows=[r for r in source if r['scene']==sid]
        folder=c.native.SOURCE/'scenes'/sid
        capture=c.read(folder/'plan.json')['plan']
        assert {r['uid'] for r in native['plan']['objects'] if r['scene']==sid}<={r['uid'] for r in rows}
        pairs={ (a['uid'],b['uid']):[] for a,b in itertools.combinations(rows,2) if a['uid'] in native_uids or b['uid'] in native_uids}
        for frame in range(capture['frame_count']):
            masks={}; areas={}; grown={}; boxes={}
            for row in rows:
                with Image.open(folder/'masks'/row['uid']/f'{frame:03d}.png') as im:
                    m=np.asarray(im.resize((518,518),Image.Resampling.NEAREST))>127
                area=int(m.sum())
                if area<128: continue
                yy,xx=np.nonzero(m)
                masks[row['uid']]=m; areas[row['uid']]=area
                grown[row['uid']]=binary_dilation(m,iterations=5)
                boxes[row['uid']]=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]
            for a,b in pairs:
                if a not in masks or b not in masks:continue
                touching=int((grown[a]&masks[b]).sum()+(grown[b]&masks[a]).sum())
                ax,ay,ar,ab=boxes[a];bx,by,br,bb=boxes[b]
                gap=float(np.hypot(max(ax-br,bx-ar,0),max(ay-bb,by-ab,0)))
                pairs[a,b].append({'frame':frame,'visible_pixels':[areas[a],areas[b]],
                    'mask_boundary_proximity_pixels':touching,'bbox_gap_px':gap,
                    'score':float(touching/max(min(areas[a],areas[b]),1))})
            if frame%20==0:print(f'PAIR_MASKS {sid} {frame}/{capture["frame_count"]}',flush=True)
        for (a,b),frames in pairs.items():
            ordered=sorted(frames,key=lambda x:(-x['score'],x['bbox_gap_px'],-min(x['visible_pixels']),x['frame']))
            selected=[];eyes=np.asarray(capture['world_eye'])
            for item in ordered:
                if any(np.linalg.norm(eyes[item['frame']]-eyes[v['frame']])<.2 for v in selected):continue
                selected.append(item)
                if len(selected)==4:break
            both=a in native_uids and b in native_uids
            coverage.append({'scene':sid,'a':a,'b':b,'co_visible_frames':len(frames),'distinct_views':len(selected),'both_native_ready':both})
            if len(selected)<2:continue
            proposals.append({'scene':sid,'split':rows[0]['split'],'a':a,'b':b,
                 'both_native_ready':both,'selected_frames':selected,'co_visible_frames':len(frames),
                 'input_evidence':'visible masks only; proximity is NOT proof of occlusion order'})
    proposals.sort(key=lambda r:(r['scene'],-max(f['score'] for f in r['selected_frames']),
                                not r['both_native_ready'],r['a'],r['b']))
    selected=[]
    for sid in scenes:
        selected.extend([r for r in proposals if r['scene']==sid][:4])
    for r in selected:
        folder=c.native.SOURCE/'scenes'/r['scene'];frame=r['selected_frames'][0]['frame']
        with Image.open(folder/'rgb'/f'{frame:03d}.png') as im:canvas=im.convert('RGB').resize((777,777))
        arr=np.asarray(canvas).copy()
        for uid,color in [(r['a'],[255,70,60]),(r['b'],[45,160,255])]:
            with Image.open(folder/'masks'/uid/f'{frame:03d}.png') as im:m=np.asarray(im.resize((777,777),Image.Resampling.NEAREST))>127
            arr[m]=(.6*arr[m]+.4*np.asarray(color)).astype(np.uint8)
        canvas=Image.fromarray(arr);draw=ImageDraw.Draw(canvas)
        draw.rectangle((0,0,777,42),fill='white');draw.text((5,5),f'{r["a"]} (red) / {r["b"]} (blue)',fill='black')
        draw.text((5,23),f'frame {frame}; visible-mask proposal, depth order not yet certified',fill='black')
        fig=c.OUT/'pair_figures_v2'/f'{r["a"]}__{r["b"]}.jpg';fig.parent.mkdir(parents=True,exist_ok=True);canvas.save(fig)
        r['figure']=str(fig)
    used={x for r in selected for x in (r['a'],r['b'])}
    c.write(c.DATA/'pair_proposals_v2.json',{'native_plan_identity':native['identity'],
        'selection_rule':'Native12 anchor plus all eligible pre-rendered candidates; per-scene up to4 by visible-mask boundary score; >=2 camera positions',
        'cross_split_checks':'candidate asset-family and exact-geometry hash must occur in only one split; source candidate plan also binds full scene-context exclusions',
        'groups':selected,'all_pair_coverage':coverage,'all_proposals':proposals,
        'endpoint_assets':{r['uid']:r for r in source if r['uid'] in used},
        'missing_native_uids':sorted(used-native_uids),'occlusion_verified':False})
    print('Prepared baseline8/val4 and pair proposals='+str(len(selected)),flush=True)


if __name__=='__main__':prepare()
