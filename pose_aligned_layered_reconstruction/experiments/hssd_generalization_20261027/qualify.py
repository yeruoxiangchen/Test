"""Reuse the audited official-compatible encoder; never train on these targets."""
import argparse
from collections import Counter
import json
from pathlib import Path

from . import common as c


def targets():
    row=c.read(c.DATASET/'targets.json')
    if c.digest(row['plan'])!=row['identity']:raise ValueError('Target denominator changed')
    return row


def encode(uid):
    import os
    os.environ.update(c.environment())
    import torch
    from ..hssd_pilot_20261020 import latents as old
    from ..hssd_input_adaptation_int32_20261025.tiled_native import LargeInputNative
    torch.set_num_threads(2)
    row=next(r for r in targets()['plan']['objects'] if r['uid']==uid)
    storage=c.read(c.OUT/'encoder_storage.json')
    old.object_dir=lambda item:c.DATASET/'objects'/item['uid']
    def offline_dino(*,device,**kwargs):
        from types import SimpleNamespace
        encoder=torch.hub.load(storage['local_dino_repository'],'dinov2_vitl14_reg',source='local',pretrained=True)
        encoder.to(device).eval().requires_grad_(False)
        return SimpleNamespace(models={'image_cond_model':encoder})
    old.build_frozen_dino_encoder=offline_dino
    load=old.load_trellis_model
    class SafeMeshDecoder(torch.nn.Module):
        def __init__(self,inner):super().__init__();self.inner=inner
        def forward(self,value):
            with LargeInputNative():return self.inner(value)
    def safe_load(pretrained,name):
        if name=='ss_enc_conv3d_16l8_fp16':pretrained=storage['local_ss_encoder']
        model,binding=load(pretrained,name)
        return (SafeMeshDecoder(model) if name=='slat_dec_mesh_swin8_B_64l8m256c_fp16' else model),binding
    old.load_trellis_model=safe_load
    old.encode_one(row)
    receipt=c.DATASET/'objects'/uid/'latent_audit.json'
    audit=c.read(receipt)
    if not audit['ss']['passed'] or not audit['coordinate']['passed'] or not audit['finite_and_nonempty']:
        raise ValueError('Mesh-to-latent qualification failed; no target removal or training')
    c.event('latent_qualified',uid=uid,split=row['split'],ss_iou=audit['ss']['iou'],audit=c.binding(receipt))


def summarize():
    import numpy as np
    p=targets();rows=[]
    for obj in p['plan']['objects']:
        root=c.DATASET/'objects'/obj['uid'];a=c.read(root/'latent_audit.json')
        if not a['ss']['passed'] or not a['coordinate']['passed'] or not a['finite_and_nonempty']:
            raise ValueError('Invalid frozen target; no silent exclusions')
        for name in ('ss_target','slat_target','raw_mesh','decoded_mesh'):c.verify(a[name])
        visual=c.read(c.OUT/'visuals'/f'{obj["uid"]}.json');c.verify(visual['sheet'])
        rows.append({'uid':obj['uid'],'split':obj['split'],'scene':obj['scene'],'category':obj['category'],
                     'asset_family':obj['asset_family'],'source_content_id':obj['source_content_id'],
                     'candidate_geometry_hash':obj['candidate_geometry_hash'],'ss_iou':a['ss']['iou'],
                     'surface':a['surface'],'coordinate':a['coordinate'],'audit':c.binding(root/'latent_audit.json'),
                     'visual':c.binding(c.OUT/'visuals'/f'{obj["uid"]}.json')})
    counts=dict(Counter(r['split'] for r in rows))
    if counts!={'train':128,'val':32,'test':32}:raise ValueError('Complete split denominator required')
    for field in ('scene','asset_family','source_content_id','candidate_geometry_hash'):
        groups={subset:{r[field] for r in rows if r['split']==subset} for subset in counts}
        if any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')]):
            raise ValueError('Split leakage: '+field)
    result={'time':c.now(),'success':True,'target_plan':c.binding(c.DATASET/'targets.json'),'counts':counts,
            'objects':rows,'ss_iou_min':min(r['ss_iou'] for r in rows),
            'ss_iou_median':float(np.median([r['ss_iou'] for r in rows])),
            'scope':'Dataset target-codec qualification, not model fitting/generalization results.',
            'training_launched':False,'ready_for_formal_training':False,
            'next_gate':'Review native mesh/decoded-target quality, shared photographs, occlusion distribution and input conditioning before formal training.',
            'no_targets_dropped_after_latent_encoding':True}
    c.write(c.OUT/'data_qualification.json',result)
    report=['# HSSD192 数据准备与质量验收','',f'完成时间：{result["time"]}。',
            '',f'训练128／验证32／测试32；24个独立场景，SS回环最小IoU={result["ss_iou_min"]:.5f}。',
            '', '这份报告只说明数据和标签准备完成，尚无新模型的泛化成绩。',
            '', '| 对象 | 划分 | SS IoU | 目标解码CD | 目标解码F@.02 |','|---|---|---:|---:|---:|']
    for row in rows:report.append(f'| {row["uid"]} | {row["split"]} | {row["ss_iou"]:.5f} | {row["surface"]["chamfer_l1"]:.6f} | {row["surface"]["fscore_0p02"]:.5f} |')
    c.data_path(c.OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    c.event('data_qualification_complete_pending_review',counts=counts,training_launched=False)


def visual(uid):
    from PIL import Image,ImageDraw
    from ..round2_v2.render_mesh_audit import render_file
    root=c.DATASET/'objects'/uid;a=c.read(root/'latent_audit.json');records=[]
    for kind,key in [('raw','raw_mesh'),('decoded','decoded_mesh')]:
        b=a[key];c.verify(b)
        records.append(render_file(b['path'],uid+'_'+kind,output=c.OUT/'visuals',expected_sha256=b['sha256'],
                                   identity={'target_plan':targets()['identity'],'uid':uid,'kind':kind}))
    side=320;sheet=Image.new('RGB',(side*3,(side+28)*2),'white');draw=ImageDraw.Draw(sheet)
    for ri,record in enumerate(records):
        for j,view in enumerate(record['views']):
            draw.text((j*side+4,ri*(side+28)+6),f'{uid} / {"raw" if ri==0 else "decoded"} / view{j}',fill='black')
            with Image.open(view['png']) as image:sheet.paste(image.convert('RGB').resize((side,side)),(j*side,ri*(side+28)+28))
    path=c.OUT/'figures'/f'{uid}_target_codec.jpg';path.parent.mkdir(parents=True,exist_ok=True);sheet.save(path,quality=95)
    c.write(c.OUT/'visuals'/f'{uid}.json',{'uid':uid,'sheet':c.binding(path),'records':records,
                                       'native_frame':True,'per_result_fitted_transform':False,'human_review_completed':False})
    c.event('target_visual_ready',uid=uid,figure=str(path))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['encode','summary','visual']);parser.add_argument('--uid')
    args=parser.parse_args();{'encode':lambda:encode(args.uid),'visual':lambda:visual(args.uid),'summary':summarize}[args.action]()
