"""Aggregate all frozen cases and render actual meshes with identical cameras."""
import os
from . import common as c


def main():
    os.environ.update(c.environment())
    import numpy as np
    import torch,trimesh
    from PIL import Image,ImageDraw,ImageFont
    from trellis2.renderers.mesh_renderer import MeshRenderer
    from trellis2.representations.mesh import Mesh
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
    records=[];objects=c.plan()['objects']
    renderer=MeshRenderer({'resolution':384,'near':.1,'far':10.,'ssaa':1})
    cameras,intrinsics=yaw_pitch_r_fov_to_extrinsics_intrinsics(
        np.deg2rad([25.,145.,265.]).tolist(),np.deg2rad([20.,20.,20.]).tolist(),2.,40.)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for obj in objects:
        uid=obj['uid'];gtpath=c.native.DATA/'objects'/uid/'target_t2.ply'
        for seed in c.CONFIG['evaluation_seeds']:
            cases=[c.read(c.OUT/'evaluations'/arm/uid/f'seed_{seed}'/'result.json') for arm in c.CONFIG['arms']]
            for case,arm in zip(cases,c.CONFIG['arms']):
                assert case['uid']==uid and case['seed']==seed and case['arm']==arm and case['split']==obj['split']
                assert case['runtime_identity']==c.read(c.OUT/'runtime.json')['identity']
            records.extend(cases)
            canvas=Image.new('RGB',(1152,1290),'white');draw=ImageDraw.Draw(canvas)
            draw.text((10,8),f'{uid} | {obj["split"]} | seed {seed} | no fitted alignment',fill='black',font=font)
            rows=[('GT complete object',gtpath)]+[(case['arm']+(' FAILED: '+case['failure'] if not case['success'] else ''),case['mesh']) for case in cases]
            for row,(label,path) in enumerate(rows):
                y=48+414*row;draw.text((10,y),label,fill='black',font=font)
                if path is None:continue
                m=trimesh.load(path,force='mesh',process=False)
                mesh=Mesh(torch.tensor(np.array(m.vertices),device='cuda',dtype=torch.float32),
                          torch.tensor(np.array(m.faces),device='cuda',dtype=torch.int32))
                with torch.no_grad():
                    for view,(E,K) in enumerate(zip(cameras,intrinsics)):
                        rendered=renderer.render(mesh,E,K,return_types=['mask','normal'])
                        mask=rendered['mask'].cpu().numpy()[...,None]
                        normal=rendered['normal'].permute(1,2,0).cpu().numpy()*2-1
                        light=np.array([.25,-.4,.88]);light/=np.linalg.norm(light)
                        shade=.3+.7*np.abs(normal@light)[...,None].clip(0,1)
                        rgb=(shade*np.array([.62,.7,.8])*mask+1-mask).clip(0,1)
                        canvas.paste(Image.fromarray(np.uint8(rgb*255)),(view*384,y+28))
                del m,mesh,rendered;torch.cuda.empty_cache()
            path=c.OUT/'figures'/f'{uid}_seed{seed}.jpg';path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
            print(f'FIGURE_COMPLETE {uid} {seed}',flush=True)
    aggregates={};paired={}
    for split in ('train','val'):
        aggregates[split]={}
        for arm in c.CONFIG['arms']:
            rows=[r for r in records if r['split']==split and r['arm']==arm]
            aggregates[split][arm]={'cases':len(rows),'successes':sum(r['success'] for r in rows),
                **{key:float(np.mean([r['metrics'][key] for r in rows])) for key in ('cd_l1','f@0.01','f@0.02')}}
        deltas=[]
        for obj in (x for x in objects if x['split']==split):
            scores={arm:np.mean([r['metrics']['cd_l1'] for r in records if r['uid']==obj['uid'] and r['arm']==arm]) for arm in c.CONFIG['arms']}
            deltas.append({'uid':obj['uid'],'cd_improvement':float(scores['official_reference']-scores['mv_independent'])})
        paired[split]={'object_count':len(deltas),'objects_improved':sum(x['cd_improvement']>0 for x in deltas),
                       'mean_cd_improvement':float(np.mean([x['cd_improvement'] for x in deltas])),'per_object':deltas}
    summary={'time':c.now(),'aggregates':aggregates,'paired':paired,'cases':records,
             'interpretation':'reference is frozen single-view official model; mv is trained independent multiview baseline, NOT relation model',
             'failure_policy':'empty/overlimit samples remain denominator with F=0 and canonical CD penalty sqrt(3)',
             'scope':'8 train / 4 val; 2 seeds per object are paired repeats, not independent scenes; GT camera and object poses'}
    c.write(c.OUT/'summary.json',summary)
    lines=['# HSSD TRELLIS.2多视图独立基线','',summary['scope'],'',
           'official_reference是冻结单图官方参考；mv_independent是训练后的多视图独立重建。它们的差异不能归因为跨物体关系模块。','',
           '| split | arm | 成功/总数 | CD-L1 ↓ | F@.01 ↑ | F@.02 ↑ |','| --- | --- | --- | ---: | ---: | ---: |']
    for split,arms in aggregates.items():
        for arm,r in arms.items():lines.append(f'| {split} | {arm} | {r["successes"]}/{r["cases"]} | {r["cd_l1"]:.6f} | {r["f@0.01"]:.6f} | {r["f@0.02"]:.6f} |')
    lines+=['','每物体先平均2个seed再统计；验证只有1个场景4个对象，不能据此作论文泛化结论。失败保留分母，CD惩罚sqrt(3)，F=0。',
            '','全部24张真实网格对照见figures/；world mesh在各case目录。GT位姿固定，无ICP或逐对象对齐。',
            '','下一阶段在相同多视图base、数据和预算下加入关系模型。此次没有训练关系分支。','']
    (c.OUT/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
