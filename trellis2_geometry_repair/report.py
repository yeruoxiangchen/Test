from collections import defaultdict
import numpy as np
from .config import OUT,ROOT,PARAMS,folder,read,write,now


def grouped(rows,function):
    groups=defaultdict(list)
    for r in rows:
        value=function(r)
        if value is not None and np.isfinite(value):groups[r['uid']].append(value)
    return {uid:float(np.mean(v)) for uid,v in groups.items()}


def build():
    rows=[read(folder(i,s)/'result.json') for i in PARAMS['indices'] for s in PARAMS['seeds'] if (folder(i,s)/'result.json').exists()]
    groups={}
    for ep in ('accepted','candidate','proposal'):
        cd=grouped(rows,lambda r:r['original']['surface']['chamfer_l1']-r['outputs'][ep]['surface']['chamfer_l1'])
        cd_relative=grouped(rows,lambda r:(r['original']['surface']['chamfer_l1']-r['outputs'][ep]['surface']['chamfer_l1'])/
                            max(r['original']['surface']['chamfer_l1'],1.e-8))
        raw_cd=grouped(rows,lambda r:(r['outputs']['raw']['surface']['chamfer_l1']-r['outputs'][ep]['surface']['chamfer_l1'])/
                       max(r['outputs']['raw']['surface']['chamfer_l1'],1.e-8))
        depth=grouped(rows,lambda r:(r['original']['novel_depth']['bad']-r['outputs'][ep]['novel_depth']['bad'])/max(r['original']['novel_depth']['bad'],1.e-8)
                      if r['original']['novel_depth']['bad'] is not None else None)
        groups[ep]=dict(cd_gain=float(np.mean(list(cd.values()))) if cd else None,
                       aligned_cd_relative_gain=float(np.mean(list(cd_relative.values()))) if cd_relative else None,
                       bad_depth_relative_gain=float(np.mean(list(depth.values()))) if depth else None,
                       raw_base_cd_relative_gain=float(np.mean(list(raw_cd.values()))) if raw_cd else None,
                       cd_by_object=cd,aligned_cd_relative_by_object=cd_relative,
                       bad_depth_by_object=depth,raw_base_cd_by_object=raw_cd)
    value=dict(time=now(),cases=len(rows),expected=8,complete=len(rows)==8,accepted=sum(r['accepted_edit'] for r in rows),
               groups=groups,records=rows,development_pilot=True,real_photos=False)
    write(OUT/'summary.json',value)
    text=['# TRELLIS.2原生几何局部纠错pilot','',f'时间：{now()}；评估 {len(rows)}/8；接受 {value["accepted"]}。','',
          '4个已检查开发物体×2种子；原生单图生成，输入多视图对齐和纠错。GT仅在输入决策封存后评估。',
          '主比较共享同一全局变换；原始未对齐输出另保留。每次优化和选择都使用原生解码mesh；固定离散分支只用于梯度数值检查。','',
          '|输出|相对输入对齐base的CD绝对改善|相对输入对齐base的CD下降(%)|相对原始base的CD下降(%)|原错误像素深度下降(%)|',
          '|---|---:|---:|---:|---:|']
    def percent(value):return f'{100*value:.6f}' if value is not None else 'NA'
    for ep,g in groups.items():
        text.append(f'|{ep}|{g["cd_gain"]}|{percent(g["aligned_cd_relative_gain"])}|{percent(g["raw_base_cd_relative_gain"])}|{percent(g["bad_depth_relative_gain"])}|')
    text+=['','正值为改善。当前开发pilot不能作为独立泛化或真实拍摄结论。技术恢复使用生成几何目标，不能当成重建精度收益。',
           '“相对原始base”包含全局旋转、平移和尺度对齐，不能将该列的收益归因于局部latent编辑。',
           'accepted为实际交付，包含所有回退；candidate和proposal未通过门限时不能当成交付成绩。',
           '逐对象、原始/对齐基线、退化和回退均保留在summary.json。','']
    (OUT/'REPORT.md').write_text('\n'.join(text));print(dict(cases=len(rows),complete=value['complete']),flush=True)
    return value


def verify_artifacts():
    import torch
    torch.set_num_threads(2);checked=0
    for i in PARAMS['indices']:
        for s in PARAMS['seeds']:
            root=folder(i,s);original=torch.load(root/'original_latent.pt',weights_only=False)
            with np.load(root/'evidence_sealed.npz') as a:active=torch.tensor(a['active'])
            for ep in ('accepted','candidate','proposal'):
                result=torch.load(root/f'{ep}_latent.pt',weights_only=False)
                if not torch.equal(result['coords'],original['coords']):raise RuntimeError('SS coordinate mismatch')
                if not torch.equal(result['features'][~active],original['features'][~active]):raise RuntimeError('outside latent mismatch')
                if not bool(torch.isfinite(result['features']).all()):raise RuntimeError('nonfinite latent')
                if not (root/f'{ep}.obj').is_file():raise RuntimeError('missing native mesh')
                checked+=1
    write(OUT/'validation/artifact_checks.json',dict(time=now(),passed=True,checked=checked,ss_fixed=True,outside_rows_exact=True))
    return dict(passed=True,endpoints_checked=checked)


if __name__=='__main__':build()
