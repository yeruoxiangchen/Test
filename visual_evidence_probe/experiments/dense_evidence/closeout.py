"""Verify the complete cohort and append an immutable audit section."""
import hashlib
import re

from .config import OUT,STORE,ARMS,PARAMS,folder,registry,read,write,now,register
from .report import build,fmt


def main():
    summary=build(final=True);checks=[]
    figures=read(OUT/'figures/manifest.json')
    if not figures['complete']:raise ValueError('figure cohort incomplete')
    for obj in registry()['rows']:
        index=obj['index']
        for seed in PARAMS['seeds']:
            root=folder(index,seed)
            state=read(root/'input_complete.json');parity=read(root/'decoder_parity.json')
            if state['gt_used'] or not parity['passed']:raise ValueError('input provenance or decoder parity failure')
            evidence=read(root/'evidence_complete.json')
            if set(evidence['train_views'])&set(evidence['heldout_views']):raise ValueError('train/heldout overlap')
            for source in ('sparse','dense'):
                optimized=read(root/source/'complete.json')
                if not optimized['frozen_rows_exact'] or optimized['uses_gt']:raise ValueError('optimization provenance failure')
                for step in PARAMS['checkpoints']:
                    for name in (f'step_{step:03d}.obj',f'step_{step:03d}_latent.pt'):
                        path=root/source/name
                        if not path.is_file():raise ValueError(f'missing checkpoint {path}')
                        checks.append(str(path))
            for arm in ARMS:
                selection=read(root/arm/'selection.json');result=read(root/arm/'result.json')
                if not result['gt_used_only_after_selection'] or not result['original_frame_preserved']:raise ValueError('evaluation provenance failure')
                expected=root/selection['source']/f'step_{selection["selected_step"]:03d}.obj'
                if (root/arm/'accepted.obj').resolve()!=expected.resolve():raise ValueError('selected mesh link mismatch')
                if not selection['accepted_edit'] and (root/arm/'accepted.obj').resolve()!=(root/'original.obj').resolve():raise ValueError('fallback differs from original')
                checks.append(str(root/arm/'result.json'))
            fig=figures['cases'][f'{index:03d}_{seed}']
            for name in ('meshes','errors','inputs','evidence'):
                from pathlib import Path
                if not Path(fig[name]).is_file():raise ValueError('missing figure')
                checks.append(fig[name])
            if obj['kind']=='real_scene':
                from pathlib import Path
                if not fig.get('raw_inputs') or not Path(fig['raw_inputs']).is_file():raise ValueError('real raw input panel missing')
                checks.append(fig['raw_inputs'])
    quality=read(OUT/'depth_quality.json')
    if len(quality['records'])!=len(registry()['rows'])*2:raise ValueError('depth diagnostics incomplete')
    if any(r.get('metric_definition')!='clipped_fixed_missing_v2' for r in quality['records']):raise ValueError('stale provider quality metric definition')
    tests=(OUT/'validation/tests.log').read_text()
    if not re.search(r'Ran \d+ tests',tests) or not re.search(r'^OK$',tests,re.M):raise ValueError('protocol tests not passed')
    conclusion=['# 第七次验证结论','',f'完成时间：{now()}','',
        '已完成8个开发物体、16个新局部修复验证物体，以及同一YCB-V电钻3组输入，均为2 seeds和4组对照。',
        '冻结原30K，固定SS，局部编辑SLat；稠密组采用NCC锚点加可靠DA3深度；所有输入决策先于对应GT评估。','',summary['interpretation'],'']
    for split,title in (('development','开发集'),('validation','新验证集'),('real_pilot','真实电钻试点')):
        conclusion += [f'## {title}','','|组别|接受/案例|新视角局部深度下降|CD改善|','|---|---:|---:|---:|']
        for arm in ARMS:
            group=summary['groups'][split][arm];e=group['endpoints']['accepted'];relative=e['novel.local_depth_relative_reduction']['mean']
            rel='NA' if relative is None else f'{relative:+.2%}'
            conclusion.append(f'|{arm}|{group["accepted_edits"]}/{group["cases"]}|{rel}|{fmt(e["surface.chamfer_l1"]["mean"])}|')
        conclusion.append('')
    conclusion += ['真实试点只有一个物体，使用BOP标注位姿、完整包围框和模型尺度；不能据此声称无标注真实泛化或总体抗遮挡能力。',
        '极小CD差异可能受表面采样影响；需同时检查固定相机的局部深度变化。候选改善不能替代实际接受结果。',
        f'完整报告：`{OUT/"REPORT.md"}`',f'结果和配对bootstrap：`{OUT/"summary.json"}`',
        f'同相机渲染：`{OUT/"figures"}`',f'协议：`/home/zjr/Tracker/visual_evidence_probe/experiments/dense_evidence/PROTOCOL.md`','']
    conclusion_path=STORE/'第七次验证结论.md';conclusion_path.write_text('\n'.join(conclusion))
    audit=STORE/'审计.md';old=audit.read_bytes();marker=b'<!-- dense-evidence-round7-complete -->'
    if marker not in old:
        section=('\n\n# 7 第七轮：稠密输入几何与观测接受策略\n\n'+marker.decode()+'\n\n'+
            '\n'.join(conclusion[2:])+'\n').encode()
        with audit.open('ab') as handle:handle.write(section)
        if audit.read_bytes()[:len(old)]!=old:raise ValueError('audit prior contents changed')
        old_preserved=True
    else:old_preserved=True
    source_receipt=register('closeout')
    receipt=dict(time=now(),complete=True,states=summary['states'],checked_artifacts=len(checks),
        checks=checks,audit=str(audit),prior_audit_bytes_preserved=old_preserved,
        audit_preappend_bytes=len(old),audit_preappend_sha256=hashlib.sha256(old).hexdigest(),
        conclusion=str(conclusion_path),figures=figures['expected'],source_snapshot=source_receipt['time'])
    write(OUT/'closeout.json',receipt)
    from pathlib import Path
    link=Path('/home/zjr/Tracker/visual_evidence_probe/第七次验证结论.md')
    if not link.exists():link.symlink_to(conclusion_path)
    print({key:value for key,value in receipt.items() if key!='checks'},flush=True)


if __name__=='__main__':main()
