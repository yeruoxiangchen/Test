import hashlib
from pathlib import Path
from .config import OUT,STORE,PARAMS,ARMS,folder,read,write,now,register
from .report import build


def main():
    summary=build()
    if not summary['complete']:raise ValueError('development cohort incomplete')
    figures=read(OUT/'figures/manifest.json')
    if not figures['complete']:raise ValueError('figures incomplete')
    if 'OK' not in (OUT/'validation/tests.log').read_text():raise ValueError('constraint tests not passed')
    checked=0
    for index in PARAMS['dev_indices']:
        for seed in PARAMS['seeds']:
            root=folder(index,seed)
            if read(root/'input_complete.json')['uses_gt'] or not read(root/'decoder_parity.json')['passed']:raise ValueError('input provenance or parity failure')
            for arm in ARMS:
                optimization=read(root/arm/'complete.json');result=read(root/arm/'result.json')
                if not optimization['frozen_rows_exact'] or optimization['uses_gt']:raise ValueError('frozen rows or GT provenance failure')
                if optimization.get('rollbacks') and not optimization.get('optimizer_snapshot_load_deepcopied'):raise ValueError('uncorrected optimizer backup aliasing')
                if not result['gt_only_after_selection'] or not result['no_new_alignment']:raise ValueError('metric frame/provenance failure')
                for step in PARAMS['checkpoints']:
                    for name in (f'step_{step:03d}.obj',f'step_{step:03d}_latent.pt'):
                        if not (root/arm/name).is_file():raise ValueError('checkpoint missing')
                        checked+=1
                if result['selected_step']==0 and (root/arm/'accepted.obj').resolve()!=(root/'original.obj').resolve():raise ValueError('no-op fallback differs')
            for key in ('meshes','weights'):
                if not Path(figures['cases'][f'{index:03d}_{seed}'][key]).is_file():raise ValueError('figure missing')
                checked+=1
    register('closeout')
    audit=STORE/'审计.md';before=audit.read_bytes();marker=b'<!-- constrained-evidence-round8-development-complete -->'
    section='\n\n# 8 第八轮开发实验：三维表面锚点与深度可靠性\n\n'+marker.decode()+'\n\n'
    section+=f'完成时间：{now()}。8个已检查开发物体×2 seeds×3组=48条结果。不是独立验证集。\n\n'
    section+=(OUT/'REPORT.md').read_text()
    section+='\n'+(OUT/'REAL_INPUT_AUDIT.md').read_text()+'\n'
    if marker not in before:
        with audit.open('ab') as stream:stream.write(section.encode())
        if audit.read_bytes()[:len(before)]!=before:raise ValueError('old audit content changed')
    conclusion=STORE/'第八次开发验证结论.md';conclusion.write_text((OUT/'REPORT.md').read_text())
    link=Path('/home/zjr/Tracker/visual_evidence_probe/第八次开发验证结论.md')
    if not link.exists():link.symlink_to(conclusion)
    value=dict(time=now(),complete=True,states=48,checked_artifacts=checked,
        prior_audit_bytes_preserved=True,audit_preappend_sha256=hashlib.sha256(before).hexdigest(),
        conclusion=str(conclusion),scope='development only; new holdout required for any generalization claim')
    write(OUT/'closeout.json',value);print(value,flush=True)


if __name__=='__main__':main()
