import hashlib
import os
from pathlib import Path
import numpy as np
from .config import OUT,STORE,PARAMS,ARMS,folder,read,write,now,register
from .report import build


def main():
    from .replay_audit import check
    replay=check(os.environ.get('CUDA_VISIBLE_DEVICES','0'))
    summary=build()
    if not summary['complete']:raise ValueError('32 diagnostic states incomplete')
    figures=read(OUT/'figures/manifest.json')
    if not figures['complete']:raise ValueError('diagnostic figures incomplete')
    if 'OK' not in (OUT/'validation/tests.log').read_text():raise ValueError('pairing tests failed')
    checked=0
    for index in PARAMS['indices']:
        for seed in PARAMS['seeds']:
            root=folder(index,seed);meta=read(root/'targets_complete.json')
            with np.load(root/'estimated_target.npz') as data:a=data['valid']
            with np.load(root/'oracle_target.npz') as data:b=data['valid']
            if not np.array_equal(a,b) or a[meta['heldout_views']].any():raise ValueError('unmatched support or heldout target leak')
            if hashlib.sha256(a.tobytes()).hexdigest()!=meta['support_sha256']:raise ValueError('frozen support changed')
            if not read(root/'decoder_parity.json')['passed']:raise ValueError('decoder parity failed')
            normalizations=[]
            for arm in ARMS:
                source=read(root/arm/'complete.json');result=read(root/arm/'result.json')
                if not source['privileged_diagnostic'] or not result['privileged_diagnostic']:raise ValueError('incorrect deployable labeling')
                if source['gt_depth_in_gradient']!=arm.startswith('oracle_'):raise ValueError('wrong depth source')
                if source['novel_geometry_in_gradient'] or source['heldout_rgb_in_gradient']:raise ValueError('optimization leak')
                if not source['frozen_rows_exact'] or not source['original_ss_fixed'] or not result['no_new_alignment']:raise ValueError('frozen support/frame violated')
                if source['support_sha256']!=meta['support_sha256']:raise ValueError('support differs across arms')
                normalizations.append(source['shared_normalization'])
                for step in PARAMS['checkpoints']:
                    for name in (f'step_{step:03d}.obj',f'step_{step:03d}_latent.pt'):
                        if not (root/arm/name).is_file():raise ValueError('checkpoint missing')
                        checked+=1
            if max(normalizations)-min(normalizations)>1.e-10:raise ValueError('loss normalization differs across evidence source')
            for key in ('candidate','proposal','targets'):
                if not Path(figures['cases'][f'{index:03d}_{seed}'][key]).is_file():raise ValueError('figure missing')
                checked+=1
    register('closeout')
    conclusion=STORE/'第九次机制诊断结论.md';conclusion.write_text((OUT/'REPORT.md').read_text())
    audit=STORE/'审计.md';before=audit.read_bytes();marker=b'<!-- matched-depth-editor-round9-diagnostic-complete -->'
    if marker not in before:
        section='\n\n# 9 第九轮机制诊断：深度准确性与编辑器约束\n\n'+marker.decode()+f'\n\n完成时间：{now()}\n\n'+(OUT/'REPORT.md').read_text()
        with audit.open('ab') as stream:stream.write(section.encode())
        if audit.read_bytes()[:len(before)]!=before:raise ValueError('prior audit changed')
    link=Path('/home/zjr/Tracker/visual_evidence_probe/第九次机制诊断结论.md')
    if not link.exists():link.symlink_to(conclusion)
    value=dict(time=now(),complete=True,states=32,checked_artifacts=checked,
        numerical_replay_complete=replay['complete'],
        all_arms_privileged_diagnostic=True,prior_audit_contents_preserved=True,
        audit_preappend_sha256=hashlib.sha256(before).hexdigest(),conclusion=str(conclusion))
    write(OUT/'closeout.json',value);print(value,flush=True)


if __name__=='__main__':main()
