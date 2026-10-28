"""Verify saved states, then append the dated sixth experiment to the audit."""
import json
from pathlib import Path

import numpy as np
import torch

from .config import ARMS, OUT, STORE, BASE, PARAMS, folder, read, write, digest, now, register
from .report import build


def verify_latents(index, seed):
    root = folder(index, seed)
    original = torch.load(root / 'original_latent.pt', map_location='cpu', weights_only=False)
    with np.load(root / 'input_sealed.npz') as a: active = torch.as_tensor(a['active_rows'])
    std = torch.tensor(read(STORE / 'outputs/local_slat_repair_20261026/model_binding.json')['normalization']['std'])
    checks = []
    for arm in ARMS:
        result = read(root / arm / 'result.json')
        for endpoint in ('proposal', 'selected', 'accepted'):
            current = torch.load(root / arm / f'{endpoint}_latent.pt', map_location='cpu', weights_only=False)
            if not torch.equal(current['coords'], original['coords']): raise ValueError('SS coordinate support changed')
            if not torch.equal(current['features'][~active], original['features'][~active]): raise ValueError('frozen rows changed')
            normalized = (current['features'].float() - original['features'].float()) / std[None]
            maximum = float(normalized.abs().max())
            if maximum > PARAMS['normalized_feature_cap'] + .002: raise ValueError('latent bound violated')
            fallback = endpoint == 'accepted' and not result['input_acceptance']['accepted']
            if fallback and not torch.equal(current['features'], original['features']):
                raise ValueError('rejected candidate did not restore original latent exactly')
            checks.append(dict(arm=arm, endpoint=endpoint, max_normalized_update=maximum,
                support_exact=True, frozen_rows_exact=True, exact_fallback=bool(fallback)))
    sealed = read(root / 'input_sealed.json')
    if digest(root / 'input_sealed.npz') != sealed['sealed_arrays_sha256']:
        raise ValueError('input seal changed after GT access')
    camera = read(folder(index) / 'novel_cameras.json')
    if camera['minimum_input_angle_degrees'] < 10: raise ValueError('evaluation cameras overlap input')
    return dict(index=index, seed=seed, checks=checks, input_seal_unchanged=True,
        new_camera_minimum_angle=camera['minimum_input_angle_degrees'])


def main():
    summary = build(final=True)
    manifest = read(OUT / 'figures/manifest.json')
    if not manifest.get('complete'): raise ValueError('figures incomplete')
    required, checks = [], []
    for index in PARAMS['indices']:
        for seed in PARAMS['seeds']:
            root = folder(index, seed)
            fit_source = BASE / 'objects' / f'{index:03d}' / f'fitted/{seed}.json'
            write(root / 'coordinate_frames.json', dict(
                latent_frame='native TRELLIS sparse grid cube', mesh_frame='shared input-camera world/canonical target frame',
                latent_native_to_mesh_world=read(fit_source)['chosen']['transform'],
                transform_source=str(fit_source), fit_uses_gt=False,
                all_arms_use_same_transform=True, additional_alignment=False,
                cameras='original crop cameras or object-level novel_cameras.json'))
            required += [root / 'complete.json', root / 'input_finished_before_gt.json', root / 'oracle_targets.json']
            for arm in ARMS:
                required += [root / arm / filename for filename in ('proposal.obj', 'selected.obj', 'accepted.obj',
                    'proposal_latent.pt', 'selected_latent.pt', 'accepted_latent.pt', 'optimization.json', 'result.json')]
            figure = manifest['cases'][f'{index:03d}_{seed}']
            required += [Path(figure[key]) for key in ('meshes', 'errors', 'inputs')]
            checks.append(verify_latents(index, seed))
    missing = [str(path) for path in required if not path.exists()]
    if missing: raise ValueError(f'missing artifacts: {missing[:8]}')
    for path in [Path(manifest['summary'])] + [Path(r[k]) for r in manifest['cases'].values() for k in ('meshes', 'errors', 'inputs')]:
        from PIL import Image
        with Image.open(path) as image: image.verify()
    from .interpretation import main as interpret
    prose, recommendation = interpret(summary)
    body = (OUT / 'REPORT.md').read_text()
    (OUT / 'REPORT.md').write_text(prose + body)
    snapshot = register('closeout')
    import shutil
    for source in snapshot['sources']:
        source_path = Path(source['path'])
        archive_path = OUT / 'source_archive' / source_path.relative_to('/home/zjr/Tracker')
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, archive_path)
    model_binding = STORE / 'outputs/local_slat_repair_20261026/model_binding.json'
    shutil.copyfile(model_binding, OUT / 'model_binding.json')
    validation_log = OUT / 'validation/tests.log'
    log_text = validation_log.read_text()
    if 'Ran 8 tests' not in log_text or not log_text.rstrip().endswith('OK'):
        raise ValueError('final eight protocol validations have not passed')
    value = dict(time=now(), complete=True, cases=16, states=64, figures=49,
        required_artifacts_checked=len(required), missing=missing,
        latent_and_seal_verification=checks, no_model_weight_training=True,
        all_fallbacks_exact=True, ss_support_unchanged=True,
        validation_tests=8, validation_log_sha256=digest(validation_log),
        new_alignment=False, independent_objects=False,
        explanatory_note='A/B are privileged diagnostics, C is input-only. New views are independent only of this edit optimization, not a new-object test set.')
    write(OUT / 'completion.json', value)
    report = (OUT / 'REPORT.md').read_text()
    conclusion = STORE / '第六次验证结论.md'
    conclusion.write_text(report)
    audit = STORE / '审计.md'
    marker = '<!-- geometry_evidence_round6_complete_20261027 -->'
    original_audit_bytes = audit.read_bytes()
    current = original_audit_bytes.decode('utf-8')
    audit_before = dict(bytes=len(original_audit_bytes), sha256=digest(audit))
    if marker not in current:
        rows = ['','', '# 6，第六轮：真实局部几何错误与输入约束诊断', '', f'时间：{value["time"]}', '', marker, '',
            '8个已检查Omni物体×2 seeds×4组=64条结果完成，49张图。30K权重冻结；仅优化原SS上的局部latent。',
            '新增80步/0.5std预算，A密集GT深度+法线、A-depth密集GT深度、B原匹配像素处GT深度、C输入估计深度。',
            '每案例的C分支在访问本案例GT之前完成；输入封存校验、原SS支持、未更新行及拒绝后原latent恢复均已验证。', '',
            '|组别|原输入局部深度下降|新视角局部深度下降|CD改善|选出/输入门接受|', '|---|---:|---:|---:|---:|']
        for name in ARMS:
            arm = summary['all_cases'][name]; endpoint = arm['endpoints']['selected']
            i = endpoint['input.local_depth_relative_reduction']['mean']
            n = endpoint['novel.local_depth_relative_reduction']['mean']
            cd = endpoint['surface.chamfer_l1']['mean']
            rows.append(f'|{name}|{i:+.2%}|{n:+.2%}|{cd:+.6f}|{arm["selected_edits"]}/{arm["accepted_edits"]}|')
        rows += ['', '以上为训练目标选择的候选；A/B含GT，不能作为可部署方法成绩。8个新视角只用于最终评价，物体仍为已检查开发集。',
            'B/C无GT交点情况及严格配对子集单列。失败、无操作、被输入门拒绝的案例全部保留。',
            recommendation,
            f'完整报告：`{conclusion}`；结果：`{OUT}`。', '']
        with audit.open('a') as stream: stream.write('\n'.join(rows))
    if not audit.read_bytes().startswith(original_audit_bytes): raise ValueError('previous audit text was modified')
    value.update(audit_before=audit_before, audit_after_sha256=digest(audit), audit_prefix_preserved=True)
    write(OUT / 'completion.json', value)
    link = Path('/home/zjr/Tracker/visual_evidence_probe/第六次验证结论.md')
    if not link.exists(): link.symlink_to(conclusion)
    print(json.dumps({key: value[key] for key in ('time', 'complete', 'cases', 'states', 'figures')}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
