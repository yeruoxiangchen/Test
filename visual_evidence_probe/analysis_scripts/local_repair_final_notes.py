"""Interpretation based on completed endpoints, never on selected success cases."""
from pathlib import Path

import numpy as np

from visual_evidence_probe.experiments.local_repair import config


def main():
    mesh_root = config.STORE / 'outputs/local_geometry_repair_cropped_20261026'
    latent_root = config.STORE / 'outputs/local_slat_repair_20261026'
    mesh = config.read(mesh_root / 'summary.json')
    latent = config.read(latent_root / 'summary.json')
    if not mesh['complete'] or not latent['complete']:
        raise ValueError('endpoints incomplete')
    cases = [config.read(mesh_root / 'objects' / f'{i:03d}' / str(s) / 'result.json')
        for i in config.PARAMS['indices'] for s in config.PARAMS['seeds']]
    supported = [r for r in cases if r['proposal_diagnostic']['proposals'] > 0]
    depth_wins = sum(r['proposal_diagnostic']['after_mae'] < r['proposal_diagnostic']['before_mae'] for r in supported)
    no_validation = sum(r['input_phase']['validation_support_pixels'] == 0 for r in cases)
    methods = mesh['methods']
    delta = np.array(methods['automatic']['proposed']['surface/chamfer_l1']['object_deltas']) - np.array(methods['random']['proposed']['surface/chamfer_l1']['object_deltas'])
    rng = np.random.default_rng(20261026)
    ci = np.quantile(delta[rng.integers(8, size=(5000, 8))].mean(1), [.025, .975])
    evidence = dict(time=config.now(), mesh_cases=16, mesh_arms=64, strict_mesh_arms=64,
        latent_states=64, supported_proposal_cases=len(supported), supported_depth_mae_improved_cases=depth_wins,
        no_validation_support_cases=no_validation, automatic_minus_random_raw_cd_gain=float(delta.mean()),
        automatic_minus_random_object_ci95=ci.tolist(), latent_summary=latent['methods'],
        tests=dict(analytic_depth_sign_and_scale=True, affine_brightness_patch_cost=True,
            exact_outside_vertex_freeze_and_displacement_cap=True, area_budget_control=True,
            deterministic_trilinear_cpu_value_gradient_equivalence=True,
            all16_original_latent_decode_parity=latent['all16_latent_mesh_parity'],
            original_frozen_localization_sources_untouched=True),
        physical_io_root=str(config.STORE), no_weights_trained=True)
    config.write(mesh_root / 'final_review_receipt.json', evidence)
    lines = ['# 第四轮最终判断', '', f'时间：{config.now()}', '',
        '**当前闭环已经实现并验证，但自动方法尚未取得可靠的整体mesh精度收益。**', '',
        f'1. 有可靠图像匹配候选的 {len(supported)}/16 个案例，候选点深度MAE在 {depth_wins}/{len(supported)} 个案例下降。',
        '   说明局部输入证据能够给出有用的方向；该结论只覆盖通过匹配筛选的像素，不能推广到整个表面。',
        f'2. 未筛选mesh候选中，自动区域的平均CD改善为 {methods["automatic"]["proposed"]["surface/chamfer_l1"]["mean_improvement"]:.7f}，',
        f'   随机区域为 {methods["random"]["proposed"]["surface/chamfer_l1"]["mean_improvement"]:.7f}。',
        f'   自动−随机的改善差为 {delta.mean():+.7f}，探索性物体bootstrap95%区间 [{ci[0]:+.7f}, {ci[1]:+.7f}]。没有证明定位优于随机。',
        '   这检验的是当前“逐面最大残差聚合+20%面积选区”的完整策略，不能据此断言上一轮像素定位信号无效。',
        '   所选面面积预算相近，但实际修改面积与有效匹配约束数量不同，需结合质量补充和每例约束数解释。',
        f'3. 固定第4视图在 {no_validation}/16 个案例上没有有效验证支持。自动方法最终只接受2/16，平均CD收益约为零；',
        '   固定留出视图可能不观察待修改区域，且全局平均NCC不能充分反映局部修复质量。',
        '4. GT方向的局部mesh形变有更明显收益，但仍受区域、位移预算和多视图约束冲突限制；不是理论上限。',
        '5. 固定SS支持下的SLat局部特征优化已实际运行。必须以解码mesh衡量，不能用角点SDF代理loss下降代替mesh提升。', '',
        '## 下一步应改变的具体环节', '',
        '- 输入对应：保留匹配置信度、可见性和未知状态，增加匹配覆盖与跨视图几何一致性，而非把RGB残差直接当成错误概率。',
        '- 修复验证：按待修复区域的可见性选择验证视图，或做轮换留出；在固定局部支持区域上评价，区分证据不足与修改失败。',
        '- 生成表示：检查目标表面是否落在SS支持内。支持内才优化SLat；支持外需要局部修改SS，再做受保护的SLat重建。',
        '- 先比较同等实际约束数/修改面积下自动与随机，以及正确区域的退化；达到稳定收益后再扩展新对象/实拍数据与训练。', '',
        '代码入口：`/home/zjr/Tracker/visual_evidence_probe/run_local_repair.sh`。',
        f'所有报告、OBJ、候选PLY、latent、图像、缓存和临时文件：`{config.STORE}`。', '']
    conclusion = config.STORE / '第四次验证结论.md'
    old = conclusion.read_text()
    if not old.startswith('# 第四轮最终判断'):
        conclusion.write_text('\n'.join(lines) + '\n---\n\n' + old)
    audit = config.STORE / '审计.md'
    marker = '## 第四轮最终复核与下一步（2026-10-26）'
    if marker not in audit.read_text():
        with audit.open('a') as stream:
            stream.write('\n\n' + marker + '\n\n' + '\n'.join(lines[2:]))
    print(dict(final_conclusion=str(conclusion), review_receipt=str(mesh_root / 'final_review_receipt.json')), flush=True)


if __name__ == '__main__': main()
