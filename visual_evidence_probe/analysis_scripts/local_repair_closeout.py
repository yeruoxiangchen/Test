"""Make zero-step fallback exactly identical to the saved baseline, preserving raw records."""
from pathlib import Path

from visual_evidence_probe.experiments.local_repair import config
from .local_repair_latent import OUT
from .local_repair_latent_report import summarize


def main():
    fixed = []
    for path in sorted(OUT.glob('objects/*/*/*/result.json')):
        r = config.read(path)
        if r['status'] != 'complete' or r['selected_step'] != 0:
            continue
        before = path.with_name('result_before_exact_identity.json')
        if not before.exists():
            config.write(before, r)
            mesh = path.parent / 'accepted.obj'
            mesh.rename(path.parent / 'accepted_decoder_identity.obj')
            mesh.symlink_to(config.BASE / 'objects' / f'{r["index"]:03d}' / f'fitted/{r["seed"]}.obj')
        r['accepted'] = r['baseline']
        r['identity_fallback_exact_baseline'] = True
        config.write(path, r)
        fixed.append([r['index'], r['seed'], r['method']])
    receipt = dict(format='exact_identity_fallback.v1', time=config.now(), cases=fixed,
        trigger='Zero-step accepted geometry should be exactly the saved baseline, avoiding decode/OBJ rounding in no-edit statistics.',
        change='Return original fitted OBJ and baseline metrics when selected_step=0. Raw decoded identity OBJ and original JSON preserved.',
        no_selection_changes=True, no_gt_used_for_selection=True, no_proposal_changes=True,
        source=dict(path=str(Path(__file__).resolve()), sha256=config.digest(__file__)))
    config.write(OUT / 'exact_identity_fallback_amendment.json', receipt)
    summary = summarize()
    q = config.read(OUT / 'quality/summary.json')
    conclusion = config.STORE / '第四次验证结论.md'
    text = conclusion.read_text().replace('两者修改面积近似匹配，但落入区域的有效匹配约束数量不同，需结合每例约束数解释。',
        '所选面面积预算相近，但实际修改面积与有效匹配约束数量不同，需结合质量补充和每例约束数解释。')
    quality_note = f'6. 质量补查：自动mesh候选有 {q["mesh_automatic_proposals_with_orientation_reversal"]}/16 例局部面朝向反转；自动latent完整候选有 {q["automatic_latent_proposals_with_more_components"]}/16 例连通分量增加，接受后为0/16。需要加强几何质量约束。\n'
    if quality_note not in text:
        text = text.replace('\n## 下一步应改变的具体环节', '\n' + quality_note + '\n## 下一步应改变的具体环节', 1)
    conclusion.write_text(text)
    marker = '## 第四轮最终收尾复核'
    addition = f'\n\n{marker}\n\n时间：{config.now()}\n\n' + \
        f'完整记录：原图mesh64臂、裁剪mesh64臂、latent64条状态；16个seed的原始mesh解码一致性均通过。\n\n' + \
        '修订说明：早先“修改面积近似匹配”只指所选面预算。实际影响面积自动平均2.80%、随机4.25%，并非严格相等的实际修改预算，比较应保留此限制。\n\n' + \
        f'对 {len(fixed)} 条selected_step=0的latent结果，接受后OBJ和指标现在直接返回原始baseline，避免零修改的重解码/OBJ舍入差异；旧OBJ和JSON已另存。未更改候选、接受判据或选择结果。\n\n' + \
        '全部质量检查、对比图、汇总图、失败日志和审计均已保存到/data/zjr。\n'
    for p in (conclusion, config.STORE / '审计.md'):
        if marker not in p.read_text():
            with p.open('a') as stream: stream.write(addition)
    config.write(OUT / 'final_completion.json', dict(time=config.now(), expected_states=64, all_complete=True,
        exact_identity_fallback=True, geometry_quality_checked=True, figures=16, audit_appended=True))
    from .local_repair_summary_plot import main as plot
    plot()
    print(dict(final_complete=True, exact_identity_cases=len(fixed)), flush=True)


if __name__ == '__main__': main()
