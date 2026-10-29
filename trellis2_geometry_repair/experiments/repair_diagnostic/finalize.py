import re
from .config import OUT,ROOT,read,write,now,digest
from .report import build


def main():
    closeout=read(OUT/'closeout.json');support=read(OUT/'support_audit.json')
    if not closeout['complete'] or len(support['records'])!=8:raise RuntimeError('complete cohort and support audit required')
    test_path=OUT/'validation/tests.log';test_text=test_path.read_text()
    match=re.search(r'Ran (\d+) tests',test_text)
    if match is None or '\nOK\n' not in test_text:raise RuntimeError('passing regression tests required')
    result=build();arms=result['arms']
    accepted={a:arms[a]['accepted'] for a in ('latent_input','mesh_input')}
    old_zero=sum(a['old_zero_gradient_fraction'] for a in result['audits'])/len(result['audits'])
    new_zero=sum(a['new_zero_gradient_fraction'] for a in result['audits'])/len(result['audits'])
    worse=sum(r['evidence_quality']['target_raw_mae']>r['evidence_quality']['base_raw_mae'] for r in result['records'])
    support_rows={(r['index'],r['seed']):r for r in support['records']}
    lines=['# 本轮最终结论','',f'汇总时间：{now()}。32/32组、8/8案例完成；96端点不变量校验通过；{match.group(1)}项测试通过。','',
       f'1. 已修复大误差深度监督的硬截断零梯度。原有效目标射线的零梯度比例案例均值约{100*old_zero:.2f}%，新损失为{100*new_zero:.2f}%。',
       f'2. 实际输入的latent与mesh两组分别{accepted["latent_input"]}/8、{accepted["mesh_input"]}/8接受；交付相对同一输入对齐base的CD改善分别为{100*arms["latent_input"]["endpoints"]["accepted"]["cd_relative_gain"]["mean"]:.4f}%、{100*arms["mesh_input"]["endpoints"]["accepted"]["cd_relative_gain"]["mean"]:.4f}%。修复代码问题不等于精度问题已解决。',
       f'3. 未过滤末步proposal的平均CD改善：latent输入{100*arms["latent_input"]["endpoints"]["proposal"]["cd_relative_gain"]["mean"]:.4f}%，mesh输入{100*arms["mesh_input"]["endpoints"]["proposal"]["cd_relative_gain"]["mean"]:.4f}%。后者可出现尖刺，不能当成交付成绩。',
       f'4. GT目标诊断的对应末步CD改善：latent{100*arms["latent_oracle"]["endpoints"]["proposal"]["cd_relative_gain"]["mean"]:.4f}%，mesh{100*arms["mesh_oracle"]["endpoints"]["proposal"]["cd_relative_gain"]["mean"]:.4f}%。它们不属于实际方法成绩，也不能证明整个latent表示的理论上限。',
       f'5. 混合DA3/NCC目标在{worse}/8案例上比原mesh的共同射线深度更差；更强的目标拟合可能放大错误证据。','',
       '固定SS距离审计另见support_audit.json和REPORT.md。相同冻结配准下，部分正确目标已经远离当前占据单元；另一些案例是GT在支持内、错误输入目标在支持外。',
       f'005/42与010/42的GT目标超出保守SS邻域比例分别为{100*support_rows[5,42]["gt"]["beyond_conservative_margin_fraction"]:.2f}%、{100*support_rows[10,42]["gt"]["beyond_conservative_margin_fraction"]:.2f}%；必须先复核配准，不能期望固定SS的小幅SLat更新恢复远处表面。',
       f'006/42、006/43的GT超出比例均为0%，但输入目标超出比例分别为{100*support_rows[6,42]["teacher"]["beyond_conservative_margin_fraction"]:.2f}%、{100*support_rows[6,43]["teacher"]["beyond_conservative_margin_fraction"]:.2f}%，提示不能按这些深度盲目扩展结构。',
       '这些情况分别需要复查全局配准/粗结构、改进证据可靠性、以及扩大生成式编辑能力，不能用统一增大步数或放宽门限解决。','',
       '下一版应先区分全局配准错误、缺失支持与支持内局部错误：确认配准后，支持外的可靠错误交给SS/局部再生成；支持内使用有生成先验的重采样与几何约束。',
       '自由mesh变形仅保留为诊断参照。先在当前开发物体验证，再固定新方法并扩展独立测试，不启动尚未定义目标的长训。','',
       '原8例pilot、失败集成尝试、初始检查与评估精度修正均保留，没有覆盖旧结论。','']
    (OUT/'FINAL_ANALYSIS.md').write_text('\n'.join(lines))
    marker='### 4.3 最终解释与补充支持审计'
    audit=ROOT/'审计.md'
    if marker not in audit.read_text():
        with audit.open('a') as stream:stream.write('\n\n'+marker+'\n\n'+'\n'.join(lines).rstrip()+'\n')
    receipt=dict(time=now(),complete=True,groups=32,cases=8,artifact_endpoints=96,
                 regression_tests=int(match.group(1)),test_log_sha256=digest(test_path),
                 support_audit_sha256=digest(OUT/'support_audit.json'),original_closeout=str(OUT/'closeout.json'),
                 analysis=str(OUT/'FINAL_ANALYSIS.md'),actual_input_accepted=accepted)
    write(OUT/'supplementary_closeout.json',receipt);print(receipt,flush=True)


if __name__=='__main__':main()
