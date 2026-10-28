"""Record completed round-seven interpretation without rewriting old audit."""
import hashlib
from pathlib import Path
from .config import OUT,STORE,PRIOR,read,write,now


def main():
    summary=read(PRIOR/'summary.json')
    if not summary['complete']:raise ValueError('round seven incomplete')
    validation=summary['groups']['validation'];dense=validation['dense_observed'];sparse=validation['sparse_observed']
    de=dense['endpoints']['accepted'];sp=sparse['endpoints']['accepted']
    difference=summary['factorial']['validation']['density_under_observed']['surface.chamfer_l1']
    text=f'''# 第七轮完整结果补充分析与第八轮动机

时间：{now()}

第七轮54个案例、216组指标、54套科学对照图及审计已收尾。以下补充不修改原始结果或原有审计文字。

稀疏证据配合新接受门在16物体/32案例上接受{sparse['accepted_edits']}例，新视角局部深度误差平均下降{sp['novel.local_depth_relative_reduction']['mean']:.2%}；
稠密证据配合相同接受门接受{dense['accepted_edits']}例，下降{de['novel.local_depth_relative_reduction']['mean']:.2%}。
两者对原始30K的全局CD均有小幅改善，但稠密减稀疏的配对CD改善差为{difference['mean']:+.8f}，95%物体bootstrap区间{difference['ci95']}，跨零。
因此，当前结果支持输入约束的局部纠错和接受策略更新；尚未证明加入DA3稳定优于稀疏证据。

稠密分支训练目标选出的候选平均新视角局部深度下降{dense['endpoints']['candidate']['novel.local_depth_relative_reduction']['mean']:.2%}，接受后只保留上述约2%的收益。
事后GT诊断发现16个同时改善CD和新视角局部深度的训练候选未被接受。全部被拒候选中，轮廓条件失败10例、区域外漂移失败9例，原因可以重叠。
这不等于这些候选都应当放行：整体指标改善仍可能伴随轮廓偏移、外部表面变化或额外褶皱。应优先约束编辑方向和几何影响范围，而不是直接放宽接受门。
表面CD采用50000点采样，极小差异还须结合固定相机深度和实际渲染判断。

## 真实电钻审查补充

原始GT投影对标注完整mask的平均IoU约0.982，K/R/t核对与1.7--2.0mm的传感器深度中位误差支持基本坐标和单位链路正确。
A组DA3的可靠射线深度中位数仍约为GT的1.322倍；跨视图一致不保证绝对深度正确。尚不能把误差唯一归因为轨迹尺度拟合。
同时，原30K在真实组的观测域轮廓IoU仅约0.12--0.29。第七轮真实组使用identity全局变换，合成组有输入mask整体拟合；真实试点混合了初始整体注册问题，不能当作纯遮挡或纯局部纠错的因果测试。
真实后续测试应先核对输入约束的整体注册，所有分支共用同一变换，不能用GT ICP改善成绩。
详细审查：{OUT/'REAL_INPUT_AUDIT.md'}。

## 已启动的后续实验

第八轮代码：/home/zjr/Tracker/visual_evidence_probe/experiments/constrained_evidence。
输出：{OUT}。
三组为第七轮原稠密参考、三维表面锚点和信任区域约束、再加训练视图深度可靠性权重。
原30K、解码器和SS固定；每新分支80次尝试更新，heldout不参与梯度、权重或信任区域判断。
先四物体seed42开发试跑，再以同一配置完成8个已检查开发物体×2 seeds。开发结果不是新独立验证，若显示收益，再冻结新的未检查物体。
三维锚点使用三角面片表面最近点近似；5项表面距离、梯度及heldout隔离测试通过，单样本实际GPU链路已跑通。
当前只完成部分案例，不能宣布第八轮整体有效；首张对照图仍能看到上部褶皱，需要如实保留这一视觉问题。
'''
    path=STORE/'第七轮补充分析_20261028.md';path.write_text(text)
    audit=STORE/'审计.md';before=audit.read_bytes();marker=b'<!-- round7-interpretation-supplement-20261028 -->'
    if marker not in before:
        with audit.open('ab') as stream:stream.write(('\n\n# 7.1 第七轮补充分析\n\n'+marker.decode()+'\n\n'+text).encode())
        if audit.read_bytes()[:len(before)]!=before:raise ValueError('prior audit text changed')
    link=Path('/home/zjr/Tracker/visual_evidence_probe/第七轮补充分析.md')
    if not link.exists():link.symlink_to(path)
    write(OUT/'audit_round7_supplement.json',dict(time=now(),path=str(path),prior_audit_bytes_preserved=True,
        audit_preappend_sha256=hashlib.sha256(before).hexdigest()))
    print(dict(path=str(path),audit=str(audit),prior_text_preserved=True),flush=True)


if __name__=='__main__':main()
