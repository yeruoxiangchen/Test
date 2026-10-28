"""Final interpretation and data-only plots after the fixed experiment review."""
import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from visual_evidence_probe.common import read,write,now,binding
from visual_evidence_probe.experiments.localization.config import OUT
from visual_evidence_probe.experiments.localization.visuals import colors


def rgb_plot(root,key,zoom=False):
    source=root/'figures'/f'{key}{"_zoom" if zoom else ""}.jpg'
    with np.load(root/'scores'/f'{key}.npz') as a:
        score=a['cross_rgb'];valid=a['common'];union=a['pred_mask']|a['observed_mask']
    sheet=Image.open(source).convert('RGB');draw=ImageDraw.Draw(sheet)
    for view in range(len(score)):
        y=32+view*188
        panel=Image.fromarray(colors(score[view],valid[view])).resize((160,160),Image.Resampling.NEAREST)
        if zoom:
            ys,xs=np.where(union[view]);side=union.shape[-1]
            if len(xs):
                left,right=xs.min(),xs.max()+1;top,bottom=ys.min(),ys.max()+1
                pad=max(2,int(.12*max(right-left,bottom-top)))
                box=np.round(np.array([max(0,left-pad),max(0,top-pad),min(side,right+pad),min(side,bottom+pad)])*160/side).astype(int)
                panel=panel.crop(tuple(box)).resize((160,160),Image.Resampling.NEAREST)
        draw.rectangle((1280,y,1440,y+27),fill='white');draw.text((1284,y+4),'cross_rgb',fill='black')
        sheet.paste(panel,(1280,y+28))
    path=source.with_name(source.stem+'_with_rgb.jpg');sheet.save(path,quality=94)


def main():
    store=Path('/data/zjr/visual_evidence_probe')
    s=read(OUT/'summary.json');c=read(OUT/'input_crop_resolution_control/summary.json')
    for index in range(4,12):
        for seed in ('42','43'):
            rgb_plot(OUT/'objects'/f'{index:03d}',seed,zoom=True)
            rgb_plot(OUT/'input_crop_resolution_control/objects'/f'{index:03d}',seed)
    report=store/'第三次验证结论.md';text=report.read_text()
    intro='''结论：**输入与生成mesh之间存在可用于定位的信号，但当前直接/跨图DINO差异不够稳定。跨视图RGB一致性这个简单基线更值得作为后续起点。**

原图主结果中，跨视图RGB的AP=0.5978、AUROC=0.6417；边界基线AP=0.5058。7个可定义对象全部优于边界，物体级平均AP增益0.0920，探索性bootstrap95%区间[0.0526,0.1370]。芦笋原图内部像素不足，二分类指标不可定义，仍保留记录。

裁剪分辨率控制中8个对象均可定义：跨图RGB的AP=0.5562、AUROC=0.6654，边界基线AP=0.4550；7/8对象改善。这个控制存在已记录的重采样边界限制，不能替代原始主结果。

跨图DINO相对边界的AP增益，主结果为0.0389、裁剪控制为0.0134，两者物体级区间均跨0；固定DINO融合也未稳定改善。当前不能据此训练一个直接把DINO残差当作错误概率的自动修复器。

这些是**可见射线深度错误的定位指标**，不是mesh修复后的CD/F-score。下一步应围绕跨视图对应、可见性和光照/材质鲁棒性验证局部更新；正确几何仍可有外观残差，因此需要可拒绝修复的判据，不能无条件编辑最高分的20%。现有SLat体素索引只验证了坐标连接，尤其DINO融合索引不应当作已验证的纠错mask。

本轮统计、图像、体素和新报告均已切换到`/data/zjr/visual_evidence_probe/`。历史大OBJ/PLY仍在低优先级后台迁移；大mesh链接会随迁移完成而齐全，当前统计和图像不依赖继续读取home大文件。
'''
    head,rest=text.split('\n',1)
    text=head+'\n\n'+intro+'\n'+rest
    text+='\n## 补充RGB基线可视化\n\n以下各图第九列为cross_rgb；原DINO融合图保留在原始图中。颜色按各图5–95%范围显示，不是校准错误概率。\n\n'
    for index in range(4,12):
        text+=f'- 对象{index:03d}：[原图放大](outputs/visible_error_localization_20261026/objects/{index:03d}/figures/42_zoom_with_rgb.jpg)、[输入裁剪](outputs/visible_error_localization_20261026/input_crop_resolution_control/objects/{index:03d}/figures/42_with_rgb.jpg)。\n'
    report.write_text(text)
    source=Path('/home/zjr/Tracker/visual_evidence_probe/审计.md');audit=store/'审计.md'
    old=source.read_bytes()
    if not audit.exists():audit.write_bytes(old)
    elif not audit.read_bytes().startswith(old):raise ValueError('audit prefix differs; refuse overwrite')
    with audit.open('a') as f:
        f.write(f'''\n## {now()}：第三次实验完整复核与I/O迁移\n\n40条主记录（24自然、16控制）及16条全部新物体裁剪控制完成。主实验原始相机检查通过，
反投影最大误差6.87e-5像素；原始源码与30K权重保持一致。裁剪风筝视图IoU0.8526异常保留，
补充同尺度重采样验证IoU0.9738、原尺度0.9790，裁剪仿射误差<1e-3；0.90门槛未下调，
此项明确作为事后分辨率/重采样控制，不混同主实验。\n
主结果：crossRGB AP0.5978/AUROC0.6417，边界AP0.5058，7/7可定义对象AP改善；
裁剪控制：crossRGB AP0.5562/AUROC0.6654，边界AP0.4550，7/8对象改善。
DINO相对边界的物体级区间包含0，不能直接据此长训/自动编辑。无模型微调或latent修复，
只输出输入分数、可见支持、三维热图和经逆全局变换对应原SLat的token索引。\n
用户指出home I/O负担后，停止home大文件复核。新统计、图像、token和报告全部在/data/zjr；
15.1GB历史输出用低I/O优先级后台迁移，原件在传输/清单校验前保留。
缓存/临时文件统一/data/zjr/visual_evidence_probe；不删除其他作业共享的home缓存。
规则已写入Tracker/AGENTS.md。当前审计正文原内容逐字保留并移到数据盘，原审计路径保留链接。\n
详细结果：/data/zjr/visual_evidence_probe/第三次验证结论.md\n
迁移状态：/data/zjr/visual_evidence_probe/io_relocation.json\n''')
    link=source.with_name('审计.md.data_link')
    link.symlink_to(audit);os.replace(link,source)
    write(store/'finalization_receipt.json',dict(time=now(),source=binding(__file__),
        report=str(report),audit=str(audit),original_audit_prefix_sha256=hashlib.sha256(old).hexdigest(),
        audit_prefix_preserved=True,all_new_artifacts_on_data=True,legacy_migration='background'))
    (store/'IO_POLICY.md').write_text('''# I/O storage policy

Active data, outputs, model/runtime caches and temporary files: /data/zjr/visual_evidence_probe/
Source code: /home/zjr/Tracker/visual_evidence_probe/

Use `bash visual_evidence_probe/run_from_data.sh <analysis-module> [args]`.
This preserves logical frozen bindings while routing physical reads/writes to /data.
Model launches wait until the dedicated model-cache migration finishes, with no fallback to home caches.

Historical transfer: `systemctl --user status visual-evidence-io-migration-20261026.service`
Progress/state: io_relocation.json and migration/outputs.log in this directory.
No repeated full mesh SHA pass was performed after the frozen report had already verified it once.
''')
    print(dict(report=str(report),audit=str(audit),status='complete'),flush=True)


if __name__=='__main__':main()
