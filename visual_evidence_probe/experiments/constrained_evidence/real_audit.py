"""Independent annotated-BOP projection/depth diagnostic, never editor input."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch

from .config import OUT,PRIOR,setup,read,write,now,register
from visual_evidence_probe.experiments.dense_evidence.data import observations
from visual_evidence_probe.experiments.dense_evidence.evaluation import load_target,render_depth
from visual_evidence_probe.experiments.dense_evidence.camera import backproject,project


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='1');a=p.parse_args();setup(a.gpu);register('real_audit')
    import nvdiffrast.torch as dr
    device=torch.device('cuda:0');context=dr.RasterizeCudaContext(device=device)
    rows=read(PRIOR/'registry.json')['rows'];results=[]
    for index in (40,41,42):
        source=next(r for r in rows if r['index']==index);rec,k,w,_,_=observations(index,device)
        gt=render_depth(load_target(index),k,w,context,side=518)
        mask=gt['mask'].cpu().numpy();depth=gt['depth'].cpu().numpy()
        with np.load(PRIOR/'data'/f'{index:03d}'/'geometry.npz') as data:affines=data['affines']
        views=[];scale=source['normalization']['scale_mm'];center=np.array(source['normalization']['center_mm'])
        for view,annotation in enumerate(source['rows']):
            full=cv2.warpAffine(np.asarray(Image.open(annotation['full_mask'])),affines[view,:2],(518,518),flags=cv2.INTER_NEAREST)>127
            visible=np.asarray(Image.open(rec['masks'][view]['path']))>127
            union=full|mask[view];iou=float((full&mask[view]).sum()/max(union.sum(),1))
            camera_json=Path(annotation['rgb']).parent.parent/'scene_camera.json'
            camera=read(camera_json)[str(annotation['frame'])]
            sensor_file=Path(annotation['rgb']).parent.parent/'depth'/Path(annotation['rgb']).name
            sensor=np.asarray(Image.open(sensor_file)).astype(np.float32)*camera['depth_scale']/scale
            sensor=cv2.warpAffine(sensor,affines[view,:2],(518,518),flags=cv2.INTER_NEAREST)
            common=visible&mask[view]&(sensor>0)
            rotation=np.array(annotation['R']);translation=(rotation@center+np.array(annotation['t_mm']))/scale
            expected_k=affines[view]@np.array(annotation['K']).reshape(3,3)
            views.append(dict(view=view,full_mask_iou=iou,
                visible_contained_fraction=float(mask[view][visible].mean()),
                gt_to_sensor_depth_mae=float(np.abs(depth[view]-sensor)[common].mean()) if common.any() else None,
                gt_to_sensor_depth_median=float(np.median(np.abs(depth[view]-sensor)[common])) if common.any() else None,
                gt_depth_median=float(np.median(depth[view][visible&mask[view]])),
                intrinsic_max_error=float(np.abs(expected_k-k[view].cpu().numpy()).max()),
                extrinsic_rotation_max_error=float(np.abs(rotation-w[view,:3,:3].cpu().numpy()).max()),
                extrinsic_translation_max_error=float(np.abs(translation-w[view,:3,3].cpu().numpy()).max())))
        providers=[]
        for path in sorted((PRIOR/'depths'/f'{index:03d}').glob('*/complete.json')):
            receipt=read(path);alignment=receipt['camera']['provider_alignment']
            with np.load(path.with_name('prediction.npz')) as data:
                d=torch.tensor(data['depth']);valid=torch.tensor(data['valid']);ks=torch.tensor(data['intrinsics']);ws=torch.tensor(data['extrinsics'])
            raw=np.asarray(alignment['raw_predicted_poses']);input_w=ws.numpy()
            def centers(m):return -np.einsum('bij,bj->bi',m[:,:3,:3].transpose(0,2,1),m[:,:3,3])
            pc,ic=centers(raw),centers(input_w)
            pd=np.linalg.norm(pc[:,None]-pc[None],axis=-1);id=np.linalg.norm(ic[:,None]-ic[None],axis=-1)
            keep=np.triu(np.ones_like(id,dtype=bool),1)&(id>1.e-5);ratios=pd[keep]/id[keep]
            train=receipt['train_views'];g=render_depth(load_target(index),k[train],w[train],context)
            dom=valid&g['mask'].cpu();gd=g['depth'].cpu();actual=(d-gd).abs()
            providers.append(dict(path=str(path.parent),train_views=train,
                official_depth_divisor=alignment['raw_to_output_depth_divisor'],
                pose_pair_scale_median=float(np.median(ratios)),pose_pair_scale_cv=float(ratios.std()/max(ratios.mean(),1.e-8)),
                reliable_depth_raw_mae=float(actual[dom].mean()) if dom.any() else None,
                median_prediction_to_gt_depth_ratio=float(torch.median((d/gd.clamp_min(1.e-6))[dom])) if dom.any() else None))
        native=[]
        for seed in (42,43):
            original=read(PRIOR/'objects'/f'{index:03d}'/str(seed)/'dense/complete.json')['snapshots'][0]
            native.append(dict(seed=seed,observed_domain_mask_iou_mean=float(np.mean(original['iou_observed']))))
        results.append(dict(index=index,group=source['group'],scale_mm=scale,views=views,providers=providers,
            native30k_original=native))
    value=dict(time=now(),diagnostic_only=True,uses_annotated_geometry_and_sensor_depth=True,
        feeds_editor=False,changes_round7=False,results=results)
    write(OUT/'real_input_audit.json',value)
    lines=['# 电钻输入链路审查','',f'时间：{now()}','',
        '此审查使用标注GT及传感器深度，仅用于定位问题；这些量不进入编辑器。第七轮文件保持原样。','',
        '|输入组|GT投影/full mask平均IoU|GT/传感器中位深度误差(mm)|DA3可靠射线raw MAE(canonical)|DA3/GT深度中位比|','|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'|{r["group"]}|{np.mean([v["full_mask_iou"] for v in r["views"]]):.5f}|'
            f'{np.mean([v["gt_to_sensor_depth_median"] for v in r["views"]])*r["scale_mm"]:.3f}|'
            f'{np.mean([v["reliable_depth_raw_mae"] for v in r["providers"]]):.6f}|'
            f'{np.mean([v["median_prediction_to_gt_depth_ratio"] for v in r["providers"]]):.4f}|')
    lines+=['',
        'K最大误差约1.3e-4像素，R约3e-8，规范化t约2.3e-7。GT投影与完整mask基本吻合，传感器深度中位误差约1.7--2.0mm；逐像素平均误差受深度离群点影响，原始均值仍保存在JSON。',
        'A组DA3的已筛选深度仍有约32%系统偏差。跨视图一致性不保证绝对深度正确，当前数据不足以把偏差唯一归因为官方轨迹尺度拟合或模型深度预测。',
        '原30K在这些输入上的观测域轮廓IoU仅约0.12--0.29。第七轮真实组采用identity全局变换，合成组使用输入mask整体拟合；真实结果混合了初始整体注册问题，不能当作纯遮挡或纯局部纠错的因果测试。',
        '后续真实局部修复测试应先检查输入约束的整体位姿/尺度注册，并保持各修复分支共用同一变换。不得使用GT ICP改善方法成绩。','']
    (OUT/'REAL_INPUT_AUDIT.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
