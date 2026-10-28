# HSSD full 分支诊断，2026-10-21

目的：定位原 HSSD16 中 SS full 落后于 no_layer 的环节。读取原 checkpoint，
不训练、不生成新 mesh，不改原 HSSD 数据、协议、门槛或训练代码。

结果目录：`pose_aligned_layered_reconstruction/outputs/hssd_full_diagnostic_20261021/`。
完整结果及结论见同目录的 `report.json`、`REPORT.md` 和 `figures/diagnostic_overview.png`。

## 固定范围

- 原 clean16、occluded16 全部对象、原始顺序、每对象全部 8 个视图。
- 几何：各域最终 LaRI1000 raw checkpoint，FP32。检查 GT64 表面体素中心、
  对应 SS16 格中心，以及距全部 GT64 表面中心超过 0.1 的 SS16 格中心。
- 统计：每查询取 8 视图最大支撑，再在对象内求均值，最后对象等权平均。
  支撑阈值 0.1 是诊断阈值；不是完整 mesh 指标，也不是新验收门。
- 几何干预：分别替换深度、存在性、归属、置信度及组合；容差 .02/.05/.1。
  单独换深度仅修改采样 GT 存在概率 >.5 的层，避免复制无效层占位深度。
- SS 条件：最终 full500/no_layer500 raw checkpoint，全部对象；bf16 autocast，
  使用上述缓存的 FP32 LaRI 预测。分别检查 gate、block 注入、oracle、置信度、
  匹配尺度和融合支撑。记录 W×condition 与 bias，避免将 bias 误当作有效几何。
- SS 前向：固定前 4 个对象，时间 .25/.75，噪声种子 `20261021 + index * 1009`，
  两域两组相同噪声。仅测正条件 velocity MSE，不做 CFG/完整采样，不能当作 mesh 成绩。
- 任何 oracle 或后验干预均会改变条件分布，只能用于诊断。
- 原实验 full 与 no_layer 分别使用几何摘要、可训练外观输入，并非同一外观分支 ± 几何。
- 本轮不声称解决独立的遮挡 SLat 拟合失败，不替代正式 heldout 实验。

## 复现

在仓库根目录执行。选择空闲物理卡，单进程运行；已有记录会核验身份后跳过。
以下程序均有固定有限范围，不会接续旧 pilot 队列。

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m \
  pose_aligned_layered_reconstruction.experiments.hssd_full_diagnostic_20261021.diagnose geometry

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/zjr/anaconda3/envs/reconviagen/bin/python -m \
  pose_aligned_layered_reconstruction.experiments.hssd_full_diagnostic_20261021.diagnose summarize

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SPCONV_ALGO=native CUDA_HOME=/usr/local/cuda-12.6 \
  /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m \
  pose_aligned_layered_reconstruction.experiments.hssd_full_diagnostic_20261021.forward_probe

CUDA_VISIBLE_DEVICES='' /home/zjr/anaconda3/envs/reconviagen/bin/python -m \
  pose_aligned_layered_reconstruction.experiments.hssd_full_diagnostic_20261021.summarize

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/zjr/anaconda3/envs/reconviagen/bin/python -m unittest \
  pose_aligned_layered_reconstruction.experiments.hssd_full_diagnostic_20261021.test_diagnose \
  pose_aligned_layered_reconstruction.experiments.hssd_full_diagnostic_20261021.test_forward_probe
```

每条记录绑定代码、manifest、checkpoint、预测缓存等 SHA256；更改绑定文件后拒绝混用。
汇总要求 32 条几何记录、64 条条件记录和 176 次 SS 前向完整存在，并核验两组父模型
固定噪声 loss 完全一致、原 pilot 的 launch/frozen/smoke 源码合同保持一致。
旧结果需要不同诊断版本时，应另建输出目录，不能绕过身份检查覆盖。
