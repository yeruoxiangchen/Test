# Pose-Aligned Layered Reconstruction

基于 TRELLIS 版本 pose-constrained 模型的几何精度研究原型：保留已有图像条件，新增目标条件分层射线预测与查询匹配。主要目标是提高最终 mesh 精度，遮挡是可选测试条件。模型和数据代码均在本目录，运行时不导入 `pose_aligned_reconstruction` 或 `pose_point_depth_mv`。

实现范围：成对图片预处理、冻结 DINO 编码、Target-LaRI 监督训练、冻结 SS30K 上的新条件模块训练、SS latent / 64³ occupancy 推理，以及独立的冻结 SLat30K 配对 mesh 导出与评估。没有训练真实场景的大规模遮挡模型。实验与限制见 [审计.md](审计.md)。

已完成 [第一轮实验协议](EXPERIMENT_ROUND1.md) 的过拟合检查、2,000 对象训练与 reserved Dev64 occupancy 对照。当前将两组 SS 的既有预测坐标接入同一个冻结 SLat30K，进行 256 个 mesh 的后台对比；协议与监视方式见 [MESH_COMPARISON.md](MESH_COMPARISON.md)。mesh GT 为官方 target-VAE 解码网格，不是原始未压缩 source mesh。实际结果与进度以 [审计.md](审计.md) 为准。

## 两种输入怎样使用

每个相机视图只拍一张原图。由同一张原图和目标**可见区域** mask 派生两份图像，使用完全相同的裁剪和缩放；K 同步更新。

| 输入 | 内容 | 用途 |
| --- | --- | --- |
| raw RGB | 保留目标、遮挡物及裁剪区域内的背景 | 冻结 DINO 后送入分层几何预测头 |
| masked RGB | 只保留目标当前可见的像素，其余为黑色 | 原有 TRELLIS appearance / image-token context，同时提供分层头的目标识别线索 |
| visible mask、K、T | 与上述两张图逐视图对应 | 目标识别、相机条件和 3D 查询投影 |

第二份不是“去掉遮挡物后补全出来的干净图”。隐藏表面的 RGB 不作为输入。训练可用完整场景几何生成多层深度与目标归属标签；推理只需 raw、visible mask、K、T。

## 环境

本机已验证环境：`/home/zjr/anaconda3/envs/reconviagen/bin/python`，PyTorch 2.4.0、RTX 3090、BF16。使用 Tracker 下的外部 `ReconViaGen` 提供 TRELLIS Flow、sampler、decoder 与 DINO 包装；其导入链还需要 `ReconViaGen/wheels/vggt`。本方法不运行 VGGT 模型。

```bash
cd /home/zjr/Tracker
PALR_PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ATTN_BACKEND=flash_attn SPCONV_ALGO=native
```

离线变量适用于本机已有模型缓存。其他机器需安装 ReconViaGen 所需环境并准备模型文件。默认 TRELLIS 基座固定到 `Stable-X/trellis-vggt-v0-2` 的 revision `647659a5ad5fbf67e22793e7b5e2cee4b30c5d13`，按完整权重严格加载并记录 SHA-256。

## 数据准备

先按 [source_manifest.example.json](examples/source_manifest.example.json) 整理源数据。`root` 填数据根目录的绝对路径，各路径列表的顺序必须相同。

- `raw_images`、`visible_masks`：每视图原图与二值可见目标 mask。
- `camera`：NPZ 中 `intrinsics [V,3,3]`、`extrinsics [V,4,4]`。
- `tracking_boxes_xyxy`：可选，来自检测或跟踪的目标范围；遮挡较重时比仅按可见碎片裁剪合适。省略则按可见 mask 裁剪。完全不可见的视图需要提供 box，且整组至少有一帧看得到目标。
- `layers`：监督训练/oracle 实验使用；NPZ 中 `target_depths [V,Lt,H,W]`、`non_target_depths [V,Lo,H,W]`，必须与**原图**同分辨率、同相机。值是正的相机前向 z 深度，无交点填 0。可见和被遮挡的交点都需要保留。多个非目标物体的交点应合并进 non-target stack。
- `ss_target`：SS 训练使用，完整目标对象的 SS 编码 latent `[8,16,16,16]`；支持 NPZ 的 `target` 字段、NPY、或 PT tensor / `{"target": tensor}`。

目标几何、相机平移、分层深度必须处于同一个归一化物体坐标系/尺度。`grid_transform` 采用 `identity` 或与已有缓存一致的 `pixal3d_rotation`。不能把相机到点的欧氏距离当 z-depth；不能直接混用米制深度与归一化相机平移。此处不要求重力或平面相机轨迹。

推理源 manifest **省略 `layers` 和 `ss_target`** 即可。缓存制作会派生 masked 图，分别编码 raw/masked，并保存输入文件哈希、共享 crop、K/T、目标身份。

```bash
"$PALR_PY" -m pose_aligned_layered_reconstruction.prepare_paired_cache \
  --source_manifest /path/to/source_train.json \
  --output_dir /path/to/new_train_cache --device cuda:0
```

所有新实验输出目录都必须尚不存在，以免覆盖先前结果。

## 训练和推理

第一阶段监督训练分层预测头：

```bash
"$PALR_PY" -m pose_aligned_layered_reconstruction.train_target_lari \
  --manifest /path/to/new_train_cache/manifest.json \
  --output_dir /path/to/lari_run --device cuda:0 --max_steps 10000
```

先做 oracle 层条件实验，检验分层几何接入位置是否有用：

```bash
"$PALR_PY" -m pose_aligned_layered_reconstruction.train_layered_ss \
  --manifest /path/to/new_train_cache/manifest.json \
  --output_dir /path/to/ss_oracle_run --layer_source oracle \
  --device cuda:0 --max_steps 1000 --gradient_checkpointing
```

再换成第一阶段训练的预测层。SS30K 及分层预测头冻结，只训练新条件模块：

```bash
"$PALR_PY" -m pose_aligned_layered_reconstruction.train_layered_ss \
  --manifest /path/to/new_train_cache/manifest.json \
  --target_lari_checkpoint /path/to/lari_run/checkpoints/last.pt \
  --output_dir /path/to/ss_predicted_run --layer_source predicted \
  --device cuda:0 --max_steps 1000 --gradient_checkpointing
```

默认父权重是本机已确认 SHA-256 的 SS30K EMA；迁移机器可通过 `--parent_ss30k` 指定文件。上述入口为单 GPU，SS 支持 gradient accumulation。另有 `train_distributed_round.py`，通过 `torch.distributed.run` 启动，以合并后的 trainable gradient 做同步平均；正式模式要求已通过过拟合门。没有正式训练断点续跑或新 adapter EMA。

第一轮官方数据使用独立的 `prepare_official_experiment.py`：保留父模型 masked DINO 和外观相机约定，只为新增分层几何显式绑定 `layer_camera_y_sign=-1`。分层监督来自官方 target-VAE 解码几何，SS 监督是实际稀疏结构 latent。`evaluate_ss_occupancy.py` 在完整采样后计算 IoU、Precision、Recall、F1 与体素数量，并移除预测分支输入中的 GT 字段。`assess_overfit.py` 检查 32 对象/多 seed 结果完整性与通过条件。

使用同一个分层预测头与对应 SS adapter，对无标签缓存推理：

```bash
"$PALR_PY" -m pose_aligned_layered_reconstruction.infer_layered_ss \
  --manifest /path/to/inference_cache/manifest.json \
  --target_lari_checkpoint /path/to/lari_run/checkpoints/last.pt \
  --layered_ss_checkpoint /path/to/ss_predicted_run/checkpoints/last.pt \
  --output_dir /path/to/inference_run --device cuda:0 --steps 25
```

输出每对象 NPZ：`latent [1,8,16,16,16]`、`coords [N,4]`（batch,x,y,z，后三列范围 0–63）、`uid`，以及 `report.json`。预测分支会校验 SS30K、基座与 Target-LaRI 的权重身份，避免误配 checkpoint。oracle 与 predicted 间的切换默认拒绝，仅明确做该消融时使用 `--allow_layer_source_change`。

## 检查与复现

```bash
"$PALR_PY" -m unittest discover -v pose_aligned_layered_reconstruction/tests
"$PALR_PY" -m pose_aligned_layered_reconstruction.verify_independence
"$PALR_PY" -m pose_aligned_layered_reconstruction.smoke_parent30k \
  --device cuda:0 --views 8 --condition_tokens 4096 --run_backward
```

分层预测评估（正式验证加对象级训练集隔离检查）：

```bash
"$PALR_PY" -m pose_aligned_layered_reconstruction.evaluate_target_lari \
  --manifest /path/to/heldout_cache/manifest.json \
  --checkpoint /path/to/lari_run/checkpoints/last.pt \
  --output_report /path/to/new_eval.json --device cuda:0 \
  --require_disjoint_from_training --raw_view_ablation
```

`make_source_fixture.py` 可生成与本次测试相同的两个射线/盒体相交场景。它包含随机 SS latent，专门测试训练链路，**不能衡量重建质量**。完整已执行命令、结果、失败修复与后续实验协议见 [审计.md](审计.md)。
# 第一轮后台实验

2026-09-09 的32对象过拟合复核、2,000对象正式训练与Dev64评估现由独立后台服务调度。
当前状态、日志及停止命令见 [BACKGROUND_ROUND1.md](BACKGROUND_ROUND1.md)；
是否已进入正式训练以 `outputs/round1_20260909/background_status.json` 为准。
