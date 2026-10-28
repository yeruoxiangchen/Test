# SS 接入冻结 SLat30K 的 mesh 配对评估

本实验只回答：第一轮 SS occupancy 的提升能否传递到最终 mesh。
不微调 SLat，不进行第二轮训练，也不引入 mesh alignment/refinement。

## 对照与评估协议

- 父模型：原 SS30K EMA → 原 SLat30K EMA → 冻结 mesh decoder。
- 候选：第一轮 LaRI4000 + layered SS3000 → **同一个原 SLat30K EMA** → 同一个 decoder。
- 直接读取第一轮 Dev64 的预测坐标：64 个对象 × seeds 42/43 × 父/候选，共 256 个 mesh。
- 输入保持八视图 masked-target DINO、相机 K/T 和原 SLat appearance 投影约定。
  raw RGB 的特征、层深度 GT、SS target、官方 SLat target 均不进入 SLat 条件。
- 按 64³ 坐标从同一个高斯噪声场取值，共有坐标的噪声逐位相同。
  噪声 seed 沿用历史 mesh 协议：seed × 2000003 + index × 2017 + 7919。
- SLat 原采样配置：25 步、CFG5、interval[0.5,1]、rescale_t3、guidance_rescale0，Flow BF16。
- 使用 PyTorch 确定性运算模式及 CUBLAS_WORKSPACE_CONFIG=:4096:8。
  原 CUDA scatter_reduce(mean) 在固定输入下也有浮点波动；双方统一启用确定性实现，权重和前向公式不变。
- 解码时暂存冻结 Flow 到 CPU，后续采样前恢复 GPU；CUDA allocator 使用 expandable_segments。
  只改变显存驻留，不改变 decoder 精度/算子、不减点、不降分辨率，确定性模式仍保留。
- mesh 使用 decoder 原生坐标（transform_pose=False），不旋转、不分别归一化、不 ICP、不挑最大连通分量。
- 每个 mesh 均匀按面积采样 20,000 个表面点，统计 Chamfer-L1/L2、F-score@0.01/0.02/0.05、
  对称绝对法线一致性；同时记录顶点/面数、连通分量和边界边。距离单位为归一化物体坐标，非米。
- Chamfer-L1 = 两个方向最近邻欧氏距离均值之和的一半；L2 为平方距离对应量。
  bootstrap 以 64 个物体为独立单位，先平均每个物体的两个采样 seed，5,000 次重采样。
- 不剔除失败对象。任何推理/解码异常都会阻止“完整成功”汇总，保存错误并停止后台后续任务。

## GT 的边界

本地 ProObjaverse 30K 配套数据只有 renders_random_env 与 lh-slats，当前 Dev64 的 GT 是
**官方 SLat 原始 features 经冻结 stock mesh decoder 解码的网格**，不是原始未压缩 source mesh。
官方 features 已经是 decoder 所需尺度，不能再次反归一化；预测 flow latent 则需要按原 mean/std 反归一化。
本结果衡量对 target-VAE geometry 的恢复，不能当作对原始 source mesh 的误差，也不证明外部遮挡鲁棒性。

## 代码与产物

- `frozen_slat.py`：独立 SLat v2 / 30K EMA 推理，严格核对 298 个 EMA 参数及基座/decoder 完整权重 hash。
- `evaluate_mesh.py`：冻结输入、worker 推理、完整性校验与汇总。
- `mesh_metrics.py`：独立的 mesh 指标实现。
- `background_mesh.py`：六 GPU 后台评估与异常停止。
- `monitor_mesh.py`：中文只读进度面板。
- `tests/smoke_slat_reference.py`：**仅测试**时使用旧项目作为数值 oracle；生产代码不依赖两个旧项目。

正式结果目录：`outputs/mesh_comparison_20260910/dev64_v2/`。
上一级保留 oracle、显存测试和第一次未通过大对象显存门的四条 smoke 结果；不混入 v2 汇总。

其中 `plan.json` 固定所有输入、代码和 checkpoint 身份；`meshes/` 存父/候选 PLY，
`targets/` 存 GT PLY，`latents/` 存可复核的预测 SLat，`records/` 存逐例指标与文件 hash，
`aggregate.json` 仅在 256 条完整成功且产物 hash 校验通过后生成。
`smoke_reference_v1/v2/v3` 是非确定性排查失败记录，不是正式评估结果，不覆盖删除。

## 正式结果（2026-09-10）

后台服务正常退出，64 对象 × 2 seeds × 父/候选共 256 个 mesh 全部成功。候选模型相对原
SS30K 的 Chamfer-L1 从 `0.017190` 降到 `0.013895`（相对下降 19.17%），F-score@0.02
从 `0.759795` 提升到 `0.841688`（提高 8.19 个百分点），Normal Consistency 从
`0.790882` 提升到 `0.814497`。按对象平均两个 seed 后，Chamfer-L1 为 53 胜/10 负/1 平，
F-score@0.02 为 59 胜/4 负/1 平；相应对象级 bootstrap 95% 区间均不跨 0。

同批 SS occupancy IoU 从 `0.372916` 提升到 `0.440586`。128 个配对中，SS IoU 收益与
Chamfer-L1 收益的 Pearson 相关系数为 `0.730`，与 F-score@0.02 收益的相关系数为
`0.844`，支持 SS support 收益已经传递到冻结 SLat30K 的最终 mesh。该结论仍仅适用于
ProObjaverse Dev64 的无外部遮挡输入和官方 SLat 解码 GT，不证明真实遮挡或跨数据集收益。

完整指标、置信区间、seed 稳定性、正反例和限制见
[`outputs/mesh_comparison_20260910/dev64_v2/RESULTS_ANALYSIS.md`](outputs/mesh_comparison_20260910/dev64_v2/RESULTS_ANALYSIS.md)。

## 监视与停止

从 Tracker 运行：

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python -m pose_aligned_layered_reconstruction.monitor_mesh --watch 5
```

退出监视器不会停止后台工作。正式后台服务名为 `tracker-palr-mesh-20260910.service`，
状态与日志（服务启动后可用）：

```bash
systemctl --user status tracker-palr-mesh-20260910.service --no-pager
journalctl --user -u tracker-palr-mesh-20260910.service -f
```

需要停止时：

```bash
systemctl --user stop tracker-palr-mesh-20260910.service
```

服务不依赖终端保持打开；临时服务不提供整机重启后的自动恢复。
若中断，可在确认 plan 与代码未变后以新的 attempt 名继续未完成对象；
已有成功记录必须通过输入/产物身份验证才会复用，不从头训练任何模型。
