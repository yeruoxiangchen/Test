# V2 Formal2K：贡献范围与单卡公平对照协议

协议日期：2026-10-18 UTC。用户授权：冻结贡献和实验协议，单卡启动 2K 训练及公平对照。
机器冻结文件为 `outputs/round2_v2_formal2k_20261018/protocol.json`；其中保存本文件、
数据计划、实际执行命令、GPU 验收和源代码 SHA。正式启动后不原地更改协议；结果和
运行事件另写时间记录。本文固定要检验的主张，不预先承诺实验会得到正结果。

## 论文主张与解释边界

主问题：在已有位姿约束生成父模型之上，目标相关的多层射线几何经过已知相机投影
关联到三维查询后，能否改善稀疏结构与最终 mesh；这种改善是否超出额外条件参数、
同数据继续训练及首个表面条件的作用。

已有相机位姿条件与多层射线表示分别属于父方法和已有 LaRI 工作，不能仅凭组合名称
声称全新表示。本轮研究增量是分层几何的查询对应与生成条件作用。强遮挡额外优势、
ownership 单独贡献、SLat 新适配及完整 AR 世界坐标系统均不是本轮已经证明的结论。
本轮只训练共享几何前端和 SS 新条件模块；SLat/mesh decoder 保持冻结。

V2 Formal2K 使用共享且冻结的几何前端，与此前 32 对象 LaRI/SS 联合训练 V2 不同。
两者不能作为仅改变数据规模的严格消融。V1 仅作历史参照，超过 V1 不是本轮唯一门槛。

## 数据固定

- 使用旧协议的全部 2000 个训练物体，每物体固定 8 个相机视图、1 个干净场景和 2 个
  三维 box 遮挡场景，共 6000 个样本。物体等权；所有组使用同一个不可变 manifest。
- 2000 对象全部属于父 SS30K/SLat30K 的 29861 训练 UID；本轮是该集合内的适配。
  Dev64 与上述训练 UID 及本轮 2000 对象均不重叠，但已参与开发，不能称最终盲测。
- 遮挡 recipe 沿用当前 V2：seed=`20261017 + object_index*1009 + variant_index`，
  `box_count=2`，5 层。经源图像/缓存/recipe/相机/监督校验后可复用旧 32×2 样本，
  其余 3936 个新建；最终仍严格为每对象 3 个样本，不额外上采样旧 32 对象。
- raw/masked 图像来自同一场景和裁剪；同相机 K/T、可见目标 mask 与冻结 DINO。
  完整 mask、层标签、SS latent 仅作监督，不进入部署推理输入。
- 标签来自官方 SLat VAE 代理几何，不能写成原始高精度扫描真值。旧线 proxy-depth
  composite 的轮廓/深度冲突区域沿用监督排除规则；统计实际可见率，不声称满足
  固定强遮挡配额或真实场景分布。五层截断和 overflow/censor 标记沿用现有合同。
- 数据准备完成并验证全部行数、对象权重、UID 隔离及文件绑定后，才发布 train6000
  manifest 并开始 optimizer 更新。准备过程不追加/改变正在训练的 manifest。

## 对照矩阵

| 名称 | SS 新条件 | 几何前端 | SS 更新 |
| --- | --- | --- | --- |
| Parent30K | 原有条件不变 | 不新增 | 0，冻结参照 |
| full | 全部 5 层预测交点的查询匹配 | 共享前端冻结 | 新适配器 9000 步 |
| no_layer | 同 raw/masked/visible-mask 按已知 K/T 投影，训练外观适配 | 不调用，不消费预测/层标签 | 新适配器 9000 步 |
| single_layer | 共享预测器的第一 scene 交点，存在概率合并为 0/至少1 | 同一共享前端冻结 | 新适配器 9000 步 |

`single_layer` 是**共享多层前端的第一交点读出对照**。第一交点可能是遮挡物；不根据
GT 或预测 ownership 偷取后方目标交点。该对照检验消费多层证据的作用，不能代表
所有独立训练的单深度模型，也不能单独证明多层前端监督不可替代。

`no_layer` 保留真实可训练路径：完整 1024 维 raw、1024 维 masked、1 维可见 mask
经可训练 2049→12 投影，再使用相同 view gate、12→128 encoder 及逐 block 条件
映射。它比 full/single 多 24600 个可训练参数，实际参数与峰值显存、耗时如实报告。
共享适配器初始化须逐位一致；不靠关闭所有适配器冒充“同预算重新训练”。

full/single 享受共享几何监督预训练，no_layer 不使用该前端。匹配的是 SS 阶段的数据、
更新、优化器和共有适配器初始化，**不声称三组总计算或预训练监督完全相等**。几何
前端训练成本须单列；参数量差异须列出。若 full 优于 no_layer，只能先支持几何条件
整体的作用；多层消费作用由 full/single 对照进一步检验。

## 固定训练预算与执行顺序

物理 GPU 固定为 4；单进程 `world_size=1`，同一时刻仅一个 GPU 子进程。顺序为：
遮挡缓存准备 → 共享 LaRI → full SS → no_layer SS → single_layer SS。

| 阶段 | 初始化 | updates | 梯度累积 | 场景样本曝光 | 6000样本遍历数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 共享 LaRI | 随机初始化，5 层配置 | 12000 | 2 | 24000 | 4 |
| full SS | 父 SS30K EMA＋零初始化条件输出 | 9000 | 2 | 18000 | 3 |
| no_layer SS | 同上，共有适配器相同初值 | 9000 | 2 | 18000 | 3 |
| single_layer SS | 同上，共有适配器相同初值 | 9000 | 2 | 18000 | 3 |

总计 39000 optimizer updates、78000 场景曝光。日志 `object_group_exposures` 在此
表示场景样本曝光，不是独立物体数量。该预算为固定有限预算，不宣称充分收敛。

统一 seed=20261018、AdamW、lr=1e-4、betas=(0.9,0.95)、weight_decay=0.01、
warmup=100、cosine 到 0.1 倍学习率、BF16、梯度裁剪 1、梯度 checkpointing。
SS 时间分布 uniform、容差 0.02、p_uncond=0，8 视图全部使用。三组数据随机顺序、
噪声和时间抽样相同。共享前端取最终 step12000 **raw** 参数，SS 前端不更新；父
SS Flow、已有 LoRA/聚合器和后续 SLat/decoder 均冻结。不开启无界 GPU 几何缓存。

LaRI 使用 layer_weight=1、cross_view_weight=0.1。full/single 的 SS 总 loss 包含
冻结前端的常数几何项；跨组优化曲线比较使用 `flow_loss`。每 500 步保存 checkpoint。
LaRI 每 500 步在 Dev64 固定前 8 对象记录深度 MAE，仅作运行诊断，不据此选最终模型。

所有 SS 最终比较固定 step9000 **raw**；不按 Dev64 挑选有利 checkpoint，不根据
中途结果追加某一组训练预算。第一轮正式矩阵使用一个训练 seed，训练重复性仍待补充。

## 后续评估协议（本服务不自动执行）

固定四方法、同对象/输入/噪声。开发集使用 Dev64 clean 和先前固定的 Strong16，
seed42/43，共 640 条 mesh 记录。SS/SLat 均 25 步、CFG5、CFG 区间 [0.5,1]、
rescale_t=3，SLat guidance_rescale=0。使用同一冻结 SLat30K EMA、官方 decoder。
SLat 密集 64³ 噪声 seed=`seed*2000003+object_index*2017+7919`，相同坐标使用
完全相同噪声值；保存 PLY 后重读评分，表面采样 20000 点，采样 seed 为
`seed*1009+object_index*9173`，F-score 阈值 0.01/0.02/0.05。

原生帧 CD/F-score 为主，proper SE(3) 为辅助诊断；禁止缩放、镜像、非刚性形变、
删部件及逐 mesh 居中。对象内先平均两个推理 seed，再作 5000 次对象配对 bootstrap
（seed20261018），报告 95% 配对区间，不将未校正的多项比较写成整体显著性。
刚体拟合沿用已经审计的 4096 点、最多 35 次 ICP、质心平移与 24 个 proper PCA
旋转初值；取拟合分数前三与恒等变换，用独立 20K 点评分选择。此为 GT 辅助诊断，
不是部署时纠正，也不保证找到全局最优。全部参数和 Strong16 manifest 绑定机器协议。
所有失败保留在分母和记录中，不按结果过滤对象。按固定效应排名
取代表图并提供全量图册，同相机、尺度、材质和灯光。
允许按原参数重试运行故障；仍无法完成评分时不发布“完整队列”的指标均值，单独
报告失败数量和原因。成功子集的统计只能标为诊断，不替代预定完整队列。

主问题依次为 full−parent、full−no_layer、full−single_layer。遮挡条件交互用同一
16 对象计算，不能把 clean64 与 strong16 的不同队列均值直接当成交互效应。论文的
独立原始 GT 测试和真实 AR 留出视角评估另行冻结，本轮不假装它们已经完成。

## 后台、状态与恢复

独立用户服务：`tracker-palr-v2-formal2k-20261018.service`。关闭监视器/终端不停止服务；
不承诺机器重启自动续训。每个阶段启动前再次校验冻结源代码和数据计划，只在固定
GPU 连续两次空闲时启动。任一阶段失败停止后续阶段，保留日志与错误现场。
每阶段成功后以不可覆盖的 `stage_receipts/<stage>.json` 封存 manifest、实际配置和
checkpoint 的 SHA；后续阶段及恢复启动均重新核验。三组完成时核对完整数据身份、
父权重、共有适配器初始 SHA 与实际参数量，不仅依赖种子声明。恢复前单独保存旧
`last.pt` 和运行配置的身份；已有完成文件却缺少封存回执时停止自动恢复。

```bash
cd /home/zjr/Tracker
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  pose_aligned_layered_reconstruction/experiments/round2_v2/formal2k_run.py monitor --watch 5
```

```bash
systemctl --user status tracker-palr-v2-formal2k-20261018.service --no-pager
journalctl --user -u tracker-palr-v2-formal2k-20261018.service -f
```

总状态为 `outputs/round2_v2_formal2k_20261018/status.json`；数据状态在 `data/status.json`，
每阶段 stdout/stderr 在 `logs/<stage>.log`，优化步记录在 `<stage>/history.jsonl`。
监视器显示最后心跳时间；运行状态超过 120 秒未更新会提示检查后台服务。
`training_complete_pending_evaluation` 仅表示四项训练完成，不表示精度评测通过。

中断后可在身份检查下从最近每500步 checkpoint 显式恢复；首次 checkpoint 前失败
不能冒充续训，需保留失败目录后重新启动该阶段。从 checkpoint 重跑的日志可能有
重复 step，分析按恢复事件/同 step 最后一条处理，不累加日志行数充当样本曝光。
