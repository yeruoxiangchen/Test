# V2 实现与运行

本轮新增真实物体坐标射线、显式跨视图候选点采样、分层几何预测、LaRI/SS 联合训练和
SLat 几何适配。所有输出建议放在本包 `outputs/` 下；第一轮 checkpoint 不用于 V2 loader。

网络输入是同裁剪的 raw/visible-masked 图像及 K、位姿和可见 mask。完整目标 mask、分层 GT、
SS/SLat latent 只作训练监督。当前 DINO 是冻结的 **DINOv2 ViT-L/14-reg**，37×37×1024。

几何计算固定 FP32，网络可以 BF16。标签与查询均用相机正向 z 深度；单位射线只用于编码。
相机保持真实距离，目标候选范围来自 canonical cube 的射线交区间，场景交点头使用更宽的真实
中心深度区间。最后的层数类别是“至少 L 层”，另保留 overflow 标记。L 默认 5，与旧 2K
标签兼容；改成 8 时必须重建相同层数的监督，不能只改网络参数。

跨视图模块先压缩特征，再分查询块、逐源视图聚合；训练开启块级重计算。所有其他视图无效时
返回零支持。可见 mask 不会硬删除隐藏目标候选。当前 view gate 的结构和位置沿用第一版。
SS 和 SLat 共享三维查询证据接口，匹配容差默认固定物体空间 0.02，不随相机距离扩张。

## 环境与入口

在 Tracker 根目录运行，`run.sh` 已设置本机 conda、CUDA、离线缓存和 `SPCONV_ALGO=native`。
本环境 spconv/cumm 的 implicit-GEMM 在 `tensorview.from_numpy` 崩溃，native 算法通过独立
配置选择，不修改外部库或原始 30K 权重。实际验证状态见 `验证记录.md` 和包根 `审计.md`。

训练入口支持单卡及 `V2_NPROC` 指定的 torchrun 多卡，每卡一组变长视图，`--grad_accum` 控制
梯度累积。checkpoint 同时保存几何前端、适配器、EMA、优化器、数据身份、游标和 RNG；
多卡 RNG 存在对应 `.rankN.rng.pt` 中，恢复时需保留这些文件。`--resume` 要求训练策略一致。
CUDA grid_sample 反向有原子累加舍入差异；GPU 恢复实测原始权重最大差约 1.86e-9，EMA
完全相同，不承诺整个 CUDA 训练过程 bit-exact。

以下是正式训练调用示例，应先准备、检查数据和 train/val 对象分割。
已完成实验的实际范围见文末诊断记录，示例命令本身不代表已执行正式训练。

```bash
CUDA_VISIBLE_DEVICES=5 bash pose_aligned_layered_reconstruction/experiments/round2_v2/run.sh lari \
  --manifest /absolute/path/train.json --val_manifest /absolute/path/val.json \
  --geometry_config pose_aligned_layered_reconstruction/experiments/round2_v2/configs/geometry.json \
  --output_dir pose_aligned_layered_reconstruction/outputs/round2_lari \
  --max_steps 20000 --grad_accum 2

CUDA_VISIBLE_DEVICES=5 bash pose_aligned_layered_reconstruction/experiments/round2_v2/run.sh ss \
  --manifest /absolute/path/train.json --val_manifest /absolute/path/val.json \
  --init pose_aligned_layered_reconstruction/outputs/round2_lari/checkpoints/last.pt \
  --output_dir pose_aligned_layered_reconstruction/outputs/round2_ss \
  --max_steps 10000 --grad_accum 2

CUDA_VISIBLE_DEVICES=5 bash pose_aligned_layered_reconstruction/experiments/round2_v2/run.sh slat \
  --manifest /absolute/path/legacy_2k_with_slat.json \
  --init pose_aligned_layered_reconstruction/outputs/round2_ss/checkpoints/last.pt \
  --output_dir pose_aligned_layered_reconstruction/outputs/round2_slat \
  --max_steps 10000 --grad_accum 2
```

SLat 阶段固定几何前端，先训练几何注入；`--finetune_lora` 可同时微调已有 LoRA。SS 默认联合
更新几何前端及新适配器，`--freeze_geometry` 是固定前端的对照。新资产当前具备 SS 标签；
没有 SLat 标签的对象不能直接用于 SLat 训练。旧 2K 已有的 SLat 坐标/特征及 hash 可以复用，
不把 latent 最近邻搬到生成的 SS 坐标上。SLat 训练用 GT support，评估用预测 SS。

## 新资产数据

`configs/assets_smoke.json` 展示目录格式。每个物理对象需显式指定 `object_uid`、`mesh`、
`split`。先确定对象级分割；同对象的所有场景、轨迹和干净/遮挡样本进入同一 split。
不同别名的同字节资产也不能跨 split。源资产与原 30K 的重合仍需建立身份映射后确认。

```bash
CUDA_VISIBLE_DEVICES=6 bash pose_aligned_layered_reconstruction/experiments/round2_v2/run.sh dataset \
  --assets /absolute/path/assets.json --split train --max_objects 1000 --variants 6 \
  --output_dir pose_aligned_layered_reconstruction/outputs/round2_scene_train
```

每个变体保存同一相机下的干净/遮挡场景、原生 RGB、可见/完整 mask、相机、分层深度与
ownership，然后计算配对 DINO。目标只编码一次 SS。表面体素化不填充内部，SS encoder
使用后验均值，并检查 VAE roundtrip。生成器遇到低于阈值的对象会停止并保留现场，不静默过滤。
`--resume` 可复用带身份校验记录的完成对象和样本。当前固定比例生成 clean/occluded 对，
实际遮挡程度记录为逐视图可见比例；尚未按可见比例分桶保证 0/2/8 帧配额。

旧 2K 增广入口：

```bash
CUDA_VISIBLE_DEVICES=7 /home/zjr/anaconda3/envs/reconviagen/bin/python -m \
  pose_aligned_layered_reconstruction.data_v2.build_legacy_occlusion \
  --manifest pose_aligned_layered_reconstruction/outputs/round1_20260909/data_v2/train2000.json \
  --output_dir pose_aligned_layered_reconstruction/outputs/round2_legacy_occlusion \
  --max_objects 2000 --variants 4
```

多条数据线用 `python -m pose_aligned_layered_reconstruction.data_v2.merge_manifests` 的
`--manifests` 合并，`--exclude_objects_from` 指定 val/test manifest 作对象排重，`--output`
指定新 manifest。旧缓存通过显式 V1→V2 适配复用，无需改写已有样本。

旧线使用 VAE 代理 mesh 排深度，RGB 来自原图，因此显式标为 proxy composite。轮廓冲突及
目标/遮挡物深度接近区域不参与精确层监督；两套 DINO 特征都重新计算。当前随机遮挡物是固定
三维 box；新场景 recipe 还支持有纹理的 mesh occluders。材质 alpha 当前统一按不透明处理，
RGB 与几何使用相同策略，不模拟透明物体。

## 推理

```bash
CUDA_VISIBLE_DEVICES=5 bash pose_aligned_layered_reconstruction/experiments/round2_v2/run.sh infer \
  --manifest /absolute/path/eval.json \
  --checkpoint pose_aligned_layered_reconstruction/outputs/round2_ss/checkpoints/last.pt \
  --slat_checkpoint pose_aligned_layered_reconstruction/outputs/round2_slat/checkpoints/last.pt \
  --output_dir pose_aligned_layered_reconstruction/outputs/round2_mesh --max_samples 8 --seed 42
```

输出 SS latent/坐标及 `mesh_canonical.obj`、PLY。省略 `--slat_checkpoint` 使用冻结 SLat30K；
`--parent_ss_only` 使用原始 SS30K 条件路径。对应比较应使用同 seed、同输入和采样参数。
`--mode layers` 只预测分层几何，`--mode ss` 不解码 mesh。当前 OBJ 导出要求 identity grid。
推理输入有明确白名单，即使读取带标签的数据集，也不会传入 GT mask、层标签或 latent。

## 验证命令

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python -m unittest discover \
  -s pose_aligned_layered_reconstruction/v2/tests -t . -v

# 各 smoke 使用独立的固定输出路径；已有结果会拒绝覆盖。
bash pose_aligned_layered_reconstruction/experiments/round2_v2/run_smoke.sh lari 5
```

GPU smoke 的阶段还包括 ss、slat、render、encode、cache、legacy、dataset、resume、infer、ddp。
几何前端两步 smoke 的 checkpoint 仅用于接口测试，不能据此解释重建精度或遮挡性能。

## 2026-10-18 诊断对照

入口 `run_diagnostic_train.sh ARM GPU`，独立输出到
`outputs/round2_v2_diagnostics_20261018/`。先运行 `smoke` 的两步检查。
以下四个新实验各训练 500 步、梯度累积 2，共 1000 次样本曝光；使用同一 train96、
同一初始化种子、同一 LaRI step2000。500 步是诊断预算，不代替 2000 步收敛检查。

| 对照 | LaRI 初始化 | 几何更新 | 时间采样 / 学习率 | SS 固定容差 |
| --- | --- | --- | --- | --- |
| 已有 SS step500 | EMA | 联合 | uniform / cosine | 0.02 |
| raw_joint | raw | 联合 | uniform / cosine | 0.02 |
| raw_frozen | raw | 冻结 | uniform / cosine | 0.02 |
| raw_frozen_v1recipe | raw | 冻结 | sigmoid(N(1,1)) / constant | 0.02 |
| raw_frozen_v1recipe_tol004 | raw | 冻结 | sigmoid(N(1,1)) / constant | 0.04 |

cosine 各组使用 `--lr_schedule_steps 2000`，确保前 500 步学习率与原实验相同。
V1 recipe 指对齐时间分布和学习率；数据增广、有效 batch 和前端仍与历史 V1 不同，
不能将两代最终模型的差异全部归因于架构。冻结组的几何缓存只保存确定性的预测与监督
统计，适配器及其条件每步重新计算；测试检查缓存前后的 loss 与适配器梯度一致。

训练新增参数默认保留原有行为。`--init_weights raw|ema` 显式选择跨阶段初始化；
`--t_sampling uniform|logit_normal` 和 `--lr_schedule cosine|constant` 控制训练配方；
`--ss_absolute_tolerance` 独立控制 SS 查询，不改几何预测器。checkpoint 恢复检查这些策略，
旧 checkpoint 缺少新字段时采用历史默认值；推理从 checkpoint 读取匹配容差。

`diagnostic_suite` 提供 `prepare_mesh`、`mesh`、`ss` 三个入口。mesh 的固定队列包括全部
32 个对象的干净输入和第一个遮挡变体、seed 42/43、parent/V1/V2 raw，共 384 个结果。
冻结 SLat30K 使用公共稠密噪声按坐标索引，25 步采样；评估原生坐标，不对齐到 GT。
GT 仍是官方 SLat VAE 代理表面。32 个对象全部见于 SLat30K 训练集，结果须标为训练对象
比较。SS 的 Dev64 测试与本次 train32 对象互斥，使用全部 64 个对象和两个 seed。

`strong_occlusion --prepare_only` 固定 Dev64 前 16 个对象及 recipe；GPU 阶段每对象生成一个
固定三维遮挡场景，至少四个视图可见目标不足 50%，平均可见比例至少 15%，至少四个视图
可见比例超过 2%。候选选择只看可见比例；64 次尝试失败时停止并保存拒绝记录，不删除
困难对象。该测试沿用原相机，是合成代理遮挡压力测试，不代表自然场景或新相机泛化。

所有评估保存输入/权重 hash、逐样本结果及预测坐标。汇总须检查完整配对，置信区间按
对象重采样；两个推理 seed 不等于两个训练 seed。正式 2K 扩展仍需独立评估结果支持。

本轮候选统一评估 SS 的 **raw 参数**；手动调用 `infer` 时显式加 `--weights raw`。
这与 `--init_weights raw` 是不同选择：前者选择推理参数，后者选择 SS 训练开始时的
LaRI 参数。历史默认值保持不变，不能将 EMA 初始化与 EMA 推理混为同一实验。

额外机制诊断包括 `bias_probe`（关闭几何证据，仅保留已训练的新分支偏置）和
`alignment_probe`（图像、相机、parent 投影保持一致，只将所有预测分层字段在视图间
循环错配一位）。后者是明确标注的追加诊断，用于检查正确对应关系的作用。
两个对照都使用完整 Dev64、两颗推理 seed。偏置对照先复现正常模型的输出坐标。

进度可用以下命令查看，同时更新 `outputs/round2_v2_diagnostics_20261018/progress.json`：

```bash
python pose_aligned_layered_reconstruction/experiments/round2_v2/monitor_diagnostics.py
```

### 已完成的 2000 步模型诊断（2026-10-18）

下表中的 V2 使用原始 SS step2000 的 raw 推理参数；该 SS 训练实际由 LaRI EMA 初始化。
各方法使用同输入和 seed 42/43。这里的 V1 是历史 overfit32 step2000，不能视为只改变架构的对照。

| 完整评估 | Parent30K | V1 step2000 | V2 raw step2000 |
| --- | ---: | ---: | ---: |
| Dev64 SS IoU ↑ | 0.372916 | 0.392910 | 0.390034 |
| 强遮挡 Dev16 SS IoU ↑ | 0.209919 | 0.218633 | 0.236142 |
| Train32 干净 + 首个遮挡变体 mesh CD-L1 ↓ | 0.016144 | 0.013224 | 0.013591 |
| 同上 mesh F-score@0.02 ↑ | 0.762060 | 0.832130 | 0.822670 |

Dev64 的 UID 与父 SS30K、父 SLat30K、V1 和 V2 的训练对象均无交集。V2 相对父模型的
IoU 收益为 +0.017118，50 胜 / 2 平 / 12 负，对象 bootstrap 95% CI
[+0.009769, +0.024415]；相对 V1 为 -0.002876，CI [-0.011922, +0.006348]。
因此有未见对象收益，但没有证明优于 V1。

强遮挡固定使用 Dev64 前 16 个对象，不按模型结果选对象。每个对象至少四个视图可见比例低于
50%，最终 128 个视图中 78 个满足；平均可见比例 0.4702，无对象被丢弃。
V2 相对父模型 +0.026223，CI [+0.005498, +0.057386]；相对 V1 +0.017509，
CI [-0.000857, +0.047843]，后者仍跨零。与同 16 个对象干净输入的收益之差也跨零，
不能宣称已经证明了随遮挡加重而增大的额外优势。五层标签在非空射线上的 overflow 比例
为 38.86%；该测试仍是固定相机下的合成代理遮挡。
第 16 个对象（index 15）的 GT 解码曾遇到显存不足，最终复用已有的同一官方 VAE 代理
mesh；对象 UID、SLat 源文件、解码器及 mesh 文件 hash 均已核对。失败现场保留在
`strong_data/incomplete_attempts/`，未丢弃对象或依据重建得分选择遮挡候选。

384 个最终 mesh 全部成功，原生坐标下 V2 相对父模型 CD-L1 改善 15.81%，32 个对象中
30 个改善；F-score 增加 0.060609。与 V1 的 CD-L1 改善差为 -0.000367，
CI [-0.001009, +0.000505]，没有证明 V2 更好。这些对象全部见于 SLat30K 训练，
参考表面是官方 SLat VAE 代理 mesh，不能将本表称为未见对象最终 mesh 结果。

完整 Dev64 的机制消融：只保留已训练的新分支偏置时 IoU 为 0.371289；保持图像、相机和
parent 条件不变，只把预测分层图在视图间循环错配时为 0.373794。完整模型分别高出
0.018745（CI [+0.011443, +0.026101]）和 0.016240（CI [+0.009575, +0.022905]）。
错配模型相对父模型的区间跨零。结果支持正确对应的条件信息确实参与收益，不能用新分支
常量偏置解释；该消融不证明 V2 的每个结构部件都不可替代，也不证明优于 V1。

以上 CI 在先对同一对象的变体和推理 seed 求平均后，按对象重采样 5000 次，固定种子
20261018，采用 object_uid 规范顺序。它们只描述固定训练实例下的对象差异；没有覆盖
不同训练 seed 的变动，也未对探索性多项比较做校正。

机器结果位于 `outputs/round2_v2_diagnostics_20261018/`：`dev64/aggregate.json`、
`strong16/aggregate.json`、`strong16/clean_vs_occluded.json`、`mesh/aggregate.json`。
`dev64_split_audit.json` 记录对象排重；`dev64_parent_reproduction.json` 记录父模型
128 条结果与历史正式评估的指标差最大值为零。偏置消融还先通过了正常模型坐标逐位复现。

### 已完成的 500 步配方对照与决定

四个新训练均已到 step500，原实验 step500 作为第五组。每组训练曝光 1000 次，
评估全部 32 对象的干净输入和第一个遮挡变体、两颗推理 seed，共 128 条记录/模型。
下表 SS 参数均取 raw；“初始化”专指训练开始时选择的 LaRI 权重。

| 500 步设置 | 全部 IoU | 干净 IoU | 遮挡 IoU |
| --- | ---: | ---: | ---: |
| EMA 初始化 + 联合 + 原配方 | 0.376928 | 0.389358 | 0.364499 |
| raw 初始化 + 联合 + 原配方 | 0.378361 | 0.390016 | 0.366706 |
| raw 初始化 + 冻结 + 原配方 | 0.377659 | 0.390082 | 0.365236 |
| raw 初始化 + 冻结 + V1 配方 | 0.388681 | 0.401787 | 0.375575 |
| raw 初始化 + 冻结 + V1 配方 + 容差 0.04 | 0.394645 | 0.406488 | 0.382802 |

| 配对改动 | IoU 差值 | 对象 bootstrap 95% CI | 胜 / 平 / 负 |
| --- | ---: | --- | --- |
| raw 初始化减 EMA 初始化 | +0.001433 | [+0.000091, +0.002825] | 20 / 0 / 12 |
| 冻结减联合更新 | -0.000702 | [-0.001905, +0.000407] | 13 / 0 / 19 |
| V1 配方减原配方（冻结） | +0.011022 | [+0.007189, +0.014941] | 29 / 0 / 3 |
| 容差 0.04 减 0.02（冻结 + V1 配方） | +0.005964 | [+0.002752, +0.009428] | 25 / 0 / 7 |

初始化有小幅正向效果，区间下界接近零。500 步内未检出联合更新相对冻结的明确优势，
这不等于证明长期训练等价。raw 联合组耗时 5364 秒，冻结组约 941–944 秒；冻结组还采用
确定性几何缓存，不能把该耗时差全部归因于是否反向传播。V1 配方同时改变时间采样和
学习率，结果只支持组合收益，尚未拆分二者贡献。不同时间分布下的 flow loss 也不能直接排名。

**优先继续验证的候选**是 `raw_frozen_v1recipe_tol004/checkpoints/step_000500.pt`，
对应 raw LaRI 初始化、固定几何、sigmoid(N(1,1)) 时间采样、常量学习率 1e-4、
SS 匹配容差 0.04。它比已有 step500 基线高 0.017717，但仍只是在训练对象上的
500 步候选，尚未做该新候选的 Dev64/最终 mesh 评估，不能与已有 step2000 当作同预算比较。

方向判断：保留按位姿对应的分层几何条件；优先把配方和容差候选验证到相同 2000 步预算，
再用未见对象、强遮挡和最终 mesh 判断。几何监督仍有提升空间：原 step2000 的 96 样本
完整评估中，目标深度 MAE 为 0.073801，落在 ±0.02 内的目标层仅 20.20%，隐藏目标
MAE 为 0.109976。加重遮挡数据和改善层数截断应作为后续工作；改变层数时需重建标签。
当前证据支持继续该方向，尚不支持 V2 已优于 V1，也不足以直接支持正式 2K 对象扩展。

工程核对：15 项单元测试全部通过；8 个实验入口语法检查通过；所有 500 步数据游标一致；
raw 联合/冻结组及两个 V1 配方容差组的时间抽样分别完全一致；三个冻结组的几何参数
均与 LaRI raw 初始化逐位相同；全部最终检查点有限。原 train96、LaRI step2000、
SS step500/2000 的 hash 保持不变，评估期间绑定的 18 个代码文件未改变。

本轮新增 1376 条 SS 评估记录，另复用 128 条配对父模型记录；完成 384 个最终 mesh。
全部预定对象与 seed 配对完整，无空 SS 预测。机器汇总见
[`final_report.json`](../../outputs/round2_v2_diagnostics_20261018/final_report.json)，
结果图见 [`diagnostic_results.png`](../../outputs/round2_v2_diagnostics_20261018/diagnostic_results.png)
和 [`paired_effects.png`](../../outputs/round2_v2_diagnostics_20261018/paired_effects.png)。
逐组结果见 `pilot32/aggregate.json`；训练核对见 `training_aggregate.json`、
`control_integrity.json`；测试日志为 `tests_final.log`。

### Mesh 实际效果与旋转、平移核查（2026-10-18）

本轮以 **V2 相对父模型的改善** 为主要论文问题，V1 保留作历史参照。
已对同一批 32 对象 × 干净/首个遮挡输入 × 两颗 seed × 三个模型的 384 个 mesh
完成原生坐标复算和 GT 辅助 SE(3) 拟合。原生结果保留姿态与形状误差；拟合结果用于
检查去除刚体差异后的形状与尺度残差，不能替代带位姿约束任务的原生分数。

| 同一批 384 个 mesh | Parent30K | V1 | V2 |
| --- | ---: | ---: | ---: |
| 原生 CD ↓ | 0.016144 | 0.013224 | 0.013591 |
| SE(3) 拟合后 CD ↓ | 0.013335 | 0.011657 | 0.011982 |
| 原生 F-score@0.02 ↑ | 0.762060 | 0.832130 | 0.822670 |
| SE(3) 拟合后 F-score@0.02 ↑ | 0.825735 | 0.873049 | 0.860494 |

V2 的原生 CD 相对父模型降低 **15.81%**；拟合后仍降低 **10.14%**，对象平均改善
0.001353，95% 对象 bootstrap CI [0.000793, 0.002042]，29 胜 / 3 负。
拟合后的 F-score 提升 0.034759，CI [0.020070, 0.052662]，28 胜 / 4 负。
因此收益不能只用整体旋转或平移解释，同时也不能将全部原生收益称为与位姿无关的形状收益。
对齐前后差值的比率不是姿态/形状贡献的严格因果分解。CI 仍按对象平均后重采样
5000 次、seed=20261018，只覆盖固定训练实例下的对象差异。

**坐标与计算核查。** 64 个输入全部为 identity 网格、w2c 相机；32 对干净/遮挡输入的
相机和 target coords 完全一致。896 个输入/输出文件的 SHA 核对通过，旧实验绑定的
18 个代码文件未改变。GT 和三种预测均使用同一冻结解码器及
`to_trimesh(transform_pose=False)`。从保存的 PLY 复算全部 384 项原生指标，最大绝对差为 **0**。
这排除了本轮的主要坐标约定/序列化风险，不代表每个模型输出都没有个体位姿偏差。

拟合仅允许 `p' = R p + t`，验证 `RᵀR=I`、`det(R)=+1`，禁止缩放、镜像、局部形变、
删连通块或网格修补。使用 4096 个注册表面点、identity 加 24 个 proper PCA 轴初始化、
每初值至多 35 次双向 ICP；取三个注册候选与不变换的 identity，在独立的 20000 个评分点
上仅按 CD 选一次变换，F-score 不单独挑选。多初值 ICP 不保证全局最优。
已知 107.51° 旋转加平移的恢复测试通过，缩放与镜像被拒绝；单进程/多进程同一实例
的指标和矩阵完全一致。

拟合旋转中位数：父模型 2.2785°、V2 1.4011°；平移范数中位数：0.016748、0.012572。
每方法 128 个实例中，分别有 17、12 个最优拟合角大于 45°。近对称物体可能得到等价的大角度
修正，形状差异也会影响拟合；这些值是**最佳拟合修正量，不是真实姿态误差**。
距离单位是规范化网格坐标，`0.02` 不能写成 2 cm。代码名 `chamfer_l1` 的实际定义为
两向最近邻欧氏距离均值的半和，不是逐坐标 Manhattan 距离。

**实际效果。** 已生成 1248 张原生三视角渲染和 180 张代表对象的 SE(3) 拟合渲染，统一
相机、正交尺度、材质、灯光及背景；没有逐 mesh 居中、缩放、平滑或删部件。全部 1428 张
图片无空图、裁剪、画布边缘触碰，拟合渲染矩阵与评分矩阵逐项相同。
代表对象在看图前按 V2 的对象平均原生 CD 改善排序，固定选第 1/9/17/25/32 名
（对象 26/1/27/19/0），全量图册仍展示全部 32 对象。

- 对象 0 的遮挡样例中，父模型缺失的支撑轮廓在 V2 中恢复，三个视角及拟合后均可见；
  但盆腔仍偏深、腿截面仍不准确。
- 对象 19 的躯干/脚部比例、对象 27 的沙发靠背/坐垫更贴近 GT；手指/足趾、扶手处
  下垂布料和小靠垫仍有缺失或融合。
- 对象 1 没有醒目的细节优势；对象 26 两者基本相当，跨 seed/条件均值轻微退步。
  对象排名是四个实例的均值；单张展示图可能与均值胜负不同。

可直接查看[实际 mesh 主图](../../outputs/round2_v2_mesh_visual_audit_20261018/figures/mesh_comparison.png)、
[原生/拟合及轮廓叠加](../../outputs/round2_v2_mesh_visual_audit_20261018/figures/pose_object_00.png)、
[全量交互图册](../../outputs/round2_v2_mesh_visual_audit_20261018/index.html)和
[姿态/形状统计图](../../outputs/round2_v2_mesh_visual_audit_20261018/figures/pose_shape_audit.png)。
图册支持 32 对象、两种输入、两颗 seed、三个视角，并链接原始 PLY；原生/拟合图切换覆盖
预选的五个对象。`figures/` 同时提供 PNG/PDF，全量原生图册共八页。
输入缩略图只显示八个输入视图中的第 0 张，该张可能仍无遮挡，且其相机不同于 mesh 显示相机。

新增工具为 `mesh_pose_audit.py`、`finalize_mesh_pose_audit.py`、`render_mesh_audit.py`、`recover_mesh_input_previews.py`、
`build_mesh_visual_report.py` 和 `verify_mesh_gallery.cjs`，均未修改原训练/生成代码。
全部机器证据及文件绑定见
[`final_report.json`](../../outputs/round2_v2_mesh_visual_audit_20261018/final_report.json)。
当前证据支持 V2 相对父模型存在实际形状改善；但这些仍是训练对象、合成遮挡和官方 SLat
VAE 代理 GT，不能替代未见对象与原始高精度 GT 上的最终 mesh 验证。

### Dev64 / Strong16 最终 mesh 验证（2026-10-18）

固定当前 V2 SS step2000 **raw** 与 Parent30K，冻结 SLat30K EMA 和官方 mesh 解码器，
评估 clean64 × seed42/43 × 两模型（256 项）及既定 strong16 × 两 seed × 两模型
（64 项），共 320 项。输入、SS 坐标、模型和代码 SHA 见
[`plan.json`](../../outputs/round2_v2_heldout_mesh_20261018/plan.json)。
复用严格验证一致的 128 个干净父模型 mesh 和 64 个 GT，实际新增 192 个 mesh。

64 个对象与父 SS/SLat 检查点记录的 29861 训练 UID，以及 V2 LaRI/SS 的 32 训练 UID
均不重叠。该核查不涵盖未提供 UID 的公共基础模型预训练集。Dev64 已用于开发诊断，
故称未见已审计训练对象的**开发集泛化验证**，不称未触碰的最终测试集。GT 仍为官方
SLat VAE 代理表面。干净和强遮挡分别统计，另在相同 16 对象上分析遮挡条件交互。

新预测导出 PLY 后重新加载评分。沿用原 Dev64 `object_index` 决定密集主噪声和表面点
随机种子；强遮挡行号 64–79 仅用于命名，其噪声索引仍为 0–15。两方法公共坐标噪声
逐位相同；同一对象干净/强遮挡的 K、外参、目标坐标、网格约定逐项一致。
原生指标为主要结果；全量 proper SE(3) 诊断沿用前一轮固定算法，不缩放、不镜像、
不形变、不删部件，并检查 PLY 原生复算与保存评分一致。

对象内先平均两颗 seed，再按对象配对 bootstrap 5000 次（seed=20261018）。显示用
固定相机/尺度/灯光/材质。代表对象在看正式图前按 V2 的平均原生 CD 改善升序排序，
clean64 固定取第 1/17/33/49/64 名、strong16 取第 1/9/16 名；全部 80 个对象/条件行
仍进入图册。已生成 1152 张原生图、96 张代表对象 SE(3) 图和 20 页全量图册。

新增执行入口：`heldout_mesh_eval.py`（prepare/worker/aggregate）、
`run_heldout_mesh_workers.py`（只在候选空闲 GPU 上调度不重叠分片）、
`heldout_mesh_pose_audit.py`（流式 proper SE3 / 单独聚合）、
`heldout_mesh_visuals.py`（固定渲染、排名、图册核查）、
`summarize_heldout_mesh.py`（独立复算配对统计、作图及最终报告）。
前一轮训练、生成与渲染代码和旧实验结果保持冻结。

2026-10-18 07:02 UTC 在 GPU 4、5 启动，07:38 UTC 全部 320 项正式结果补齐，随后
完成独立聚合与全量 SE(3) 核查。完整过程见
[`scheduler_status.json`](../../outputs/round2_v2_heldout_mesh_20261018/scheduler_status.json)、
[`run_journal.jsonl`](../../outputs/round2_v2_heldout_mesh_20261018/run_journal.jsonl) 和
[`审计.md` 第 5.4 节](../../审计.md)。

**V2 相对父模型的最终结果。** 下表的 CD 改善为父模型减 V2，F-score 改善为 V2 减
父模型；CI 为对象配对 95% bootstrap 区间。距离单位是规范化网格坐标，不是米；
CD 为两向最近邻欧氏距离均值的半和。

| 队列 / 坐标 | CD：父模型 → V2 ↓ | CD 相对降低 | CD 改善的 95% CI | 改善对象 |
| --- | ---: | ---: | ---: | ---: |
| Clean64 原生 | 0.01719040 → 0.01630598 | 5.14% | [0.00053913, 0.00127797] | 49/64 |
| Clean64 SE(3) | 0.01441073 → 0.01398481 | 2.96% | [0.00015032, 0.00077359] | 43/64 |
| Strong16 原生 | 0.03123715 → 0.02949441 | 5.58% | [0.00000766, 0.00345979] | 11/16 |
| Strong16 SE(3) | 0.02585318 → 0.02396751 | 7.29% | [0.00031342, 0.00381722] | 11/16 |

| 队列 / 坐标 | F-score@0.02：父模型 → V2 ↑ | 绝对改善的 95% CI |
| --- | ---: | ---: |
| Clean64 原生 | 0.759795 → 0.781014 | [0.015112, 0.027406] |
| Clean64 SE(3) | 0.822834 → 0.833387 | [0.005607, 0.015948] |
| Strong16 原生 | 0.525130 → 0.550666 | [0.011503, 0.040282] |
| Strong16 SE(3) | 0.600109 → 0.621764 | [-0.000392, 0.049146] |

Clean64 的原生和刚体拟合后 CD/F-score 均支持正收益，因此改善不能仅用整体旋转、
平移解释。Strong16 的 CD 改善区间也为正，但原生 CD 下界很接近零，拟合后的 F-score
区间跨零。Strong16 原生 CD 的对象改善中位数为 0.00125331；去掉最大改善对象后
均值仍为 0.00125979，去掉最大两对象后为 0.00083881，主结果没有删去任何对象。
这支持平均收益并非仅由一个极端对象维持，但不消除小队列的不确定性。

相同 16 对象上，“强遮挡时的父/V2差值减去干净时的父/V2差值”的原生 CD 均值为
0.00122266，CI [-0.00055402, 0.00305591]；F-score 为 0.00747909，CI
[-0.01100813, 0.02539172]。SE(3) 下两指标的交互区间同样跨零，**不能宣称强遮挡
相对干净输入有显著更大的方法优势**。

**坐标与刚体检查。** 全部 320 个 PLY 原生指标复算最大绝对差为 0；709 个唯一源
文件 SHA 通过。所有拟合均为 proper SE(3)，最大 `|det(R)-1|=2.11e-15`，无缩放、
镜像或非刚性形变；31 例保留 identity。绘图应用的矩阵与指标计算矩阵相同。
Clean64 的拟合旋转中位数为父模型 1.825° / V2 1.593°，平移范数中位数为
0.014997 / 0.013597；Strong16 对应为 6.209° / 4.967°、0.029922 / 0.029260。
拟合修正量受对称性和形状误差影响，不是真实姿态误差；GT 辅助 SE(3) 仅作诊断，
原生帧仍是主结果。有限多初值 ICP 不保证全局最优。

**mesh 实际效果。** 固定排名选出的 clean 对象为 54/45/47/4/52，strong 对象为
7/5/1，包含最差、中间和最佳改善对象。Clean 对象 52 的细长分支更完整，三个视角
和拟合图均可见；对象 45 的两方法差异较小，对象 54 的箱体结构/布置出现退化。
Strong 对象 1 的部分伪面减少，但台阶和底座仍有明显缺损；对象 7 仍严重变形。
对象 4 的代理 GT 自身包含明显薄片状结构，不能把指标提高直接解释为恢复了完整
真实形状。当前优势主要是部分结构/轮廓改善，尚不是全面高精度细节恢复。

查看[clean 实际 mesh](../../outputs/round2_v2_heldout_mesh_20261018/figures/representative_native_clean_v0.png)、
[strong 实际 mesh](../../outputs/round2_v2_heldout_mesh_20261018/figures/representative_native_strong_occluded_v0.png)、
[原生与刚体拟合三视角](../../outputs/round2_v2_heldout_mesh_20261018/figures/pose_clean_52.png)、
[完整离线图册](../../outputs/round2_v2_heldout_mesh_20261018/index.html)与
[效应及置信区间图](../../outputs/round2_v2_heldout_mesh_20261018/figures/generalization_effects.png)。
原生图册覆盖 clean64 / strong16、两颗 seed、三个视角；拟合切换限预选代表对象。
图册还提供 1056 张轮廓叠加图和原始 PLY 链接。显示尺度、世界原点、相机、材质、
灯光固定；未逐 mesh 重新居中或缩放，1248 张渲染均无空图或裁剪。

**推理复现性和运行恢复。** GPU0 两次真实 CUDA 非法访问均在独立进程/其他设备上
按原协议恢复；另有三次协调器交接导致的进程中断。无样本排除。主协调器保留
`worker_failed` / exit1 历史，独立补跑 exit0；最终完成依据为全 320 例和证据核验，
不是把主协调器历史改写成成功。

六对自然出现的相同 SS support 在相同冻结 SLat 输入/权重/噪声下仍观察到小幅运行
差异，最大 CD 差为 4.65745e-5。额外四次原参数复跑中一例 mesh 逐位相同，其余
三例有差异，最大 CD 差为 4.70507e-5、F-score@0.02 差为 0.00059995。
这些观察不能作为其他样本的误差上界，也不能宣称确定性开关保证全链路逐位复现。
仅将六对理论相同支持的增益置零作敏感性分析，clean CD 平均改善从 0.0008844236
变为 0.0008839391（变化约 0.055%），这六对波动没有驱动本轮均值收益。
正式记录/mesh/主统计均未替换；四个复跑 mesh 不计入正式 320 项。

机器报告与可复查数据：
[`final_report.json`](../../outputs/round2_v2_heldout_mesh_20261018/final_report.json)、
[`metrics_summary.csv`](../../outputs/round2_v2_heldout_mesh_20261018/metrics_summary.csv)、
[`独立统计核查`](../../outputs/round2_v2_heldout_mesh_20261018/independent_statistics.json)、
[`相同支持敏感性`](../../outputs/round2_v2_heldout_mesh_20261018/identical_support_runtime_audit.json)、
[`四例复跑`](../../outputs/round2_v2_heldout_mesh_20261018/runtime_repeatability/compare.json)、
[`极端对象敏感性`](../../outputs/round2_v2_heldout_mesh_20261018/object_gain_robustness.json)。

论文层面的结论可写为：**当前 V2 相对父模型，在未参与已审计训练的开发对象上改善
最终 mesh，且刚体拟合后仍有 CD 收益。** 本轮不涉及新训练，也没有证明 V2 超过 V1；
是否超过 V1 并非“相对父模型有效”的必要条件，但机制新意仍需单独论证。Dev64 已参与
开发、GT 为代理、强遮挡仅 16 个合成对象、训练 seed 仅一个，均应保留在论文限制中。
后续正式泛化结论应使用冻结方案后的独立测试集及原始高精度 GT，不能继续把本 Dev64
调参结果称为最终盲测。

## Formal2K 双卡训练及公平对照（2026-10-18）

贡献、数据和评估设计见原冻结协议
[`FORMAL2K_PROTOCOL_20261018.md`](FORMAL2K_PROTOCOL_20261018.md)，
时间记录见 [`审计.md` 第5.5节](../../审计.md)。机器协议及所有输出位于
[`round2_v2_formal2k_20261018`](../../outputs/round2_v2_formal2k_20261018)。

用户最新要求把等待中的队列迁至 **GPU 4、7**，并自行负责长期监视。本次切换由agent
恢复现有队列：缓存准备在GPU4运行，完整发布train6000后，四个训练阶段均用GPU4/7
同步训练。切换起点为130/4000个遮挡样本（64个历史复用+66个新增），训练0步；
实时进度以监视器为准。
11:32 UTC处理object65时mesh解码发生实际CUDA OOM；12:00 UTC已完成修复回归并
冻结`runtime_amendments/003_data_memory.json`。
新数据worker让DINO在编码调用之外驻留CPU，并将mesh属性的确定性归约逐通道执行。
保持原归约函数、贡献顺序、精度、分辨率与全部训练命令。原协议、001与002历史修订
均保留；当前004硬件迁移继承003数据准备修复及002双卡训练设置。

以后需要恢复已停止的队列时，在工作区根目录执行：

```bash
cd /home/zjr/Tracker
bash pose_aligned_layered_reconstruction/experiments/round2_v2/run_formal2k_gpu47.sh
```

使用上述新入口恢复；旧`run_formal2k_memoryfix.sh`仍固定GPU1/2。
脚本检查已有服务状态和运行环境、校验修复及双卡协议，再提交独立后台服务。
没有额外显存额度或单独容量探测步骤。阶段启动前检查所需卡空闲；已有其它任务时
等待，不中断其它任务。退出终端/监视器不会停止后台服务，不承诺机器重启自动恢复。
数据并行仍在每卡加载完整模型，两卡显存不会合并供单对象使用；实际流程出错即停，
保留缓存、日志与checkpoint，不自动减少对象、降低分辨率或变更训练预算。

实时监视：

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python -m \
  pose_aligned_layered_reconstruction.experiments.round2_v2.formal2k_runtime_gpu47 monitor --watch 5
```

服务和当前准备日志：

```bash
systemctl --user status tracker-palr-v2-formal2k-20261018.service --no-pager
tail -F pose_aligned_layered_reconstruction/outputs/round2_v2_formal2k_20261018/logs/prepare_occlusion.log
nvidia-smi -i 4,7 -l 2
```

阶段依次为：遮挡缓存→共享LaRI 12000更新→full SS 9000→no_layer SS 9000→
single_layer SS 9000。2000对象各1个clean+2个occlusion，共6000场景样本。
训练设置从单卡`world_size=1 / grad_accum=2`调整为双卡`world_size=2 / grad_accum=1`，
全局有效batch保持2；学习率1e-4、步数及LaRI 24000/每组SS 18000曝光预算不变。
6000为偶数，每步的全局样本对保持一致，但rank1独立随机流与浮点归约改变，不能声称
与原单卡逐位等价。三组SS统一使用同一双卡设置、数据顺序、rank种子、优化器和父权重。
训练日志中的`cuda_peak_allocated_mib`仅为rank0，不能当作两卡全局峰值。

full/single使用共享最终raw LaRI，并在SS阶段冻结；single只读第一scene交点。
no_layer使用可训练raw/masked/mask外观投影，不消费分层预测或层监督。可训练参数为
3,189,377 / 3,213,977 / 3,189,377；几何前端成本单列，不声称三组总计算完全相等。
Formal2K的SS冻结几何，与之前Train32联合训练不同，不能只归因于数据规模变化。
最终raw checkpoint固定用于后续评估，不按Dev64选点；有限预算不代表充分收敛。

原三组真实单卡GPU各3步验收、20项CPU测试及控制器故障注入记录保留。
双卡启动前做配置/命令和恢复校验；没有把单卡短跑称作已完成的双卡GPU验证。
双卡训练会在初始化和结束时校验两rank参数哈希一致，梯度先跨卡求均值再裁剪和更新。
完成回执绑定最终checkpoint及两个rank RNG文件。恢复也要求两个RNG文件完整、
时间不早于主checkpoint；中断若造成文件不一致则拒绝恢复，保留现场后人工选择完整
checkpoint，不能把原单卡checkpoint改元数据后冒充双卡恢复。

GPU7历史：第一次12GiB额度在6个新缓存后触及上限；16GiB探针完成对象35解码
（峰值15150.448MiB）后因独立启动PATH缺少ninja失败。后续22GiB候选未运行、未冻结。
这些历史在审计和输出目录保留，当前双卡入口不使用旧GPU7的probe/cap脚本。

本次修复的[实际GPU回归报告](../../outputs/round2_v2_formal2k_20261018/memory_fix/regression_gpu7/report.json)
于11:59 UTC通过，独立输出不计入正式130个缓存。对象35的同次原版/优化版mesh全部
张量，以及已有生产缓存的全部张量、图像、标签和scene元数据逐位一致。对象65、1918
完成完整增强/特征缓存，并在真实全部cube行、全部10属性通道上通过原函数双通道参考
与新逐通道结果的逐位比较。PyTorch allocated峰值分别为7535.46、9298.64、10586.71MiB；
后两例包含额外参考计算，不能当作纯生产流程的峰值。35的原版单独解码峰值为
15150.45MiB，覆盖范围与优化版完整流程不同，不据此声称精确加速/显存降低比例。
三个对象通过不等于已完成4000个遮挡样本；正式运行仍保留原失败停止与恢复机制。

首次修复回归错误地将历史clean mesh数量当作本次原实现的输出检查，已保留失败证据
并改为严格的同次原版对照。两次原版35的数量曾不同，第二次与历史值一致；原因尚未
单独定位，不声称跨进程逐位复现。修复没有放宽同次mesh和既有生产缓存的一致性要求。

`training_complete_pending_evaluation`仅表示四项训练完成。后续父模型及三组对照、
Dev64/Strong16、两推理seed共640条mesh记录的指标/刚体诊断/全量图册设计已冻结，
本服务不自动执行评估；独立原始GT、真实AR和重复训练seed仍是后续论文工作。
