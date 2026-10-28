# SharedScene4 输入适配对照（2026-10-25）

目的：在修改模型或扩大训练之前，检验共享场景照片中的目标尺寸、空目标视图和原图分辨率是否限制现有模型。此轮不训练，不启用 LaRI，不重新编码 mesh latent。

## 冻结范围

- 沿用 SharedScene4 的一个 HSSD 场景、四个旧训练物体（chair/table/bench/stool）、相机帧 0/4/7/11 和种子 42/43/44。
- 两个模型分别为冻结 yxc30k 与冻结 appearance_control；各自比较输入变体。两者训练历史不同，跨模型比较不作为容量匹配的消融证据。
- SLat30K、采样步数 25、CFG 5、区间 0.5–1、时间重标定 3、初始噪声公式、固定 Native 稀疏算子和 mesh 解码方法沿用原实验。
- 沿用 GT 辅助的相机、可见实例 mask 和物体 canonical frame。不是端到端 RGB-only 系统，也不是未见物体泛化实验。

## 四种输入

| 变体 | 原图及输入 | 视图 | 用途 |
|---|---|---|---|
| full_all | 原 518 全图 | 全部四帧 | 显式引用旧基线 |
| full_visible | 原 518 全图 | 原 mask 非空帧 | 只排除完全不可见视图 |
| crop518_visible | 原 518 全图裁剪并缩放至 518 | 同上一行 | 在固定原始观测下测试目标尺度适配 |
| crop2072_visible | 同相机重新渲染 2072，再按相同区域裁剪至 518 | 同上一行 | 测试更高质量的原始栅格；同时包含重渲染抗锯齿/噪声变化 |

非空帧在读取高分辨率结果前冻结：chair/table/bench 均四帧；stool 仅帧 0、11。高分辨率新出现的像素不能改变选帧。所有 raw/masked feature、mask、K、外参、SS 全局 context 和 SLat context 必须同步选帧。

ROI 使用已有 canonical frame 的单位立方体投影，取边长为包围盒长边 1.10 倍的正方形，限制在原始照片内。禁止使用 GT 精细 mesh 边界或 amodal mask 决定 ROI。方框过大则缩到照片大小，并平移至照片范围内；不补虚构的黑色边界。原来位于画面外的部分不会被恢复。

PIL 的像素中心约定为 `u_new = (u_old-left+0.5)*scale-0.5`，用同一仿射左乘 K；2072 原图 K 使用 4 倍缩放及 1.5 像素中心偏移，高分辨率 crop box 为低分辨率 box 的 4 倍，两者最终 K 相同。相机外参、物体坐标系不变。

## 数量及复用

共 96 个逻辑比较位置（4 变体 × 2 模型 × 4 物体 × 3 种子），包含 24 个旧结果引用、18 个完全相同输入的显式别名，以及 **54 个新增完整 SS→SLat→mesh 样本**。不把引用或别名当作新增独立重复。独立物体仍只有 4 个、场景只有 1 个。

正式队列前执行短检查：原始 chair/yxc30k/seed42 完整重放，要求 SS 坐标和 mesh 顶点/面、指标与旧结果完全一致；以及 crop518/stool/yxc30k/seed42 的两视图完整推理。后者计入 54 个正式样本并在长队列中复用。

## 检查和停止条件

1. 冻结旧输入、模型、代码及结果的哈希；续跑不替换已完成样本。
2. 裁剪坐标检查、实际 mask 不被 ROI 截掉、低/高分辨率 K 一致。
3. 2072 Blender mask 与独立场景几何 rasterization 的 IoU 每项至少 0.95。首帧可先做短预览，其余三帧在长队列中处理。
4. DINO 由同一冻结编码器产生，绑定实际本地权重。GT SS/SLat target 不变，GT 深度不作为条件输入。
5. 原始重放失败、非有限输出、空 SS/mesh、输入或 checkpoint 哈希变化时停止并保留失败对象，禁止悄悄丢弃。

## 评估和解释

- 主指标：原生 canonical 坐标下 CD↓、F@0.02↑、SS IoU↑。不进行 ICP、自动旋转、平移或缩放拟合。
- 可见/外部遮挡表面 recall 以及 free-space violation 始终使用原来的四帧 GT 分区，不能随各变体的选帧改变评估集合。
- 分别报告每个模型内三组相邻对照：空帧处理、裁剪、原图分辨率。先按物体汇总三个种子，再报告四物体宏平均；给出逐物体差值和种子胜负。空帧影响另报唯一受影响的 stool，不夸大样本数。
- 同时导出每物体/模型/种子的五行三视角 mesh 对比图（GT+四输入变体），所有行固定相机、尺度和材质；提供已知物体坐标变换下的场景 GLB。这些布局是已知输入，不是模型估计。
- 不预设这些处理必然改善。裁剪无法补回遮挡信息，尤其 bench 原始可见像素很少。若可见部分仍严重错误，先调整实例观测及条件接口；若可见部分恢复而遮挡部分差，才为后续遮挡建模提供依据。此实验本身不证明 LaRI 有效，也不构成论文主结果。

## 运行

单卡串行，默认物理 GPU 0，可用 `HSSD_INPUT_GPU` 显式修改。启动前只读取占用，不分配显存探测，不自动切卡。长队列由用户启动和监视，断点续跑使用相同 start 命令。

```bash
HSSD_INPUT_GPU=0 bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_input_adaptation_20261025/run.sh start
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_input_adaptation_20261025/run.sh watch
```

完成状态为 `all_complete_review_required`：新增 SS=54/54、mesh=54/54，逻辑比较=96/96，汇总和可视化均已导出，仍需人工审阅 mesh。
