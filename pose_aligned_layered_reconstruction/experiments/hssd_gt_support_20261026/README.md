# 共享照片下的 GT 支持 → 冻结 SLat 诊断

前一轮 SS×SLat 交叉已表明：桌子和凳子的主要输入相关几何变化跟随 SS 支持。
本轮把 SS 替换为已有 GT 占据坐标，在当前共享照片接口下检查冻结 SLat 的生成能力，
决定后续能否集中改进 SS。**这是使用特权支持的诊断，不是新训练，也不作为实际方法成绩。**

## 当前结果：2026-10-26 06:13 UTC

03:02:08 UTC全量结束，新增12/12、逻辑24/24；12张图、6个GLB齐全。全部图审和独立CPU统计/latent/坐标验收已完成，详见[结果审阅](../../outputs/hssd_gt_support_20261026/review_20261026/REPORT.md)。

预测支持→GT支持：宏CD 0.074228→0.007628，F@.02 30.62%→99.79%，12/12配对改善；但GT支持已给出隐藏表面位置，不能作为RGB方法成绩。更严格F@.01仅73.76%，局部细节仍有残差。12生成latent均不同于GT编码，各seed mesh不同，坐标转换复核通过。

下一步集中于当前共享输入上的匹配SS短训：appearance-only与几何条件分支同输入同预算比较，先冻结SLat。当前外观适配器训练于旧8视图对象中心输入，不能忽略输入分布变化。原LaRI不直接恢复长训。本次未启动新训练，无需再次start本队列。

## 历史启动交接：2026-10-26 02:55 UTC

CPU核验及GPU短检查已通过。旧table2072/seed42的canonical/world顶点、面和指标精确重放；stool/seed42的真实GT支持采样完成，计入新增1/12。12组预测/GT支持公共坐标噪声一致。

助手已于02:54:20 UTC在GPU0启动后台完整队列，PID3383489；02:54:53核验进程身份匹配、心跳0秒、阶段mesh、新增1/12。用户只需运行下面的watch，不需要再次start。随后自动补齐11个、汇总及生成12张对比图和6个场景GLB。

身份：`bba6b0443f9611f30edaa4c4096a05128024163ca75c5ad4a1d35ce527b33d96`。
证据：[交接记录](../../outputs/hssd_gt_support_20261026/HANDOFF.md)、[验收](../../outputs/hssd_gt_support_20261026/acceptance.json)。

## 固定范围

- 使用原来的全部四个物体，保留椅子、桌子、长凳、凳子及 seeds 42/43/44，不删难例。
- 所有物体统一使用 `crop2072_visible` 条件；2072 是源渲染尺寸，网络输入仍为 518。
- 对照为同条件、同冻结 SLat 的 appearance_control 预测 SS 旧结果，共 12 个引用。
- 只新增 12 次 GT 支持上的完整 SLat→mesh 采样；不重新生成 SS、不重新编码 GT latent。
- 保持原四 UID 加载上下文、原对象 index、采样器、权重、decoder、坐标变换和 GT 评估分区。
- 同一 canonical 坐标使用相同初始噪声；预测/GT 支持不必等长或具有公共坐标。

12 个新增结果加 12 个旧引用，共 24 个逻辑结果；采样 seeds 是对象内重复，不是独立物体。
固定 GT 支持只检验支持已知时的生成能力，不保证每个指标必然优于预测支持。
当前输入仍含 GT 辅助相机、mask 和 canonical frame，不支持 RGB-only 泛化结论。

## 启动与监视

默认单卡 GPU0，通过 `HSSD_GT_SUPPORT_GPU` 显式指定。助手负责实现、短检查和启动长队列，用户只需执行：

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_gt_support_20261026/run.sh watch
```

只查询一次状态用 `status` 替换 `watch`；`Ctrl+C` 仅退出监视，不停止后台队列。
本目录入口使用绝对路径后，可以从任意工作目录调用。

以下为助手启动/恢复命令，用户不需要重复执行：

```bash
HSSD_GT_SUPPORT_GPU=0 bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_gt_support_20261026/run.sh start
```

`start` 在后台串行运行 `prepare → smoke → mesh → summary`，以排他锁防止重复队列。
启动时只读取选定卡的显存占用，不分配显存探测、不自动换卡。
已完成 mesh 和采样 latent 通过身份及输入绑定检查后复用；遇到错误则停止并保留失败记录。
`stop` 只停止本实验身份匹配的后台队列及它创建的当前子进程，保留完成产物。

CPU 准备和短检查可分别运行 `prepare`、`smoke`；只重新生成统计与图片可用 `summary`。
直接执行这些动作是前台任务，助手在交付长队列前完成必要检查。

## 短检查与结束标志

短检查包括旧 table/seed42 的 `SS2072 + condition2072` 完整重放，以及 stool/seed42 的真实 GT 支持采样。
重放必须匹配旧 canonical/world mesh 和指标；GT 样本计入新增 12 个结果，无需重复计算。
通过后其余 GT 支持样本才进入长队列。实际通过状态以 `smoke.json` 及其完整绑定为准，
实际完成状态与时间见文首交接；这些要求已由本轮短检查验证。

`all_complete_review_required` 表示计算和汇总已结束；应显示新增 GT 支持 mesh 12/12、旧预测支持引用 12、逻辑结果 24/24。
此时阶段为空，状态时间停止刷新是正常现象，无需继续等心跳或再次启动。
最终仍需审阅逐物体/逐 seed 指标和 GT/预测支持/GT支持 mesh 对比图，不能仅凭完成状态认定模型通过。

输出目录：`/home/zjr/Tracker/pose_aligned_layered_reconstruction/outputs/hssd_gt_support_20261026`。
队列日志在 `logs/`；进程身份和阶段在 `status.json`；最终统计为 `summary.json`、`REPORT.md`。
协议见 [PROTOCOL.md](PROTOCOL.md)。冻结后不要修改 Python、run.sh 或协议文件；README 和时间记录可继续补充交接状态。
