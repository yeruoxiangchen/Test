# SS支持 × SLat条件交叉诊断

检查上一轮桌子高分辨率退化、凳子改善主要跟随SS支持，还是跟随SLat观测条件。
仅使用原table i075/stool i065及seed42/43/44，不训练、不重采SS、不重编码GT latent。
协议见 [PROTOCOL.md](PROTOCOL.md)。

| SS支持来源 | SLat条件来源 | 本轮处理 |
|---|---|---|
| crop518 | crop518 | 引用旧结果 |
| crop2072 | crop2072 | 引用旧结果 |
| crop518 | crop2072 | 新增交叉 |
| crop2072 | crop518 | 新增交叉 |

共24个逻辑位置，包含12个旧对角引用和12个新增mesh；仍只有2个物体、1个场景。
两轴不是独立网络训练；支持改变占据量/形状/稀疏拓扑，条件同时替换全局及投影DINO。
保持原相机、GT分区、原物体index5/7、SLat30K及采样/解码路径；同一canonical坐标获取相同噪声。
采用已验证的LargeInputNative，原数据、旧实验状态及绑定产物都不修改。

## 当前交接

2026-10-26 01:58 UTC：全量计算已于01:14:44 UTC完成；新增mesh12/12、旧引用12、逻辑24/24，6张图和12个场景GLB齐全。独立统计/绑定/坐标复核及六图视觉审阅完成，见[结果分析](../../outputs/hssd_ss_slat_cross_20261026/review_20261026/REPORT.md)。

主要变化跟随SS支持：桌子换SS时F下降14.11–15.82个百分点，凳子上升25.34–25.51个百分点；固定SS只换条件的平均F变化分别为−0.45至+1.26、+0.02至+0.19个百分点。SLat latent确实变化，不是重复使用同一缓存。该结论仅适用于两个旧对象的本次输入干预，不证明SLat整体无问题或LaRI有效。

当前`all_complete_review_required`、无活动阶段表示队列已结束；旧心跳不代表卡住。无需再次start。没有启动下一轮GPU任务。下面保留短检查交接历史和运行命令供查阅。

### 历史短检查（00:46 UTC）

2026-10-26 00:46 UTC：CPU数据核查和GPU短检查通过。
table/seed42/SS518→condition518完整重放的canonical/world顶点、面及所有指标精确复现。
stool/seed42/SS518→condition2072已完成，计入新增结果1/12；长队列尚未启动，剩余11个。
6组支持的坐标噪声检查通过；stool seed43两域支持无交集，记录为无可比较公共点，不声称噪声向量相同。

当时`smoke_complete_review_required`且阶段为空，表示短检查结束；不会自动启动长队列。
当时GPU4短进程已退出。完成记录见 [smoke.json](../../outputs/hssd_ss_slat_cross_20261026/smoke.json)，
数据/合同检查见同目录preflight.json、contract_checks.json、acceptance.json。

## 启动与监视

从任何工作目录执行，默认单卡GPU4；换卡时改`HSSD_CROSS_GPU`。

```bash
HSSD_CROSS_GPU=4 bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_ss_slat_cross_20261026/run.sh start
```

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_ss_slat_cross_20261026/run.sh watch
```

流程：CPU冻结/数据核查 → 复核已完成短检查 → 补齐剩余交叉mesh → 汇总和可视化。
`start`在后台运行；只查询选定卡的显存占用，不分配探测显存，不自动换卡。
`status`看一次，`watch`的Ctrl+C只退出监视；`stop`只结束本实验队列和其子进程。
同一`start`可续跑，已完成mesh与解码前SLat缓存先校验再复用；错误条件来源或身份会拒绝复用。

完成标志：`all_complete_review_required`，新增mesh12/12、旧引用12、逻辑24/24。
输出：`/home/zjr/Tracker/pose_aligned_layered_reconstruction/outputs/hssd_ss_slat_cross_20261026`。

## 最终产物与解释

- summary.json、REPORT.md：逐物体/seed的完整2×2结果，固定SS时条件差、固定条件时SS差、交互HH−HL−LH+LL。
- figures/：6张GT+四角的三固定视角对比图；复用12个旧对角渲染，仅渲染12个交叉mesh。
- world/：12个双物体场景GLB，采用已知位置，不是预测场景布局。
- visual_audit.json：渲染、绑定与裁切记录，计算结束仍需查看真实mesh。

若固定支持时条件变化仍造成桌子退化，优先检查SLat观测接口；若固定条件后变化随支持而走，
优先定位SS占据生成；两者均影响时报告交互。不能用两个事后选择的旧训练对象声称泛化或LaRI贡献。
本轮不包含GT support oracle或长凳追加采样，也不删改上一轮长凳失败记录。

冻结后不要编辑本目录Python、run.sh、PROTOCOL.md；README及项目时间记录可继续更新。
