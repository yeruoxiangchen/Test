# HSSD TRELLIS.2 MVBase12

2026-10-28：Native12原生标签审阅之后的下一阶段。构建同骨干多视图独立物体基线，同时准备跨物体邻居组。**本轮尚无关系模型训练，不是论文最终泛化实验。**

13:02 UTC启动核验：完整队列PID1336208，进程身份匹配、心跳刷新，状态`waiting_for_gpu/smoke_ss`、0/56阶段。当前所有卡均未达到空卡条件，尚未开始GPU短检查或训练。运行身份`93e1a3a1e72e21e84cc4de27dd4e5060164edc67ff8a8c8ee3bb236022c087fd`。这是启动快照，之后以watch为准。

已完成312个共享帧的候选mask检查：302对候选中43对满足至少两个拍摄位置，共固定12对；其中仅1对两端已有原生标签，涉及7个新增邻居尚需后续编码。全部12对已保存照片/mask叠加图，GT遮挡资格核验排入GPU队列，当前不能称12对都已确认遮挡或已经具备关系训练所需的完整缓存。

日常只需监视（任意工作目录可运行）：

```bash
bash /home/zjr/Tracker/pose_constrained_trellis2/experiments/hssd_mv_baseline_20261028/run.sh watch
```

`Ctrl-C`只退出监视。`waiting_for_gpu`表示尚未分配GPU，不是卡住；队列单卡串行，在0–7中等待首张合格卡。需要总显存<512MiB、空余>=22000MiB、利用率0且没有大显存计算进程，容许少量空闲驻留上下文，不停止其他任务。没有分配式显存探测。

## 自动执行内容

1. 最大原生token训练对象（6800 tokens）的SS/Shape各2步，实际留出对象两臂完整12/12采样，确认新评估入口可以在严格训练绑定下评估验证对象。
2. 根据已有共享照片/mask选出的邻居候选，用GT几何单列核验遮挡。GT深度只用于数据诊断，禁止当作测试输入。
3. 从官方权重新建SS和Shape，8训练对象各400步、50轮，单卡分阶段，保存200/400检查点。独立smoke权重不进入正式训练。
4. 全部12对象×2个seed×2臂=48次完整生成，自动导出规范/world网格、数值报告和24张真正三角网格三视角对比图。

`official_reference`是冻结的官方单图参考；`mv_independent`是训练后的多视图独立基线。之后关系模型的主对照是后者，不能把本轮相对单图参考的变化都算成关系贡献。仍使用GT相机/物体变换；4个验证对象只来自一个场景，不能据此声称广泛泛化。固定raw400权重，不按验证结果挑checkpoint。

所有目标和失败留在统计分母；程序或CUDA异常停队列，模型空输出/超8192 token明确计为失败。无ICP或逐物体拟合旋转/尺度。训练/验证分别汇总；100k点mesh指标与旧20k点codec指标不直接比较。详细定义见[冻结协议](PROTOCOL.md)。

## 数据与结果

- 复用原生标签：`/data/zjr/pose_constrained_trellis2/datasets/hssd_scene_native_v1`。
- 新清单及邻居索引：`/data/zjr/pose_constrained_trellis2/datasets/hssd_mv_baseline_v1`。
- 输出：`/data/zjr/pose_constrained_trellis2/outputs/hssd_mv_baseline_20261028`。
- 结束报告：上述输出中的`REPORT.md`、`summary.json`、`figures/`；终态`all_complete_review_required`后无需继续等待。
- 所有训练/模型/绘图临时目录和缓存继续使用项目/data目录；没有修改旧任务或共享模型环境。

邻居候选v1仅覆盖每场景已有8目标，容易遗漏真正遮挡物。保留v1历史，实际v2使用12个anchor与三个场景所有合格、已有照片/mask的候选资产，按输入mask边界关系选组；跨划分家族及精确几何重复者剔除。`pair_proposals_v2.json`记录共视输入候选、尚缺原生标签的邻居及全部候选覆盖；`pair_qualification_gt_only.json`稍后记录实际遮挡资格。可见mask边界相邻不自动证明遮挡，新增邻居没有编码时不声称关系训练数据齐备。

## 启动与恢复

已由助手启动后不要重复start。代码、清单未变且队列失败/停止时，可续跑：

```bash
bash /home/zjr/Tracker/pose_constrained_trellis2/experiments/hssd_mv_baseline_20261028/run.sh start
```

仅在需要限制候选卡时，在启动命令前设置`T2_BASE_GPUS=编号`，逗号分隔表示任选一张，不表示多卡训练。`run.sh stop`只终止该队列及它启动的计算子进程，保留产物；完整阶段按源码绑定回执复用，训练从最近已保存检查点恢复。

## 启动前已执行检查

6项CPU检查通过：实际train/val场景与资产隔离、推断不读取目标latent、检查点保持训练身份并拒绝错误验证绑定、48case分母、/data存储/组内划分，以及GPU准入不抢占大显存任务。Python/bash语法检查通过。这些不能代替队列中的实际GPU短检查。

准备期间GPU0新增vLLM占用，最初仅候选GPU0的空等待队列已停止并归档；当时没有任何计算阶段启动或产物被删除。新队列自动等待0–7中合格卡，并将短检查、训练、评估串接，用户无需手动切换阶段。
