# HSSD LaRI：算子诊断与固定内核公平复核

本轮不训练，也不重新编码 mesh→latent。先定位相同 SS/噪声产生不同 mesh 的原因，再重新比较此前选定的四个训练诊断物体。

实验输出：[hssd_operator_audit_20261025](../../outputs/hssd_operator_audit_20261025/)。
数值诊断结论见该目录的 `REPORT.md`；长复核完成后另生成 `fair/REPORT.md`。

## 已完成的诊断

- 三个独立进程、每进程重复两次，记录第一个稀疏残差块的子模块输入/输出、有效邻接关系及实际 spconv GEMM 内核。
- 两个独立进程从同一个保存的 latent 解码，21 个记录点全部一致，并与先前固定 latent 解码一致。
- 首次数值差异位于 `input_blocks.0.conv1`。邻接表的列顺序会变化，但每个偏移的输入→输出连接集合相同；不能把排列变化当作特征数值变化。
- 固定两种实际观察到的内核，按 A/B/A 顺序执行原残差块，精确复现各自观察到的输出。原始块输出也分别与先前完整 SLat 回放的两种输出一致。

`kernel_policy.py` 只在当前推理进程内固定 native GEMM 的兼容候选，保留原卷积、权重、张量精度及 mesh 转换。首选已观察到的 fp16 内核；其他兼容布局按固定描述符顺序选择。C++ tuner 每个签名只有一个候选，不再由计时竞争决定选择。它不改安装包，不适用于训练或 backward。

该结果解释本案例的已观察差异，不证明所有模型、对象和 GPU 均已确定性，也不改变 LaRI 尚未超过同容量外观对照的结果。

## 长复核的固定范围

`appearance_control / uniform / balanced` × 原索引 `0,2,5,14` × 采样 seed `42,43,44` × 两个独立进程，共 **72 个新 mesh**。

第二进程反转对象、seed 和方法顺序。三组全部使用相同新内核规则重新生成，不复用旧对照 mesh，不合并相同 SS 的案例。每个对象/seed 共用保存的实际 CUDA 坐标噪声；每次保存初始稀疏张量、最终 latent、mesh、内核信息、权重身份和指标。相同输入跨进程必须产生相同 latent、mesh 和指标。GT 与预测使用原生坐标，不拟合旋转/平移/尺度，不做 ICP。

`fair-smoke` 只执行柜架 i083、seed42 的 uniform/balanced 两条完整采样，各在独立进程中运行。由于两组 SS 完全相同，这是完整路径的相同输入对照。它通过后才允许启动长复核。

2026-10-25 05:27:39 UTC：上述 smoke 已通过，初始张量、最终 latent、mesh、PLY SHA、全部指标一致，短检查已退出。长复核尚未启动。

本轮仍是四个选定训练物体的诊断，不是 heldout 主实验。三个采样 seed 不等于三次独立训练，两个执行顺序不增加独立样本数。

## 用户启动与监视

在任何目录都可以执行，默认 GPU 3。长复核由用户启动：

```bash
HSSD_OPERATOR_GPU=3 bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_operator_audit_20261025/run.sh start-fair
```

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_operator_audit_20261025/run.sh watch
```

单次查看：将 `watch` 换为 `status`。长复核最终状态为 `fair_complete_review_required`，完整结果在 `fair/run_summary.json`，24 张共享相机对比图在 `fair/figures/`。

如果被中断，使用同一个 `start-fair` 命令继续；已完成项先校验文件绑定再复用，新的进程片段独立记录。不要同时启动多个队列。若状态为 failed，先读所列日志；不删除失败项、不自动丢弃对象、不跳过复现门槛。

以下仅用于诊断复现，日常无需重跑：

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_operator_audit_20261025/run.sh audit
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_operator_audit_20261025/run.sh pincheck
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_operator_audit_20261025/run.sh fair-smoke
```

## 继续研究的边界

目标/隐藏目标深度改善、oracle 几何的 SS 收益说明几何条件仍值得检验；预测 LaRI 的 SS 和现有 mesh 结果尚未稳定超过外观对照。数值修复只保证评估可比较，不能当作方法贡献。

本次公平复核用于决定当前配置是否收尾。若仍未超过同容量外观对照，停止当前配置的扩量和追加长训练，转向条件注入或论文范围的重新定义；不得因某个 seed 或某一幅图更好而继续堆算力。即使本次胜出，仍需独立场景和训练重复，才有资格进入论文主实验。
