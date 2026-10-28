# HSSD 单案例 mesh 回放

2026-10-25 03:09:45 UTC已完成本轮限定回放，结果见
[诊断报告](../../outputs/hssd_mesh_replay_20261025/REPORT.md)。
跨进程最早差异位于SLat的`input_blocks.0`，具体子算子尚未定位；无需重复启动本轮已完成队列。

柜架i083、原始全局index=2、seed=42。三个完整SLat采样、两次额外固定latent解码；没有训练。
旧DepthBalance16代码、权重和输出保持原样，协议见[PROTOCOL.md](PROTOCOL.md)。

前台运行（已完成阶段会验证并复用）：

```bash
HSSD_REPLAY_GPU=3 bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_mesh_replay_20261025/run.sh run
```

后台运行与监视：

```bash
HSSD_REPLAY_GPU=3 bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_mesh_replay_20261025/run.sh start
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_mesh_replay_20261025/run.sh watch
```

一次状态查询用`status`。更换GPU会影响跨进程环境一致性检查；同一次成对回放应固定物理GPU。
输出目录是`pose_aligned_layered_reconstruction/outputs/hssd_mesh_replay_20261025`。
`summary.json`保存逐段精确一致性和数值差异；`attempts/`保存中间张量、latent、mesh及每次结果。
`replay_complete_review_required`表示全部限定步骤结束，并不预设它们必须数值相同。

失败不自动启动后续阶段，完整运行超时会终止本队列自己的子进程并保留部分结果。
再次`run`会验证已完成阶段；未完成阶段写入新attempt，不覆盖已有trace。
历史输入未保存中间tensor，当前回放即使完全一致，也不能证明旧差异的原因已经修复。
