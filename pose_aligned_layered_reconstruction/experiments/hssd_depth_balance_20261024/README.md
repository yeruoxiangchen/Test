# HSSD DepthBalance16

本轮只检验目标/隐藏目标深度监督平衡，详见[冻结协议](PROTOCOL.md)。
预检完成后，单卡启动两组LaRI1000＋两组SS adapter200＋LaRI审计及96条新SS：

```bash
HSSD_DEPTH_GPU=3 bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_depth_balance_20261024/run.sh start-run
```

监视（Ctrl-C只退出监视）：

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_depth_balance_20261024/run.sh watch
```

GPU是物理编号。只在启动命令改HSSD_DEPTH_GPU；不会自动换卡或抢占其他任务。
命令可以从任意目录运行。正式run完成停在ss_complete_review_required，不自动启动mesh。
SS审阅后，再单独启动24个新mesh及12张图库；既有36个参照mesh核验后复用：

```bash
HSSD_DEPTH_GPU=3 bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_depth_balance_20261024/run.sh start-mesh
```

中断后以同一命令恢复，已完成阶段核验后跳过。不要同时启动两个队列。
也支持start-train、start-ss、status；preflight/start-preflight只用于工程预检，用户无需重复执行。
训练结果位于outputs/hssd_depth_balance_20261024/train/，分组审计audit_summary.json，
SS比较ss_summary.json，最终mesh比较mesh_summary.json，固定相机图在figures/。
完成不代表假设成立，不自动扩量；三个seed是采样seed，数据仍是训练对象。
