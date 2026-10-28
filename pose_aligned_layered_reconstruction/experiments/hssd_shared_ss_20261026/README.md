# 共享场景输入上的 SS 匹配短训

本轮已于 **2026-10-26 09:23:38 UTC** 完成。结果与全部12图审阅见 [review_20261026/REPORT.md](../../outputs/hssd_shared_ss_20261026/review_20261026/REPORT.md)。外观匹配适配带来主要提升；几何分支有局部增益，强遮挡长凳仍失败。无需再次 start。

固定原四个 HSSD 物体、共享照片、父模型和 SLat，比较两组各 200 步训练：新增外观残差，与同容量的 RGB 预测几何残差。后者使用本地 VGGT 的深度预测，将可见目标表面、前方自由空间、遮挡后未知区域分开编码。这里不恢复旧 LaRI 多层交点训练。

这是同一场景、四个旧物体的可学习性实验；不能当作未见场景泛化结果。相机、可见 mask 和规范坐标仍沿用已知输入。几何组增加了冻结 VGGT 和全图上下文，同训练容量不代表同推理开销。详细约束见 [PROTOCOL.md](PROTOCOL.md)。

输出目录：`/home/zjr/Tracker/pose_aligned_layered_reconstruction/outputs/hssd_shared_ss_20261026`。

监视命令，在任意目录可执行：

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_shared_ss_20261026/run.sh watch
```

后台队列依次完成输入核验、预测深度、几何缓存、GPU 短检查、两组训练、SS 采样、mesh 生成与汇总。两组各新增 12 个 SS/mesh，另引用 12 个旧冻结结果，共 36 个逻辑结果。最终生成 12 张对比图和 9 个已知场景摆放的 GLB。

单次状态用 `status`；完成状态是 `all_complete_review_required`，结束后心跳时间不再刷新。`Ctrl-C` 退出 watch 不会停止后台队列。

首次启动或失败修复后的续跑入口为 `run.sh start`，默认 GPU 0；换卡可显式设置 `HSSD_SHARED_SS_GPU`。已经运行时不要重复启动。所有复用缓存必须匹配冻结输入和代码；`smoke` 的两个短训练检查与正式训练分目录，不为正式训练热启动。
