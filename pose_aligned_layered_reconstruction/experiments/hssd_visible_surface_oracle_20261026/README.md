# HSSD 可见首表面 oracle 对照

2026-10-26 12:39 UTC：本轮已完成并审阅，12新SS/mesh＋24旧引用齐全。Oracle相对预测几何的宏F@.02提高6.75pp、遮挡召回提高9.22pp，但长凳仍结构失败、自由空间违规增加0.42pp。12图已逐张检查，独立统计与坐标核验通过。详见[完整审阅](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_visible_surface_oracle_20261026/review_20261026/REPORT.md)。无需继续监视或再次启动；未启动下一组实验。

只新增 `geometry_oracle_visible`：同一结构、同一 200 步预算，用 GT 可见首表面深度与有效性置信度替代预测几何；遮挡后方仍未知。复用 SharedSS 的外观组和预测几何组，共 36 个逻辑结果，其中 12 个新 SS/mesh。

这是旧四物体、单场景上的受控诊断。它用于区分观测质量与几何注入问题，不能当作 RGB 方法成绩或泛化证据。协议见 [PROTOCOL.md](PROTOCOL.md)。

输出、权重、临时文件、模型缓存均位于 `/data/zjr/pose_aligned_layered_reconstruction/`。旧输入通过 SHA 校验的物理路径映射复用，不改旧实验。

监视（任意当前目录均可）：

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_visible_surface_oracle_20261026/run.sh watch
```

启动完整队列用同一脚本的 `start`；默认等待空闲单卡，自动执行短检查、200 步训练、12 个 SS/mesh 和汇总。`status` 查看一次；`stop` 只停止本队列。`storage`、`prepare`、`evidence` 是独立 CPU 阶段。已有任务运行时重复启动会被锁拒绝。
