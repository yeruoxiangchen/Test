# 深度 × 置信度交叉诊断

2026-10-27 01:53 UTC：计算和结果审阅均已完成。两新臂各200步、24新增SS/mesh、36旧引用完整；独立统计/训练/坐标验收通过，12张mesh拼图全部查看。本轮不用继续监视或重新启动。结论：深度与置信度处理相互影响；固定1的F收益集中在椅子，宏平均CD反而变差，长凳和桌底结构仍失败。下一步转未见对象/场景验证，不继续四旧物体调参。详见[审阅报告](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_depth_confidence_20261026/review_20261026/REPORT.md)及[过程交接](HANDOFF.md)。

本轮仅新增两组200步训练：GT可见首表面深度＋预测置信度，以及预测深度＋有效置信度设为1。已有预测/预测与GT/1两组复用，外观组作为背景基线。总计24新SS/mesh＋36旧引用＝60结果，其中主2×2对照48个。

首先用新路径重算两旧对角的全部8个物体条件缓存，要求逐值一致，再进入GPU短检查和正式队列。保持全部对象、seed和原预算，不据结果调阈值。详见[冻结协议](PROTOCOL.md)。

监视（任意当前目录）：

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_depth_confidence_20261026/run.sh watch
```

输出：`/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_depth_confidence_20261026/`。同脚本`start`启动完整后台队列，`status`查看一次，`stop`只停止本队列。单卡运行，优先空闲卡，必要时采用≥18GiB可用显存的共享卡；每阶段结束后可换卡，不重启已完成训练。

这是同场景四旧物体上的最后一轮预定机制定位；两个GT深度cell是特权诊断。随后需要未见场景/对象验证，本轮不据此声称泛化或原LaRI贡献成立。
