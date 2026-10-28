# 2026-10-26 可见首表面 oracle 交接

更新 12:39 UTC：队列12:18:44正常结束，全部计算与汇总完成；36case、12图、9GLB已审阅。Oracle相对预测几何：宏F@.02 55.59→62.34%、CD下降5.08%、SS IoU23.51→27.64%、遮挡召回56.51→65.73%；free违规15.32→15.74%变差。椅子/高凳提升清楚，桌子底座仍错、长凳三seed仍是块体/墙片/宽平台。当前注入能利用可靠可见几何，但这不是纯深度消融，不证明原LaRI或跨场景泛化成立。建议最后补一轮深度×置信度两格交叉诊断，然后进入留出场景/对象验证；尚未实现或启动下一轮。独立统计、821项产物检查、12新mesh全4255424顶点坐标核验通过。见[审阅报告](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_visible_surface_oracle_20261026/review_20261026/REPORT.md)。以下为历史启动记录。

更新 12:07 UTC：用户要求切GPU。原队列在训练200/200、SS12/12完成后因GPU0被其他任务占用而等待，已仅停止本队列并保留产物。通过独立 `operations/resume_mesh.sh` 调度入口在GPU2续跑，PID43915，当前阶段 `mesh_geometry_oracle_visible`，心跳/进程身份匹配。只调用原mesh与summary模块，不重训、不重采SS，原运行身份不变。该操作入口允许共享但要求启动前可用显存≥18GiB；实际12:07启动时GPU2已空闲（仅4MiB已用、24123MiB可用）。监视命令保持原样，原stop也可识别新队列。变更收据保存在/data实验输出的 `operations/gpu_resume_1793016421602036580.json`。

11:27:36 UTC 已启动完整后台队列 PID 4173995；11:28:10 复核进程身份匹配、心跳正常，选中物理 GPU 0，正在 `smoke`。正式训练尚未开始，GPU 短检查尚未全部通过。

本轮只新增 `geometry_oracle_visible`：同图像、同参数量/初值、同200步预算，用 GT 可见首表面深度及有效性置信度替代预测几何；遮挡后方仍未知。与已完成的外观组、预测几何组比较，检验观测质量是否限制当前几何注入。旧两臂24结果复用，新生成12个SS/mesh。该oracle是特权诊断，不是RGB方法成绩；它同时改变深度和置信度。

CPU验收已完成：455项存储映射；4对象原输入/latent/支持坐标；24旧参考；四帧剥离首表面缓存；12通道证据；无效深度及插值unknown、隐藏数据读取边界、stool仅两帧等。CUDA尚未在CPU阶段初始化。bench的oracle证据已有非零surface，但仍极稀疏；这只说明输入生成符合预期，不能提前判断mesh改进。

运行身份：`c05d02da649f96f5ad75d01acc53c20cc1606e78720d64472b991fc17cb2f4b2`。

GPU短检查将验证实际零注入/梯度隔离、独立两步时序以及table seed42完整SS/mesh父模型重放。全部通过才自动执行从零初始化的200步训练、12SS、12mesh、12对比图及9场景GLB；失败即停，不绕过检查，不从smoke权重开始正式训练。长任务由用户监视。

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_visible_surface_oracle_20261026/run.sh watch
```

所有新权重/输出/临时文件/模型及Numba缓存位于 `/data/zjr/pose_aligned_layered_reconstruction/`，实验输出为其 `outputs/hssd_visible_surface_oracle_20261026/`。迁移新增复制约3.73GB；旧home原件保留，原逻辑路径与SHA不变。后续仅通过/data物理映射读取大payload。

证据：[CPU验收](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_visible_surface_oracle_20261026/preflight.json)、[存储清单](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_visible_surface_oracle_20261026/storage.json)、[运行状态](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_visible_surface_oracle_20261026/status.json)、[协议](PROTOCOL.md)。本交接为启动阶段记录，最终状态以 `status.json` 为准。
