# HSSD Generalization192 数据准备

2026-10-28研究决策更新：用户将推进标准改为完整方案相对base的整体收益；当前数据准备继续，后续优先base与完整方案主实验，同容量外观/置信度消融改为可选。详见[新决策](RESEARCH_DECISION_20261028.md)，覆盖下文及旧协议中的三分支必跑优先级，不改动当前冻结数据队列。

下一步从四旧物体机制定位转到独立场景、独立目标资产的首批数据集。固定HSSD-hab版本，官方125 train/43 val边界不混用；选16训练场景、4验证场景、4本项目测试场景，每场景最终8个目标，预算128/32/32。

2026-10-27 11:18 UTC：第8场景渲染因Blender CUDA非法内存访问退出，助手已用原冻结代码从断点恢复，PID2204262、GPU1。原失败frame44（第45张）已通过并进入后续帧，前7场景及既有照片复用。仍为数据准备阶段，未开始新模型训练；用户直接watch即可，无需再次start。详见[过程交接与错误说明](HANDOFF.md)。

已完成元数据调查与候选资格筛选：24场景、627候选，目标编码前再按固定照片覆盖规则选满192。禁止在latent或生成结果出来后换对象、删失败。目标家族及相同GLB字节若在另一划分的背景中出现，也排除为目标；旧三个开发场景及其目标家族排除。原始GLB依赖约1.92GiB，图片、特征、latent和临时产物预计显著更大。

数据根目录：`/data/zjr/pose_aligned_layered_reconstruction/datasets/hssd_generalization_v1/`。

运行状态/日志：`/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_generalization_20261027/`。

CPU验收23项通过；实际Blender镜像/非均匀缩放、地面射线和相机投影检查通过。SS编码器与DINO按旧审计hash迁到/data，156个DINO源码一致，home原件保留。本轮实际GPU渲染与新物体codec检查由后台队列继续完成，不能把CPU检查写成完整GPU验收。

完整准备队列：元数据核验→候选计划→编码器存储→下载原始依赖→只改纹理的GLB转换→canonical mesh→每场景共享照片及可见mask→固定192目标→train/val/test各一个完整codec短检查→其余目标150视图/SS/SLat编码及回环→192张原始/解码mesh三视角图→数据报告。每阶段可恢复，失败停止，不直接启动正式训练。

启动一次：

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_generalization_20261027/run.sh start-prepare
```

监视（任意当前目录）：

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_generalization_20261027/run.sh watch
```

单次状态：同脚本`status`；停止本队列：`stop`。Ctrl-C退出watch不停止后台队列。每个GPU阶段优先选择空卡，无空卡时要求至少18GiB可用；仅查询驱动，不分配显存探测或抢占他人任务。可在首次启动或停止后恢复时设置`HSSD_GENERALIZATION_GPU=3`等固定卡；默认auto。

完成状态`data_qualification_complete_pending_review`表示数据准备和编码完成，等待质量审阅。后续正式训练预算已经预登记为每分支50epochs/6400次对象暴露，但需先验收新数据、实际遮挡分布和条件缓存；本脚本没有自动训练阶段。详见[冻结协议](PROTOCOL.md)。
