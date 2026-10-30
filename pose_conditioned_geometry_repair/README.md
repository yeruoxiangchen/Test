# Pose-Conditioned Geometry Repair

原生 TRELLIS.2 几何修复训练：损坏状态 + 修改区域 + 已标定多视图条件。
冻结官方 SS / Shape 主干，训练条件适配器和 attention LoRA；不加载旧 30K
Flow，不训练纹理。框架和冻结实验规则见 [PROTOCOL.md](PROTOCOL.md)，进度与
核验见 [审计.md](审计.md)。

当前默认队列为 2026-10-30 的残差桥修复实验。第一轮 Shape 过拟合未通过，
原记录保留；`status-v1` 可查看第一轮，`status` 查看当前轮。
当前协议补充见 [BRIDGE_PROTOCOL.md](BRIDGE_PROTOCOL.md)。

代码独立于正在运行的 `pose_constrained_trellis2` 源码；必要的现有核心模块
以带 SHA256 记录的快照放在 `vendor_pose/`。外部运行依赖为 Tracker/TRELLIS2、
现有 trellis2 Python 环境和本地模型/数据。只读复用本地原生 Shape 标签与权重。

```bash
# 后台队列：32 对象过拟合 → 门槛评估 → 2K / Dev64 → 两组正式训练和评估
bash /home/zjr/Tracker/pose_conditioned_geometry_repair/run.sh launch

# 连续监视
watch -n 10 -d 'bash /home/zjr/Tracker/pose_conditioned_geometry_repair/run.sh status'

# 队列服务日志
journalctl --user -fu pose-conditioned-geometry-repair-bridge-v2-20261030

# CPU 验证（不额外安装 pytest）
bash /home/zjr/Tracker/pose_conditioned_geometry_repair/run.sh tests
```

指定允许使用的 GPU，例如仅 GPU 3：

```bash
T2_REPAIR_GPUS=3 bash pose_conditioned_geometry_repair/run.sh launch
```

未指定时队列只考虑 GPU 3、2、5、6、7，按实际空闲显存和利用率等待。不会
终止其他工作。单独调试命令使用 `--gpu 3`；长期任务交给队列管理。

当前输出：`/data/zjr/pose_conditioned_geometry_repair/outputs/bridge_v2_20261030/`。
第一轮保留在 `outputs/round1_20261029/`。
数据清单：`/data/zjr/pose_conditioned_geometry_repair/datasets/native_objaverse_v1/`。
每个评估对象输出 OBJ、原生 latent、原始 GT 与明确标注的 codec/teacher-support
对照。第一轮使用合成损坏和已知 ROI，真实生成错误及自动定位的收益需要另测。

第二轮保持第一轮 SS600 不变，只改变 Shape 的训练/推理轨迹：从损坏状态到
干净状态学习残差。未训练的残差算子严格返回输入；它的监督基线是“不修改”，
同时保留原版局部生成的 mesh 对照。预测 SS 的结果与真正删除缺失坐标的输入
对比；保留 GT 支持的零特征对照另列。通过 32 对象门槛后，队列自动继续 2K
数据、两组正式训练和独立 Dev64。任一门槛失败则保留结果，停止正式训练。
