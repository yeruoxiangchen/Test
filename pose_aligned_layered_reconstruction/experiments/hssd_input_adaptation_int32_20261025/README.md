# InputAdapt4 大张量解码修复

**2026-10-26更新：修复队列已于00:04 UTC完成，新增mesh54/54。大长凳精确回归通过，105283点超限项成功解码且保留全部邻接。完整效果分析见[完成审阅](../../outputs/hssd_input_adaptation_20261025/review_20261026/REPORT.md)。以下保留修复实现和启动时的交接说明。**

原实验 SS=54/54、mesh=50/54。剩余长凳 seed44 的输入特征超过 Native
GEMM 的 int32 字节寻址范围。本目录是单独记录的运行修订，不改原冻结源码、
已完成 mesh、SS 支持点、模型、精度、解码分辨率或评估坐标系。

按输入行分块执行问题卷积，保留全部邻接，保持原 center mm 和各 kernel 的累加顺序。
普通尺寸沿用原 FixedNative。仅限本实验审计过的 CUDA FP16 原生推理路径，不能用于训练反传。
详见 [PROTOCOL.md](PROTOCOL.md) 和 [实现](tiled_native.py)。

## 已完成的短验证

- 3125 点、随机缺口和乱序坐标，bias 开/关、257/1024 两种行块：四组输出逐值完全一致。
- 已保存的真实椅子 SLat（5055 SS 点）：目标卷积输入323520行，强制65536行分块。
  6983178 条非中心邻接全部处理；卷积特征/坐标、原始 mesh 顶点/面/属性和旧归档 mesh 精确一致。

这些验证支持继续大输入回归，不能提前宣告105283点超限样本已完成。
运行记录位于原输出目录的 `int32_fix_20261025/`。

## 长队列启动和监视

默认 GPU4；设置 `HSSD_INPUT_GPU` 可显式换卡。启动只查询显存占用，不分配显存做探测。
长任务交用户启动和监视，从任何目录运行：

```bash
HSSD_INPUT_GPU=4 bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_input_adaptation_int32_20261025/run.sh start
```

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_input_adaptation_int32_20261025/run.sh watch
```

流程：核验短回归 → 85267点长凳seed42的原路径/分块路径及旧mesh精确回归 →
补齐长凳seed44和凳子三个seed → 原指标、24张对比图和24个场景GLB汇总。
大样本回归有任何数值差异都会停止，不能将变化当作模型改进。

重启仍用 `start`，已有结果校验复用；采样的SLat在解码前保存，已绑定输入、SS、权重和运行修订。
此队列复用原实验排他锁，避免和旧队列同时运行。不要再用旧 `run.sh start` 重试本次int32故障。
`status` 查看一次；`watch` 的 Ctrl+C 仅结束监视；`stop` 仅停止本修复队列及其子进程。

最终状态应为 `all_complete_review_required`、mesh=54/54、逻辑结果=96/96。
此时原输出目录 `REPORT.md`、`summary.json`、`figures/`、`world/` 已完成。
运行修订、数值门槛和新增4份结果绑定都将写入汇总。
长凳过度预测的105283个SS点不会被裁掉，也不会从平均结果中排除。

## 回归入口

`check-kernel` 和 `check-small` 是独立GPU短验证；`audit` 是短验证完成后的CPU哈希核验。
`preflight` 依次核验/运行两项短验证。`check-large` 是长凳大输入回归，不属于短检查。
`summary` 仅在四个修复结果均完整时运行汇总。冻结 `runtime_amendment.json` 后不要编辑本目录
Python、run.sh 或 PROTOCOL.md；README 和时间记录可继续补充。
