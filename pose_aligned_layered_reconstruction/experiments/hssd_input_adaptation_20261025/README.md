# SharedScene4 输入适配对照

**最新状态（2026-10-26）：于00:04 UTC完整结束，SS/mesh均54/54，逻辑96/96，24张图和24个场景GLB齐全；三道精确回归及独立身份/坐标核验通过。已审阅全部24张mesh图：裁剪明显改善椅子，高分辨率收益依赖对象，长凳仍失败。完整分析与下一步12次SS×SLat条件交叉诊断建议见[完成结果审阅](../../outputs/hssd_input_adaptation_20261025/review_20261026/REPORT.md)。下文启动/恢复命令为历史运行入口，当前没有待补mesh；新诊断尚未启动。**

检验冻结模型在共享场景照片上的失败，有多少来自目标过小、空目标视图和原图分辨率。实验协议见 [PROTOCOL.md](PROTOCOL.md)。不训练新模型，不启用 LaRI，不重新编码 mesh→latent。

四个物体、两组冻结权重、三个种子，依次比较整帧、去空视图、518 原图裁剪、2072 原图裁剪。共 54 个新增 SS→SLat→mesh 样本；96 个逻辑比较位置另含 24 个旧结果引用和 18 个相同输入别名。不能把它们当作 96 个独立样本。

## 用户启动与监视

从任何工作目录均可运行。默认单卡 GPU 0，需要换卡时修改 `HSSD_INPUT_GPU`；不会自动换卡。启动前读取显存占用，不运行显存分配探测。

```bash
HSSD_INPUT_GPU=0 bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_input_adaptation_20261025/run.sh start
```

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_input_adaptation_20261025/run.sh watch
```

只查看一次用 `status`。`watch` 的 Ctrl+C 仅退出监视。停止本实验队列用 `stop`；它检查 PID 的命令归属后只结束本实验进程。继续运行仍用同一个 `start`，已完成且哈希一致的结果会复用。

队列顺序：数据/裁剪检查 → 补齐四帧高分辨率渲染 → 独立投影核验 → DINO 缓存及原特征重放 → 旧基线完整重放 + 两视图短检查 → 两模型 SS → mesh → 指标、对比图及场景 GLB 汇总。

最终状态：`all_complete_review_required`，新增 SS=54/54、mesh=54/54、逻辑结果=96/96。此时计算完成，仍需查看 mesh 图后决定下一轮研究方向。

输出目录：`/home/zjr/Tracker/pose_aligned_layered_reconstruction/outputs/hssd_input_adaptation_20261025`。

## 已提供的校验与产物

- `preflight.json`：冻结相机、可见视图、ROI 和引用；检查裁剪前后投影及实际目标像素保留。
- `hires_preview_audit.json` / `hires_audit.json`：首帧 / 四帧独立 rasterization 与 Blender mask 的一致性。
- `encoder_identity.json` / `encoder_replay.json`：实际 DINO 权重和源码绑定；原始四视图特征重放，排除编码器变化。
- `smoke.json`：仅在旧 SS 坐标、完整 mesh 顶点/面及指标重放一致，且两视图裁剪样本完整推理成功后生成通过记录。正式队列必须通过它。
- `crop518_input_sheet.jpg` / `crop2072_input_sheet.jpg`：原尺寸与高分辨率裁剪输入。预览阶段高分辨率图库只有第一列，不能把空列理解为目标不可见。
- `summary.json` / `REPORT.md`：全量完成后生成配对指标、逐物体结果及复用说明。
- `figures/`：24 张 GT+四输入变体的三视角对比图，统一 native 相机、材质和尺度。
- `world/`：24 个场景组合 GLB，放置来自固定已知物体坐标系，并非预测的布局。

状态监视仅依赖 Python 标准库，不导入 Torch。模型加载遇到共享磁盘 I/O 延迟时，仍能查看当前阶段和日志。心跳表示调度器存活；进度与日志用于判断子任务是否推进，不能仅凭 running 推断计算有效。

## 单独运行短检查

`prepare`：CPU 数据冻结、六项回归测试及低分辨率输入图。
`preview`：上述检查加单帧 2072 渲染和独立 mask 核验。
`smoke`：原图特征重放，以及两个完整 SS→mesh 样本，不展开 54 样本长队列。
`test`：回归测试；`data`：准备全部输入；`summary`：在样本完整后重新检查并输出汇总。

实验运行期间不要编辑冻结的源码或 `PROTOCOL.md`。代码、输入或权重变化会被检测并停止，不会混入已有结果。`README.md` 和项目时间记录可继续补充实验进展。

## 2026-10-25 12:29 UTC 交接状态

六项回归测试通过；16 个对象视图的 ROI 保留全部原可见目标像素，无人工 padding。首帧 2072 独立 mask IoU 为 0.99878–0.99976。已实际查看原图/高分辨率裁剪预览。

原始四视图 DINO raw/masked 特征均与归档逐值一致，四个 crop518 缓存完成；独立 CPU 检查确认 SS target/coords、SLat binding、相机外参完全沿用，模型真实投影与世界相机加裁剪公式最大差异约 0.0001034 像素。

完整短检查于 12:28:13 UTC 通过：原 chair/yxc30k/seed42 的 SS 坐标、mesh 顶点/面及指标精确重现；crop518/stool/yxc30k/seed42 的两视图 SS→mesh 完成。此新样本计入正式队列，目前新增结果 1/54；长队列将继续其余 53 个。完整队列尚未启动，GPU 0 已释放（交接时查询 1 MiB、0% 利用率）。

原先模型库冷启动受到共享磁盘读取延迟；监视入口已改为不导入 Torch 的标准库脚本。该延迟不是模型不拟合证据。短检查通过也不是输入方案效果已证实，需待 54 个样本及完整图库完成后分析。

证据：[短检查](../../outputs/hssd_input_adaptation_20261025/smoke.json)、[DINO 重放](../../outputs/hssd_input_adaptation_20261025/encoder_replay.json)、[实际缓存与投影](../../outputs/hssd_input_adaptation_20261025/input_cache_audit.json)、[2072 首帧核验](../../outputs/hssd_input_adaptation_20261025/hires_preview_audit.json)。

## 2026-10-25 14:28 UTC CUDA 失败恢复说明

用户正式队列在 SS 54/54、mesh 6/54 时因 mesh decoder 的 CUDA illegal memory access 中止。
失败项为 `full_visible / appearance_control / i065 / seed44`。冻结源码和已保存结果哈希均正常。
GPU 4 的独立新进程、同步 CUDA 重放已通过：旧控制样本精确复现，原失败样本成功生成，未改模型/数据或正式结果。
原根因未确定，不将此事归因于模型不拟合或显卡硬件损坏。

当前建议的续跑命令如下，监视命令不变；长队列留给用户运行。

```bash
CUDA_LAUNCH_BLOCKING=1 HSSD_INPUT_GPU=4 bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_input_adaptation_20261025/run.sh start
```

完整记录：[错误与独立重放报告](../../outputs/hssd_input_adaptation_20261025/recovery_20261025/REPORT.md)。
