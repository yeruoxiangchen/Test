# SharedScene4：共享 RGB 轨迹的冻结模型验证

本轮用于把输入方式改成“一次拍摄同一场景，再逐个生成其中的物体”。当前 LaRI 不进入推理队列，不启动训练。先确认共享相机、遮挡输入和世界坐标输出能工作，再决定新几何模块的实验。

实验输出：`../../outputs/hssd_shared_scene_20261025/`。完整协议见 [PROTOCOL.md](PROTOCOL.md)。

2026-10-25 更新：用户在 GPU1 完成全部 24 个 mesh，09:36 UTC 队列正常结束。CPU 复算、输入/坐标检查与全部图库审阅已完成，见[结果分析](../../outputs/hssd_shared_scene_20261025/review_20261025/REPORT.md)。外观对照优于父模型，但两者共享场景重建质量不足；下一步先诊断实例裁剪、输入分辨率和空视图，不追加 LaRI 长训练。

## 数据与范围

- HSSD 场景 `102344049` 的餐桌区域，固定四个既有目标：椅子 i076、桌子 i075、长凳 i080、凳子 i065。
- 一条公共相机轨迹共 12 张 RGB，四个目标全部使用相同的第 0、4、7、11 张照片。没有为每个目标另找好视角；完全不可见的单个视图也保留。
- 复用此前已经验收的 mesh、SS latent 和 SLat latent，不重新做 mesh→latent。保存 SHA 绑定。
- 本轮提供渲染真值相机、可见实例掩码和 GT 推导的 canonical frame。它是输入接口与冻结模型的诊断，不是纯 RGB 自动定位系统，也不是独立场景泛化实验；目标曾参与 HSSD 拟合。

## 启动与监视

脚本会自行切换到项目根目录，从任意当前目录执行均可。GPU 3 是默认值，修改环境变量可显式换卡。后台启动会检查指定卡是否空闲，单卡顺序执行，无自动多进程抢卡。

首次交付时，12 张图像与坐标核对已完成，DINO 缓存曾因磁盘读取等待中断；用户现已续跑完成。以下保留原启动命令用于复现，本轮无需重新启动（已完成产物会跳过）：

```bash
HSSD_SHARED_GPU=3 bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_shared_scene_20261025/run.sh start
```

监视：

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_shared_scene_20261025/run.sh watch
```

只看一次状态，将 `watch` 改为 `status`。退出监视用 Ctrl-C，后台任务继续。

完整基线总量为 **2 种模型 × 4 个物体 × 3 个采样 seed = 24 个 mesh**：

- `yxc30k`：原有 SS30K 父模型，关闭新增层分支。
- `appearance_control`：已完成的 HSSD 外观对照检查点。

两者共用冻结 SLat30K、相同视角、相同采样参数、相同每坐标噪声和已审阅的固定 spconv native 内核规则。两个 SS 模型训练经历不同，这个比较用于测量输入改变后的基线表现，不是证明新方法有效的容量匹配消融。

`start` 执行数据→基线全部步骤；已有完整 `manifest.json` 后可用 `start-baseline` 只跑推理。只准备数据用 `start-data`。单个完整 SS→SLat→mesh 检查用 `start-smoke`，只执行第一个父模型案例，成功结果可被后续 `start-baseline` 复用。当前全部 24 个完整采样案例均已执行成功；这表示链路完成，几何质量需按审阅报告解释。

失败时先看状态中的具体日志；修复环境后重新执行同一命令会核对身份并跳过已完成结果。采样开始后会冻结 `evaluation_identity.json`，源码或协议变化会拒绝混跑。不要为了继续而删除身份或检查失败的目标；实验设计改变应另开版本。

## 输出怎样看

- `shared_input_contact_sheet.jpg`：第一行是公共原图，其余四行是每个实例的可见区域。
- `geometry_audit.json`：Blender 掩码与独立光栅化的投影一致性、遮挡比例与近处几何检查。
- `evaluations/<arm>/<uid>/<seed>/`：SS、canonical mesh、world mesh、指标和数据/权重身份。
- `figures/`：12 张同相机对比图，每张包括 GT、父模型、外观对照的三个固定观察方向。
- `world/`：6 个组合 GLB，每个含同一场景的四个独立生成物体；位置由已知 canonical→world 变换给定，不是模型预测的布局。
- `summary.json` / `REPORT.md`：完整结果、可见面/外部遮挡面/其他未观测面的距离与召回率，以及预测表面侵入观测自由空间的比例。

`all_complete_review_required`（使用 `start`）或 `baseline_complete_review_required`（使用 `start-baseline`）表示 24 个 mesh、图库和汇总完成，等待审阅；`stage=None` 表示没有正在执行的阶段。3 个采样 seed 不等于 3 次独立训练。

## GENA3D 接入状态

已固定官方源码、Pi3 源码并实现输入导出与独立官方示例检查入口。包发现检查完成；实际类导入因 Open3D→sklearn→pandas 的磁盘读取等待而人工中断，故尚未验证运行时兼容性，模型权重未下载。真实 HSSD 对照仍缺**预测的 2D amodal completion 与 occlusion mask**，以及其归一化坐标到本项目坐标的明确转换。本轮不把 GT 完整图、GT amodal mask 或直接复制原图充作预测补全。

`gena_inputs/` 中目前只导出真实遮挡 RGB 与可见目标掩码，`gena_input_manifest.json` 明确列出每个视图缺的两项文件。官方推荐的 OAAD 是独立前端，尚未完成安装和验证。当前 24 个 mesh 队列不依赖它。

单独查看环境审计和实际类导入检查：

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_shared_scene_20261025/run.sh external-status
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_shared_scene_20261025/run.sh external-check
```

较长的权重下载与官方示例推理由用户另行启动，顺序执行并等待前一阶段完成：

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_shared_scene_20261025/run.sh start-gena-download
HSSD_SHARED_GPU=3 bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_shared_scene_20261025/run.sh start-gena-smoke
```

下载器先查询文件体积与 revision；加上 SS encoder/u2net 余量超过 10 GB 时放到 `/data/zjr/scene_shared_baselines_20261025`，否则放本实验输出目录。官方示例成功只代表兼容性验证，不能报为 HSSD 实验成绩。下载/示例队列与基线队列使用同一互斥锁，按次序运行。

来源：[GENA3D 官方代码](https://github.com/Colezwhy/GENAmodal3D)、[Pi3](https://github.com/yyfz/Pi3)、[OAAD](https://github.com/saraao/amodal)。
