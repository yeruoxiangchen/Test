# 2026-10-26 深度×置信度交叉诊断交接

更新2026-10-27 01:53 UTC：本轮于10月26日14:57完成计算，现已完成结果审阅，24新增SS/mesh＋36旧引用共60个结果完整。独立CPU复核通过：2598项检查、15925项数值或字段比较（含9615项训练字段）、947个文件哈希；24个新mesh的7866229个顶点坐标全量核对，world最大误差2.38418437e−7米。12张对比图已由Codex全部查看，不冒充用户人工验收。没有启动额外GPU任务。

预测深度＋固定1相对旧预测/预测：F@.02增加3.785个百分点，但CD恶化4.09%；椅子贡献净F增益103.45%，去掉椅子后三对象平均变化−0.174个百分点。固定1条件下GT深度相对预测深度在12/12配对上改善CD与F，说明准确深度仍然有价值。桌面新增凸柱和长凳三seed失败均已记录。暂保留原预测/预测为下一留出协议的保守RGB几何主配置，固定1作消融；不重启原LaRI长训。下一步按场景及物体资产划分未见评估，不继续在本四物体上搜阈值或追加预算。本轮无需继续watch或start；以下为历史运行记录。

审阅产物：[中文报告](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_depth_confidence_20261026/review_20261026/REPORT.md)、[独立统计](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_depth_confidence_20261026/review_20261026/analysis.json)、[坐标及产物核验](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_depth_confidence_20261026/review_20261026/verification.json)、[逐图记录](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_depth_confidence_20261026/review_20261026/visual_review.json)、[审阅回执](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_depth_confidence_20261026/review_20261026/review.json)。原始summary/status保持不变，仍保留review_required的队列历史；本独立回执记录已完成的审阅。

更新14:24 UTC：14:22:18完整GPU短检查通过。两臂零注入与父模型逐值相同，冻结参数无梯度且版本不变，新投影梯度有限且非零，参数量与初始化一致；各2步训练的对象/噪声/时间/学习率与旧对照匹配，原table的SS和mesh完整重放一致。此前8项旧几何对角逐值重放也通过。14:24:07主机复核队列PID246470身份匹配、心跳2秒，第一臂正式训练62/200，第二臂尚未开始。正式训练阶段重新选择空闲GPU0，GPU2上的短检查已经结束；这是按阶段择卡，没有中途迁移训练。

接下来队列自动完成两组各200步、24个新增SS与mesh及最终汇总。此处“检查通过”只证明执行路径、初始化和对照一致性，尚不代表方法质量改善。最新运行身份为 `d491eb3df35c80e74853cdec49839e223c7a62659bf814cf0f9ee86ada9b9052`。用户只需运行下方watch命令。[GPU短检查回执](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_depth_confidence_20261026/smoke.json)。以下保留启动与修复的历史记录。

更新14:19 UTC：首次队列于13:06:23在第一臂初始化哈希阶段失败：new common从独立加载的prior common直接别名tensor_hash，函数内相对导入没有package上下文。没有正式训练、SS、mesh或成功GPU检查产物。旧源、身份、CPU验收、状态及失败日志已保留于/data输出的 `repairs/hash_import_20261026/`。修复仅用绝对package名解析同一旧tensor_hash实现；21个不同dtype/shape/非连续/带梯度张量哈希与原实现完全一致，CPU未初始化CUDA。新旧runtime只common.py的导入调用变化，plan、模型/采样/输入合同不变。

14:18:19重新启动完整队列PID246470，14:19:02主机复核running/smoke、GPU2（空闲卡准入）、身份匹配、心跳1秒。GPU短检查仍待全部通过，两臂正式训练尚为0/200。新运行身份为 `d491eb3df35c80e74853cdec49839e223c7a62659bf814cf0f9ee86ada9b9052`，以下a12开头的身份属于归档的失败版本。原watch命令保持不变，无需用户再次start。

13:05:16 UTC 已由助手启动完整后台队列，PID133330。13:05:58在主机进程空间复核：进程身份匹配、心跳2秒，GPU0，当前阶段smoke。该阶段启动时GPU0空闲、可用显存24126MiB。GPU短检查尚未全部通过，正式训练仍为两组各0/200；不能把CPU验收写成完整GPU验收。

两新组：

- geometry_gt_depth_pred_confidence：GT可见首表面深度＋旧预测置信度。
- geometry_pred_depth_unit_confidence：旧预测深度＋有效观测权重1。

两旧对角（预测深度/预测置信度、GT深度/权重1）以及外观组直接复用。主2×2对照48结果，加外观背景12，共60；新增24个SS/mesh。保持原四物体、三个采样seed、每组200步、同参数量/初始化/输入target/顺序/噪声，以及冻结SLat。长凳、所有失败和负面指标均保留。

已完成CPU实查：原输入与latent/支持坐标、36旧参考、几何语义/字段读取边界、无效插值unknown、stool仅两帧、统计交互逻辑。新实现重算两旧对角各4物体，共8项，features/weights/valid全部torch.equal；随后生成两新组8项几何输入缓存。预测深度＋权重1路径不会读取GT观测/valid。没有重新运行VGGT或mesh→latent编码。

复用既有455项/data映射，仅补拷12个预测depth/rank与旧几何缓存文件，共22543858字节；原home文件保留，训练/输入/模型大payload继续从/data读取。缓存包括Torch/HF/扩展/Matplotlib/Numba均在/data。

首次失败版本的归档身份：`a12d02bc56b6a34e58193c67aa04d9623f30162f3019cd4edbe27bfee39623e7`。

队列约定先完成两臂实际零注入/梯度隔离、各2步时序匹配、旧table完整SS/mesh重放；这些检查现已于14:22:18全部通过。两臂独立从零初始化训练各200步，再完成24SS、24mesh、12张六行对比图和15个已知摆放GLB。任一后续检查失败即停，不从smoke checkpoint开始正式训练。

单卡调度每阶段重新择卡，优先空闲卡；没有空闲卡时允许至少18GiB可用显存的共享卡，取余量最大者。只读取驱动状态，没有CUDA分配探测；不会切换正在运行的阶段或停止他人任务。

用户只需监视，无需再start：

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_depth_confidence_20261026/run.sh watch
```

输出目录：`/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_depth_confidence_20261026/`。

证据：[CPU验收](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_depth_confidence_20261026/preflight.json)、[8项对角逐值重放](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_depth_confidence_20261026/evidence/diagonal_check.json)、[实际状态](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_depth_confidence_20261026/status.json)、[冻结协议](PROTOCOL.md)。本交接为启动记录，最终状态以status.json为准。

本轮是最后一轮预定四旧物体机制定位，仍不是泛化证据。完成后应转到场景/对象隔离的留出评估；没有自动接着调tau、扩展训练预算或恢复原LaRI长训。
