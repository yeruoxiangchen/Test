# HSSD TRELLIS.2 Native12

**2026-10-28已完成并审阅：90/90阶段、12/12原生标签、SS/Shape短训恢复和完整采样均结束。标签支持下一阶段训练；三步生成尚未恢复完整椅子，不是关系模型效果。见[结果审阅与真实网格图](REVIEW_20261028.md)。无需继续watch或再次start。**

2026-10-28 实施场景协同重建方案的第一个执行阶段。3场景、12物体（8 train / 4 val），从已有完整mesh生成T2原生监督，验证坐标、DINOv3条件、实际GPU短训/恢复及完整采样。

```bash
bash /home/zjr/Tracker/pose_constrained_trellis2/experiments/hssd_scene_20261028/run.sh watch
```

已启动后只需监视。`Ctrl-C`退出watch不会停止队列。失败后定位日志；若代码未改，可用相同脚本的`start`续跑，完成阶段保留。完成状态为`native_complete_review_required`，无需再次启动。

所有产物：`/data/zjr/pose_constrained_trellis2/`。模型大权重优先只读引用/data既有文件，补充DINO和SS encoder只复制一次并校验；peft/accelerate安装到项目packages，未改共享Conda环境。TMP、HF、Torch、Triton、CUDA、FlexGEMM缓存均在/data。

90阶段包括12物体各7阶段、manifest汇总、SS/Shape各2步与恢复第3步、12/12步完整采样。原生编码使用官方mesh→dual-grid→Shape VAE；SS监督取Shape1024原生支持。没有重新生成150张目标训练图。

本轮不是关系模型的正式训练，也不代表T2迁移已全部验收：

- 只使用已知相机、已知物体canonical/world变换的受控协议；RGB输入的自动定位仍待实现。
- 解码指标是mesh→latent标签保真度；3步训练只检查执行，不证明拟合/泛化。
- 输出`objects/<uid>/codec.png`为GT/解码表面点采样的同相机三视角图，并同时保存真正canonical/world mesh。正式方法评估仍需完整mesh渲染与场景图。
- `selection.json`保存同场景对象对及选定帧共视情况。部分对象距离较远、共视为空；本轮适合codec覆盖，不直接宣称合格的遮挡关系训练集。后续从候选实例和共享帧构建更紧密的对象对，另行固定新清单。
- 未自动启动旧30K或新200步正式训练。完成后审阅原生标签，再进入多视图base及关系分支；以完整方案对base的总体结果判断。

## 已完成的实际检查和修复

1. peft 0.18.1、accelerate 1.12.0及现有torch 2.6.0+cu124、o_voxel、cumesh、flex_gemm实际导入；主机CUDA可用。
2. 源码语法及bash语法检查通过。
3. `operations/preflight.py`在真实12物体和其输入相机上独立重算world与裁剪投影。发现旧crop helper漏掉Pillow像素中心平移，在本实验`images.py`修正，不修改旧模型运行源。修正后全部通过：world往返最大3.56e-15；相机原图投影最大5.92e-11px；裁剪投影最大0.000599px。该数值修复不构成旧模型效益失败的归因证据。
4. O-Voxel的CPU转换入口也导入Triton，隐藏CUDA会报`0 active drivers`；已让该阶段也等待空卡，不改变CPU转换数学定义。首轮失败和随后crop修正前的身份/阶段回执归档于输出`repairs/`。
5. 首个物体已有真实GPU编码产物：Shape512为681个32通道token，Shape1024为3067个32通道token。完整队列仍在运行，未提前声明12对象解码或训练检查通过。

2026-10-28 10:13 UTC 当前修订队列PID1047476，GPU7，runtime identity `6e78174f12bad7faed4ba5a62ace04d2543d89aa20f90100136ece05f2430fa5`。这是当时快照，实时状态以watch为准。

10:14首对象已完成原生解码与指标：294469顶点、653928面，SS roundtrip IoU=1.0，CD-L1=0.00301546（规范空间）、F@.01=0.999400、F@.02=1.0。随后检查图因缺少matplotlib停止；补齐`requirements-runtime.txt`所列绘图依赖到项目packages后续跑，保留已完成5阶段，未改源码/身份/对象清单。恢复队列PID1066027。该依赖补齐没有改变任何几何计算，也未修改共享环境。

复建环境时，先在项目/data下配置TMPDIR、PIP_CACHE_DIR，再使用trellis2 Python按`requirements-runtime.txt`安装到`/data/zjr/pose_constrained_trellis2/packages`（`--no-deps --target`，保留已有torch/transformers）；`run.sh setup`提供模型路径准备。pip锁版本和环境检查不能代替实际GPU执行。

10:25 GPU7已被其他任务占用，队列处于等待状态。仅重启该等待进程，改为允许0、6、7中任意满足原空卡条件的GPU，PID1071674；没有打断计算或改动数值协议。记录见输出`gpu_wait_change.json`。后续手动续跑若希望沿用三卡候选，可使用 `T2_SCENE_GPUS=0,6,7 bash /home/zjr/Tracker/pose_constrained_trellis2/experiments/hssd_scene_20261028/run.sh start`，日常只运行watch。

## 2026-10-28 10:33 UTC：按用户要求停止旧队列，新队列固定GPU0

用户明确要求停止旧任务并启动新任务。旧HSSD192通过其原生stop命令停止，主/子进程退出、状态stopped，保留66/192旧编码及全部照片/mesh。未删除任何原始数据或旧完成结果。

新队列此前在pose阶段暴露Transformers5.9的DINOv3接口变化：层列表移至`model.model.layer`，原提取器访问`model.layer`失败。新增实验内`dino.py`，用具名RoPE参数，保持TRELLIS.2原有prenorm token后functional LayerNorm（没有改用带可学习仿射的最终norm）。实际512输入GPU检查与安装版encoder参考路径逐值一致，1029×1024、有限，最大误差0；记录`dino_compatibility.json`。

`operations/repair_dino.py`验证只改动worker的pose函数，六个已完成codec阶段函数AST完全相同；逐文件验证已有产物并归档旧身份/回执，明确迁移到新身份`272cfe8dbc42e077eeb57c19bdc354e85f6ee458b7a7829df93c60cade098c64`。没有改换对象、标签、decoder或重做完整编码。

新队列使用`T2_SCENE_GPUS=0`启动，PID1089497。10:33:24核验running/第二对象encode512、GPU0、8/90，进程身份匹配。首个对象的原生codec、检查图和DINO条件均完成。启动脚本的提示文字仍固定写“默认GPU7”，实际选卡以环境变量和status中的GPU0为准；未为提示文字修改已冻结运行代码。日常仍只需上方watch命令。

方案：[总体设计](../../SCENE_RELATIONS_PLAN_20261028.md)。执行范围：[冻结协议](PROTOCOL.md)。旧HSSD192起初保持运行，随后已于10:33按用户明确要求停止，保留66/192旧编码和全部原始资产。
