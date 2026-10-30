# HSSD192 TRELLIS.2 关系模型主对照

2026-10-29：在已完成的Native12/MVBase12基础上，准备192个主要目标与邻居的原生标签，训练同预算的独立多视图base和关系候选。旧实验代码和结果未修改。长队列由用户启动；当前完成度与短检查结果见下方记录，不把实现完成写成论文收益。

## 长任务启动与监视

全部主线（短检查回执复用→24场景输入分组→原生标签及条件→数据检查→两臂SS/Shape训练→train诊断/val完整mesh评估）：

```bash
T2_REL_GPUS=6 bash /home/zjr/Tracker/pose_constrained_trellis2/experiments/hssd_relations192_20261029/run.sh start
```

另一个终端监视，任意工作目录可运行：

```bash
bash /home/zjr/Tracker/pose_constrained_trellis2/experiments/hssd_relations192_20261029/run.sh watch
```

`Ctrl-C`只退出监视。终态`all_complete_review_required`表示默认主线已结束，读取输出`REPORT.md`、`summary.json`与`figures/`；测试集不会由此命令自动评估。共享GPU默认启动需空余18GiB、PyTorch分配上限16GiB，不预分配、不终止其他任务；环境变量可指定其他候选卡，多个编号表示任选一张，始终单卡串行。

任务失败时看watch显示的具体错误；源码未改动、异常原因已解决后重复对应start命令会复用完整阶段和最新训练检查点，不需要删除整个输出目录。需要停止时：

```bash
bash /home/zjr/Tracker/pose_constrained_trellis2/experiments/hssd_relations192_20261029/run.sh stop
```

要分阶段运行，可依次使用`start-data`、`start-train`、`start-eval`。`start-data`结束于`data_complete_review_required`，检查`data_summary.json`和`codec_quality.json`；`start-train`结束于`training_complete_pending_evaluation`。源码、协议或数据身份改变会拒绝静默续跑。`start-test`只在方法固定后显式评估保留的32个test主要目标。

## 本轮结构

```text
同一场景的共同RGB帧和实例mask
  → 各物体crop/K/DINOv3 + 已知GT相机/物体变换
  → 每物体独立SS/Shape状态，共享TRELLIS.2权重
      independent：已有多视图pose条件 + LoRA
      relational ：额外读取邻居同一时刻的状态、图像观测和相对位置
  → 全组同步提交每一步采样更新
  → 独立canonical/world mesh + 共同相机可见所有权检查
```

关系模块位于[scene_relations/model.py](../../scene_relations/model.py)对应的源目录（从仓库根为`pose_constrained_trellis2/scene_relations/model.py`）：每邻居最多64个池化token，128维4头attention，五个DiT block零初始化注入。训练输入是邻居当前带噪状态，推断输入是邻居当前预测状态；没有输入干净GT邻居latent、GT深度顺序或完整隐藏面。共享坐标仍依赖GT变换，是受控实验。当前没有LaRI深度预测器、VGGT前端、联合渲染loss或自动位姿优化。

## 数据与预算

- 原始主要目标保持192：train128／val32／test32，来自16／4／4个场景。
- 每主要目标最多一个邻居；按原目标输入帧的可见mask选组，至少两个相机位置。无合格邻居则保留单物体组，分母不变。
- 冻结候选623个（原627个候选中排除跨split资产家族/几何重复者）；实际需编码物体数量在24场景选组后确定，可能多于192。邻居不会另算成主要验证对象。
- 另保留此前12对开发候选及7个缺标签邻居。可复用12个Native12标签目录，全部只读；新的组内共享帧条件会另存。
- 两臂从官方权重初始化，SS和Shape各50个组epoch，即128组×50=6400步，共四个正式训练任务。两臂组序、视图、时间和噪声种子相同，记录实际对象暴露；固定最终raw权重。
- 默认评估8个预先按哈希选出的训练诊断目标和全部32个val目标，2个采样seed×2臂，共160个主要目标case；同时输出同组邻居mesh。每case包含完整12＋12步采样。
- 全部32个test目标保持独立显式阶段，不用它们调参。

Shape token支持完整32³格（最多32768），不再把官方训练数据的8192筛选门槛作为原生编码有效性条件。明显codec误差、非有限值、非法/重复坐标、坐标投影不符仍会停止并保留失败，不截token、不替换目标。具体数据检查阈值、失败记分、可见所有权accuracy的局限见[冻结协议](PROTOCOL.md)。这些是标签正确性检查，不是新增模块必须在小样本上获胜的门槛。

2026-10-29修复验收与续跑见[TOKEN_SUPPORT_REPAIR_20261029.md](TOKEN_SUPPORT_REPAIR_20261029.md)：10457-token柜子完整codec、共视条件和两臂各2步训练已通过；DINO.to调用同时修正。完整all队列PID3501851于05:54:37确认在GPU5正常运行，已完成520阶段，历史产物和协议记录保留。

## 存储与完成状态

新数据：`/data/zjr/pose_constrained_trellis2/datasets/hssd_relations_native_v1`。

新输出：`/data/zjr/pose_constrained_trellis2/outputs/hssd_relations192_20261029`。

所有临时、模型、HF、Torch、Triton和绘图缓存使用该项目/data目录；源码和小文档留在Tracker。每次GPU阶段记录`memory/`峰值，stage回执绑定源码、计划与命令。模型保存`last.pt`用于恢复、`final.pt`为最终文件的硬链接，不每200步保留一个大型历史副本。

## 实现验收记录

2026-10-30 00:01 UTC阶段性核查：全部289物体数据检查通过；独立SS已完成6400步，独立Shape约4491/6400，关系训练和最终mesh评估尚未开始。详见[阶段性分析](INTERIM_REVIEW_20261030.md)。队列继续运行，无需重启。

2026-10-29：7项CPU行为检查已通过：零初始化保持base且梯度有效、邻居状态/位置生效、共同世界变换不变性、邻居顺序不变性、同步更新语义、推断不读目标标签、摘要来自当前状态且detach。192主要目标划分和623合格候选已冻结；一个真实训练场景`104348133_171513054`的输入mask分组完成，8/8主要目标得到双物体组。该小型CPU检查输出可由正式队列按同一规则复核。

GPU preflight仅使用已编码训练对象，不使用val对象训练；包含最大6800-token物体和同场景另一训练物体。两臂SS/Shape各2步已完成，关系分支梯度非零。完整两物体12/12步采样仍以preflight回执为准；长192编码、6400步正式训练和主评估尚未启动。之后的实际验收结果追加于此。

00:57 UTC验收：preflight已于00:44:36结束6/6；两臂各两物体的完整12+12采样均成功，共4个非空mesh。真实GPU第一步SS/Shape loss在两臂逐值一致（0.089883808／0.597359449）；第二步关系梯度范数SS=0.0298432、Shape=0.000336773，确认关系分支参与了优化。最大PyTorch分配5626.6MiB、保留5742MiB；所有latent有限且符合原生支持/通道约定。旧MVBase12源码身份复核一致。本轮新身份`10ac94b9021541c3faf330a5067548c79984d45df830c31b54c7e61e40deb3c0`。见[短检查验收记录](/data/zjr/pose_constrained_trellis2/outputs/hssd_relations192_20261029/smoke/verification.json)。这不是关系质量收益或192目标训练结果；长主线尚未启动，用户执行上方start即可复用全部6个完成阶段。

00:59 UTC补充：新数据队列的共享相机投影诊断也已用两个已编码物体、两个共同帧实际执行，4次可见mask→完整GT投影命中率为0.9900–0.9992。该val对象对仅用于坐标诊断，没有被训练。记录：[projection_probe.json](/data/zjr/pose_constrained_trellis2/outputs/hssd_relations192_20261029/smoke/projection_probe.json)。有界检查进程均已结束。

用户随后授权由助手启动长任务：完整all队列PID2874639已启动。01:38:56 UTC实查进程身份匹配、心跳刷新，GPU6上正在`hssd_102816036_i605_dual_1024`，已完成113阶段。24场景选组已结束，含邻居共需289个物体标签，原主要目标仍为192；当时Shape512编码25个、codec24个（含只读复用）。本快照之后以watch为准。当前无需重复start，用户只需监视；正式训练和模型效果尚待后续阶段。

2026-10-29 03:53 UTC换卡记录：此前按用户要求从GPU6迁至GPU3（PID2963114）。GPU3随后被其他训练和vLLM进程占用，空余约2020MiB，队列在第370个已完成阶段之后等待`hssd_102344280_i070_encode_512`，此时编码512和codec均67/289。用户明确要求切GPU4后，停止自有等待队列，保留全部回执，以`T2_REL_GPUS=4`续跑完整all队列，新PID3234365。03:53:42状态确认`running`、GPU4、上述encode_512阶段，子进程3234495；源码身份与18GiB启动门槛、16GiB分配上限不变。未改动其他任务，监视命令不变；后续以watch为准。

2026-10-29 04:57 UTC换卡记录：GPU4再次低于启动门槛时，已完成465阶段、83个codec。用户要求从screenshot_code_agent workflow将释放的四卡中选一张；只读确认其使用2/3/5/6，曾将自有队列切到2（PID3333436）。用户随后提醒GPU2可能已被其他任务选定，因此改为固定GPU5。切换期间又完成若干阶段，保留471阶段回执；新队列PID3374940于04:57:17确认`running`，GPU5上执行`hssd_103997403_171030405_i036_dual_512`，子进程3375068。未停止或修改screenshot_code_agent及其他任务。源码身份、训练协议和显存门槛不变，所有已有数据保留，监视命令不变。

2026-10-29 07:16 UTC换卡记录：用户报告GPU5再次等待、GPU4空闲，实查GPU4空余24126MiB。停止自有等待队列PID3501851后，以`T2_REL_GPUS=4`恢复all，新PID3675596。07:16:49宿主机确认进程身份匹配、心跳刷新、GPU4正在`hssd_104348463_171513588_i029_dual_1024`，阶段已从645推进到665，Shape512编码116、codec115（含复用）。源码身份、实验协议和显存门槛不变，未改动其他任务；监视命令不变。
