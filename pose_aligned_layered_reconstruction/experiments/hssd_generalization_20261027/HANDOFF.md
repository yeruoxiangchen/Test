# 2026-10-27 HSSD192 数据扩展交接

**2026-10-28更新：** 24场景2504张共享照片全部完成，192目标按128/32/32冻结；07:57约39个目标完成渲染/编码/解码检查，原PID2204262继续运行。用户随后明确调整推进标准为完整方案相对base的整体收益，后续优先主对照，取消三分支消融必须先通过的要求。见[研究决策](RESEARCH_DECISION_20261028.md)。以下02:51/11:17阶段说明与三分支预算保留为历史记录，后续工作以新决策优先。

11:17 UTC 助手已从渲染断点恢复后台数据准备队列，PID2204262，当前GPU1。原PID1340112于10:27因Blender CUDA非法内存访问退出。恢复后原失败帧44（第45张）已完成，11:17:43确认正在渲染帧45；已完成场景和照片复用，用户无需再次start，只需watch。

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_generalization_20261027/run.sh watch
```

这次执行的是正式跨场景实验前的数据准备与目标codec验收，尚未开始新模型训练。完成状态为`data_qualification_complete_pending_review`；同容量外观/预测几何/权重1三分支的50epoch/6400暴露预算已经预登记，但须先审阅数据质量、真实遮挡分布和新输入缓存。

## 已确定的数据与隔离

- HSSD-hab固定revision4369cb9876214c7fbebcf552eb532380e4d287e4；172个清单/场景/元数据文件共24385531字节已下载，168普通场景、官方125train/43val。
- 保持官方train/val边界，项目选16train/4val/4test场景；每场景最终选8个独立目标，预算128/32/32。测试是公开val隔离子集，不是官方隐藏测试。
- 当前627个候选，按固定照片可见性/几何重复规则在编码前选出192个完整分母；实际192个对象尚未生成。旧三个开发场景及其家族排除。候选目标家族和相同原GLB字节若在另一划分背景中出现，也排除为目标。
- 元数据资格筛选发现一个训练场景只有8个候选家族、一个测试场景14个，依预先固定的至少16家族余量规则，分别换为同官方划分中的下一个哈希场景。只看元数据，未看latent、渲染或模型成绩；原草案保存在planning_history。最终候选最少18个/场景，预算及方法配置未变。
- 场景建筑与全部其他实例原位置保留。共享拍摄由房间格点统一确定，不按目标重新搜索机位；每张2072RGB只渲染一次，所有候选mask同步输出。

## 当前验证与物理存储

23项CPU检查通过，涵盖实际场景隔离、候选家族/内容在其他划分背景中的出现、旧家族排除、输出目录边界、GPU准入、镜像/非均匀缩放与proper相机R。实际Blender API的地面raycast、实例变换及相机投影短检查也通过：实例误差2.38418579e−7，投影误差4.56984242e−5像素。短检查没有加载CUDA模型；不能视为新192对象的完整GPU/codec验收。

SS编码器配置/权重、DINO权重按旧HSSD16/InputAdapt审计hash一次性复制到/data；156项DINO源码一致，home原件保留。正式DINO只从/data本地源码和权重加载，HF保持离线，不回退下载到home。旧SLat encoder和冻结stock decoder复用/data既有模型。

数据根目录：`/data/zjr/pose_aligned_layered_reconstruction/datasets/hssd_generalization_v1/`。raw、converted_glb、texture_cache、scenes照片/mask、objects目标渲染/features/SS/SLat均在此目录。Torch/HF/扩展/Matplotlib/Numba/Blender/TMPDIR均在/data项目cache/tmp。

日志和回执根目录：`/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_generalization_20261027/`。当前3640个GLB原始依赖共2065223048字节（1.923GiB），只是原始模型量；共享照片、150视图目标渲染、DINO特征及latent会明显扩大。开始时/data约5.3TiB可用，原始下载限40GiB、至少100GiB余量；没有将新大产物写回home。

候选计划身份：`edf32b6bf34d46f530cca4cab1e49cad200116f5fd7b97d26c18c9f53c129c4c`。

运行源码身份：`2b25abd4392f11e851aa54764492ee1ca2071775d27e767bab729512427eb76e`。

## 队列如何继续

当前下载→纹理转换（原几何二进制保留）→canonical目标→24场景共享拍摄→照片资格与目标分母冻结→train/val/test各首目标150视图/SS/SLat/坐标回环→其余目标→192张原始/解码mesh三视角对比→汇总。复用已审计的官方兼容HSSD16链路，SS回环IoU至少0.90；任一编码/坐标失败即停止，不删目标后继续训练。

每个GPU阶段单卡串行，一个对象完成后释放模型；空卡优先，无空卡允许至少18GiB可用显存的卡。仅查询驱动，未做分配式显存探测，不抢占其他任务。长任务由用户监视，Ctrl-C退出watch不停止后台任务。同脚本status查看一次、stop停止本队列；失败后修复对应问题再start-prepare，已完成阶段保留。

证据：[候选计划](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_generalization_20261027/candidate_plan.json)、[CPU验收](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_generalization_20261027/preflight.json)、[Blender实际检查](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_generalization_20261027/blender_math_smoke.json)、[编码器迁移](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_generalization_20261027/encoder_storage.json)、[冻结协议](PROTOCOL.md)。

## 2026-10-27 11:18 UTC：共享照片渲染失败与恢复

10:27:27，场景108294816_176710461（第8个场景）在frame44的Cycles着色阶段报`Illegal address in CUDA queue copy_from_device (integrator_shade_surface integrator_sorted_paths_array prefix_sum)`，随后大量`cuMemFree_v2`非法地址是同一CUDA上下文失效后的清理错误。日志没有Out of memory；不能据此判显存不足、场景资产损坏或模型训练失败。底层诱因尚未确定。此时尚未开始目标latent编码或正式模型训练。

失败前7场景712帧全部完成，第8场景44/104帧完成。恢复时检查既有756份帧回执的plan_identity，以及对应20212个RGB/mask文件存在且非空，无异常；这项检查不等同于逐图视觉质量验收。原失败日志保留，未删除照片、修改场景分母或重写已完成阶段回执。

11:17:02使用原start-prepare恢复，自动选择空闲GPU1，新队列PID2204262、Blender PID2204288。运行源码身份仍为2b25abd4392f11e851aa54764492ee1ca2071775d27e767bab729512427eb76e；原渲染器、2072分辨率、16 samples、相机、材质及随机设置均未修改。11:17:43日志确认frame44 complete并进入frame45，证明原失败帧在新进程中能够通过，不能据此保证后续所有帧永不出错。

监视中的`GPU: None`来自失败状态未保存选卡字段，不代表此前没用GPU；原失败时具体GPU无法从当前保留的status追溯。`canonical_assets_complete:627`是上一个写入progress.json的CPU阶段，表示候选资产构造完成；Blender逐帧进度写入当前日志的`SHARED_RENDER`，不会更新该字段。它既不是627个已编码训练对象，也不代表队列一直停在03:26。

证据：[原失败日志](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_generalization_20261027/logs/photographs_108294816_176710461_1793095296732024261.log)、[恢复渲染日志](/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_generalization_20261027/logs/photographs_108294816_176710461_1793099823406567726.log)。所有产物与缓存继续写/data。
