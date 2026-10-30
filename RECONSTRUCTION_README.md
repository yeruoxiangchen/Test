# 多视图遮挡物体重建：代码与实验记录快照

这是Tracker中本条重建研究主线的独立源码快照。主线为TRELLIS.2多视图独立物体基线及Relations192跨物体关系实验；保留LaRI/HSSD实验历史及必要旧源码依赖。

- [Relations192运行与监视](pose_constrained_trellis2/experiments/hssd_relations192_20261029/README.md)
- [原生支持容量修复验收](pose_constrained_trellis2/experiments/hssd_relations192_20261029/TOKEN_SUPPORT_REPAIR_20261029.md)
- [2026-10-30阶段性分析](pose_constrained_trellis2/experiments/hssd_relations192_20261029/INTERIM_REVIEW_20261030.md)
- [本次增量来源与校验](RECONSTRUCTION_UPDATE_20261030.json)
- [既有基线与监视说明](pose_constrained_trellis2/experiments/hssd_mv_baseline_20261028/README.md)
- [原生标签、坐标与真实mesh审阅](pose_constrained_trellis2/experiments/hssd_scene_20261028/REVIEW_20261028.md)
- [后续关系重建设计](pose_constrained_trellis2/SCENE_RELATIONS_PLAN_20261028.md)
- [按时间记录的实验审计](pose_aligned_layered_reconstruction/审计.md)
- [源码文件校验及外部版本](RECONSTRUCTION_SNAPSHOT.json)

## 外部代码

TRELLIS2、ReconViaGen及GENAmodal3D以固定commit的Git子模块记录，保留各自许可。

```bash
git submodule update --init --recursive
git -C ReconViaGen apply ../pose_constrained_trellis2/publication/dependency_patches/reconviagen_geometry.patch
```

ReconViaGen补丁仅包含本重建实验使用的SS支持导出及FlexiCubes索引检查，不含其他应用界面改动。TRELLIS.2原生主线不依赖此旧后端补丁。

## 数据与复现边界

此仓库不包含HSSD/ScanNet等原始数据、模型权重、latent缓存、检查点、渲染图或实验输出。它们保留在服务器/data；文档中的/data和/home路径记录原运行位置，在其他机器上不可直接访问。

冻结实验路径和协议原样保存；换机器需要按数据准备及依赖说明准备环境和路径映射，不应修改旧报告冒充原运行。依赖见各项目requirements及Native12运行环境记录。Relations192的289物体数据构建和检查已完成，正式两臂训练正在进行；最终mesh评估尚未完成。各文档保留其记录时间，最新进度以原服务器监视为准，不能把源码发布当成最终模型效果。

源文件为创建快照时的内容，校验值见JSON。发布在独立分支，未修改原Tracker工作区的分支或暂存区，也没有将其他对话未提交的业务项目纳入快照。
