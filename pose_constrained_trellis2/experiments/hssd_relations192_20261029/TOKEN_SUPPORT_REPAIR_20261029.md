# Relations192 原生支持容量修复（2026-10-29）

本次失败发生在数据编码，尚未开始正式训练。训练邻居柜子`hssd_103997403_171030405_i086`的原生Shape512有10457个token；旧门槛8192触发ValueError。该次GPU峰值分配3673.85MiB，未发生OOM，坐标/有限值检查已通过。

官方`TRELLIS2/configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json`的8192位于dataset配置，`TRELLIS2/trellis2/datasets/structured_latent.py::filter_metadata`用它筛选预训练样本。它不是VAE/flow的固定序列长度。本项目此前把这一筛选门槛沿用到原生编码、训练读取和推理，扩大到完整场景物体时暴露了适用范围问题。

修复保留全部32³有效支持（空间最大32768），三处采用一致容量；不改变模型权重、32通道规范化、分辨率、dual-grid数学或划分。不丢物体、不截token。重复/越界/非有限支持仍报错，OOM仍停止。两臂使用相同规则，固定GT世界变换不变。

旧源码、协议和502份阶段回执归档于[revision目录](/data/zjr/pose_constrained_trellis2/outputs/hssd_relations192_20261029/revisions/shape_support_32768_20261029)。旧回执未改写，由明确的逐回执摘要兼容清单授权复用。原plan保持历史身份；当前有效配置及新源码身份为`429ff02fcc1d87aa9b7da35dd739350c3161cb7f7b05a5508dc1f391841084c1`。旧Native12/MVBase12源码未改。

12项CPU检查通过，包括10457个token无截断读取、旧样本规范化逐值一致、32768完整支持推理、非法数据仍拒绝，以及旧回执篡改拒绝。原7项关系模型行为检查仍通过。

真实GPU检查使用原冻结训练组`hssd_103997403_171030405_i112`，成员为i112和i086，共视帧与分组不改。i086已完成Shape512=10457、Shape1024=45289、SS IoU=1.0、codec CD-L1=0.01152875、F@.02=0.90647070，符合原标签检查阈值。1024编码峰值12669.03MiB、保留14098MiB，低于16GiB分配上限。i112的Shape512为3686个token。已查看i086的GT/codec三视角表面采样图，整体结构和方向相符；该图是采样点对照，不代替最终生成mesh效果评审。

短检查同时提前暴露共享照片条件构建的接口错误：DinoV3FeatureExtractor.to不返回self，旧代码链式赋值导致NoneType不可调用。已改为单独构建extractor再调用to；没有改动DINO特征提取数学。第二次修订和513份已完成回执归档于`revisions/dino_transfer_20261029`，逐回执同时保留两代身份；当前有效源码身份更新为`ab9986eb4bc2ddbb8ebf9c136942938c7352c0ea86289d1356f21105a5e6378c`。12项检查再次通过。

05:49:59 UTC实际验收：共享照片条件和GT投影检查通过，两臂各2步Shape训练全部完成；两臂首步loss均为0.13337275385856628，梯度有限。关系分支梯度范数为0.000162576和0.000613047，确认有效反传。训练峰值分配independent=7057.30MiB、relational=7106.81MiB；本次codec与训练全部阶段中最大分配12669.03MiB，未提高16GiB限额。旧MVBase源码身份复核一致。证据：[support_probe/verification.json](/data/zjr/pose_constrained_trellis2/outputs/hssd_relations192_20261029/support_probe/verification.json)。

验收后已启动完整all队列PID3501851，仍固定GPU5，按原192主要目标/含邻居289标签物体继续执行；已完成阶段自动复用，监视命令不变。正式训练从官方权重按原预算开始，不续接短检查权重。本检查不能证明新模型质量提升，也不能保证所有可能的32768-token输入均可在16GiB内执行。
